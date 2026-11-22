from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from cfg import Config
import data as D
import nets as N
import physics as PH


def _to_jax(data: dict) -> dict:
    return {k: jnp.asarray(v) for k, v in data.items()}


def _branch_input(jd, batch_idx, kind: str, n_grid: int):
    return N.branch_input(jd["a"][batch_idx], jd["lam"][batch_idx],
                          jd["c"][batch_idx], kind, n_grid)


class Trainer:
    def __init__(self, mode: str, cfg: Config, data: dict, seed: Optional[int] = None):
        if mode not in ("deeponet", "pi-deeponet", "sep-pi-deeponet"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.cfg = cfg
        self.n = cfg.grid.n
        self.half = cfg.grid.half
        self.margin = cfg.physics.margin
        self.family = cfg.data.family
        self.c_scale = cfg.data.c_scale
        self.bc_mode = cfg.physics.bc_mode
        if self.bc_mode not in ("soft", "hard_anti", "seed",
                                "profile", "hard_both", "adf_aq"):
            raise ValueError(f"unknown bc_mode {self.bc_mode!r}")
        self.positive_x = bool(cfg.physics.positive_x)
        self.negative_x = bool(cfg.physics.negative_x)
        if self.positive_x and self.negative_x:
            raise ValueError("positive_x and negative_x are mutually exclusive")
        self.seed = cfg.train.seed if seed is None else seed

        self.modules, self.params = N.build_model(
            mode, cfg.model, self.n, seed=self.seed)
        self.n_params = N.count_params(self.params)


        n_steps = max(1, (cfg.data.n_train // cfg.train.batch_size)
                      * cfg.train.epochs)
        if cfg.train.lr_schedule == "constant":
            schedule = cfg.train.lr
        elif cfg.train.lr_schedule == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=cfg.train.lr, decay_steps=n_steps, alpha=0.02)
        else:
            raise ValueError(f"unknown lr_schedule {cfg.train.lr_schedule!r}")
        adam = optax.adamw(learning_rate=schedule,
                           weight_decay=cfg.train.weight_decay)
        if cfg.train.clip_norm > 0:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.train.clip_norm), adam)
        else:
            self.optimizer = adam
        self.opt_state = self.optimizer.init(self.params)
        self.jd = _to_jax(data)
        self.data_np = data
        self.online_truth = bool(cfg.physics.online_truth) and mode == "pi-deeponet"
        self._kernel_cache = {}


        self.full_xy_np, self.inside_np = D.full_grid_xy(
            data["grid"], positive_x=self.positive_x,
            negative_x=self.negative_x)
        self.full_xy = jnp.asarray(self.full_xy_np)
        self.inside = jnp.asarray(self.inside_np)


        pool_margin = self.margin if cfg.physics.interior_anchors else 0.0
        self.pool_np = D.interior_pool(data["grid"], pool_margin,
                                       positive_x=self.positive_x,
                                       negative_x=self.negative_x)
        if self.pool_np.shape[0] == 0:
            raise RuntimeError("empty anchor pool; decrease margin")
        self.pool = jnp.asarray(self.pool_np)


        if cfg.physics.fixed_colloc:
            g = np.asarray(data["grid"])


            pool_c = D.interior_pool(g, cfg.physics.colloc_margin,
                                     positive_x=self.positive_x,
                                     negative_x=self.negative_x)
            xy = np.stack([g[pool_c[:, 0]], g[pool_c[:, 1]]], axis=1)
            if cfg.physics.n_colloc < len(xy):
                rng = np.random.default_rng(self.seed + 123)
                sel = rng.choice(len(xy), cfg.physics.n_colloc, replace=False)
                xy = xy[sel]
            self.fixed_xy = jnp.asarray(xy.astype(np.float32))
            self.n_colloc = int(xy.shape[0])
        else:
            self.fixed_xy = None
            self.n_colloc = int(cfg.physics.n_colloc)

        self.history = dict(total=[], pde=[], data=[], bc_anti=[], bc_diag=[],
                            test_rel_l2=[], eval_epochs=[], epoch_time=[],
                            truth_time=[])
        self._build_jits()


    def _net_and_derivs(self, params, b_embed, xy):
        if self.mode in ("deeponet", "pi-deeponet"):
            t, tx, ty, txx, tyy = PH.trunk_derivs_2d(
                self.modules["trunk"].apply, params["trunk"], xy)
            bias = params["bias"]
            return ((t @ b_embed.T + bias).T, (tx @ b_embed.T).T,
                    (ty @ b_embed.T).T, (txx @ b_embed.T).T,
                    (tyy @ b_embed.T).T)

        x = xy[:, 0:1]
        y = xy[:, 1:2]
        txv, tx1, tx2 = PH.trunk_derivs_1d(
            self.modules["trunk_x"].apply, params["trunk_x"], x)
        tyv, ty1, ty2 = PH.trunk_derivs_1d(
            self.modules["trunk_y"].apply, params["trunk_y"], y)
        bias = params["bias"]
        net = jnp.einsum("nr,nr,br->nb", txv, tyv, b_embed) + bias
        net_x = jnp.einsum("nr,nr,br->nb", tx1, tyv, b_embed)
        net_y = jnp.einsum("nr,nr,br->nb", txv, ty1, b_embed)
        net_xx = jnp.einsum("nr,nr,br->nb", tx2, tyv, b_embed)
        net_yy = jnp.einsum("nr,nr,br->nb", txv, ty2, b_embed)
        return net.T, net_x.T, net_y.T, net_xx.T, net_yy.T

    def _net_value(self, params, b_embed, xy):
        if self.mode in ("deeponet", "pi-deeponet"):
            t = PH.trunk_value_2d(
                self.modules["trunk"].apply, params["trunk"], xy)
            return (t @ b_embed.T + params["bias"]).T
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        txv = PH.trunk_value_1d(
            self.modules["trunk_x"].apply, params["trunk_x"], x)
        tyv = PH.trunk_value_1d(
            self.modules["trunk_y"].apply, params["trunk_y"], y)
        net = jnp.einsum("nr,nr,br->nb", txv, tyv, b_embed) + params["bias"]
        return net.T

    def _apply_bc(self, params, net, net_x, net_y, net_xx, net_yy, xy, p_batch):
        if self.bc_mode != "profile":
            return PH.apply_parameterization(
                self.bc_mode, net, net_x, net_y, net_xx, net_yy,
                xy, p_batch, self.family, self.c_scale)

        x = xy[:, 0]
        y = xy[:, 1]
        m, m_x, m_y, m_xx, m_yy = PH.mask_terms(x, y)
        s, s_x, s_y, s_xx, s_yy = PH.profile_seed_terms(
            self.family, x, y, p_batch,
            self.modules["profile"].apply, params["profile"], self.c_scale)
        k = s + m[None, :] * net
        k_x = s_x + m_x[None, :] * net + m[None, :] * net_x
        k_y = s_y + m_y[None, :] * net + m[None, :] * net_y
        k_xx = (s_xx + m_xx[None, :] * net
                + 2.0 * m_x[None, :] * net_x + m[None, :] * net_xx)
        k_yy = (s_yy + m_yy[None, :] * net
                + 2.0 * m_y[None, :] * net_y + m[None, :] * net_yy)
        return k, k_x, k_y, k_xx, k_yy


    def _solve_kernel(self, idx: int) -> np.ndarray:
        key = int(idx)
        cached = self._kernel_cache.get(key)
        if cached is None:
            a = np.asarray(self.data_np["a"][key], dtype=np.float64)
            lam = np.asarray(self.data_np["lam"][key], dtype=np.float64)
            c = float(self.data_np["c"][key])
            cached = D.solve_kernel_exact(
                self.cfg.grid.M, a, lam, c).astype(np.float32)
            self._kernel_cache[key] = cached
        return cached

    def _truth_batch_np(self, batch_idx) -> np.ndarray:
        batch_idx = np.asarray(batch_idx, dtype=np.int64)
        if self.online_truth:
            ks = [self._solve_kernel(i) for i in batch_idx]
            return np.stack(ks).astype(np.float32)
        return np.asarray(self.data_np["kernels"])[batch_idx]

    def _truth_np(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if "kernels" in self.data_np:
            return np.asarray(self.data_np["kernels"])[idx]
        ks = [self._solve_kernel(i) for i in idx]
        return np.stack(ks).astype(np.float32)

    def _pde_and_data_pred(self, params, batch_idx, xy, ai, aj, xb, k_batch):
        b_embed = self.modules["branch"].apply(
            params["branch"],
            _branch_input(self.jd, batch_idx, self.cfg.model.branch_kind, self.n))
        p_batch = self.jd["p"][batch_idx]

        net, net_x, net_y, net_xx, net_yy = self._net_and_derivs(
            params, b_embed, xy)
        k, k_x, k_y, k_xx, k_yy = self._apply_bc(
            params, net, net_x, net_y, net_xx, net_yy, xy, p_batch)
        R = PH.pde_residual(k, k_y, k_xx, k_yy, xy, p_batch,
                            self.family, self.c_scale)
        if (self.cfg.physics.pde_norm == "fixed_seed"
                and self.bc_mode == "adf_aq"):


            z = jnp.zeros_like(net)
            sk, _, sky, skxx, skyy = PH.apply_parameterization(
                "adf_aq", z, z, z, z, z, xy, p_batch,
                self.family, self.c_scale)
            Rs = PH.pde_residual(sk, sky, skxx, skyy, xy, p_batch,
                                 self.family, self.c_scale)
            num = jnp.mean(R ** 2, axis=1)
            den = jnp.mean(Rs ** 2, axis=1) + 1e-12
            loss_pde = jnp.mean(num / den)
        else:
            loss_pde = PH.relative_residual(
                R, k, k_y, k_xx, k_yy, xy, p_batch, self.family, self.c_scale,
                norm=self.cfg.physics.pde_norm)


        if self.bc_mode in ("seed", "profile", "hard_both") or xb.shape[0] == 0:
            loss_anti = jnp.zeros(())
            loss_diag = jnp.zeros(())
        else:
            xy_a = jnp.stack([xb, -xb], axis=1)
            net_a = self._net_value(params, b_embed, xy_a)
            if self.bc_mode == "soft":
                k_a = net_a
            else:
                k_a = jnp.zeros_like(net_a)
            xy_diag = jnp.stack([xb, xb], axis=1)
            net_dg, nx_dg, ny_dg, nxx_dg, nyy_dg = self._net_and_derivs(
                params, b_embed, xy_diag)
            k_dg, kx_dg, ky_dg, _, _ = self._apply_bc(
                params, net_dg, nx_dg, ny_dg, nxx_dg, nyy_dg,
                xy_diag, p_batch)
            loss_anti, loss_diag = PH.bc_losses(
                k_a, k_dg, kx_dg, ky_dg, xb, p_batch,
                self.family, self.c_scale)


        if ai.shape[0] == 0:
            loss_data = jnp.zeros(())
            return loss_pde, loss_data, loss_anti, loss_diag

        xd = self.jd["grid"][ai]
        yd = self.jd["grid"][aj]
        xy_d = jnp.stack([xd, yd], axis=1)
        net_d = self._net_value(params, b_embed, xy_d)
        if self.bc_mode == "soft":
            k_d = net_d
        elif self.bc_mode == "hard_anti":
            k_d = (xd + yd)[None, :] * net_d
        elif self.bc_mode == "profile":
            m_d, _, _, _, _ = PH.mask_terms(xd, yd)
            s_d, _, _, _, _ = PH.profile_seed_terms(
                self.family, xd, yd, p_batch,
                self.modules["profile"].apply, params["profile"], self.c_scale)
            k_d = s_d + m_d[None, :] * net_d
        elif self.bc_mode == "hard_both":
            (s_d, _, _, _, _, m_d, _, _, _, _) = PH.hard_both_terms(
                self.family, xd, yd, p_batch, self.c_scale)
            k_d = s_d + m_d[None, :] * net_d
        elif self.bc_mode == "adf_aq":
            k_d, _, _ = PH.adf_aq_value(xd, yd, p_batch, net_d)
        else:
            m_d, _, _, _, _ = PH.mask_terms(xd, yd)
            s_d, _, _, _ = PH.seed_terms(self.family, xd, yd, p_batch, self.c_scale)
            k_d = s_d + m_d[None, :] * net_d

        k_true = k_batch[:, ai, aj]
        k_scale = jnp.mean(k_batch ** 2, axis=(1, 2))
        num = jnp.mean((k_d - k_true) ** 2, axis=1)
        den = jnp.mean(k_true ** 2, axis=1) + 0.01 * k_scale + 1e-12
        loss_data = jnp.mean(num / den)
        return loss_pde, loss_data, loss_anti, loss_diag

    def _full_data_loss(self, params, batch_idx, k_batch):
        b_embed = self.modules["branch"].apply(
            params["branch"],
            _branch_input(self.jd, batch_idx, self.cfg.model.branch_kind, self.n))
        if self.mode in ("deeponet", "pi-deeponet"):
            t = PH.trunk_value_2d(
                self.modules["trunk"].apply, params["trunk"], self.full_xy)
            net = (t @ b_embed.T + params["bias"]).T
        else:
            g = self.jd["grid"]
            xg = g[:, None]
            txv = PH.trunk_value_1d(
                self.modules["trunk_x"].apply, params["trunk_x"], xg)
            tyv = PH.trunk_value_1d(
                self.modules["trunk_y"].apply, params["trunk_y"], xg)
            net = jnp.einsum("ir,jr,br->ijb", txv, tyv, b_embed) + params["bias"]
            net = net.reshape(-1, b_embed.shape[0]).T

        k_true = k_batch.reshape(batch_idx.shape[0], -1)
        if self.positive_x:
            mask = self.inside_np.reshape(-1)
            net = net[:, mask]
            k_true = k_true[:, mask]
        num = jnp.mean((net - k_true) ** 2, axis=1)
        den = jnp.mean(k_true ** 2, axis=1) + 1e-12
        return jnp.mean(num / den)


    def _build_jits(self):
        cfg = self.cfg
        mode = self.mode
        n_data = cfg.physics.n_data
        n_colloc = cfg.physics.n_colloc
        half = self.half
        margin = self.margin
        fixed_xy = self.fixed_xy
        pool_size = self.pool.shape[0]
        w_pde = cfg.physics.w_pde
        w_data = cfg.physics.w_data
        w_bc_anti = cfg.physics.w_bc_anti
        w_bc_diag = cfg.physics.w_bc_diag
        n_bc = cfg.physics.n_bc
        bc_mode = self.bc_mode

        @jax.jit
        def train_step(params, opt_state, batch_idx, key, k_batch):
            if mode == "deeponet":
                def loss_fn(params):
                    loss = self._full_data_loss(params, batch_idx, k_batch)
                    z = jnp.zeros(())
                    return loss, (loss, z, z, z)
            else:
                key_c, key_d, key_b = jax.random.split(key, 3)
                if fixed_xy is None:
                    xy = PH.sample_colloc(key_c, n_colloc, half, margin,
                                          positive_x=self.positive_x,
                                          negative_x=self.negative_x)
                else:
                    xy = fixed_xy
                if n_data > 0:
                    sel = jax.random.choice(key_d, pool_size,
                                            shape=(n_data,), replace=True)
                    ai = self.pool[sel, 0]
                    aj = self.pool[sel, 1]
                else:
                    ai = jnp.zeros((0,), dtype=jnp.int32)
                    aj = jnp.zeros((0,), dtype=jnp.int32)
                if bc_mode in ("seed", "profile", "hard_both"):
                    xb = jnp.zeros((0,))
                else:
                    xb = PH.sample_boundary(
                        key_b, n_bc, half, positive_x=self.positive_x,
                        negative_x=self.negative_x)

                def loss_fn(params):
                    lp, ld, la, ldiag = self._pde_and_data_pred(
                        params, batch_idx, xy, ai, aj, xb, k_batch)
                    loss = (w_pde * lp + w_data * ld
                            + w_bc_anti * la + w_bc_diag * ldiag)
                    return loss, (lp, ld, la, ldiag)

            (loss, (lp, ld, la, ldiag)), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(params)
            updates, opt_state = self.optimizer.update(
                grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, lp, ld, la, ldiag

        @jax.jit
        def predict_grid(params, batch_idx):
            b_embed = self.modules["branch"].apply(
                params["branch"],
                _branch_input(self.jd, batch_idx, self.cfg.model.branch_kind, self.n))
            p_batch = self.jd["p"][batch_idx]
            if self.mode in ("deeponet", "pi-deeponet"):
                t = PH.trunk_value_2d(
                    self.modules["trunk"].apply, params["trunk"], self.full_xy)
                net = (t @ b_embed.T + params["bias"]).T
            else:
                g = self.jd["grid"]
                xg = g[:, None]
                txv = PH.trunk_value_1d(
                    self.modules["trunk_x"].apply, params["trunk_x"], xg)
                tyv = PH.trunk_value_1d(
                    self.modules["trunk_y"].apply, params["trunk_y"], xg)
                net = jnp.einsum("ir,jr,br->ijb", txv, tyv, b_embed) + params["bias"]
                net = net.reshape(-1, b_embed.shape[0]).T

            if self.mode == "deeponet" or self.bc_mode == "soft":
                k = net
            elif self.bc_mode == "hard_anti":
                x = self.full_xy[:, 0]
                y = self.full_xy[:, 1]
                k = (x + y)[None, :] * net
            elif self.bc_mode == "profile":
                x = self.full_xy[:, 0]
                y = self.full_xy[:, 1]
                m, _, _, _, _ = PH.mask_terms(x, y)
                s, _, _, _, _ = PH.profile_seed_terms(
                    self.family, x, y, p_batch,
                    self.modules["profile"].apply, params["profile"],
                    self.c_scale)
                k = s + m[None, :] * net
            elif self.bc_mode == "hard_both":
                x = self.full_xy[:, 0]
                y = self.full_xy[:, 1]
                (s, _, _, _, _, m, _, _, _, _) = PH.hard_both_terms(
                    self.family, x, y, p_batch, self.c_scale)
                k = s + m[None, :] * net
            elif self.bc_mode == "adf_aq":
                x = self.full_xy[:, 0]
                y = self.full_xy[:, 1]
                k, _, _ = PH.adf_aq_value(x, y, p_batch, net)
            else:
                x = self.full_xy[:, 0]
                y = self.full_xy[:, 1]
                m, _, _, _, _ = PH.mask_terms(x, y)
                s, _, _, _ = PH.seed_terms(self.family, x, y, p_batch, self.c_scale)
                k = s + m[None, :] * net

            return k.reshape(batch_idx.shape[0], self.n, self.n)

        @jax.jit
        def pde_diag(params, batch_idx, xy):
            b_embed = self.modules["branch"].apply(
                params["branch"],
                _branch_input(self.jd, batch_idx, self.cfg.model.branch_kind, self.n))
            p_batch = self.jd["p"][batch_idx]
            net, net_x, net_y, net_xx, net_yy = self._net_and_derivs(
                params, b_embed, xy)
            if self.mode == "deeponet":
                k, k_y, k_xx, k_yy = net, net_y, net_xx, net_yy
            else:
                k, _, k_y, k_xx, k_yy = self._apply_bc(
                    params, net, net_x, net_y, net_xx, net_yy, xy, p_batch)
            R = PH.pde_residual(k, k_y, k_xx, k_yy, xy, p_batch,
                                self.family, self.c_scale)
            diagnostic_norm = ("terms" if self.cfg.physics.pde_norm == "fixed_seed"
                               else self.cfg.physics.pde_norm)
            rel = PH.relative_residual(
                R, k, k_y, k_xx, k_yy, xy, p_batch, self.family, self.c_scale,
                norm=diagnostic_norm)
            return rel, jnp.mean(k ** 2)

        @jax.jit
        def pde_weights(params):
            return jnp.asarray(cfg.physics.w_pde)

        self.train_step = train_step
        self.predict_grid_jit = predict_grid
        self.pde_diag_jit = pde_diag
        self.pde_weights = pde_weights


    def predict_full(self, params=None) -> np.ndarray:
        params = self.params if params is None else params
        idx = jnp.arange(self.jd["kernels"].shape[0])
        out = self.predict_grid_jit(params, idx)
        return np.asarray(out)

    def test_metrics(self, params=None, idx=None, batch: int = 25) -> dict:
        params = self.params if params is None else params
        if idx is None:
            idx = np.asarray(self.jd["test_idx"])
        idx = np.asarray(idx)
        preds = []
        for start in range(0, len(idx), batch):
            sub = jnp.asarray(idx[start:start + batch])
            preds.append(np.asarray(self.predict_grid_jit(params, sub)))
        pred = np.concatenate(preds, axis=0)
        true = self._truth_np(idx)
        mask = self.inside_np
        err = (pred - true)[:, mask]
        tgt = true[:, mask]
        num = np.sum(err ** 2, axis=1)
        den = np.sum(tgt ** 2, axis=1) + 1e-12
        rel = np.sqrt(num / den)
        return dict(
            rel_l2_mean=float(rel.mean()),
            rel_l2_median=float(np.median(rel)),
            rel_l2_max=float(rel.max()),
            rel_l2_global=float(np.sqrt(np.sum(num) / (np.sum(den) + 1e-12))),
            abs_err_max=float(np.abs(err).max()),
            abs_err_mean=float(np.abs(err).mean()),
            pred=pred,
            true=true,
            idx=idx,
        )

    def boundary_metrics(self, pred: np.ndarray, idx: np.ndarray) -> dict:
        Nn = self.n
        k = pred
        anti = k[:, np.arange(Nn), Nn - 1 - np.arange(Nn)]
        diag = k[:, np.arange(Nn), np.arange(Nn)]
        h = self.cfg.grid.h

        ddiag = (diag[:, 2:] - diag[:, :-2]) / (2.0 * h)
        xg = np.asarray(self.jd["grid"])
        a_full, a1_full, _, lam_full, c_np = D.family_coeffs_np(
            np.asarray(self.jd["p"])[idx], xg, self.cfg.data)


        i_int = np.arange(1, Nn - 1)
        if self.positive_x:
            i_int = i_int[xg[i_int] > 0.0]
        elif self.negative_x:
            i_int = i_int[xg[i_int] < 0.0]
        lhs = 2.0 * a_full[:, i_int] * ddiag[:, i_int - 1]
        rhs = (-a1_full[:, i_int] * diag[:, i_int]
               + lam_full[:, i_int] + c_np[:, None])
        return dict(
            anti_diag_max=float(np.abs(anti).max()),
            diag_bc_rel=float(np.linalg.norm(lhs - rhs)
                              / (np.linalg.norm(rhs) + 1e-12)),
        )

    def pde_metrics(self, params=None, idx=None, n_points: int = 512) -> dict:
        params = self.params if params is None else params
        if idx is None:
            idx = np.asarray(self.jd["test_idx"][:10])
        idx = np.asarray(idx)
        rng = np.random.default_rng(self.seed + 777)
        xy = jnp.asarray(D.sample_colloc_np(
            rng, n_points, self.half, self.margin,
            positive_x=self.positive_x, negative_x=self.negative_x))
        rel, k2 = self.pde_diag_jit(params, jnp.asarray(idx), xy)
        return dict(pde_rel_mean=float(jnp.mean(rel)),
                    pde_rel_max=float(jnp.max(rel)),
                    pde_k_rms=float(jnp.sqrt(jnp.mean(k2))))


    def fit(self, verbose: bool = True) -> dict:
        cfg = self.cfg
        key = jax.random.PRNGKey(self.seed + 1)
        train_idx = np.array(np.asarray(self.jd["train_idx"]), copy=True)
        rng = np.random.default_rng(self.seed + 2)
        bs = cfg.train.batch_size


        n_steps_per_epoch = max(1, len(train_idx) // bs)
        total_steps = cfg.train.epochs * n_steps_per_epoch


        params0 = jax.tree_util.tree_map(lambda x: x, self.params)
        opt0 = jax.tree_util.tree_map(lambda x: x, self.opt_state)
        need_truth = self.mode == "deeponet" or cfg.physics.n_data > 0
        warm_batch = jnp.asarray(train_idx[:bs])
        if need_truth:
            warm_k = jnp.asarray(self._truth_batch_np(train_idx[:bs]))
        else:
            warm_k = jnp.zeros((warm_batch.shape[0], 1, 1), dtype=jnp.float32)
        t_compile0 = time.perf_counter()
        for _ in range(max(1, cfg.train.warmup_steps)):
            key, sub = jax.random.split(key)
            self.params, self.opt_state, *_ = self.train_step(
                self.params, self.opt_state, warm_batch, sub, warm_k)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), self.params)
        compile_time = time.perf_counter() - t_compile0
        self.params, self.opt_state = params0, opt0
        key = jax.random.PRNGKey(self.seed + 1)

        t0 = time.perf_counter()
        for ep in range(cfg.train.epochs):
            ep_t0 = time.perf_counter()
            rng.shuffle(train_idx)
            tot = pde = dat = bca = bcd = 0.0
            ep_truth = 0.0
            for start in range(0, n_steps_per_epoch * bs, bs):
                batch_np = train_idx[start:start + bs]
                batch = jnp.asarray(batch_np)
                if need_truth:
                    t_truth0 = time.perf_counter()
                    k_batch = jnp.asarray(self._truth_batch_np(batch_np))
                    ep_truth += time.perf_counter() - t_truth0
                else:
                    k_batch = jnp.zeros((batch.shape[0], 1, 1),
                                        dtype=jnp.float32)
                key, sub = jax.random.split(key)
                self.params, self.opt_state, loss, lp, ld, la, ldiag = self.train_step(
                    self.params, self.opt_state, batch, sub, k_batch)
                tot += float(loss)
                pde += float(lp)
                dat += float(ld)
                bca += float(la)
                bcd += float(ldiag)
            tot /= n_steps_per_epoch
            pde /= n_steps_per_epoch
            dat /= n_steps_per_epoch
            bca /= n_steps_per_epoch
            bcd /= n_steps_per_epoch
            epoch_time = time.perf_counter() - ep_t0

            self.history["total"].append(tot)
            self.history["pde"].append(pde)
            self.history["data"].append(dat)
            self.history["bc_anti"].append(bca)
            self.history["bc_diag"].append(bcd)
            self.history["epoch_time"].append(epoch_time)
            self.history["truth_time"].append(ep_truth)

            if (ep + 1) % cfg.train.eval_every == 0 or ep == cfg.train.epochs - 1:
                sub_idx = np.asarray(self.jd["test_idx"])[:cfg.train.eval_samples]
                m = self.test_metrics(idx=sub_idx, batch=min(25, len(sub_idx)))
                self.history["test_rel_l2"].append(m["rel_l2_mean"])
                self.history["eval_epochs"].append(ep + 1)
                if verbose:
                    print(f"  ep {ep+1:4d}/{cfg.train.epochs} "
                          f"loss={tot:.4e} pde={pde:.4e} data={dat:.4e} "
                          f"test_relL2={m['rel_l2_mean']:.4e} "
                          f"({epoch_time:.2f}s, truth {ep_truth:.2f}s)")
            elif verbose and (ep + 1) % cfg.train.log_every == 0:
                print(f"  ep {ep+1:4d}/{cfg.train.epochs} "
                      f"loss={tot:.4e} pde={pde:.4e} data={dat:.4e} "
                      f"({epoch_time:.2f}s, truth {ep_truth:.2f}s)")

        train_time = time.perf_counter() - t0
        self.history["train_time"] = train_time
        self.history["compile_time"] = compile_time
        self.history["ms_per_step"] = 1000.0 * train_time / max(1, total_steps)
        self.history["total_steps"] = total_steps
        return self.history


    def save(self, outdir: str, tag: str) -> str:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{tag}_params.msgpack")
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(self.params))
        hist_path = os.path.join(outdir, f"{tag}_history.npz")
        np.savez(hist_path, **{k: np.asarray(v) for k, v in self.history.items()
                               if isinstance(v, (list, np.ndarray))})
        return path

