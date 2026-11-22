from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from cfg import ModelConfig

_ACTS = {
    "tanh": nn.tanh,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "sigmoid": nn.sigmoid,
    "sin": jnp.sin,
}


def get_act(name: str):
    if name not in _ACTS:
        raise ValueError(f"unknown activation {name!r}; choose from {list(_ACTS)}")
    return _ACTS[name]


N_GLOBAL = 11


class BranchNet(nn.Module):
    cfg: ModelConfig
    n_grid: int = 0

    @nn.compact
    def __call__(self, x):
        cfg = self.cfg
        act = get_act(cfg.branch_activation)

        if cfg.branch_kind == "cnn":

            x = jnp.transpose(x, (0, 2, 3, 1))
            x = jnp.stack([
                x[..., 0] / cfg.input_scale_a,
                x[..., 1] / cfg.input_scale_lam,
                x[..., 2] / cfg.input_scale_c,
            ], axis=-1)
            x = act(nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2),
                            padding="VALID")(x))
            x = act(nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2),
                            padding="VALID")(x))
            x = x.reshape(x.shape[0], -1)
            x = act(nn.Dense(1028)(x))
            return nn.Dense(
                cfg.feature_dim,
                kernel_init=nn.initializers.normal(stddev=cfg.branch_out_scale))(x)


        n = self.n_grid or ((x.shape[-1] - 1) // 2)
        a = x[..., :n] / cfg.input_scale_a
        lam = x[..., n:2 * n] / cfg.input_scale_lam
        c = x[..., 2 * n:2 * n + 1] / cfg.input_scale_c
        extra = x[..., 2 * n + 1:]
        z = jnp.concatenate([a, lam, c], axis=-1)
        if extra.shape[-1] > 0:

            z = jnp.concatenate([z, extra / cfg.input_scale_a], axis=-1)
        for h in cfg.branch_layers:
            z = act(nn.Dense(h)(z))
        return nn.Dense(
            cfg.feature_dim,
            kernel_init=nn.initializers.normal(stddev=cfg.branch_out_scale))(z)


class TrunkNet2D(nn.Module):
    cfg: ModelConfig
    fourier_b: Any = None

    @nn.compact
    def __call__(self, xy):
        cfg = self.cfg
        act = get_act(cfg.activation)
        x = xy
        if self.fourier_b is not None:
            z = x @ self.fourier_b.T
            x = jnp.concatenate([jnp.cos(2.0 * np.pi * z),
                                 jnp.sin(2.0 * np.pi * z)], axis=-1)
        for h in cfg.trunk_layers:
            x = act(nn.Dense(h)(x))
        return nn.Dense(cfg.feature_dim)(x)


class TrunkNet1D(nn.Module):
    cfg: ModelConfig
    fourier_b: Any = None

    @nn.compact
    def __call__(self, x):
        cfg = self.cfg
        act = get_act(cfg.activation)
        z = x
        if self.fourier_b is not None:
            zz = z @ self.fourier_b.T
            z = jnp.concatenate([jnp.cos(2.0 * np.pi * zz),
                                 jnp.sin(2.0 * np.pi * zz)], axis=-1)
        for h in cfg.trunk_layers:
            z = act(nn.Dense(h)(z))
        return nn.Dense(cfg.feature_dim)(z)


class ProfileNet(nn.Module):
    cfg: ModelConfig
    hidden: int = 32

    @nn.compact
    def __call__(self, z):
        t = z[..., 0:1]
        act = get_act(self.cfg.activation)
        h = act(nn.Dense(self.hidden)(z))
        h = act(nn.Dense(self.hidden)(h))
        u = nn.Dense(1, kernel_init=nn.initializers.normal(stddev=0.01))(h)
        S = 3.0 * t ** 2 - 2.0 * t ** 3
        return S + t * (1.0 - t) ** 2 * u


def _fourier_matrix(cfg: ModelConfig, seed: int, dim: int = 2):
    if cfg.fourier_m <= 0:
        return None
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 900 + dim)
    return (jax.random.normal(key, (cfg.fourier_m, dim), dtype=jnp.float32)
            * cfg.fourier_sigma)


def build_model(mode: str, cfg: ModelConfig, n_grid: int, seed: int = 0):
    branch = BranchNet(cfg=cfg, n_grid=n_grid)
    key = jax.random.PRNGKey(seed)

    if cfg.branch_kind == "mlp":
        branch_example = jnp.zeros((1, 2 * n_grid + 1 + N_GLOBAL))
    elif cfg.branch_kind == "cnn":
        branch_example = jnp.zeros((1, 3, n_grid, n_grid))
    else:
        raise ValueError(f"unknown branch_kind {cfg.branch_kind!r}")

    bparams = branch.init(key, branch_example)
    modules: Dict[str, Any] = {"branch": branch}
    params: Dict[str, Any] = {"branch": bparams, "bias": jnp.zeros(())}

    if mode != "deeponet":
        profile = ProfileNet(cfg=cfg)
        params["profile"] = profile.init(jax.random.fold_in(key, 4),
                                         jnp.zeros((1, 2)))
        modules["profile"] = profile

    if mode in ("deeponet", "pi-deeponet"):
        trunk = TrunkNet2D(cfg=cfg, fourier_b=_fourier_matrix(cfg, seed))
        tparams = trunk.init(jax.random.fold_in(key, 1), jnp.zeros((1, 2)))
        modules["trunk"] = trunk
        params["trunk"] = tparams
    elif mode == "sep-pi-deeponet":
        fb = _fourier_matrix(cfg, seed, dim=1)
        trunk_x = TrunkNet1D(cfg=cfg, fourier_b=fb)
        trunk_y = TrunkNet1D(cfg=cfg, fourier_b=fb)
        tparams_x = trunk_x.init(jax.random.fold_in(key, 2), jnp.zeros((1,)))
        tparams_y = trunk_y.init(jax.random.fold_in(key, 3), jnp.zeros((1,)))
        modules["trunk_x"] = trunk_x
        modules["trunk_y"] = trunk_y
        params["trunk_x"] = tparams_x
        params["trunk_y"] = tparams_y
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return modules, params


def count_params(params) -> int:
    return int(sum(jnp.size(x) for x in jax.tree_util.tree_leaves(params)))


def branch_input(a_batch, lam_batch, c_batch, kind: str, n_grid: int):
    if kind == "mlp":
        n = a_batch.shape[-1]
        glob = jnp.stack([
            a_batch[:, n // 2], lam_batch[:, n // 2], c_batch,
            jnp.mean(a_batch, axis=1), jnp.std(a_batch, axis=1),
            jnp.min(a_batch, axis=1), jnp.max(a_batch, axis=1),
            jnp.mean(lam_batch, axis=1), jnp.std(lam_batch, axis=1),
            jnp.min(lam_batch, axis=1), jnp.max(lam_batch, axis=1),
        ], axis=-1)
        return jnp.concatenate([a_batch, lam_batch, c_batch[:, None], glob],
                               axis=-1)
    if kind == "cnn":
        a_img = jnp.repeat(a_batch[:, :, None], n_grid, axis=2)
        l_img = jnp.repeat(lam_batch[:, :, None], n_grid, axis=2)
        c_img = jnp.broadcast_to(c_batch[:, None, None],
                                 (a_batch.shape[0], n_grid, n_grid))
        return jnp.stack([a_img, l_img, c_img], axis=1)
    raise ValueError(f"unknown branch kind {kind!r}")

