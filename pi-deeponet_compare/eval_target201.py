from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import jax.numpy as jnp
from flax import serialization

import data as D
import nets as N
import physics as PH
from cfg import config_for_mode
from solveIntegralFD import solveIntegralFD


def predict_201(cfg, modules, params, a_train, lam_train, c, p_test):
    b = modules["branch"].apply(
        params["branch"],
        N.branch_input(jnp.asarray(a_train), jnp.asarray(lam_train),
                       jnp.asarray(c), cfg.model.branch_kind, cfg.grid.n))
    g = np.linspace(-0.5, 0.5, 201)
    X, Y = np.meshgrid(g, g, indexing="ij")
    xy = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    if cfg.model.branch_kind == "cnn":
        pass
    if "trunk" in params:
        t = PH.trunk_value_2d(modules["trunk"].apply, params["trunk"], xy)
        net = (t @ b.T + params["bias"]).T
    else:
        gx = jnp.asarray(g)[:, None]
        tx = PH.trunk_value_1d(modules["trunk_x"].apply, params["trunk_x"], gx)
        ty = PH.trunk_value_1d(modules["trunk_y"].apply, params["trunk_y"], gx)
        net = (jnp.einsum("ir,jr,br->ijb", tx, ty, b)
               + params["bias"]).reshape(201, 201, b.shape[0])
        net = net.transpose(2, 0, 1).reshape(b.shape[0], -1)
    if cfg._mode == "deeponet":
        k = net
    elif cfg.physics.bc_mode == "adf_aq":
        k, _, _ = PH.adf_aq_value(X.ravel(), Y.ravel(),
                                  jnp.asarray(p_test), net)
    else:
        k = (jnp.asarray(X.ravel()) + jnp.asarray(Y.ravel()))[None, :] * net
    return np.asarray(k).reshape(b.shape[0], 201, 201)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["deeponet", "pi-deeponet"])
    ap.add_argument("--M-train", type=int, required=True)
    ap.add_argument("--branch", default="mlp", choices=["mlp", "cnn"])
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    branch = args.branch
    cfg = config_for_mode(args.mode, M=args.M_train, branch_kind=branch,
                          bc_mode="adf_aq", n_samples=args.n_samples,
                          n_test=args.n_test, seed=args.seed)
    cfg._mode = args.mode
    data = D.load_or_generate(cfg, args.outdir, verbose=True)
    idx = np.asarray(data["test_idx"])
    a = np.asarray(data["a"])[idx]
    lam = np.asarray(data["lam"])[idx]
    c = np.asarray(data["c"])[idx]
    p_test = np.asarray(data["p"])[idx]

    g201 = np.linspace(-0.5, 0.5, 201)
    a201, _, _, lam201, mu201 = D.family_coeffs_np(p_test, g201, cfg.data)
    true201 = np.stack([
        solveIntegralFD(100, a201[i], lam201[i], mu201[i],
                        return_full=True, compute_derivative=False)
        for i in range(len(idx))
    ]).astype(np.float32)
    X, Y = np.meshgrid(g201, g201, indexing="ij")
    mask = np.abs(Y) <= np.abs(X)

    modules, skeleton = N.build_model(args.mode, cfg.model, cfg.grid.n, seed=0)
    path = os.path.join(args.outdir, f"{args.tag}_params.msgpack")
    with open(path, "rb") as f:
        params = serialization.from_bytes(skeleton, f.read())
    pred = predict_201(cfg, modules, params, a, lam, c, p_test)
    err = (pred - true201)[:, mask]
    tgt = true201[:, mask]
    num = np.sum(err ** 2, axis=1)
    den = np.sum(tgt ** 2, axis=1) + 1e-12
    rel = np.sqrt(num / den)
    out = dict(
        mode=args.mode, M_train=args.M_train, grid_train=cfg.grid.n,
        tag=args.tag,
        rel_l2_mean=float(rel.mean()),
        rel_l2_global=float(np.sqrt(np.sum(num) / np.sum(den))),
        rel_l2_median=float(np.median(rel)),
    )
    sp = os.path.join(args.outdir, f"{args.tag}_summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        out["train_time_s"] = float(s["train_time_s"])
        out["n_params"] = int(s["n_params"])
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        json.dump(out, open(args.out_json, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

