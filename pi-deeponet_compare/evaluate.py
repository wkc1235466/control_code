from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import Trainer


def _masked(a, mask):
    out = np.asarray(a, dtype=float).copy()
    out[~mask] = np.nan
    return out


def save_plots(trainer: Trainer, test: dict, outdir: str, tag: str):
    os.makedirs(outdir, exist_ok=True)
    hist = trainer.history

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(hist["total"], label="total")
    if np.any(np.asarray(hist["pde"]) > 0):
        ax[0].semilogy(hist["pde"], label="pde")
    if np.any(np.asarray(hist["data"]) > 0):
        ax[0].semilogy(hist["data"], label="data")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("loss")
    ax[0].legend()
    ax[0].grid(True, ls=":", alpha=0.5)

    if hist["test_rel_l2"]:
        ax[1].semilogy(hist["eval_epochs"], hist["test_rel_l2"], marker="o")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("test rel L2")
    ax[1].set_title("test accuracy")
    ax[1].grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{tag}_loss.png"), dpi=150)
    plt.close(fig)


    mask = trainer.inside_np
    pred, true = test["pred"], test["true"]
    err = (pred - true)[:, mask]
    tgt = true[:, mask]
    rel = np.sqrt(np.sum(err ** 2, axis=1) / (np.sum(tgt ** 2, axis=1) + 1e-12))
    pick = int(np.argsort(rel)[len(rel) // 2])

    grid = np.asarray(trainer.jd["grid"])
    data = [true[pick], pred[pick], pred[pick] - true[pick]]
    titles = ["true k", "pred k", "error (pred-true)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, arr, title in zip(axes, data, titles):
        im = ax.imshow(_masked(arr, mask), origin="lower",
                       extent=[-0.5, 0.5, -0.5, 0.5], cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{tag}_kernel.png"), dpi=150)
    plt.close(fig)


def evaluate(trainer: Trainer, tag: str, outdir: str,
             make_plots: bool = True) -> dict:
    test = trainer.test_metrics()
    bounds = trainer.boundary_metrics(test["pred"], np.asarray(test["idx"]))
    if trainer.mode == "deeponet":


        pde = dict(pde_rel_mean=0.0, pde_rel_max=0.0, pde_k_rms=0.0)
    else:
        pde = trainer.pde_metrics()

    rel = test["rel_l2_mean"]
    summary = dict(
        mode=trainer.mode,
        tag=tag,
        family=trainer.family,
        c_scale=trainer.c_scale,
        bc_mode=trainer.bc_mode,
        positive_x=bool(trainer.positive_x),
        negative_x=bool(trainer.negative_x),
        n_data=("full" if trainer.mode == "deeponet"
                else int(trainer.cfg.physics.n_data)),
        n_colloc=int(trainer.n_colloc),
        branch_kind=trainer.cfg.model.branch_kind,
        fourier_m=int(trainer.cfg.model.fourier_m),
        n_params=int(trainer.n_params),
        n_samples=int(trainer.cfg.data.n_samples),
        n_train=int(trainer.cfg.data.n_train),
        n_test=int(trainer.cfg.data.n_test),
        grid=int(trainer.n),
        epochs=int(trainer.cfg.train.epochs),
        total_steps=int(trainer.history["total_steps"]),
        train_time_s=float(trainer.history["train_time"]),
        compile_time_s=float(trainer.history["compile_time"]),
        ms_per_step=float(trainer.history["ms_per_step"]),
        test_rel_l2_mean=float(rel),
        test_rel_l2_median=float(test["rel_l2_median"]),
        test_rel_l2_max=float(test["rel_l2_max"]),
        test_rel_l2_global=float(test["rel_l2_global"]),
        abs_err_max=float(test["abs_err_max"]),
        abs_err_mean=float(test["abs_err_mean"]),
        anti_diag_max=float(bounds["anti_diag_max"]),
        diag_bc_rel=float(bounds["diag_bc_rel"]),
        pde_rel_mean=float(pde["pde_rel_mean"]),
        pde_rel_max=float(pde["pde_rel_max"]),
    )

    if make_plots:
        save_plots(trainer, test, outdir, tag)
        with open(os.path.join(outdir, f"{tag}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

