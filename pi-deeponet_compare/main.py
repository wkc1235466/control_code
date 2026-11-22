from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import data as D
import evaluate as E
import report as R
from cfg import config_for_mode
from train import Trainer

MODES = ("sep-pi-deeponet", "pi-deeponet", "deeponet")


def _apply_smoke(cfg):

    return cfg.copy(
        grid=type(cfg.grid)(half=cfg.grid.half, M=20),
        data=type(cfg.data)(**{**cfg.data.__dict__,
                               "n_samples": 30, "n_test": 5,
                               "cache": False}),
        physics=type(cfg.physics)(**{**cfg.physics.__dict__,
                                     "n_colloc": 64, "n_data": 3}),
        train=type(cfg.train)(**{**cfg.train.__dict__,
                                 "epochs": 3, "batch_size": 4,
                                 "eval_every": 1, "eval_samples": 5,
                                 "log_every": 1}),
    )


def build_cfg(args, mode):
    overrides = dict(
        M=args.M,
        family=args.family,
        c_scale=args.c_scale,
        n_samples=args.n_samples,
        n_test=args.n_test,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_colloc=args.n_colloc,
        fixed_colloc=args.fixed_colloc,
        n_data=args.n_data,
        margin=args.margin,
        bc_mode=args.bc_mode,
        positive_x=args.positive_x,
        negative_x=args.negative_x,
        n_bc=args.n_bc,
        w_pde=args.w_pde,
        w_data=args.w_data,
        w_bc_anti=args.w_bc_anti,
        w_bc_diag=args.w_bc_diag,
        pde_norm=args.pde_norm,
        colloc_margin=args.colloc_margin,
        lr_schedule=args.lr_schedule,
        clip_norm=args.clip_norm,
        branch_kind=args.branch,
        feature_dim=args.feature_dim,
        branch_out_scale=args.branch_out_scale,
        fourier_m=args.fourier_m,
        fourier_sigma=args.fourier_sigma,
        eval_every=args.eval_every,
        cache=(not args.no_cache),
        online_truth=args.online_truth,
    )
    if args.hidden:
        overrides["branch_layers"] = tuple([args.hidden] * args.depth)
        overrides["trunk_layers"] = tuple([args.hidden] * args.depth)
    cfg = config_for_mode(mode, **overrides)
    if args.smoke:
        cfg = _apply_smoke(cfg)
    return cfg


def run_one(mode, cfg, data, outdir, tag=None, make_plots=True):
    tag = tag or f"{mode}_{cfg.data.family}_cs{cfg.data.c_scale:g}_d{cfg.physics.n_data}"
    print("=" * 78)
    print(f"[{mode}]  family={cfg.data.family} c_scale={cfg.data.c_scale:g} "
          f"grid={cfg.grid.n} n_data={cfg.physics.n_data} "
          f"branch={cfg.model.branch_kind}")
    print("=" * 78)
    trainer = Trainer(mode, cfg, data)
    print(f"  parameters: {trainer.n_params:,}")
    trainer.fit(verbose=True)
    summary = E.evaluate(trainer, tag, outdir, make_plots=make_plots)
    trainer.save(outdir, tag)
    print(f"  FINAL rel-L2={summary['test_rel_l2_mean']:.4e} "
          f"train={summary['train_time_s']:.1f}s "
          f"({summary['ms_per_step']:.2f} ms/step) "
          f"pde_rel={summary['pde_rel_mean']:.3e}")
    return summary, trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="all", choices=list(MODES) + ["all"])
    ap.add_argument("--family", default="a_q", choices=["a_q"])
    ap.add_argument("--c-scale", type=float, default=12.0)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-colloc", type=int, default=512)
    ap.add_argument("--n-data", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--fixed-colloc", action="store_true",
                    help="use fixed interior grid points for PDE collocation")
    ap.add_argument("--bc-mode", default="adf_aq",
                    choices=["soft", "hard_anti", "adf_aq"])
    ap.add_argument("--positive-x", action="store_true",
                    help="train/evaluate only the x>0 half of the diamond")
    ap.add_argument("--negative-x", action="store_true",
                    help="train/evaluate only the x<0 half of the diamond")
    ap.add_argument("--n-bc", type=int, default=512)
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-data", type=float, default=10.0)
    ap.add_argument("--w-bc-anti", type=float, default=1.0)
    ap.add_argument("--w-bc-diag", type=float, default=10.0)
    ap.add_argument("--pde-norm", default="terms",
                    choices=["terms", "qk", "fixed_seed"])
    ap.add_argument("--colloc-margin", type=float, default=0.02)
    ap.add_argument("--lr-schedule", default="cosine",
                    choices=["cosine", "constant"])
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--branch", default=None, choices=["mlp", "cnn"])
    ap.add_argument("--feature-dim", type=int, default=None)
    ap.add_argument("--branch-out-scale", type=float, default=None)
    ap.add_argument("--fourier-m", type=int, default=None)
    ap.add_argument("--fourier-sigma", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=None,
                    help="width of all branch/trunk hidden layers")
    ap.add_argument("--depth", type=int, default=3,
                    help="number of hidden layers when --hidden is set")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(SCRIPT_DIR, "output"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    truth = ap.add_mutually_exclusive_group()
    truth.add_argument("--online-truth", dest="online_truth",
                       action="store_true",
                       help="pi-deeponet: solve anchor truth on first use")
    truth.add_argument("--offline-truth", dest="online_truth",
                       action="store_false",
                       help="pi-deeponet: use the precomputed kernel dataset")
    ap.set_defaults(online_truth=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    modes = list(MODES) if args.mode == "all" else [args.mode]
    configs = [build_cfg(args, m) for m in modes]


    online_pi_only = (len(modes) == 1 and modes[0] == "pi-deeponet"
                      and configs[0].physics.online_truth)
    if online_pi_only:
        data = D.generate_inputs(configs[0], verbose=True)
    else:
        data = D.load_or_generate(configs[0], args.outdir, verbose=True)
    summary_data = D.dataset_summary(data, configs[0])
    print("dataset:", json.dumps(summary_data, ensure_ascii=False))

    results = []
    for mode, cfg in zip(modes, configs):
        tag = args.tag or f"{mode}_{cfg.data.family}_cs{cfg.data.c_scale:g}_d{cfg.physics.n_data}"
        summary, _ = run_one(mode, cfg, data, args.outdir, tag=tag,
                             make_plots=not args.no_plots)
        results.append(summary)

    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if len(results) > 1:
        protocol = (
            f"- Truth family: `{configs[0].data.family}`, c_scale={configs[0].data.c_scale:g}, "
            f"FD grid {configs[0].grid.n}×{configs[0].grid.n}, "
            f"train/test {configs[0].data.n_train}/{configs[0].data.n_test}.\n"
            f"- Optimizer: AdamW lr={configs[0].train.lr:g}, "
            f"batch={configs[0].train.batch_size}, epochs={configs[0].train.epochs}.\n"
            f"- PI method: boundary mode `{configs[0].physics.bc_mode}`, "
            f"{configs[0].physics.n_colloc} PDE collocation points/step "
            f"+ {configs[0].physics.n_data} truth anchors, "
            f"w_data={configs[0].physics.w_data:g}.\n"
            f"- DeepONet: data-only, full-grid relative MSE supervision.\n"
            f"- Metrics are relative L2 over `|y| <= |x|`; times exclude data generation and JIT compilation."
        )
        R.write_report(os.path.join(args.outdir, "results.md"),
                       "sep-deeponet: PI-DeepONet / Sep-PI-DeepONet / DeepONet comparison",
                       results, protocol=protocol)
        print(f"\nreport: {os.path.join(args.outdir, 'results.md')}")

    print("\nDONE")


if __name__ == "__main__":
    main()

