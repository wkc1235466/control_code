from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDS = {10: 21, 15: 31, 25: 51}


def _run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_md(path, results, args):
    lines = [
        "# PI-DeepONet vs DeepONet: Low-Resolution Training and 201×201 Generalization",
        "",
        f"Training setup: {args.n_samples} aQ input functions, "
        f"{args.n_samples - args.n_test}/{args.n_test} split, "
        f"{args.epochs} epochs, batch={args.batch_size}; "
        "both models use an MLP branch and a 2D trunk.",
        f"PI uses ADF-aQ with n_data={args.n_data} and n_colloc={args.n_colloc}; "
        "DeepONet uses full-grid data supervision. Ground truth is the exact 201×201 FD solution.",
        "",
        "| Training grid | DeepONet | PI-DeepONet | Winner | PI improvement |",
        "|---:|---:|---:|---|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['grid']}×{r['grid']} | {r['deep']:.4e} | {r['pi']:.4e} | "
            f"{r['winner']} | {100.0 * r['improvement']:.1f}% |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.join(HERE, "output", "lowres"))
    ap.add_argument("--grids", type=int, nargs="+", default=[10, 15, 25],
                    help="FD grid parameters M; 10/15/25 -> 21/31/51")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-data", type=int, default=20)
    ap.add_argument("--n-colloc", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    py = sys.executable
    results = []

    for M in args.grids:
        if M not in GRIDS:
            raise ValueError(f"unsupported M={M}; choose from {sorted(GRIDS)}")
        tag_deep = f"low_deep_M{M}"
        tag_pi = f"low_pi_M{M}"

        common = [
            "--M", str(M),
            "--branch", "mlp",
            "--n-samples", str(args.n_samples),
            "--n-test", str(args.n_test),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--outdir", args.outdir,
        ]
        if not args.plots:
            common.append("--no-plots")

        jobs = [
            ("deeponet", tag_deep, []),
            ("pi-deeponet", tag_pi, [
                "--n-data", str(args.n_data),
                "--n-colloc", str(args.n_colloc),
                "--bc-mode", "adf_aq",
                "--offline-truth",
            ]),
        ]
        for mode, tag, extra in jobs:
            ckpt = os.path.join(args.outdir, f"{tag}_params.msgpack")
            if not (args.skip_existing and os.path.exists(ckpt)):
                _run([py, os.path.join(HERE, "main.py"),
                      "--mode", mode, "--tag", tag, *common, *extra])

        for mode, tag, _ in jobs:
            _run([py, os.path.join(HERE, "eval_target201.py"),
                  "--mode", mode, "--M-train", str(M), "--branch", "mlp",
                  "--n-samples", str(args.n_samples),
                  "--n-test", str(args.n_test), "--seed", str(args.seed),
                  "--tag", tag, "--outdir", args.outdir,
                  "--out-json", os.path.join(args.outdir,
                                             f"eval_{mode}_M{M}.json")])

        deep = _read_json(os.path.join(args.outdir, f"eval_deeponet_M{M}.json"))
        pi = _read_json(os.path.join(args.outdir, f"eval_pi-deeponet_M{M}.json"))
        d = float(deep["rel_l2_mean"])
        p = float(pi["rel_l2_mean"])
        results.append(dict(
            grid=GRIDS[M],
            deep=d,
            pi=p,
            winner="PI-DeepONet" if p < d else "DeepONet",
            improvement=(d - p) / d,
        ))

    results.sort(key=lambda r: r["grid"])
    with open(os.path.join(args.outdir, "lowres_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    _write_md(os.path.join(args.outdir, "lowres_results.md"), results, args)

    print("\n" + json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nresults: {os.path.join(args.outdir, 'lowres_results.md')}")


if __name__ == "__main__":
    main()

