from __future__ import annotations

import os
from datetime import datetime


def _f(x, nd=4):
    if x is None:
        return "—"
    x = float(x)
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def main_table(results):
    headers = ["Model", "Branch", "Boundary", "Fourier", "Anchors", "Parameters",
               "Training time (s)", "ms/step", "rel-L2 mean", "rel-L2 median",
               "rel-L2 global", "anti-diag max", "diag BC rel", "PDE rel"]
    rows = []
    for r in results:
        rows.append([
            r["mode"],
            r["branch_kind"],
            r.get("bc_mode", "—"),
            r.get("fourier_m", 0),
            r.get("n_data", "—"),
            f"{r['n_params']:,}",
            _f(r["train_time_s"], 2),
            _f(r["ms_per_step"], 2),
            _f(r["test_rel_l2_mean"]),
            _f(r["test_rel_l2_median"]),
            _f(r["test_rel_l2_global"]),
            _f(r["anti_diag_max"], 2),
            _f(r["diag_bc_rel"]),
            _f(r["pde_rel_mean"]),
        ])
    return _table(headers, rows)


def sweep_table(results, key="n_data"):
    headers = ["Model", "family", "Boundary", "positive_x", "n_data", "rel-L2 mean",
               "rel-L2 median", "PDE rel", "Training time (s)"]
    rows = []
    for r in results:
        rows.append([
            r["mode"], r["family"], r.get("bc_mode", "—"),
            r.get("positive_x", False), r[key],
            _f(r["test_rel_l2_mean"]), _f(r["test_rel_l2_median"]),
            _f(r["pde_rel_mean"]), _f(r["train_time_s"], 1),
        ])
    return _table(headers, rows)


def write_report(path: str, title: str, main_results, sweep_results=None,
                 extra_notes: str = "", protocol: str = ""):
    parts = [f"# {title}", ""]
    parts.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append("")
    if protocol:
        parts += ["## Experimental setup", "", protocol, ""]

    parts += ["## Main comparison of the three models", "", main_table(main_results), ""]

    if sweep_results:
        parts += ["## Anchor and truth-family sweep", "", sweep_table(sweep_results), ""]

    if extra_notes:
        parts += ["## Conclusions and notes", "", extra_notes, ""]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path

