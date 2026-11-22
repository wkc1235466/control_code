# Code Package for the Manuscript

This directory contains the code used for the numerical study in the paper. It is organized into three projects and one shared data-generation module. Full-line comments have been removed; copyright and attribution notices are retained.

## Structure

```
code_package/
├── README.md
├── LICENSE
├── CITATION.cff
├── NOTICE
├── data_generation/
│   └── solveIntegralFD.py      # canonical finite-difference kernel solver
├── deeponet_training/          # DeepONet training
├── pi-deeponet_compare/        # PI-DeepONet / DeepONet / Sep-PI-DeepONet comparison
└── paper_code/                 # MATLAB simulation + precomputed .mat files
```

`pi-deeponet_compare/solveIntegralFD.py` is a small shim that loads `../data_generation/solveIntegralFD.py`. Existing imports such as

```python
from solveIntegralFD import solveIntegralFD, boundary_derivatives
```

work without modification.

`deeponet_training/DeepONet.py` embeds its own copy of the solver so that the training script remains self-contained. This is intentional.

## Projects

### deeponet_training

Single-file DeepONet training script for the mapping

\[
(\bar a, Q, \mu) \mapsto k(x,y).
\]

```bash
cd deeponet_training
pip install -r requirements.txt
python DeepONet.py
```

CUDA is required. The script writes `training_results_supervised/DeepONet.pth` and `kernel_comparison.png`.

### pi-deeponet_compare

Training and evaluation of PI-DeepONet, pure-data DeepONet, and separable PI-DeepONet. This directory reproduces the low-resolution comparison in Table 2.

```bash
cd pi-deeponet_compare
pip install -r requirements.txt
python run_lowres.py
```

The precomputed result table is `pi-deeponet_compare/results/results.md`. The driver writes new runs to `output/lowres/`.

### paper_code

MATLAB simulation code. It uses the precomputed `.mat` files in `paper_code/data/` and does not require Python.

```matlab
cd paper_code
main
compare_etm
plot_kernel_comparison_3d
```

`main.m` settings:

| Variable | Values | Meaning |
|---|---|---|
| `CONTROL` | `'ctrl'` / `'oplp'` | closed-loop / open-loop |
| `PLOT_COMPARISON` | `true` / `false` | export state-evolution figures |

The closed-loop case uses `data/kernels_deeponet.mat`; the numerical-kernel branch has been removed.

## Data files

| File | Used by | Purpose |
|---|---|---|
| `kernels_deeponet.mat` | `main.m` (closed-loop), `compare_etm.m` | DeepONet-predicted kernel and boundary derivatives |
| `kernel_comparison_lam1.mat` | `plot_kernel_comparison_3d.m` | true / predicted / error data for system 1 |
| `kernel_comparison_lam2.mat` | `plot_kernel_comparison_3d.m` | true / predicted / error data for system 2 |

These three files are sufficient to reproduce Figures 2–6 and Table 1.

## Reproduction guide

| Paper item | Command | Main output |
|---|---|---|
| Figure 2 (open-loop) | `paper_code/main.m` with `CONTROL='oplp'` | `figs/main/state_evolution_oplp_*_x1.png`, `..._x2.png` |
| Figure 3 (kernel comparison) | `plot_kernel_comparison_3d.m` | `figs/plot_kernel/kernel_comparison_3d_combined_sys*_true/pred/err.png` |
| Figure 4 (closed-loop) | `main.m` with `CONTROL='ctrl'` | `figs/main/state_evolution_ctrl_deeponet_x1.png`, `..._x2.png` |
| Figure 5 (DETM states) | `compare_etm.m` | `figs/compare_etm/system_state_detm_x1.png`, `..._x2.png` |
| Figure 6 (trigger intervals) | `compare_etm.m` | `figs/fig6_interval_distribution.png` |
| Table 1 (trigger counts) | `compare_etm.m` | `figs/trigger_counts.txt` |
| Table 2 (low-resolution errors) | `pi-deeponet_compare/run_lowres.py` | `pi-deeponet_compare/results/results.md` |
| DeepONet training | `deeponet_training/DeepONet.py` | `training_results_supervised/DeepONet.pth` |

Algorithm 1 is pseudocode in the paper and has no separate script.

## Model weights

The package does not include `DeepONet.pth` (about 71 MB) or the Python scripts that regenerate the `.mat` kernels.

- The MATLAB figures and Table 1 are unaffected because they use the precomputed `.mat` files.
- Table 2 trains from scratch and does not require the model weights.
- Model weights are needed only if the DeepONet kernels must be regenerated. They can be obtained by running `deeponet_training/DeepONet.py` or from a release archive.

The root `.gitignore` excludes `*.pth`, `output/`, `cache/`, `*.npz`, and `*.png`.

## Environment

- MATLAB R2020a or later (`exportgraphics`); tested with R2024b.
- `deeponet_training` requires CUDA.
- `pi-deeponet_compare` runs with CPU-only JAX.

## License and citation

- The code is released under the **MIT License**; see `LICENSE`.
- `paper_code/create_pde.m` and related files come from the one-dimensional advection-diffusion toolbox by Bo Zhuang (Jiangnan University). They retain their original copyright notices and are **not** relicensed under the MIT License; see `NOTICE`.
- Citation metadata is provided in `CITATION.cff`. Please cite the associated article and this repository.
- Full-line comments have been removed.
