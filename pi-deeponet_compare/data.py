from __future__ import annotations

import os

import numpy as np

from cfg import Config
from solveIntegralFD import solveIntegralFD, boundary_derivatives


FAMILY_PARAM_NAMES = {"a_q": ("s", "mu")}


def sample_family(cfg: Config):
    d = cfg.data
    n = cfg.grid.n
    x = np.linspace(-cfg.grid.half, cfg.grid.half, n, dtype=np.float64)
    rng = np.random.default_rng(d.seed)
    S = d.n_samples

    s = rng.uniform(d.s_low, d.s_high, S)
    mu = rng.uniform(d.mu_low, d.mu_high, S)
    p = np.stack([s, mu], axis=1)
    a, _, _, lam, c = family_coeffs_np(p, x, d)
    return p, a, lam, c


def family_coeffs_np(p: np.ndarray, x: np.ndarray, d):
    p = np.asarray(p, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    s = p[:, 0:1]
    mu = p[:, 1]

    a = s * (d.A0 + d.C2 * x ** 2)
    a_x = 2.0 * d.C2 * s * x
    a_xx = 2.0 * d.C2 * s * np.ones_like(x)

    lambda_bar = d.L0 + d.L1 * s * x
    lambda_bar_prime = d.L1 * s * np.ones_like(x)
    Q = (0.5 * lambda_bar_prime
         - 0.5 * (a_x / a) * lambda_bar
         - 0.25 * lambda_bar ** 2 / a)
    return a, a_x, a_xx, Q, mu


def target_inputs(M: int = 50, mu: float = 8.0, s: float = 1.0):
    n = 2 * M + 1
    x = np.linspace(-0.5, 0.5, n)
    p = np.array([[s, mu]])

    from cfg import DataConfig
    d = DataConfig()
    a, _, _, Q, mu_arr = family_coeffs_np(p, x, d)
    return x, a[0], Q[0], float(mu_arr[0])


def solve_kernel_exact(M: int, a_vals, Q_vals, mu: float):
    return solveIntegralFD(int(M), np.asarray(a_vals, dtype=float),
                           np.asarray(Q_vals, dtype=float), float(mu),
                           return_full=True, compute_derivative=False)


def generate_inputs(cfg: Config, verbose: bool = True) -> dict:
    p, a, lam, c = sample_family(cfg)
    S, N = a.shape
    grid = np.linspace(-cfg.grid.half, cfg.grid.half, N, dtype=np.float64)
    rng = np.random.default_rng(cfg.data.seed + 17)
    perm = rng.permutation(S)
    test_idx = perm[:cfg.data.n_test]
    train_idx = perm[cfg.data.n_test:]
    if verbose:
        print(f"  generated {S} inputs (labels deferred)")

    return dict(
        grid=grid.astype(np.float32),
        p=p,
        a=a,
        lam=lam,
        c=c,
        train_idx=train_idx.astype(np.int32),
        test_idx=test_idx.astype(np.int32),
    )


def generate_dataset(cfg: Config, verbose: bool = True) -> dict:
    data = generate_inputs(cfg, verbose=False)
    a, lam, c = data["a"], data["lam"], data["c"]
    S, N = a.shape
    kernels = np.zeros((S, N, N), dtype=np.float64)
    for i in range(S):
        kernels[i] = solve_kernel_exact(cfg.grid.M, a[i], lam[i], c[i])
        if verbose and (i + 1) % max(1, S // 10) == 0:
            print(f"  generated {i + 1}/{S} kernels")

    if not np.isfinite(kernels).all():
        raise RuntimeError("non-finite kernel values")
    if np.abs(kernels).max() > 1e4:
        raise RuntimeError("kernel magnitude too large; check aQ family")

    data["kernels"] = kernels.astype(np.float32)
    return data


def cache_path(cfg: Config, outdir: str) -> str:
    d = cfg.data
    return os.path.join(
        outdir, "cache",
        f"aq_data_M{cfg.grid.M}_n{d.n_samples}_nt{d.n_test}_seed{d.seed}.npz",
    )


def load_or_generate(cfg: Config, outdir: str, verbose: bool = True) -> dict:
    path = cache_path(cfg, outdir)
    if cfg.data.cache and os.path.exists(path):
        if verbose:
            print(f"  loading cached data: {path}")
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    if verbose:
        print(f"  generating {cfg.data.n_samples} aQ kernels ...")
    data = generate_dataset(cfg, verbose=verbose)
    if cfg.data.cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **data)
        if verbose:
            print(f"  cached: {path}")
    return data


def full_grid_xy(grid: np.ndarray, positive_x: bool = False,
                 negative_x: bool = False):
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    inside = np.abs(Y) <= np.abs(X)
    if positive_x:
        inside &= X > 0.0
    if negative_x:
        inside &= X < 0.0
    return xy.astype(np.float32), inside


def interior_pool(grid: np.ndarray, margin: float,
                  positive_x: bool = False,
                  negative_x: bool = False) -> np.ndarray:
    N = len(grid)
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    x_i = grid[ii]
    y_j = grid[jj]
    valid = (
        (np.abs(x_i) > margin)
        & (np.abs(y_j) <= np.abs(x_i))
        & (np.minimum(np.abs(x_i - y_j), np.abs(x_i + y_j)) > margin)
    )
    if positive_x:
        valid &= x_i > 0.0
    if negative_x:
        valid &= x_i < 0.0
    return np.stack([ii[valid], jj[valid]], axis=1).astype(np.int32)


def sample_colloc_np(rng, n, half, margin, positive_x=False, negative_x=False):
    u = rng.random((n, 3))
    absx = 2.0 * margin + (half - 2.0 * margin) * u[:, 0]
    if positive_x:
        x = absx
    elif negative_x:
        x = -absx
    else:
        x = np.where(u[:, 1] < 0.5, -absx, absx)
    y = (2.0 * u[:, 2] - 1.0) * (absx - margin)
    return np.stack([x, y], axis=1).astype(np.float32)


def dataset_summary(data: dict, cfg: Config) -> dict:
    kernels = data.get("kernels")
    return dict(
        n_samples=int(cfg.data.n_samples),
        n_train=int(cfg.data.n_train),
        n_test=int(cfg.data.n_test),
        grid=cfg.grid.n,
        kernel_abs_max=(float(np.abs(kernels).max())
                        if kernels is not None else None),
        kernel_rms=(float(np.sqrt(np.mean(kernels ** 2)))
                    if kernels is not None else None),
        a_range=(float(data["a"].min()), float(data["a"].max())),
        Q_range=(float(data["lam"].min()), float(data["lam"].max())),
        mu_range=(float(data["c"].min()), float(data["c"].max())),
    )

