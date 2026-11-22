import numpy as np

def solveIntegralFD(
    M,
    a_func,
    lambda_func,
    c,
    return_full=True,
    compute_derivative=False
):

    M = int(M)
    if M < 2:
        raise ValueError("M must be >= 2")

    M2 = 2 * M
    N = M2 + 1

    h = 1.0 / M2

    x = np.linspace(-0.5, 0.5, N)
    y = x.copy()

    a_vals = np.asarray(a_func, dtype=float).reshape(-1)
    lambda_vals = np.asarray(lambda_func, dtype=float).reshape(-1)

    if len(a_vals) != N:
        raise ValueError(
            f"Invalid a_func length: expected {N}, got {len(a_vals)}"
        )

    if len(lambda_vals) != N:
        raise ValueError(
            f"Invalid lambda_func length: expected {N}, got {len(lambda_vals)}"
        )

    if np.any(np.abs(a_vals) < 1e-14):
        raise ValueError("a(x) contains values near zero; division is not possible")

    a_prime = np.gradient(a_vals, h, edge_order=2)

    q_vals = lambda_vals + c

    k = np.zeros((N, N), dtype=float)

    for i in range(N):
        j_anti = M2 - i

        if 0 <= j_anti < N:
            k[i, j_anti] = 0.0

    k[M, M] = 0.0

    def ki1j(i, j):
        if j - 1 < 0 or j + 1 >= N:
            return 0.0

        kij = 0.5 * (
            k[i, j + 1] + k[i, j - 1]
        )

        return (
            (
                (
                    2.0 * (a_vals[i] - a_vals[j])
                    + h * h * q_vals[j]
                ) * kij
                + a_vals[j + 1] * k[i, j + 1]
                + a_vals[j - 1] * k[i, j - 1]
            ) / a_vals[i]
            - k[i - 1, j]
        )

    for i in range(M, N - 1):

        new_i = i + 1

        j_diag = new_i
        j_anti = M2 - new_i

        if 0 <= j_anti < N:
            k[new_i, j_anti] = 0.0

        denom = (
            1.0
            + h * a_prime[j_diag]
            / (4.0 * a_vals[j_diag])
        )

        k[new_i, j_diag] = (
            (
                1.0
                - h * a_prime[i]
                / (4.0 * a_vals[i])
            )
            * k[i, i]
            + h / 4.0 * (
                q_vals[i] / a_vals[i]
                + q_vals[j_diag] / a_vals[j_diag]
            )
        ) / denom

        j_near_diag = new_i - 1

        if (
            j_near_diag > j_anti
            and 0 <= j_near_diag < N
        ):
            k[new_i, j_near_diag] = (
                k[i, i]
                - h * a_prime[i]
                / (4.0 * a_vals[i])
                * k[i, i]
                + h * q_vals[i]
                / (4.0 * a_vals[i])
            )

        j_start = j_anti + 1
        j_end = j_diag - 1

        for j in range(j_start, j_end):
            if j < 0 or j >= N:
                continue

            if j == j_anti or j == j_diag:
                continue

            if (
                i - 1 >= 0
                and j - 1 >= 0
                and j + 1 < N
            ):
                k[new_i, j] = ki1j(i, j)

    for i in range(M, 0, -1):

        new_i = i - 1

        j_diag = new_i
        j_anti = M2 - new_i

        if 0 <= j_anti < N:
            k[new_i, j_anti] = 0.0

        denom_left = (
            1.0
            - h * a_prime[j_diag]
            / (4.0 * a_vals[j_diag])
        )

        k[new_i, j_diag] = (
            (
                1.0
                + h * a_prime[i]
                / (4.0 * a_vals[i])
            )
            * k[i, i]
            - h / 4.0 * (
                q_vals[i] / a_vals[i]
                + q_vals[j_diag] / a_vals[j_diag]
            )
        ) / denom_left

        j_near_diag = i

        if (
            j_diag < j_near_diag < j_anti
            and 0 <= j_near_diag < N
        ):
            k[new_i, j_near_diag] = (
                k[i, i]
                + h * a_prime[i]
                / (4.0 * a_vals[i])
                * k[i, i]
                - h * q_vals[i]
                / (4.0 * a_vals[i])
            )

        j_start = i + 1
        j_end = j_anti

        for j in range(j_start, j_end):

            if j < 0 or j >= N:
                continue

            if j == j_diag or j == j_anti:
                continue

            if (
                i + 1 < N
                and j - 1 >= 0
                and j + 1 < N
            ):

                k[new_i, j] = (
                    (
                        (
                            2.0 * (a_vals[i] - a_vals[j])
                            + h * h * q_vals[j]
                        )
                        * 0.5
                        * (
                            k[i, j + 1]
                            + k[i, j - 1]
                        )
                        + a_vals[j + 1] * k[i, j + 1]
                        + a_vals[j - 1] * k[i, j - 1]
                    )
                    / a_vals[i]
                    - k[i + 1, j]
                )

    for i in range(N):
        j_anti = M2 - i
        if 0 <= j_anti < N:
            k[i, j_anti] = 0.0

    k[M, M] = 0.0

    k_out = k if return_full else k[::2, ::2]
    if not compute_derivative:
        return k_out
    dk_05, dk_m05 = boundary_derivatives(k, verbose=False)
    if not return_full:
        dk_05, dk_m05 = dk_05[::2], dk_m05[::2]
    return k_out, dk_05, dk_m05

from scipy.interpolate import CubicSpline

def _column_spline_dx(vals, y, side, margin=6):
    N = len(y)
    out = np.full(N, np.nan)
    for j, xj in enumerate(y):
        lo = abs(xj)
        m = (y >= lo - 1e-12) if side > 0 else (y <= -lo + 1e-12)
        xs, col = y[m], vals[m, j]
        if xs.size < margin + 3 or (xs[-1] - xs[0]) < margin * (xs[1] - xs[0]):
            continue
        out[j] = CubicSpline(xs, col, bc_type="natural")(0.5 if side > 0 else -0.5, 1)
    return out

def _row_fallback(vals, y, side):
    N = len(y)
    h = 0.5 / (N - 1) * 2
    h = y[1] - y[0]
    out = np.full(N, np.nan)
    for j in range(N):
        if abs(y[j]) > 0.5 - 2 * h + 1e-12:
            continue
        if side > 0:
            out[j] = (3 * vals[-1, j] - 4 * vals[-2, j] + vals[-3, j]) / (2 * h)
        else:
            out[j] = (-3 * vals[0, j] + 4 * vals[1, j] - vals[2, j]) / (2 * h)
    return out

def _extrapolate_ends(vals, y):
    out = np.array(vals, dtype=float, copy=True)
    good = np.flatnonzero(np.isfinite(out))
    if good.size < 3:
        return out
    for j in range(len(out)):
        if np.isfinite(out[j]):
            continue
        near = good[np.argsort(np.abs(good - j))[:3]]
        near = np.sort(near)
        out[j] = float(np.polyval(np.polyfit(y[near], out[near], 1), y[j]))
    return out

def boundary_derivatives(k_full, verbose=False):
    vals = np.asarray(k_full, dtype=float)
    N = vals.shape[1]
    y = np.linspace(-0.5, 0.5, N)
    right = _column_spline_dx(vals, y, +1)
    left = _column_spline_dx(vals, y, -1)
    rr = _row_fallback(vals, y, +1)
    rl = _row_fallback(vals, y, -1)
    right = _extrapolate_ends(np.where(np.isfinite(right), right, rr), y)
    left = _extrapolate_ends(np.where(np.isfinite(left), left, rl), y)
    if verbose:
        print(f'    [info] derivative ranges: +0.5 [{right.min():+.4f}, {right.max():+.4f}]  '
              f'-0.5 [{left.min():+.4f}, {left.max():+.4f}]')
    return right, left
