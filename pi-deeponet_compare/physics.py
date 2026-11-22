from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def family_coeffs(family: str, x: jnp.ndarray, p: jnp.ndarray,
                  c_scale: float = 12.0):
    x = jnp.asarray(x)
    p = jnp.asarray(p)
    if x.ndim == 1:
        x = x[None, :]
    ones = jnp.ones_like(x)

    if family in ("a_q", "aq"):


        s = p[:, 0:1]
        mu = p[:, 1]
        A0, C2, L0, L1 = 0.40, 1.50, 0.10, 0.10
        a = s * (A0 + C2 * x ** 2)
        a_x = 2.0 * C2 * s * x
        a_xx = 2.0 * C2 * s * ones
        lambda_bar = L0 + L1 * s * x
        lambda_bar_prime = L1 * s * ones
        lam = (0.5 * lambda_bar_prime
               - 0.5 * (a_x / a) * lambda_bar
               - 0.25 * lambda_bar ** 2 / a)
        c = mu
    elif family == "poly":
        alpha = p[:, 0:1]
        beta = p[:, 1:2]
        a = alpha * (1.0 + 6.0 * x ** 2)
        a_x = 12.0 * alpha * x
        a_xx = 12.0 * alpha * ones
        lam = 8.0 * alpha * x - 7.0
        c = c_scale * beta[:, 0]
    elif family == "const":
        A, L, C = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        a = A * ones
        a_x = jnp.zeros_like(x)
        a_xx = jnp.zeros_like(x)
        lam = L * ones
        c = C[:, 0]
    elif family == "a_only":
        alpha = p[:, 0:1]
        a = alpha * (1.0 + 6.0 * x ** 2)
        a_x = 12.0 * alpha * x
        a_xx = 12.0 * alpha * ones
        lam = jnp.full((p.shape[0], x.shape[1]), -7.0)
        c = jnp.full((p.shape[0],), 18.0)
    elif family == "paper":
        g = p[:, 0:1]
        u = 1.0 - 0.5 * x ** 2
        a = g * u ** 2
        a_x = -2.0 * g * x * u
        a_xx = -2.0 * g * (1.0 - 1.5 * x ** 2)
        lam = g * x ** 2 + 6.0
        c = c_scale * g[:, 0]
    else:
        raise ValueError(f"unknown family {family!r}")

    return a, a_x, a_xx, lam, c


_AQ_A0, _AQ_C2, _AQ_L0, _AQ_L1 = 0.40, 1.50, 0.10, 0.10


def _aq_q_scalar(x, p):
    s, mu = p[0], p[1]
    B = _AQ_L1 * s
    A = _AQ_A0 + _AQ_C2 * x ** 2
    sqA = jnp.sqrt(A)
    sqA0 = jnp.sqrt(_AQ_A0)


    I1 = jnp.arcsinh(x * jnp.sqrt(_AQ_C2 / _AQ_A0)) / jnp.sqrt(_AQ_C2)
    I2 = (1.0 / sqA0 - 1.0 / sqA) / _AQ_C2
    I3 = (I1 - x / sqA) / _AQ_C2
    J = x / (_AQ_A0 * sqA)

    integ = ((0.5 * B + mu) * I1
             - _AQ_C2 * _AQ_L0 * I2
             - _AQ_C2 * B * I3
             - (_AQ_L0 ** 2 * J
                + 2.0 * _AQ_L0 * B * I2
                + B ** 2 * I3) / (4.0 * s))
    return integ / (2.0 * s * sqA)


def _aq_h_scalar(x, p):
    s, mu = p[0], p[1]
    B = _AQ_L1 * s
    Q0 = 0.5 * B - _AQ_L0 ** 2 / (4.0 * s * _AQ_A0)
    h0 = (Q0 + mu) / (4.0 * s * _AQ_A0)

    xs = jnp.where(jnp.abs(x) > 1e-6, x, 1.0)
    h = _aq_q_scalar(xs, p) / (2.0 * xs)
    return jnp.where(jnp.abs(x) > 1e-6, h, h0)


def _aq_h_terms_single(x, p):
    h = _aq_h_scalar(x, p)
    h1 = jax.jacfwd(lambda z: _aq_h_scalar(z, p))(x)
    h2 = jax.jacfwd(lambda z: jax.jacfwd(
        lambda zz: _aq_h_scalar(zz, p))(z))(x)
    return h, h1, h2


def adf_aq_h_terms(x, p):
    x = jnp.asarray(x)
    p = jnp.asarray(p)

    def one_sample(pp):
        return jax.vmap(lambda xx: _aq_h_terms_single(xx, pp))(x)

    return jax.vmap(one_sample)(p)


def adf_aq_value(x, y, p, net):
    h, _, _ = adf_aq_h_terms(x, p)
    m = x ** 2 - y ** 2
    return (x + y)[None, :] * h + m[None, :] * net, h, m


def _f_scalar(x, p, family: str, c_scale: float):
    if family == "poly":
        alpha = p[0]
        beta = p[1]
        c = c_scale * beta
        S = jnp.sqrt(1.0 + 6.0 * x ** 2)
        C1 = 2.0 / 3.0
        C2 = (c - 7.0) / (2.0 * alpha * jnp.sqrt(6.0))
        return C1 * (1.0 - 1.0 / S) + C2 * jnp.arcsinh(jnp.sqrt(6.0) * x) / S
    elif family == "const":
        A, L, C = p[0], p[1], p[2]
        return (L + C) / (2.0 * A) * x
    elif family == "a_only":
        alpha = p[0]
        q = -7.0 + 18.0
        S = jnp.sqrt(1.0 + 6.0 * x ** 2)
        return q / (2.0 * alpha * jnp.sqrt(6.0)) * jnp.arcsinh(jnp.sqrt(6.0) * x) / S
    elif family == "paper":
        g = p[0]
        c = c_scale * g
        u = 1.0 - 0.5 * x ** 2
        B = 6.0 + c
        integral = (2.0 * g + B) * jnp.sqrt(2.0) * jnp.arctanh(x / jnp.sqrt(2.0)) - 2.0 * g * x
        return integral / (2.0 * g * u)
    else:
        raise ValueError(f"unknown family {family!r}")


def _f_value_and_derivs(x, p, family, c_scale):
    g = lambda z: _f_scalar(z, p, family, c_scale)
    f0 = g(x)
    f1 = jax.jacfwd(g)(x)
    f2 = jax.jacfwd(jax.jacfwd(g))(x)
    return f0, f1, f2


def _f_derivs_at_zero(p, family, c_scale):
    g = lambda z: _f_scalar(z, p, family, c_scale)
    d1 = jax.jacfwd(g)
    d2 = jax.jacfwd(d1)
    d3 = jax.jacfwd(d2)
    return d1(0.0), d2(0.0), d3(0.0)


def f_terms(family: str, x: jnp.ndarray, p: jnp.ndarray,
            c_scale: float = 12.0):
    x = jnp.asarray(x)
    p = jnp.asarray(p)
    B = p.shape[0]
    N = x.shape[0]
    xf = jnp.broadcast_to(x[None, :], (B, N)).reshape(-1)
    pf = jnp.repeat(p, N, axis=0)

    def fd(xx, pp):
        return _f_value_and_derivs(xx, pp, family, c_scale)

    f, f1, f2 = jax.vmap(fd)(xf, pf)
    return f.reshape(B, N), f1.reshape(B, N), f2.reshape(B, N)


def seed_terms(family: str, x: jnp.ndarray, y: jnp.ndarray, p: jnp.ndarray,
               c_scale: float = 12.0):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    p = jnp.asarray(p)
    B = p.shape[0]
    N = x.shape[0]

    f, f1, f2 = f_terms(family, x, p, c_scale)
    ff = f.reshape(-1)
    ff1 = f1.reshape(-1)
    ff2 = f2.reshape(-1)
    xf = jnp.broadcast_to(x[None, :], (B, N)).reshape(-1)

    def fd0(pp):
        return _f_derivs_at_zero(pp, family, c_scale)

    f1_0, f2_0, f3_0 = jax.vmap(fd0)(p)
    f1_0 = jnp.repeat(f1_0, N)
    f2_0 = jnp.repeat(f2_0, N)
    f3_0 = jnp.repeat(f3_0, N)


    eps = 1e-3
    safe = jnp.abs(xf) > eps
    xs = jnp.where(safe, xf, 1.0)
    h_d = ff / (2.0 * xs)
    h1_d = (xs * ff1 - ff) / (2.0 * xs ** 2)
    h2_d = (xs ** 2 * ff2 - 2.0 * xs * ff1 + 2.0 * ff) / (2.0 * xs ** 3)

    h_s = f1_0 / 2.0 + f2_0 * xf / 2.0 + f3_0 * xf ** 2 / 4.0
    h1_s = f2_0 / 2.0 + f3_0 * xf / 2.0
    h2_s = f3_0 / 2.0

    h = jnp.where(safe, h_d, h_s)
    h1 = jnp.where(safe, h1_d, h1_s)
    h2 = jnp.where(safe, h2_d, h2_s)

    f = f.reshape(B, N)
    f1 = f1.reshape(B, N)
    f2 = f2.reshape(B, N)
    h = h.reshape(B, N)
    h1 = h1.reshape(B, N)
    h2 = h2.reshape(B, N)

    yb = y[None, :]
    s = f / 2.0 + h * yb
    s_x = f1 / 2.0 + h1 * yb
    s_y = h
    s_xx = f2 / 2.0 + h2 * yb
    return s, s_x, s_y, s_xx


def profile_terms(profile_apply, profile_params, t, sign):
    def scalar(tt, ss):
        z = jnp.stack([tt, ss])[None, :]
        return profile_apply(profile_params, z)[0, 0]

    def one(tt, ss):
        g = lambda z: scalar(z, ss)
        return (scalar(tt, ss), jax.jacfwd(g)(tt),
                jax.jacfwd(jax.jacfwd(g))(tt))

    return jax.vmap(one)(t, sign)


def profile_seed_terms(family, x, y, p, profile_apply, profile_params,
                       c_scale: float = 12.0):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    p = jnp.asarray(p)
    f, f1, f2 = f_terms(family, x, p, c_scale)

    eps = 1e-6
    xs = jnp.where(jnp.abs(x) > eps, x, 1.0)
    t = jnp.clip((xs + y) / (2.0 * xs), 0.0, 1.0)
    sgn = jnp.where(x >= 0.0, 1.0, -1.0)
    phi, phi1, phi2 = profile_terms(
        profile_apply, profile_params, t, sgn)

    t_x = -y / (2.0 * xs ** 2)
    t_y = 1.0 / (2.0 * xs)
    t_xx = y / xs ** 3

    ph = phi[None, :]
    s = f * ph
    s_x = f1 * ph + f * (phi1 * t_x)[None, :]
    s_y = f * (phi1 * t_y)[None, :]
    s_xx = (f2 * ph
            + 2.0 * f1 * (phi1 * t_x)[None, :]
            + f * (phi2 * t_x ** 2 + phi1 * t_xx)[None, :])
    s_yy = f * (phi2 * t_y ** 2)[None, :]
    return s, s_x, s_y, s_xx, s_yy


def hard_both_terms(family, x, y, p, c_scale: float = 12.0):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    p = jnp.asarray(p)
    f, f1, f2 = f_terms(family, x, p, c_scale)

    eps = 1e-6
    xs = jnp.where(jnp.abs(x) > eps, x, 1.0)
    t = jnp.clip((xs + y) / (2.0 * xs), 0.0, 1.0)
    phi = 3.0 * t ** 2 - 2.0 * t ** 3
    phi1 = 6.0 * t * (1.0 - t)
    phi2 = 6.0 - 12.0 * t

    t_x = -y / (2.0 * xs ** 2)
    t_y = 1.0 / (2.0 * xs)
    t_xx = y / xs ** 3

    s = f * phi[None, :]
    s_x = f1 * phi[None, :] + f * (phi1 * t_x)[None, :]
    s_y = f * (phi1 * t_y)[None, :]
    s_xx = (f2 * phi[None, :]
            + 2.0 * f1 * (phi1 * t_x)[None, :]
            + f * (phi2 * t_x ** 2 + phi1 * t_xx)[None, :])
    s_yy = f * (phi2 * t_y ** 2)[None, :]


    m = 4.0 * (x ** 2 - y ** 2)
    m_x = 8.0 * x
    m_y = -8.0 * y
    m_xx = 8.0 * jnp.ones_like(x)
    m_yy = -8.0 * jnp.ones_like(x)
    return s, s_x, s_y, s_xx, s_yy, m, m_x, m_y, m_xx, m_yy


_MASK_MAX = 0.140625


def mask_terms(x: jnp.ndarray, y: jnp.ndarray):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    d = x - y
    s = 1.0 / _MASK_MAX
    m = s * (x + y) * d ** 2
    m_x = s * d * (3.0 * x + y)
    m_y = -s * d * (x + 3.0 * y)
    m_xx = s * (6.0 * x - 2.0 * y)
    m_yy = s * 6.0 * jnp.ones_like(x)
    return m, m_x, m_y, m_xx, m_yy


def apply_parameterization(bc_mode: str, net, net_x, net_y, net_xx, net_yy,
                           xy, p, family, c_scale: float = 12.0):
    x = xy[:, 0]
    y = xy[:, 1]

    if bc_mode == "soft":
        return net, net_x, net_y, net_xx, net_yy

    if bc_mode == "hard_anti":
        s = (x + y)[None, :]
        return (s * net,
                net + s * net_x,
                net + s * net_y,
                2.0 * net_x + s * net_xx,
                2.0 * net_y + s * net_yy)

    if bc_mode == "adf_aq":

        h, h1, h2 = adf_aq_h_terms(x, p)
        s = (x + y)[None, :]
        phi = s * h
        phi_x = h + s * h1
        phi_y = h
        phi_xx = 2.0 * h1 + s * h2
        m = x ** 2 - y ** 2
        m_x = 2.0 * x
        m_y = -2.0 * y
        m_xx = 2.0 * jnp.ones_like(x)
        m_yy = -2.0 * jnp.ones_like(x)

        k = phi + m[None, :] * net
        k_x = phi_x + m_x[None, :] * net + m[None, :] * net_x
        k_y = phi_y + m_y[None, :] * net + m[None, :] * net_y
        k_xx = (phi_xx + m_xx[None, :] * net
                + 2.0 * m_x[None, :] * net_x + m[None, :] * net_xx)
        k_yy = (m_yy[None, :] * net
                + 2.0 * m_y[None, :] * net_y + m[None, :] * net_yy)
        return k, k_x, k_y, k_xx, k_yy

    if bc_mode == "seed":
        m, m_x, m_y, m_xx, m_yy = mask_terms(x, y)
        sa, sa_x, sa_y, sa_xx = seed_terms(family, x, y, p, c_scale)
        k = sa + m[None, :] * net
        k_x = sa_x + m_x[None, :] * net + m[None, :] * net_x
        k_y = sa_y + m_y[None, :] * net + m[None, :] * net_y
        k_xx = sa_xx + m_xx[None, :] * net + 2.0 * m_x[None, :] * net_x + m[None, :] * net_xx
        k_yy = m_yy[None, :] * net + 2.0 * m_y[None, :] * net_y + m[None, :] * net_yy
        return k, k_x, k_y, k_xx, k_yy

    if bc_mode == "hard_both":
        (sa, sa_x, sa_y, sa_xx, sa_yy,
         m, m_x, m_y, m_xx, m_yy) = hard_both_terms(
            family, x, y, p, c_scale)
        k = sa + m[None, :] * net
        k_x = sa_x + m_x[None, :] * net + m[None, :] * net_x
        k_y = sa_y + m_y[None, :] * net + m[None, :] * net_y
        k_xx = (sa_xx + m_xx[None, :] * net
                + 2.0 * m_x[None, :] * net_x + m[None, :] * net_xx)
        k_yy = (sa_yy + m_yy[None, :] * net
                + 2.0 * m_y[None, :] * net_y + m[None, :] * net_yy)
        return k, k_x, k_y, k_xx, k_yy

    raise ValueError(f"unknown bc_mode {bc_mode!r}")


def kernel_from_net(net, net_x, net_y, net_xx, net_yy, xy, p, family,
                    c_scale: float = 12.0):
    k, _, k_y, k_xx, k_yy = apply_parameterization(
        "seed", net, net_x, net_y, net_xx, net_yy, xy, p, family, c_scale)
    return k, k_y, k_xx, k_yy


def bc_losses(k_anti, k_diag, kx_diag, ky_diag, x_bc, p, family,
              c_scale: float = 12.0):


    a, a_x, _, lam, c = family_coeffs(family, x_bc, p, c_scale)
    q = lam + c[:, None]
    scale = jnp.mean(q ** 2, axis=1) + 1e-12
    loss_anti = jnp.mean(jnp.mean(k_anti ** 2, axis=1) / scale)


    res = 2.0 * a * (kx_diag + ky_diag) + a_x * k_diag - q
    num = jnp.mean(res ** 2, axis=1)
    den = jnp.mean(q ** 2, axis=1) + 1e-12
    loss_diag = jnp.mean(num / den)
    return loss_anti, loss_diag


def sample_boundary(key, n: int, half: float, positive_x: bool = False,
                    negative_x: bool = False):
    if positive_x:
        x = jax.random.uniform(key, (n,), minval=0.0, maxval=half)
    elif negative_x:
        x = jax.random.uniform(key, (n,), minval=-half, maxval=0.0)
    else:
        x = jax.random.uniform(key, (n,), minval=-half, maxval=half)
    return x


def pde_residual(k, k_y, k_xx, k_yy, xy, p, family, c_scale: float = 12.0):
    x = xy[:, 0]
    y = xy[:, 1]
    a_x, _, _, _, c = family_coeffs(family, x, p, c_scale)
    a_y, a1_y, a2_y, lam_y, _ = family_coeffs(family, y, p, c_scale)

    R = (a_x * k_xx
         - a_y * k_yy
         - 2.0 * a1_y * k_y
         - a2_y * k
         - (lam_y + c[:, None]) * k)
    return R


def relative_residual(R, k, k_y, k_xx, k_yy, xy, p, family,
                      c_scale: float = 12.0, norm: str = "terms"):
    x = xy[:, 0]
    y = xy[:, 1]
    a_x, _, _, _, c = family_coeffs(family, x, p, c_scale)
    a_y, a1_y, a2_y, lam_y, _ = family_coeffs(family, y, p, c_scale)

    num = jnp.mean(R ** 2, axis=1)
    if norm == "qk":
        qk = (lam_y + c[:, None]) * k
        den = jnp.mean(qk ** 2, axis=1) + 1e-12
    elif norm == "terms":
        t1 = a_x * k_xx
        t2 = a_y * k_yy + 2.0 * a1_y * k_y + a2_y * k
        t3 = (lam_y + c[:, None]) * k
        den = jnp.mean(t1 ** 2 + t2 ** 2 + t3 ** 2, axis=1) + 1e-12
    else:
        raise ValueError(f"unknown pde norm {norm!r}")
    return jnp.mean(num / den)


def trunk_derivs_2d(trunk_apply, trunk_params, xy):
    def scalar(c):
        return trunk_apply(trunk_params, c)

    def one(c):
        f = scalar(c)
        J = jax.jacfwd(scalar)(c)
        H = jax.hessian(scalar)(c)
        return f, J[:, 0], J[:, 1], H[:, 0, 0], H[:, 1, 1]

    return jax.vmap(one)(xy)


def trunk_value_2d(trunk_apply, trunk_params, xy):
    return jax.vmap(lambda c: trunk_apply(trunk_params, c))(xy)


def trunk_derivs_1d(trunk_apply, trunk_params, x):
    def scalar(c):
        return trunk_apply(trunk_params, c)

    def one(c):
        f = scalar(c)
        d1 = jax.jacfwd(scalar)(c)
        d2 = jax.jacfwd(jax.jacfwd(scalar))(c)
        return f, d1[:, 0], d2[:, 0, 0]

    return jax.vmap(one)(x)


def trunk_value_1d(trunk_apply, trunk_params, x):
    return jax.vmap(lambda c: trunk_apply(trunk_params, c))(x)


def sample_colloc(key, n: int, half: float, margin: float,
                  positive_x: bool = False, negative_x: bool = False):
    u = jax.random.uniform(key, (n, 3))
    absx = 2.0 * margin + (half - 2.0 * margin) * u[:, 0]
    if positive_x:
        x = absx
    elif negative_x:
        x = -absx
    else:
        x = jnp.where(u[:, 1] < 0.5, -absx, absx)
    y = (2.0 * u[:, 2] - 1.0) * (absx - margin)
    return jnp.stack([x, y], axis=1)

