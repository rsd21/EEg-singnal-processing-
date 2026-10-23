"""Spherical spline interpolation (Perrin et al., 1989, 1990).

Potentials on the scalp are modelled as

    V(r) = c0 + sum_i c_i * g(cos(r, r_i))
    g(x) = 1/(4 pi) * sum_n (2n + 1) / (n (n + 1))^m * P_n(x)

with Legendre polynomials P_n and stiffness m (4 is standard). The same
coefficients give the surface Laplacian / current source density through

    h(x) = 1/(4 pi) * sum_n (2n + 1) / (n (n + 1))^(m - 1) * P_n(x)

which is what spherical-spline CSD uses (Perrin 1990, Kayser & Tenke 2006).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval


def _unit(xyz: np.ndarray) -> np.ndarray:
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    n = np.linalg.norm(xyz, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return xyz / n


def calc_g(cosang: np.ndarray, stiffness: int = 4, n_terms: int = 50) -> np.ndarray:
    n = np.arange(1, n_terms + 1)
    factors = (2 * n + 1) / ((n * (n + 1)) ** stiffness * 4 * np.pi)
    return legval(np.clip(cosang, -1, 1), np.concatenate([[0.0], factors]))


def calc_h(cosang: np.ndarray, stiffness: int = 4, n_terms: int = 50) -> np.ndarray:
    n = np.arange(1, n_terms + 1)
    factors = (2 * n + 1) / ((n * (n + 1)) ** (stiffness - 1) * 4 * np.pi)
    return legval(np.clip(cosang, -1, 1), np.concatenate([[0.0], factors]))


def interpolation_matrix(pos_from: np.ndarray, pos_to: np.ndarray, alpha: float = 1e-5,
                         stiffness: int = 4, n_terms: int = 50) -> np.ndarray:
    """Matrix M so that ``values_to = M @ values_from``."""
    a = _unit(pos_from)
    b = _unit(pos_to)
    n_from = len(a)
    g_from = calc_g(a @ a.T, stiffness, n_terms)
    g_to = calc_g(b @ a.T, stiffness, n_terms)
    g_from.flat[:: n_from + 1] += alpha
    c = np.zeros((n_from + 1, n_from + 1))
    c[:n_from, :n_from] = g_from
    c[:n_from, n_from] = 1.0
    c[n_from, :n_from] = 1.0
    c_inv = np.linalg.pinv(c)
    return np.hstack([g_to, np.ones((len(b), 1))]) @ c_inv[:, :n_from]


def csd_matrix(pos: np.ndarray, lambda2: float = 1e-5, stiffness: int = 4,
               n_terms: int = 50) -> np.ndarray:
    """Matrix T so that ``csd = T @ potentials`` (unit-sphere units, µV/r²)."""
    u = _unit(pos)
    cos = u @ u.T
    g = calc_g(cos, stiffness, n_terms)
    h = calc_h(cos, stiffness, n_terms)
    g.flat[:: len(g) + 1] += lambda2
    g_inv = np.linalg.inv(g)
    tc = g_inv.sum(axis=0)
    sgi = tc.sum()
    # Coefficients C = Gi (V - c0), with c0 = sum(Gi V) / sum(Gi) (constant term)
    c_op = g_inv - np.outer(tc, tc) / sgi
    return h @ c_op
