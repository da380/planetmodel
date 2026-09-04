"""gll.py -- Gauss-Lobatto-Legendre nodes, weights and Lagrange machinery.

Everything here lives on the reference element [-1, 1] and knows
nothing of models or physics: the GLL quadrature rule, the differentiation
matrix of the associated Lagrange basis, and evaluation of that basis at
arbitrary points.
"""
from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre, roots_jacobi

__all__ = ["gll_points_weights", "lagrange_derivative_matrix", "lagrange_basis"]


def gll_points_weights(ngll: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Lobatto-Legendre points and weights on [-1, 1].

    The ngll >= 2 nodes are the endpoints plus the ngll - 2 zeros of
    P'_N with N = ngll - 1, computed stably as the zeros of the Jacobi
    polynomial P^(1,1)_(N-1); the weights are 2 / (N (N+1) P_N(x)^2).
    The quadrature integrates polynomials of degree <= 2N - 1 exactly.
    """
    if ngll < 2:
        raise ValueError("need at least two GLL points")
    N = ngll - 1
    if ngll == 2:
        x = np.array([-1.0, 1.0])
    else:
        interior = np.sort(roots_jacobi(N - 1, 1.0, 1.0)[0])
        x = np.concatenate(([-1.0], interior, [1.0]))
    w = 2.0 / (N * (N + 1) * eval_legendre(N, x) ** 2)
    return x, w


def lagrange_derivative_matrix(x) -> np.ndarray:
    """Derivative matrix D[k, i] = l_i'(x_k) on the GLL nodes x.

    Closed form for GLL nodes (e.g. Canuto et al. 1988): off the
    diagonal D[k, i] = P_N(x_k) / (P_N(x_i) (x_k - x_i)); the diagonal
    vanishes except for D[0, 0] = -N(N+1)/4 and D[N, N] = +N(N+1)/4.
    The index convention (derivative of basis function i evaluated at
    node k) matches the 'hp(knode, inode)' arrays of standard SEM
    codes, so quadrature sums read  sum_k w_k f(x_k) D[k, i] D[k, j].
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    N = n - 1
    P = eval_legendre(N, x)
    D = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            if k != i:
                D[k, i] = P[k] / (P[i] * (x[k] - x[i]))
    D[0, 0] = -0.25 * N * (N + 1)
    D[-1, -1] = 0.25 * N * (N + 1)
    return D


def lagrange_basis(nodes, x) -> np.ndarray:
    """Values L[j, i] = l_i(x_j) of the Lagrange basis at points x.

    Uses the second barycentric form for stability; points coinciding
    with a node are handled exactly.  Useful for interpolating a
    spectral-element solution off its nodes.
    """
    nodes = np.asarray(nodes, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    diff = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(diff, 1.0)
    wb = 1.0 / diff.prod(axis=1)          # barycentric weights
    d = x[:, None] - nodes[None, :]
    L = np.zeros((x.size, nodes.size))
    hit = d == 0.0
    on_node = hit.any(axis=1)
    L[on_node] = hit[on_node]
    off = ~on_node
    t = wb[None, :] / d[off]
    L[off] = t / t.sum(axis=1, keepdims=True)
    return L
