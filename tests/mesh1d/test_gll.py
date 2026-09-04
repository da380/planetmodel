"""GLL quadrature, the differentiation matrix and the Lagrange basis."""
import numpy as np
import pytest

from planetmodel.mesh1d import (gll_points_weights, lagrange_basis,
                                lagrange_derivative_matrix)


@pytest.mark.parametrize("ngll", [2, 3, 5, 8])
def test_quadrature_is_exact_to_degree_2n_minus_3(ngll):
    x, w = gll_points_weights(ngll)
    assert x[0] == -1.0 and x[-1] == 1.0
    assert np.all(np.diff(x) > 0)
    assert w.sum() == pytest.approx(2.0)
    N = ngll - 1
    for p in range(2 * N):
        exact = 0.0 if p % 2 else 2.0 / (p + 1)
        assert np.sum(w * x ** p) == pytest.approx(exact, abs=1e-13)


def test_refuses_fewer_than_two_nodes():
    with pytest.raises(ValueError):
        gll_points_weights(1)


def test_derivative_matrix_differentiates_polynomials():
    x, _ = gll_points_weights(6)
    D = lagrange_derivative_matrix(x)
    for p in range(6):
        want = p * x ** (p - 1) if p else np.zeros_like(x)
        assert np.allclose(D @ (x ** p), want, atol=1e-12)


def test_lagrange_basis_interpolates():
    x, _ = gll_points_weights(5)
    L = lagrange_basis(x, x)
    assert np.allclose(L, np.eye(5))
    pts = np.linspace(-1, 1, 11)
    L = lagrange_basis(x, pts)
    f = x ** 3 - 0.5 * x
    assert np.allclose(L @ f, pts ** 3 - 0.5 * pts)
    assert np.allclose(L.sum(axis=1), 1.0)
