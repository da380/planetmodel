"""The grid transforms of `planetmodel.harmonics` against the scipy basis."""
import numpy as np
import pytest

from planetmodel import (analyse_grid, equiangular, gauss_legendre, real_harmonics,
                         synthesise, synthesise_grid)
from planetmodel.randomfield import SphericalGRF

pytestmark = pytest.mark.harmonics
pytest.importorskip("pyshtools", reason="needs the planetmodel[harmonics] extra")


def random_coefficients(L, rng, *extra):
    c = rng.standard_normal((2, L + 1, L + 1) + extra)
    for l in range(L + 1):
        c[0, l, l + 1:] = 0.0
        c[1, l, 0] = 0.0
        c[1, l, l + 1:] = 0.0
    return c


def test_backend_is_ducc_when_installed():
    pytest.importorskip("ducc0")
    from planetmodel.harmonics import _pyshtools
    assert _pyshtools().backends.preferred_backend() == "ducc"


def test_synthesis_on_the_grid_matches_the_basis():
    L = 7
    rng = np.random.default_rng(0)
    c = random_coefficients(L, rng)
    grid = gauss_legendre(L)
    got = synthesise_grid(c, grid)
    assert got.shape == (grid.ntheta, grid.nphi)
    want = synthesise(c, grid.colatitudes[:, None], grid.longitudes[None, :])
    assert np.allclose(got, want, atol=1e-12)


def test_extra_axes_and_a_wider_grid():
    L = 5
    rng = np.random.default_rng(1)
    c = random_coefficients(L, rng, 3, 2)
    grid = gauss_legendre(9)                       # band above the coefficients
    got = synthesise_grid(c, grid)
    assert got.shape == (3, 2, grid.ntheta, grid.nphi)
    Y = real_harmonics(L, grid.colatitudes[:, None], grid.longitudes[None, :])
    want = np.einsum("slmab,slmtp->abtp", c, Y)
    assert np.allclose(got, want, atol=1e-12)


def test_analysis_inverts_synthesis():
    L = 8
    rng = np.random.default_rng(2)
    c = random_coefficients(L, rng, 4)
    grid = gauss_legendre(L)
    back = analyse_grid(synthesise_grid(c, grid), grid)
    assert back.shape == c.shape
    assert np.allclose(back, c, atol=1e-11)
    low = analyse_grid(synthesise_grid(c, grid), grid, lmax=3)
    assert low.shape == (2, 4, 4, 4) and np.allclose(low, c[:, :4, :4], atol=1e-11)


def test_refusals_name_the_grid():
    c = random_coefficients(4, np.random.default_rng(0))
    with pytest.raises(ValueError, match="Gauss-Legendre"):
        synthesise_grid(c, equiangular(8, 16))
    with pytest.raises(ValueError, match="longitudes"):
        synthesise_grid(c, gauss_legendre(4, nphi=16))
    with pytest.raises(ValueError, match="cannot hold"):
        synthesise_grid(c, gauss_legendre(3))
    with pytest.raises(ValueError, match="shape"):
        synthesise_grid(c[0], gauss_legendre(4))
    with pytest.raises(ValueError, match="trailing shape"):
        analyse_grid(np.zeros((3, 4)), gauss_legendre(4))
    with pytest.raises(ValueError, match="lmax"):
        analyse_grid(np.zeros((5, 9)), gauss_legendre(4), lmax=6)


def test_shell_sample_on_a_grid_has_the_marginal_variance():
    g = SphericalGRF(0.7, 1.0, 2.0, 0.15, sigma=1.3, lmax=10)
    grid = gauss_legendre(g.lmax)
    values = g.sample_grid(grid, rng=5)
    assert values.shape == (g.r.size, grid.ntheta, grid.nphi)
    w = grid.weights[:, None] * (2 * np.pi / grid.nphi) / (4 * np.pi)
    mean_sq = np.sum(values[3] ** 2 * w)
    # one sample: the sphere average of u^2 estimates sigma^2 to within
    # the sampling scatter of ~ (2 / N_modes)^(1/2)
    assert abs(mean_sq / g.sigma[3] ** 2 - 1.0) < 0.5
    # against the point synthesis at the grid nodes
    c = g.sample(rng=5)
    want = synthesise_grid(c, grid)
    assert np.allclose(values, want)
