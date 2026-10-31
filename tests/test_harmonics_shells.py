"""`ShellTransform`, the batched shell route, against the grid transforms."""
import numpy as np
import pytest

from planetmodel import AngularGrid, driscoll_healy, equiangular, gauss_legendre
from planetmodel.harmonics import ShellTransform, analyse_grid, synthesise_grid


def random_coefficients(L, rng, *extra):
    c = rng.standard_normal((2, L + 1, L + 1) + extra)
    for l in range(L + 1):
        c[0, l, l + 1:] = 0.0
        c[1, l, 0] = 0.0
        c[1, l, l + 1:] = 0.0
    return c


GRIDS = [gauss_legendre(0), gauss_legendre(1), gauss_legendre(4), gauss_legendre(12),
         gauss_legendre(48), gauss_legendre(72), gauss_legendre(5, phi0=0.4)]


@pytest.mark.parametrize("grid", GRIDS, ids=repr)
def test_the_batched_routes_agree_with_the_grid_transforms(grid):
    rng = np.random.default_rng(0)
    c = random_coefficients(grid.lmax, rng, 7)
    t = ShellTransform(grid)
    assert t.lmax == grid.lmax
    values = t.synthesise(c)
    assert values.shape == (7, grid.ntheta, grid.nphi)
    want = synthesise_grid(c, grid)
    scale = max(1.0, np.abs(want).max())
    assert np.allclose(values, want, rtol=0.0, atol=1e-12 * scale)
    assert np.allclose(t.analyse(values), analyse_grid(values, grid),
                       rtol=0.0, atol=1e-11 * scale)


@pytest.mark.parametrize("grid", GRIDS, ids=repr)
def test_the_batched_round_trip_returns_the_coefficients(grid):
    rng = np.random.default_rng(1)
    c = random_coefficients(grid.lmax, rng, 3)
    t = ShellTransform(grid)
    assert np.allclose(t.analyse(t.synthesise(c)), c, atol=1e-11)


def test_a_single_shell_and_the_entries_of_no_harmonic():
    L = 6
    grid = gauss_legendre(L)
    t = ShellTransform(grid)
    rng = np.random.default_rng(2)
    c = random_coefficients(L, rng, 1)
    values = t.synthesise(c)
    assert values.shape == (1, grid.ntheta, grid.nphi)
    assert np.allclose(values[0], synthesise_grid(c[..., 0], grid), atol=1e-12)
    assert np.allclose(t.analyse(values), c, atol=1e-11)
    # analysis of values of no band is the same projection, zero where no
    # harmonic is
    v = rng.standard_normal((2, grid.ntheta, grid.nphi))
    out = t.analyse(v)
    assert np.allclose(out, analyse_grid(v, grid), rtol=0.0, atol=1e-12)
    assert np.all(out[1, :, 0] == 0.0)
    l, m = np.meshgrid(np.arange(L + 1), np.arange(L + 1), indexing="ij")
    assert np.all(out[:, m > l] == 0.0)


def test_refusals_name_what_is_wrong():
    with pytest.raises(ValueError, match="Gauss-Legendre"):
        ShellTransform(driscoll_healy(4))
    with pytest.raises(ValueError, match="Gauss-Legendre"):
        ShellTransform(equiangular(9, 20))
    g = gauss_legendre(4)
    with pytest.raises(ValueError, match="Gauss-Legendre"):
        ShellTransform(AngularGrid(g.colatitudes, g.longitudes, lmax=4))
    with pytest.raises(ValueError, match="weights"):
        ShellTransform(AngularGrid(g.colatitudes, g.longitudes,
                                   kind="gauss_legendre", lmax=4))
    with pytest.raises(ValueError, match="9 longitudes"):
        ShellTransform(gauss_legendre(4, nphi=12))
    g6 = gauss_legendre(6)
    with pytest.raises(ValueError, match="5 colatitudes"):
        ShellTransform(AngularGrid(g6.colatitudes, g.longitudes,
                                   kind="gauss_legendre", lmax=4,
                                   weights=g6.weights))
    with pytest.raises(ValueError, match="colatitudes are not those"):
        ShellTransform(AngularGrid(np.linspace(0.3, 2.8, 5), g.longitudes,
                                   kind="gauss_legendre", lmax=4,
                                   weights=np.ones(5)))
    t = ShellTransform(g)
    with pytest.raises(ValueError, match="shape"):
        t.synthesise(np.zeros((2, 5, 5)))              # no shell axis
    with pytest.raises(ValueError, match="shape"):
        t.synthesise(np.zeros((2, 4, 5, 3)))           # not a coefficient array
    with pytest.raises(ValueError, match="shape"):
        t.analyse(np.zeros((5, 9)))                    # no shell axis
    with pytest.raises(ValueError, match="shape"):
        t.analyse(np.zeros((3, 5, 8)))                 # too few longitudes
