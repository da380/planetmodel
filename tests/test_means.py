"""Spherical means: exact where the field can give them, quadrature else.

The grid's quadrature is exact on the harmonics of its band; a harmonic
field's mean is its c_00 term and a gridded field's the mean of its
interpolant at every node joined by its radial rule, both against a
quadrature oracle; the perturbed kinds against their formulas; a model
spherically averaged re-derives its moduli from the means.
"""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from planetmodel import (PREM, AnalyticField, AngularGrid, GriddedField,
                         HarmonicField, PerturbedField, RadialField,
                         driscoll_healy, equiangular, gauss_legendre, perturbed,
                         spherical_mean, spherically_averaged)
from planetmodel.character import DENSITY, SCALAR, VECTOR
from planetmodel.harmonics import packing
from planetmodel.testing import check_field, check_model

INTERVAL = (0.5, 1.0)
FINE = gauss_legendre(48)


def oracle(field, r):
    """The mean by quadrature on a fine Gauss-Legendre grid."""
    v = field.evaluate(np.asarray(r)[:, None, None], FINE.colatitudes[None, :, None],
                       FINE.longitudes[None, None, :])
    return FINE.mean(v)


def smooth(r, t, p):
    st = np.sin(t)
    return (1.0 + r ** 2) * np.exp(0.3 * st * np.cos(p) - 0.2 * st * np.sin(p)
                                   + 0.1 * np.cos(t))


@pytest.mark.parametrize("grid", [gauss_legendre(6), driscoll_healy(5),
                                  equiangular(9, 16), equiangular(9, 16, poles=True)],
                         ids=repr)
def test_the_grid_mean_is_exact_on_its_band(grid):
    from planetmodel import real_harmonics
    Y = real_harmonics(grid.lmax, grid.colatitudes[:, None], grid.longitudes[None, :])
    means = grid.mean(Y)
    want = np.zeros_like(means)
    want[0, 0, 0] = np.sqrt(4 * np.pi) / (4 * np.pi)
    assert np.allclose(means, want, atol=1e-14)
    with pytest.raises(ValueError, match="no quadrature"):
        AngularGrid(grid.colatitudes, grid.longitudes).mean(Y)


def test_a_radial_field_is_its_own_mean():
    f = RadialField(INTERVAL, lambda r: r ** 2, character=DENSITY)
    assert spherical_mean(f) is f


def test_a_harmonic_field_mean_is_its_degree_zero_term():
    rng = np.random.default_rng(0)
    L = 6
    c = np.zeros((6, 2, L + 1, L + 1))
    s, l, m = packing(L)
    c[..., s, l, m] = rng.standard_normal((6, s.size))
    f = HarmonicField(INTERVAL, CubicSpline(np.linspace(*INTERVAL, 6), c, axis=0),
                      character=DENSITY, name="rho")
    fbar = spherical_mean(f)
    assert isinstance(fbar, RadialField) and fbar.character == DENSITY
    assert fbar.name == "rho" and fbar.radial_degree is None
    r = rng.uniform(*INTERVAL, 12)
    assert np.allclose(fbar(r), oracle(f, r), rtol=0, atol=1e-12)
    check_field(fbar)


@pytest.mark.parametrize("radial", ["polynomial", "linear"])
@pytest.mark.parametrize("grid", [gauss_legendre(8),
                                  AngularGrid(np.linspace(0.2, 2.9, 9),
                                              np.linspace(0.1, 6.0, 15))], ids=repr)
def test_a_gridded_field_mean_is_the_mean_of_its_rules(grid, radial):
    """At every radius the mean of the field as its lateral and radial
    rules define it: against fine quadrature of the gridded field itself."""
    g = GriddedField.sampled(AnalyticField(INTERVAL, smooth), grid,
                             breaks=[0.5, 0.7, 1.0], ngll=4, radial=radial,
                             lateral="cubic")
    fbar = spherical_mean(g)
    r = np.random.default_rng(1).uniform(*INTERVAL, 8)
    assert np.allclose(fbar(r), oracle(g, r), rtol=1e-9)
    tol = 1e-3 if radial == "polynomial" else 1e-2
    assert np.allclose(fbar(r), oracle(AnalyticField(INTERVAL, smooth), r), rtol=tol)


def test_a_field_without_a_mean_of_its_own_needs_a_grid():
    f = AnalyticField(INTERVAL, lambda r, t, p: r + r ** 2 * np.cos(t) + 0 * p)
    with pytest.raises(ValueError, match="grid="):
        spherical_mean(f)
    fbar = spherical_mean(f, grid=gauss_legendre(4))
    r = np.array([0.6, 0.9])
    assert np.allclose(fbar(r), r, atol=1e-14)
    with pytest.raises(ValueError, match="rank 0"):
        spherical_mean(AnalyticField(INTERVAL, lambda r, t, p: np.stack(
            np.broadcast_arrays(r, t, p), -1), character=VECTOR))


def test_a_quadrature_mean_costs_its_distinct_radii(monkeypatch):
    """The field is evaluated on the grid once for each distinct radius,
    however often a radius repeats, and a bounded number of values at a
    time; the mean has the shape of the radii asked for."""
    from planetmodel import means

    sizes = []

    def counted(r, t, p):
        sizes.append(np.broadcast(r, t, p).size)
        return r + r ** 2 * np.cos(t) + 0 * p

    grid = gauss_legendre(4)
    nodes = grid.ntheta * grid.nphi
    fbar = spherical_mean(AnalyticField(INTERVAL, counted), grid=grid)
    r = np.array([[0.6, 0.9, 0.6], [0.9, 0.9, 0.6]])
    sizes.clear()
    assert np.allclose(fbar(np.tile(r, (50, 1))), np.tile(r, (50, 1)), atol=1e-14)
    assert sum(sizes) == 2 * nodes

    monkeypatch.setattr(means, "_BLOCK_VALUES", 3 * nodes)
    fbar = spherical_mean(AnalyticField(INTERVAL, counted), grid=grid)
    r = np.linspace(0.5, 1.0, 10).reshape(2, 5)
    sizes.clear()
    assert np.allclose(fbar(r), r, atol=1e-14)
    assert sum(sizes) == r.size * nodes and max(sizes) <= 3 * nodes
    assert fbar(np.float64(0.7)).shape == ()


REF = RadialField(INTERVAL, lambda r: 3.0 + r ** 2, character=DENSITY, name="rho")


def dlog(r, t, p):
    return 0.2 * r * np.sin(t) * np.cos(p) + 0.1 * np.cos(t) ** 2


@pytest.mark.parametrize("kind", ["absolute", "relative"])
def test_the_perturbed_kinds_combine_their_parts(kind):
    char = DENSITY if kind == "absolute" else SCALAR
    d = AnalyticField(INTERVAL, dlog, character=char)
    f = PerturbedField(REF, d, kind=kind)
    grid = gauss_legendre(6)
    fbar = spherical_mean(f, grid=grid)
    r = np.array([0.55, 0.8, 0.95])
    dbar = 0.1 / 3.0                                 # the mean of cos^2
    want = REF(r) + dbar if kind == "absolute" else REF(r) * (1.0 + dbar)
    assert np.allclose(fbar(r), want, rtol=1e-14)
    assert fbar.character == DENSITY and fbar.name == "rho"


def test_a_log_perturbation_takes_the_mean_of_its_exponential():
    d = AnalyticField(INTERVAL, dlog)
    f = PerturbedField(REF, d, kind="log")
    with pytest.raises(ValueError, match="grid="):
        spherical_mean(f)
    r = np.array([0.55, 0.8])
    fbar = spherical_mean(f, grid=gauss_legendre(24))
    assert np.allclose(fbar(r), oracle(f, r), rtol=1e-12)
    gridded = GriddedField.sampled(d, gauss_legendre(24), ngll=4, lateral="cubic")
    own = spherical_mean(PerturbedField(REF, gridded, kind="log"))
    assert np.allclose(own(r), oracle(f, r), rtol=1e-6)
    radial = PerturbedField(REF, RadialField(INTERVAL, lambda r: 0.1 * r), kind="log")
    assert np.allclose(spherical_mean(radial)(r), REF(r) * np.exp(0.1 * r))


def test_a_model_spherically_averaged_rederives_its_moduli():
    lo, hi = 3480e3, 5701e3
    m = PREM()

    def dlnvs(r, t, p):
        return 0.02 * np.cos(t) ** 2 + 0.01 * np.sin(t) * np.cos(p) * r / hi

    mp = perturbed(m, "vsv", AnalyticField((lo, hi), dlnvs), kind="relative")
    avg = spherically_averaged(mp, grid=gauss_legendre(8))
    check_model(avg)
    r = np.linspace(3.7e6, 5.5e6, 7)
    layer, base = avg.layer(3), m.layer(3)
    vsv = base["vsv"](r) * (1.0 + 0.02 / 3.0)
    assert layer["vsv"].is_radial and layer["L"].is_radial
    assert np.allclose(layer["vsv"](r), vsv, rtol=1e-12)
    assert np.allclose(layer["L"](r), base["rho"](r) * vsv ** 2, rtol=1e-12)
    assert spherically_averaged(m).layer(3)["vsv"] is m.layer(3)["vsv"]
