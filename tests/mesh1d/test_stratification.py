"""The Bullen parameter and the buoyancy frequency, against closed forms
and a finite-difference oracle."""
import numpy as np
import pytest

from planetmodel import Elastic, Geometry, Model, SelfGravitating, Skeleton
from planetmodel.catalogue import LayeredIsotropicElastic, PREM
from planetmodel.character import DENSITY
from planetmodel.fields import RadialField
from planetmodel.layerfunction import polynomial_layer
from planetmodel.materials import kappa_mu
from planetmodel.mesh1d.gravity import gravity
from planetmodel.mesh1d.stratification import (
    brunt_vaisala_squared,
    bullen_excess,
    bullen_parameter,
)
from planetmodel.units import G_SI


class Planet(Elastic, SelfGravitating, Model):
    """A model given by rho, kappa and mu."""


def adams_williamson(excess, *, rho0=3.0, alpha=0.2, c=0.6):
    """A fluid core of density rho0 (1 - alpha x^2), x = r / c, whose bulk
    modulus is 1 + excess times the neutral one, under a uniform solid
    shell.  With g = (4 pi G rho0 / 3) r (1 - (3 alpha / 5) x^2), the
    neutral modulus -rho^2 g / rho' is
    (2 pi G rho0^2 c^2 / (3 alpha)) (1 - alpha x^2)^2 (1 - (3 alpha / 5) x^2),
    a polynomial, so the Bullen parameter is 1 + excess throughout."""
    sk = Skeleton([0.0, c, 1.0])
    core, shell = sk.interval(0), sk.interval(1)
    kappa0 = (1.0 + excess) * 2.0 * np.pi * G_SI * rho0 ** 2 * c ** 2 / (3.0 * alpha)
    a = alpha / c ** 2
    square = np.polynomial.polynomial.polymul([1.0, 0.0, -a], [1.0, 0.0, -a])
    kappa = kappa0 * np.polynomial.polynomial.polymul(square, [1.0, 0.0, -0.6 * a])

    def field(interval, coeffs, name):
        return RadialField(interval, polynomial_layer(interval, coeffs),
                           character=DENSITY, name=name)

    layers = [
        {"rho": field(core, [rho0, 0.0, -rho0 * a], "rho"),
         "kappa": field(core, kappa, "kappa"), "mu": field(core, [0.0], "mu")},
        {"rho": field(shell, [1.0], "rho"),
         "kappa": field(shell, [5.0 * G_SI], "kappa"),
         "mu": field(shell, [2.0 * G_SI], "mu")},
    ]
    return Planet(Geometry(sk), layers)


def finite_difference_bullen(model, i, r, *, step=1e-6):
    """The oracle: rho' by a central difference, the rest read directly."""
    layer = model.layers[i]
    lo, hi = layer.interval
    h = step * (hi - lo)
    rho = layer["rho"]
    slope = (rho(r + h) - rho(r - h)) / (2.0 * h)
    kappa = kappa_mu(layer)[0](r)
    return -kappa * slope / (rho(r) ** 2 * gravity(model, r))


@pytest.mark.parametrize("excess", [0.0, 0.02, 0.5])
def test_an_adams_williamson_core_has_a_uniform_bullen_parameter(excess):
    m = adams_williamson(excess)
    r = np.linspace(0.0, 0.6, 13)[:-1]
    assert np.allclose(bullen_parameter(m, r), 1.0 + excess, rtol=1e-12)
    assert np.allclose(bullen_excess(m, r), excess, atol=1e-12)
    assert np.allclose(m.bullen_excess(r), excess, atol=1e-12)


def test_the_centre_is_the_limit_of_the_ratio():
    m = adams_williamson(0.3)
    assert np.isclose(bullen_parameter(m, 0.0), bullen_parameter(m, 1e-6), rtol=1e-9)
    # a density with a slope at the centre is infinitely stratified there
    cusp = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[(2.0, -1.0), 1.0],
                                   vp=[1.0, 2.0], vs=[0.0, 1.0])
    assert bullen_parameter(cusp, 0.0) == np.inf
    assert brunt_vaisala_squared(cusp, 0.0) == 0.0


def test_against_a_finite_difference_on_prem():
    m = PREM()
    for name in ("inner_core", "outer_core"):
        i = m.layer(name).index
        lo, hi = m.layer(name).interval
        r = np.linspace(lo, hi, 9)[1:-1]
        assert np.allclose(bullen_parameter(m, r), finite_difference_bullen(m, i, r),
                           rtol=1e-6)


def test_prems_outer_core_is_unstable_in_places():
    m = PREM()
    lo, hi = m.layer("outer_core").interval
    r = np.linspace(lo, hi, 4001)[1:-1]
    eta = bullen_parameter(m, r)
    assert np.isclose(eta[0], 1.028, atol=1e-3)     # above the inner core
    assert np.isclose(eta[-1], 0.983, atol=1e-3)    # beneath the mantle
    assert np.isclose(np.mean(eta < 1.0), 0.446, atol=5e-3)
    # the unstable intervals, in km
    edges = r[np.flatnonzero(np.diff(eta < 1.0))] / 1e3
    assert np.allclose(edges, [1686.0, 2344.0, 3130.0], atol=2.0)


def test_the_buoyancy_frequency_restates_the_bullen_parameter():
    m = PREM()
    b = m.skeleton.boundaries
    r = np.concatenate([np.linspace(lo, hi, 7)[1:-1] for lo, hi in zip(b[:-1], b[1:])])
    n2 = brunt_vaisala_squared(m, r)
    g = gravity(m, r)
    expected = np.empty(r.shape)
    for j, x in enumerate(r):
        layer = m.layers[int(np.searchsorted(b, x, side="right") - 1)]
        rho, kappa = layer["rho"](x), kappa_mu(layer)[0](x)
        expected[j] = rho * g[j] ** 2 / kappa * (bullen_parameter(m, x) - 1.0)
    assert np.allclose(n2, expected, rtol=1e-10, atol=1e-18)
    assert brunt_vaisala_squared(m, 0.0) == 0.0
    assert n2.shape == r.shape
    assert bullen_parameter(m, r.reshape(-1, 1)).shape == (r.size, 1)


def test_a_boundary_is_answered_from_the_layer_above():
    m = adams_williamson(0.25)
    # the shell is uniform, so its Bullen parameter is zero
    assert bullen_parameter(m, 0.6) == 0.0
    assert np.isclose(bullen_parameter(m, 0.6 - 1e-9), 1.25, rtol=1e-6)


def test_a_radius_outside_the_model_is_refused():
    m = adams_williamson(0.0)
    with pytest.raises(ValueError, match="radii must lie"):
        bullen_parameter(m, 1.5)
