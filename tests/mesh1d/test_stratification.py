"""The Bullen parameter and the buoyancy frequency, against closed forms
and a finite-difference oracle."""
import numpy as np
import pytest
from scipy.interpolate import PPoly

from planetmodel import Elastic, Geometry, Model, SelfGravitating, Skeleton
from planetmodel.catalogue import LayeredIsotropicElastic, PREM
from planetmodel.character import DENSITY
from planetmodel.fields import RadialField
from planetmodel.layerfunction import PolynomialLayer, polynomial_layer
from planetmodel.materials import kappa_mu
from planetmodel.mesh1d.gravity import gravity
from planetmodel.mesh1d.stratification import (
    brunt_vaisala_squared,
    bullen_excess,
    bullen_parameter,
    stratified_density,
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


# --- the density from the stratification -------------------------------------

AW = dict(rho0=3.0, alpha=0.2, c=0.6)


def aw_density(r):
    return AW["rho0"] * (1.0 - AW["alpha"] * (np.asarray(r) / AW["c"]) ** 2)


def planet_with_core_density(base, density):
    """`base` with the density of its core replaced by a layer function."""
    core = base.layers[0]
    rho = RadialField(core.interval, density, character=DENSITY, name="rho")
    fields = [{"rho": rho, "kappa": core["kappa"], "mu": core["mu"]},
              {n: base.layers[1][n] for n in ("rho", "kappa", "mu")}]
    return Planet(base.geometry, fields)


@pytest.mark.parametrize("excess", [0.0, 0.3])
def test_the_density_of_an_adams_williamson_core_is_recovered(excess):
    m = adams_williamson(excess, **AW)
    core = m.layers[0]
    kappa = kappa_mu(core)[0].function
    r = np.linspace(0.0, AW["c"], 41)
    layer_mass = m.mass(radius=AW["c"])
    for anchor in ({"rho_bottom": AW["rho0"]}, {"rho_top": float(aw_density(AW["c"]))},
                   {"mass": layer_mass}):
        d = stratified_density(core.interval, excess=excess, kappa=kappa, G=G_SI,
                               **anchor)
        assert isinstance(d, PolynomialLayer)
        assert np.allclose(d(r), aw_density(r), rtol=1e-11), anchor
    # by the velocity in the place of the modulus
    d = stratified_density(core.interval, excess=excess, G=G_SI,
                           vp=lambda x: np.sqrt(kappa(x) / aw_density(x)),
                           rho_bottom=AW["rho0"])
    assert np.allclose(d(r), aw_density(r), rtol=1e-11)


def test_the_mass_asked_for_is_the_mass_of_the_layer():
    m = adams_williamson(0.1, **AW)
    core = m.layers[0]
    d = stratified_density(core.interval, excess=0.4, G=G_SI,
                           kappa=kappa_mu(core)[0].function, mass=2.5)
    assert np.isclose(planet_with_core_density(m, d).mass(radius=AW["c"]), 2.5,
                      rtol=1e-11)


def test_a_layer_neutral_below_and_stratified_above():
    # the derived density, read back by the Bullen parameter, has the
    # excess it was derived from on each piece, and a breakpoint between
    m = adams_williamson(0.0, **AW)
    core = m.layers[0]
    mid = 0.5 * AW["c"]
    excess = PPoly(np.array([[0.0, 0.25]]), [0.0, mid, AW["c"]])
    d = stratified_density(core.interval, excess=excess, G=G_SI,
                           kappa=kappa_mu(core)[0].function, rho_bottom=AW["rho0"])
    assert mid in d.ppoly.x
    assert np.isclose(d(mid - 1e-12), d(mid + 1e-12), rtol=1e-11)
    held = planet_with_core_density(m, d)
    below, above = np.linspace(0.0, mid, 9)[1:-1], np.linspace(mid, AW["c"], 9)[:-1]
    assert np.allclose(bullen_excess(held, below), 0.0, atol=1e-9)
    assert np.allclose(bullen_excess(held, above), 0.25, atol=1e-9)
    # at the centre the excess is a limit that takes the second derivative
    # of the density, which a fitted polynomial holds less closely
    assert np.isclose(bullen_excess(held, 0.0), 0.0, atol=1e-6)


def test_prems_outer_core_is_recovered_from_its_own_excess():
    # the excess of a layer, of either sign, gives the layer's density back
    m = PREM()
    layer = m.layer("outer_core")
    lo, hi = layer.interval
    rho, kappa = layer["rho"], kappa_mu(layer)[0]

    def excess(r):
        r = np.asarray(r, dtype=float)
        return (-kappa(r) * rho.derivative(nu=1)(r)
                / (rho(r) ** 2 * gravity(m, r)) - 1.0)

    d = stratified_density((lo, hi), excess=excess, kappa=kappa.function, G=m.G,
                           mass_below=m.mass(radius=lo), rho_bottom=float(rho(lo)))
    r = np.linspace(lo, hi, 101)
    assert np.allclose(d(r), rho(r), rtol=1e-11)


def test_a_density_needs_one_stiffness_and_one_anchor():
    interval = (0.0, 1.0)
    with pytest.raises(ValueError, match="exactly one of vp and kappa"):
        stratified_density(interval, excess=0.0, G=1.0, rho_bottom=1.0)
    with pytest.raises(ValueError, match="exactly one of vp and kappa"):
        stratified_density(interval, excess=0.0, G=1.0, vp=1.0, kappa=1.0,
                           rho_bottom=1.0)
    with pytest.raises(ValueError, match="exactly one of rho_bottom"):
        stratified_density(interval, excess=0.0, G=1.0, vp=1.0)
    with pytest.raises(ValueError, match="exactly one of rho_bottom"):
        stratified_density(interval, excess=0.0, G=1.0, vp=1.0, rho_bottom=1.0,
                           mass=1.0)
    with pytest.raises(ValueError, match="must be positive"):
        stratified_density(interval, excess=0.0, G=1.0, vp=1.0, mass=-1.0)
    with pytest.raises(ValueError, match="mass_below"):
        stratified_density(interval, excess=0.0, G=1.0, vp=1.0, rho_bottom=1.0,
                           mass_below=1.0)
