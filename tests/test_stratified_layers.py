"""A fluid layer described by its stratification: the density derived
from the Bullen excess, and kept in step with what it is derived from."""
import numpy as np
import pytest
from scipy.interpolate import PPoly

from planetmodel import Elastic, Geometry, Model, SelfGravitating, Skeleton, testing
from planetmodel.catalogue import LayeredIsotropicElastic
from planetmodel.character import DENSITY
from planetmodel.fields import ComposedField, RadialField
from planetmodel.layerfunction import polynomial_layer
from planetmodel.materials import kappa_mu
from planetmodel.mesh1d.stratification import BULLEN_EXCESS, is_neutral

ICB, CMB, SURFACE = 1221.5e3, 3480e3, 6371e3


def earth(*, inner=13000.0):
    """A solid inner core, a uniform fluid outer core and a mantle."""
    return LayeredIsotropicElastic(
        [0.0, ICB, CMB, SURFACE], rho=[inner, 11000.0, 4500.0],
        vp=[11000.0, 9000.0, 11000.0], vs=[3500.0, 0.0, 6000.0],
        layer_names=["inner_core", "outer_core", "mantle"])


def inside(layer, n=9):
    lo, hi = layer.interval
    return np.linspace(lo, hi, n + 2)[1:-1]


def test_a_fluid_layer_takes_its_density_from_its_bullen_excess():
    m = earth()
    s = m.with_bullen_excess("outer_core", 0.02)
    assert type(s) is type(m)
    core = s.layer("outer_core")
    r = inside(core)
    assert np.allclose(core[BULLEN_EXCESS](r), 0.02)
    assert np.allclose(s.bullen_excess(r), 0.02, atol=1e-9)
    # the layer keeps its mass, so gravity above it is what it was
    assert np.allclose(s.layer_masses(), m.layer_masses(), rtol=1e-11)
    above = inside(s.layer("mantle"))
    assert np.allclose(s.gravity(above), m.gravity(above), rtol=1e-11)
    # the velocity is the description: it is held, and the modulus follows
    assert core["vp"] is m.layer("outer_core")["vp"]
    assert np.allclose(kappa_mu(core)[0](r), core["rho"](r) * 9000.0 ** 2, rtol=1e-12)
    # the other layers are untouched
    for name in ("inner_core", "mantle"):
        assert s.layer(name).fields == m.layer(name).fields
    assert not s.is_neutral("outer_core")
    testing.check_model(s)


@pytest.mark.parametrize("anchor, at", [("rho_bottom", ICB), ("rho_top", CMB)])
def test_the_density_may_be_fixed_at_either_end(anchor, at):
    s = earth().with_bullen_excess("outer_core", 0.1, **{anchor: 10500.0})
    assert np.isclose(s.layer("outer_core")["rho"](at), 10500.0, rtol=1e-11)
    assert np.allclose(s.bullen_excess(inside(s.layer(1))), 0.1, atol=1e-9)


def test_a_layer_may_be_neutral_on_some_pieces():
    mid = 2400e3
    excess = PPoly(np.array([[0.05, 0.0]]), [ICB, mid, CMB])
    s = earth().with_bullen_excess(1, excess)
    lower, upper = np.linspace(ICB, mid, 7)[1:-1], np.linspace(mid, CMB, 7)[1:-1]
    assert np.allclose(s.bullen_excess(lower), 0.05, atol=1e-9)
    assert np.allclose(s.bullen_excess(upper), 0.0, atol=1e-9)
    assert not is_neutral(s.layer(1))
    assert earth().with_bullen_excess(1, 0.0).is_neutral(1)
    with pytest.raises(KeyError, match="holds no 'bullen_excess'"):
        earth().is_neutral(1)
    testing.check_model(s)


def test_an_edit_of_the_layer_derives_its_density_again():
    m = earth()
    s = m.with_bullen_excess(1, 0.02)
    r = inside(s.layer(1))
    # a new velocity
    faster = RadialField((ICB, CMB), polynomial_layer((ICB, CMB), [9500.0]),
                         name="vp")
    v = s.with_field(1, "vp", faster, replace=True)
    assert np.allclose(v.bullen_excess(r), 0.02, atol=1e-9)
    assert np.isclose(v.layer_masses()[1], m.layer_masses()[1], rtol=1e-11)
    assert not np.allclose(v.layer(1)["rho"](r), s.layer(1)["rho"](r), rtol=1e-6)
    # a new excess, by with_field as by with_bullen_excess
    e = RadialField((ICB, CMB), polynomial_layer((ICB, CMB), [0.3]),
                    name=BULLEN_EXCESS)
    w = s.with_field(1, BULLEN_EXCESS, e, replace=True)
    assert np.allclose(w.bullen_excess(r), 0.3, atol=1e-9)
    assert np.allclose(w.layer(1)["rho"](r),
                       m.with_bullen_excess(1, 0.3).layer(1)["rho"](r), rtol=1e-11)
    # a density given directly becomes the description
    d = s.with_field(1, "rho", m.layer(1)["rho"], replace=True)
    assert BULLEN_EXCESS not in d.layer(1)
    assert np.allclose(d.layer(1)["rho"](r), 11000.0)


def test_an_edit_beneath_derives_the_density_again():
    m = earth()
    s = m.with_bullen_excess(1, 0.02)
    heavier = s.with_field(0, "rho", m.layer(0)["rho"] * 1.1, replace=True)
    r = inside(s.layer(1))
    # more mass beneath, so more gravity: the same excess is another density
    assert np.allclose(heavier.bullen_excess(r), 0.02, atol=1e-9)
    change = heavier.layer(1)["rho"](r) / s.layer(1)["rho"](r) - 1.0
    assert np.max(np.abs(change)) > 1e-4
    assert np.isclose(heavier.layer_masses()[1], s.layer_masses()[1], rtol=1e-11)
    assert heavier.layer("mantle").fields == s.layer("mantle").fields
    testing.check_model(heavier)
    # an edit above changes nothing below
    above = s.with_field(2, "rho", m.layer(2)["rho"] * 1.1, replace=True)
    assert above.layer(1).fields == s.layer(1).fields


def test_two_stratified_layers_follow_an_edit_beneath_both():
    m = LayeredIsotropicElastic(
        [0.0, ICB, 2400e3, CMB, SURFACE], rho=[13000.0, 11500.0, 10500.0, 4500.0],
        vp=[11000.0, 9500.0, 8500.0, 11000.0], vs=[3500.0, 0.0, 0.0, 6000.0])
    s = m.with_bullen_excess(1, 0.0).with_bullen_excess(2, 0.04)
    out = s.with_field(0, "rho", m.layer(0)["rho"] * 0.9, replace=True)
    for i, e in ((1, 0.0), (2, 0.04)):
        assert np.allclose(out.bullen_excess(inside(out.layer(i))), e, atol=1e-9)
        assert np.isclose(out.layer_masses()[i], s.layer_masses()[i], rtol=1e-11)
    testing.check_model(out)


class Planet(Elastic, SelfGravitating, Model):
    """A model given by rho, kappa and mu."""


def test_a_layer_described_by_its_modulus_keeps_the_modulus():
    sk = Skeleton([0.0, CMB, SURFACE])

    def field(i, value, name):
        iv = sk.interval(i)
        return RadialField(iv, polynomial_layer(iv, [value]), character=DENSITY,
                           name=name)

    m = Planet(Geometry(sk), [
        {"rho": field(0, 11000.0, "rho"), "kappa": field(0, 9e11, "kappa"),
         "mu": field(0, 0.0, "mu")},
        {"rho": field(1, 4500.0, "rho"), "kappa": field(1, 4e11, "kappa"),
         "mu": field(1, 1.6e11, "mu")}])
    s = m.with_bullen_excess(0, 0.02)
    r = inside(s.layer(0))
    assert s.layer(0)["kappa"] is m.layer(0)["kappa"]
    assert isinstance(s.layer(0)["vp"], ComposedField)
    assert np.allclose(s.layer(0)["vp"](r), np.sqrt(9e11 / s.layer(0)["rho"](r)))
    assert np.allclose(s.bullen_excess(r), 0.02, atol=1e-9)
    assert np.isclose(s.layer_masses()[0], m.layer_masses()[0], rtol=1e-11)
    testing.check_model(s)


def test_what_cannot_be_described_by_stratification_is_refused():
    m = earth()
    with pytest.raises(ValueError, match="is not a fluid layer"):
        m.with_bullen_excess("mantle", 0.0)
    with pytest.raises(ValueError, match="negative in places"):
        m.with_bullen_excess("outer_core", -0.01)
    with pytest.raises(ValueError, match="negative in places"):
        m.with_bullen_excess(1, PPoly(np.array([[0.1, -0.1]]), [ICB, 2e6, CMB]))
    with pytest.raises(ValueError, match="exactly one of rho_bottom"):
        m.with_bullen_excess(1, 0.0, mass=1e24, rho_top=1e4)


def test_surgery_keeps_or_gives_up_the_description():
    m = earth()
    s = m.with_bullen_excess(1, 0.02)
    # a split layer is described on each part
    split = s.refined([2400e3])
    assert BULLEN_EXCESS in split.layer(1) and BULLEN_EXCESS in split.layer(2)
    testing.check_model(split)
    # cutting the outside away leaves the inside as it was
    cut = s.truncated(3000e3)
    assert BULLEN_EXCESS in cut.layer(1)
    testing.check_model(cut)
    # cutting the inside away removes the mass the density was derived under
    hollow = s.hollowed(ICB)
    assert BULLEN_EXCESS not in hollow.field_names()
    assert np.allclose(hollow.layer(0)["rho"](inside(hollow.layer(0))),
                       s.layer(1)["rho"](inside(s.layer(1))))
    # a continuation of a stratified layer is not derived
    top = s.truncated(CMB).extended([4000e3], fields="extrapolate")
    assert BULLEN_EXCESS in top.layer(1) and BULLEN_EXCESS not in top.layer(2)
    # a merge across a stratified layer is refused; one beneath it is followed
    with pytest.raises(ValueError, match="derived from its 'bullen_excess'"):
        s.coarsened(drop=[1])
    four = LayeredIsotropicElastic(
        [0.0, 600e3, ICB, CMB, SURFACE], rho=[(13200.0, -400.0), 12800.0, 11000.0,
                                             4500.0], scale=ICB,
        vp=[11000.0, 11000.0, 9000.0, 11000.0], vs=[3500.0, 3500.0, 0.0, 6000.0])
    held = four.with_bullen_excess(2, 0.02)
    coarse, _ = held.coarsened(drop=[0], degree=1)
    assert np.allclose(coarse.bullen_excess(inside(coarse.layer(1))), 0.02, atol=1e-9)
    assert np.isclose(coarse.layer_masses()[1], held.layer_masses()[2], rtol=1e-11)
    testing.check_model(coarse)


def test_the_stratification_field_reads_the_excess():
    s = earth().with_bullen_excess(1, 0.02).with_stratification()
    core = s.layer(1)
    r = inside(core)
    expected = -1.02 * core["rho"](r) ** 2 / kappa_mu(core)[0](r)
    assert np.allclose(core["drho_dphi0"](r), expected, rtol=1e-12)
    # and agrees with the derivative of the density over gravity
    assert np.allclose(core["drho_dphi0"](r),
                       core["rho"].derivative(nu=1)(r) / s.gravity(r), rtol=1e-8)


def test_other_units_keep_the_description():
    s = earth().with_bullen_excess(1, 0.02)
    nd = s.nondimensionalised()
    r = inside(nd.layer(1))
    assert np.allclose(nd.layer(1)[BULLEN_EXCESS](r), 0.02)
    assert np.allclose(nd.bullen_excess(r), 0.02, atol=1e-9)
    testing.check_model(nd)
