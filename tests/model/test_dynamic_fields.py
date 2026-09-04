"""Frequency- and time-dependent fields.

Three kinds of field, one extra argument each; lifts, algebra,
composition, restriction and assembly per kind; freezing back to a
static field.  No laws: the oracles here are hand-written omega and t
dependences whose values are known in closed form.
"""
import numpy as np
import pytest

from planetmodel import (DENSITY, PREM, AnalyticField, RadialField, ReferenceBody,
                    Skeleton)
from planetmodel.model.body import Layer
from planetmodel.model.character import SCALAR, VECTOR
from planetmodel.model.fields.dependent import SumDependentField
from planetmodel.model.fields.frequency import (ComposedFrequencyField,
                                                FrequencyDependentField,
                                                LiftedFrequencyField,
                                                at_frequency,
                                                lifted_to_frequency)
from planetmodel.model.fields.layerwise import LayerwiseField
from planetmodel.model.fields.time import (ComposedTimeField, TimeDependentField,
                                      at_time, lifted_to_time)
from planetmodel.model.units import Dimensions
from planetmodel.testing import (check_field, check_frequency_dependent_field,
                            check_time_dependent_field)

OMEGAS = (0.5, 2.0, 7.0)
TS = (0.0, 1.5, 4.0)


@pytest.fixture
def sk():
    return Skeleton([0.0, 1.0, 2.0, 3.0])


@pytest.fixture
def mu(sk):
    return RadialField(sk, [lambda r: 3.0 + r, lambda r: 2.0 + 0 * r,
                            lambda r: 1.0 + r ** 2], name="mu",
                       dimensions=Dimensions.MODULUS)


@pytest.fixture
def tau(sk):
    return RadialField(sk, [lambda r: 0.5 + 0 * r] * 3, name="tau",
                       dimensions=Dimensions.TIME)


def maxwell_like(mu, tau):
    """mu s tau / (1 + s tau), s = i omega: entire in omega, a closed form."""
    def fn(omega, m, t):
        s = 1j * omega
        return m * s * t / (1.0 + s * t)
    return ComposedFrequencyField(fn, [mu, tau], character=SCALAR,
                                  dimensions=Dimensions.MODULUS, name="mu_dyn")


# ------------------------------------------------------------- lifts

def test_a_lifted_radial_field_meets_the_contract(mu):
    f = lifted_to_frequency(mu)
    assert isinstance(f, FrequencyDependentField)
    assert f.kind == "frequency" and f.omega_domain == "complex"
    assert f.is_radial and f.domain == (0, 1, 2)
    check_frequency_dependent_field(f, omegas=OMEGAS)
    assert lifted_to_frequency(f) is f
    g = lifted_to_time(mu)
    assert isinstance(g, TimeDependentField) and g.kind == "time"
    check_time_dependent_field(g, ts=TS)


def test_a_lifted_analytic_field_meets_the_contract(sk):
    fn = lambda r, t, p: np.stack([r, np.cos(t), np.sin(p)], axis=-1)  # noqa: E731
    v = AnalyticField(fn, sk, character=VECTOR, name="v")
    f = lifted_to_frequency(v)
    check_frequency_dependent_field(f, omegas=OMEGAS)
    got = f.evaluate(1.5, 0.3, 0.2, omega=2.0, frame="cartesian")
    want = v.evaluate(1.5, 0.3, 0.2, frame="cartesian")
    assert np.allclose(got, want)
    check_time_dependent_field(lifted_to_time(v), ts=TS)


# ------------------------------------------------------- composition

def test_a_composed_field_reproduces_its_closed_form(mu, tau):
    f = maxwell_like(mu, tau)
    check_frequency_dependent_field(f, omegas=OMEGAS)
    r = np.array([0.5, 1.5, 2.5])
    for omega in OMEGAS + (0.3 + 0.7j,):        # entire: complex omega too
        s = 1j * omega
        want = mu.evaluate(r) * s * 0.5 / (1.0 + s * 0.5)
        assert np.allclose(f.evaluate(r, omega=omega), want)
    assert np.allclose(f.evaluate(r, omega=2.0, part="imag"),
                       np.imag(mu.evaluate(r) * 2j * 0.5 / (1.0 + 1j)))
    assert f.evaluate(r, omega=0.0).tolist() == [0j, 0j, 0j]


def test_a_real_axis_form_refuses_complex_omega(mu):
    def constant_q_like(omega, m):
        return m * (1.0 + (2.0 / np.pi) * np.log(omega) / 100.0 + 1j / 100.0)
    f = ComposedFrequencyField(constant_q_like, [mu], character=SCALAR,
                               omega_domain="real", name="q")
    assert f.omega_domain == "real"
    check_frequency_dependent_field(f, omegas=OMEGAS)
    with pytest.raises(ValueError, match="omega must be real"):
        f.evaluate(1.5, omega=1.0 + 1.0j)
    assert (f + lifted_to_frequency(mu)).omega_domain == "real"


def test_a_time_composition(mu, tau):
    def relax(t, m, tau):
        return m * np.exp(-t / tau)
    f = ComposedTimeField(relax, [mu, tau], character=SCALAR, name="relax")
    check_time_dependent_field(f, ts=TS)
    r = np.array([0.5, 2.5])
    assert np.allclose(f.evaluate(r, t=1.0), mu.evaluate(r) * np.exp(-2.0))
    with pytest.raises(ValueError, match="t must be real"):
        f.evaluate(r, t=1j)


def test_the_argument_is_a_scalar(mu, tau):
    f = maxwell_like(mu, tau)
    with pytest.raises(ValueError, match="scalar"):
        f.evaluate(1.5, omega=np.array([1.0, 2.0]))


# ------------------------------------------------------------ algebra

def test_the_algebra_lifts_static_operands_and_stays_closed(mu, tau):
    f = maxwell_like(mu, tau)
    r = np.array([0.5, 1.5, 2.5])
    total = (f + mu) - 0.5 * mu
    assert isinstance(total, SumDependentField) and total.kind == "frequency"
    want = f.evaluate(r, omega=2.0) + 0.5 * mu.evaluate(r)
    assert np.allclose(total.evaluate(r, omega=2.0), want)
    also = mu + f                                # static on the left
    assert also.kind == "frequency"
    assert np.allclose(also.evaluate(r, omega=2.0), f.evaluate(r, omega=2.0)
                       + mu.evaluate(r))
    twice = 2.0 * f
    assert np.allclose(twice.evaluate(r, omega=2.0), 2 * f.evaluate(r, omega=2.0))
    rotated = 1j * f
    assert np.allclose(rotated.evaluate(r, omega=2.0),
                       1j * f.evaluate(r, omega=2.0))
    assert np.allclose((-f).evaluate(r, omega=2.0), -f.evaluate(r, omega=2.0))
    assert np.allclose((f / 2.0).evaluate(r, omega=2.0),
                       0.5 * f.evaluate(r, omega=2.0))
    check_frequency_dependent_field(total, omegas=OMEGAS)


def test_frequency_and_time_do_not_mix(mu, tau):
    f = maxwell_like(mu, tau)
    g = lifted_to_time(mu)
    with pytest.raises(TypeError, match="transform"):
        f + g
    with pytest.raises(TypeError, match="transform"):
        g + f
    with pytest.raises(ValueError, match="real"):
        1j * g


def test_characters_and_dimensions_are_checked(mu, sk):
    rho = RadialField(sk, [lambda r: 1.0 + 0 * r] * 3, name="rho",
                      character=DENSITY, dimensions=Dimensions.DENSITY)
    with pytest.raises(ValueError, match="different character"):
        lifted_to_frequency(mu) + rho


# ----------------------------------------------------------- freezing

def test_freezing_gives_a_static_field(mu, tau):
    f = maxwell_like(mu, tau)
    frozen = at_frequency(f, 2.0)
    assert frozen.kind == "static" and frozen.argument == 2.0
    assert frozen.name == "mu_dyn@omega=2"
    r = np.array([0.5, 2.5])
    # The complex tensor is the object: frozen values are complex by default.
    c = frozen.evaluate(r)
    assert c.dtype == np.complex128
    assert np.allclose(c, f.evaluate(r, omega=2.0))
    real = at_frequency(f, 2.0, part="real")
    check_field(real)
    assert real.evaluate(r).dtype == np.float64
    assert np.allclose(real.evaluate(r), np.real(f.evaluate(r, omega=2.0)))
    assert np.allclose(at_frequency(f, 2.0, part="imag").evaluate(r),
                       np.imag(f.evaluate(r, omega=2.0)))
    assert frozen.restricted(1).skeleton.nlayers == 1
    assert f.at(2.0).evaluate(1.5) == frozen.evaluate(1.5)
    assert at_time(lifted_to_time(mu), 1.0).part == "real"
    with pytest.raises(ValueError, match="part"):
        at_frequency(lifted_to_time(mu), 1.0, part="imag")


def test_static_algebra_keeps_complex_values(mu, tau):
    """A frozen complex field stays complex through +, * and composition."""
    from planetmodel.model.fields.composite import ComposedField
    f = at_frequency(maxwell_like(mu, tau), 2.0)
    r = np.array([0.5, 2.5])
    want = f.evaluate(r)
    assert np.iscomplexobj(want) and np.any(np.imag(want) != 0)
    assert (2 * f).evaluate(r).dtype == np.complex128
    assert np.allclose((2 * f).evaluate(r), 2 * want)
    assert np.allclose((f + mu).evaluate(r), want + mu.evaluate(r))
    assert np.allclose((f - f).evaluate(r), 0)
    doubled = ComposedField(lambda a: 2 * a, [f], name="2mu")
    assert np.allclose(doubled.evaluate(r), 2 * want)


def test_a_real_domain_accepts_a_complex_omega_on_the_axis(mu):
    def constant_q_like(omega, m):
        return m * (1.0 + (2.0 / np.pi) * np.log(omega) / 100.0 + 1j / 100.0)
    f = ComposedFrequencyField(constant_q_like, [mu], character=SCALAR,
                               omega_domain="real", name="q")
    r = np.array([0.5, 2.5])
    assert np.allclose(f.evaluate(r, omega=complex(3.0, 0.0)),
                       f.evaluate(r, omega=3.0))
    with pytest.raises(ValueError, match="omega must be real"):
        f.evaluate(r, omega=complex(3.0, 1e-3))


def test_frozen_fields_push_forward_and_compose_like_static_ones(mu, tau):
    from planetmodel.model.fields.composite import ComposedField
    f = at_frequency(maxwell_like(mu, tau), 2.0)
    doubled = ComposedField(lambda a: 2 * a, [f], name="2mu")
    assert np.allclose(doubled.evaluate(1.5), 2 * f.evaluate(1.5))
    assert (f + mu).evaluate(1.5) == f.evaluate(1.5) + mu.evaluate(1.5)


# --------------------------------------------- pieces, layers and views

def test_restriction_and_assembly_by_layer(mu, tau, sk):
    f = maxwell_like(mu, tau)
    piece = f.restricted(1)
    assert isinstance(piece, ComposedFrequencyField)
    assert piece.skeleton.nlayers == 1
    assert piece.evaluate(1.5, omega=2.0) == f.evaluate(1.5, omega=2.0, layer=1)
    assert piece.on_interval(1.0, 1.5).evaluate(1.25, omega=2.0) == \
        f.evaluate(1.25, omega=2.0)

    body = ReferenceBody.from_fields(sk, {"mu": mu, "tau": tau})
    body.add_field("mu_dyn", f)
    assert isinstance(body.layer(2)["mu_dyn"], ComposedFrequencyField)
    view = body["mu_dyn"]
    assert isinstance(view, ComposedFrequencyField)      # reassembled by type
    assert view.kind == "frequency" and view.domain == (0, 1, 2)
    r = np.array([0.5, 1.5, 2.5])
    assert np.allclose(view.evaluate(r, omega=2.0), f.evaluate(r, omega=2.0))
    assert body.field_names == ("mu", "tau", "mu_dyn")


def test_a_view_of_mixed_lifted_and_composed_pieces(mu, tau, sk):
    """An elastic lithosphere over a Maxwell mantle over an elastic core."""
    f = maxwell_like(mu, tau)
    layers = [
        Layer(0, interval=(0.0, 1.0), fields={"mu": mu[0],
                                              "dyn": lifted_to_frequency(mu[0])}),
        Layer(1, interval=(1.0, 2.0), fields={"mu": mu[1], "dyn": f.restricted(1)}),
        Layer(2, interval=(2.0, 3.0), fields={"mu": mu[2],
                                              "dyn": lifted_to_frequency(mu[2])}),
    ]
    body = ReferenceBody.from_layers(layers)
    view = body["dyn"]
    assert isinstance(view, LayerwiseField) and view.kind == "frequency"
    assert view.omega_domain == "complex" and view.domain == (0, 1, 2)
    r = np.array([0.5, 1.5, 2.5])
    got = view.evaluate(r, omega=2.0)
    s = 2j
    want = np.array([mu.evaluate(0.5), mu.evaluate(1.5) * s * 0.5 / (1 + s * 0.5),
                     mu.evaluate(2.5)], dtype=complex)
    assert np.allclose(got, want)
    assert view[1] is layers[1]["dyn"]
    check_frequency_dependent_field(view, omegas=OMEGAS)
    frozen = view.at(2.0, part="real")
    check_field(frozen)
    assert np.allclose(frozen.evaluate(r), np.real(want))
    assert np.allclose(view.at(2.0).evaluate(r), want)
    with pytest.raises(TypeError, match="omega="):
        view.evaluate(r, t=1.0)
    with pytest.raises(TypeError, match="omega"):
        body["mu"].evaluate(r, omega=1.0)           # a static view takes none


def test_a_view_refuses_mixed_kinds(mu, sk):
    layers = [
        Layer(0, interval=(0.0, 1.0), fields={"x": mu[0]}),
        Layer(1, interval=(1.0, 2.0), fields={"x": lifted_to_frequency(mu[1])}),
        Layer(2, interval=(2.0, 3.0), fields={"x": mu[2]}),
    ]
    body = ReferenceBody.from_layers(layers)
    with pytest.raises(ValueError, match="mixes kinds"):
        body["x"].evaluate(0.5)


def test_sampling_takes_static_fields_and_frozen_dynamic_ones(mu, tau, sk):
    from planetmodel.sampling import AngularGrid
    body = ReferenceBody.from_fields(sk, {"mu": mu, "tau": tau})
    body.add_field("dyn", maxwell_like(mu, tau))
    grid = AngularGrid.gauss_legendre(2)
    sample = body.sample(grid, drmax=0.5)
    assert set(sample.fields) == {"mu", "tau"}          # the dynamic one waits
    frozen = body.sample(grid, drmax=0.5,
                         fields={"dyn2": at_frequency(body["dyn"], 2.0)})
    assert set(frozen.fields) == {"dyn2"}
    assert np.all(np.isfinite(frozen.fields["dyn2"]))


def test_surgery_and_rescaling_carry_dynamic_fields(mu, tau, sk):
    body = ReferenceBody.from_fields(sk, {"mu": mu, "tau": tau})
    body.add_field("lift", lifted_to_frequency(mu))
    body.add_field("dyn", maxwell_like(mu, tau))
    cut = body.truncated(2.5)
    assert cut["dyn"].evaluate(2.2, omega=2.0) == body["dyn"].evaluate(
        2.2, omega=2.0)
    fine = body.refined([1.5])
    assert fine["lift"].evaluate(1.7, omega=3.0) == mu.evaluate(1.7)
    from planetmodel.model.units import Scales
    nd = body.without_field("dyn").rescaled(Scales.geophysical(3.0, density=1.0))
    assert isinstance(nd["lift"], LiftedFrequencyField)
    assert nd["lift"].evaluate(1.7 / 3.0, omega=3.0) == pytest.approx(
        mu.evaluate(1.7) / Scales.geophysical(3.0, density=1.0).factor(
            Dimensions.MODULUS))
    with pytest.raises(TypeError, match="cannot rescale ComposedFrequencyField"):
        body.rescaled(Scales.geophysical(3.0, density=1.0))


def test_prem_moduli_lift_and_freeze():
    prem = PREM()
    dyn = lifted_to_frequency(prem["elastic_moduli"])
    assert dyn.character.rank == 4
    r = np.array([4.0e6])
    c = dyn.evaluate(r, omega=1.0)
    assert c.shape == (1, 6, 6) and c.dtype == np.complex128
    assert np.allclose(np.real(c), prem["elastic_moduli"].evaluate(r))
    frozen = at_frequency(dyn, 1.0, part="real")
    assert frozen.character.rank == 4
    check_field(frozen)
