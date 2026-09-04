"""Model classes.

A model class guarantees its aspects' fields layer by layer, exposes
them under stable names, survives surgery as itself, and -- for the
viscoelastic one -- lifts static moduli where a layer holds nothing
frequency-dependent.
"""
import numpy as np
import pytest

from planetmodel import PREM, RadialField, ReferenceBody, Skeleton
from planetmodel.io.deck import read_isotropic_deck
from planetmodel.model.body import Layer
from planetmodel.model.character import DENSITY
from planetmodel.model.fields.frequency import LiftedFrequencyField
from planetmodel.model.rheology import constant_q, maxwell
from planetmodel.model.units import Dimensions, Scales
from planetmodel.model.classes import (ElasticModel, HasDensity, HasModuli, ModelBase,
                           ViscoelasticModel, ViscousModel)
from planetmodel.registry import lookup
from planetmodel.testing import check_model


@pytest.fixture(scope="module")
def deck():
    return read_isotropic_deck("tests/data/prem.nocrust")


def viscous_body():
    sk = Skeleton([0.0, 1.0, 2.0, 3.0])
    rho = RadialField(sk, [lambda r: 5.0e3 + 0 * r] * 3, name="rho",
                      character=DENSITY, dimensions=Dimensions.DENSITY)
    eta = RadialField(sk, [lambda r: 1.0e21 + 0 * r] * 3, name="viscosity",
                      dimensions=Dimensions.VISCOSITY)
    return ReferenceBody.from_fields(sk, {"rho": rho, "viscosity": eta})


# ------------------------------------------------------------ guarantees

def test_elastic_model_from_a_deck(deck):
    m = deck.as_class(ElasticModel)
    assert isinstance(m, ElasticModel) and isinstance(m, ModelBase)
    assert m.required_fields() == ("rho", "elastic_moduli")
    assert m.rho is m["rho"] and m.elastic_moduli is m["elastic_moduli"]
    assert m.symmetry.name == "ISOTROPIC"
    assert m.guaranteed_layers == tuple(range(deck.skeleton.nlayers))
    assert m.skeleton == deck.skeleton and m.meta == deck.meta
    assert "guarantees ['rho', 'elastic_moduli']" in repr(m)
    check_model(m)


def test_a_layer_holding_some_but_not_all_is_refused(deck):
    plain = ReferenceBody.from_layers(deck.layers, meta=deck.meta)
    body = plain.with_layer(3, plain.layer(3).without_field("elastic_moduli"))
    with pytest.raises(ValueError, match="layer 3 holds \\['rho'\\] but not"):
        body.as_class(ElasticModel)
    with pytest.raises(ValueError, match="layer 3 holds \\['rho'\\] but not"):
        deck.with_layer(3, deck.layer(3).without_field("elastic_moduli"))
    # ... an empty shell and a vacuum pass: they hold none of them
    ok = deck.extended([6.5e6]).with_buffer(ratio=0.1).as_class(ElasticModel)
    assert ok.guaranteed_layers == tuple(range(deck.skeleton.nlayers))
    assert ok.layer(-1).is_vacuum and ok.layer(-2).fields == {}


def test_a_guarantee_of_nothing_is_refused():
    sk = Skeleton([0.0, 1.0])
    with pytest.raises(ValueError, match="guarantees nothing"):
        ViscousModel.from_fields(sk, {})
    with pytest.raises(ValueError, match="holds \\['rho'\\] but not"):
        viscous_body().as_class(ElasticModel)            # rho without moduli


def test_viscous_model_and_the_registry():
    m = viscous_body().as_class(ViscousModel)
    assert m.viscosity is m["viscosity"] and m.rho is m["rho"]
    assert m.required_fields() == ("rho", "viscosity")
    check_model(m)
    for cls in (ElasticModel, ViscoelasticModel, ViscousModel):
        assert lookup("model_class", cls.__name__) is cls
    assert ElasticModel.ASPECTS == (HasDensity, HasModuli)


# --------------------------------------------------------------- surgery

def test_surgery_preserves_the_class_and_revalidates(deck):
    m = deck.as_class(ElasticModel)
    outer = float(m.skeleton.boundaries[-1])
    for other in (m.truncated(6.0e6), m.extended([6.5e6]), m.refined([5.0e6]),
                  m.coarsened(drop=[0])[0], m.with_buffer(ratio=0.2),
                  m.annotate(0, name="core"), m.classify_states(),
                  m.rescaled(Scales.geophysical(outer, density=5.5e3))):
        assert type(other) is ElasticModel
        other.validate()
    with pytest.raises(ValueError, match="guarantees"):
        m.without_field("rho")
    with pytest.raises(ValueError, match="guarantees"):
        m.add_field("elastic_moduli", m.layer(0)["elastic_moduli"],
                    replace=True)  # rho stays
    # ... whereas a plain body stays plain
    plain = ReferenceBody.from_layers(deck.layers)
    assert type(plain.truncated(6.0e6)) is ReferenceBody


def test_add_field_revalidates(deck):
    m = deck.extended([6.5e6]).as_class(ElasticModel)
    piece = m.layer(-2)["elastic_moduli"].on_interval(*m.layer(-1).interval)
    with pytest.raises(ValueError, match="holds \\['elastic_moduli'\\] but not"):
        m.with_field(-1, "elastic_moduli", piece)                # the shell: no rho


# ---------------------------------------------------------- viscoelastic

def test_viscoelastic_model_lifts_static_moduli_at_view_time(deck):
    m = deck.as_class(ViscoelasticModel)
    # The class checks; it stores nothing.  An elastic layer holds its
    # static moduli alone, and the view lifts them when asked.
    assert not any("viscoelastic_moduli" in lay for lay in m.layers)
    assert m.layers == deck.layers
    dyn = m.viscoelastic_moduli
    assert dyn.kind == "frequency" and dyn.character.rank == 4
    assert dyn.domain == m.elastic_moduli.domain
    assert all(isinstance(dyn.restricted(i), LiftedFrequencyField)
               for i in dyn.domain)
    assert m["viscoelastic_moduli"] is dyn
    r = np.array([4.0e6])
    assert np.allclose(dyn.evaluate(r, omega=3.0), m.elastic_moduli.evaluate(r))
    frozen = m.moduli_at(3.0)
    assert frozen.kind == "static"
    assert frozen.evaluate(r).dtype == np.complex128
    assert np.allclose(frozen.evaluate(r), m.elastic_moduli.evaluate(r))
    assert m.moduli_at(3.0, part="real").evaluate(r).dtype == np.float64
    check_model(m)


def test_viscoelastic_model_keeps_a_law_where_the_builder_put_one(deck):
    law = constant_q(deck["elastic_moduli"], deck["qkappa"], deck["qmu"],
                     reference_period=1.0)
    layers = []
    for i, lay in enumerate(deck.layers):
        layers.append(lay if i == 0 else
                      lay.with_field("viscoelastic_moduli", law.restricted(i)))
    body = ReferenceBody.from_layers(layers, meta=deck.meta)
    m = body.as_class(ViscoelasticModel)
    assert "viscoelastic_moduli" not in m.layer(0)
    assert isinstance(m.viscoelastic_moduli[0], LiftedFrequencyField)
    assert m.layer(2)["viscoelastic_moduli"].law.law == "constant_q"
    dyn = m.viscoelastic_moduli
    assert dyn.omega_domain == "real" and dyn.domain == tuple(range(len(layers)))
    r2 = np.array([0.5 * sum(deck.layer(2).interval)])
    assert np.allclose(dyn.evaluate(r2, omega=3.0), law.evaluate(r2, omega=3.0))
    r0 = np.array([0.5 * deck.layer(0).interval[1]])
    assert np.all(np.imag(dyn.evaluate(r0, omega=3.0)) == 0.0)
    check_model(m)
    cut = m.truncated(6.0e6)
    assert type(cut) is ViscoelasticModel
    assert cut.layer(2)["viscoelastic_moduli"].law.law == "constant_q"


def test_viscoelastic_model_checks_what_it_is_given(deck):
    bad = deck.with_layer(1, deck.layer(1).with_field(
        "viscoelastic_moduli", deck.layer(1)["elastic_moduli"]))  # static, not lifted
    with pytest.raises(ValueError, match="not a frequency-dependent field"):
        bad.as_class(ViscoelasticModel)
    wrong = deck.with_layer(1, deck.layer(1).with_field(
        "viscoelastic_moduli",
        maxwell(deck["elastic_moduli"], deck["rho"]).restricted(1)))
    ok = wrong.as_class(ViscoelasticModel)      # rank 4, frequency: accepted
    assert ok.layer(1)["viscoelastic_moduli"].law.law == "maxwell"
    orphan = deck.with_layer(1, deck.layer(1).without_field("elastic_moduli")
                             .without_field("rho"))
    orphan = orphan.with_layer(1, orphan.layer(1).with_field(
        "viscoelastic_moduli",
        maxwell(deck["elastic_moduli"], deck["rho"]).restricted(1)))
    with pytest.raises(ValueError, match="without the static 'elastic_moduli'"):
        orphan.as_class(ViscoelasticModel)


def test_prem_promotes_to_a_viscoelastic_model():
    prem = PREM()
    m = prem.as_class(ViscoelasticModel)
    assert m.symmetry.name == "VTI" and m.meta["tref"] == 1.0
    check_model(m)


def test_layers_built_by_hand_make_a_model():
    sk = Skeleton([0.0, 1.0, 2.0])
    rho = RadialField(sk, [lambda r: 1.0 + 0 * r] * 2, name="rho",
                      character=DENSITY, dimensions=Dimensions.DENSITY)
    body = ReferenceBody.from_fields(sk, {"rho": rho})
    layers = [body.layer(0),
              Layer(1, interval=(1.0, 2.0), name="shell")]   # empty shell
    with pytest.raises(ValueError, match="guarantees"):
        ElasticModel(layers)
    v = ViscousModel([body.layer(0).with_field(
        "viscosity", RadialField(Skeleton([0.0, 1.0]), [lambda r: 1.0 + 0 * r],
                                 name="viscosity")), layers[1]])
    assert v.guaranteed_layers == (0,) and v.layer("shell").fields == {}


# --------------------------------------------------------- factories

def test_prem_is_a_factory_for_a_viscoelastic_model():
    m = PREM()
    assert type(m) is ViscoelasticModel and m.symmetry.name == "VTI"
    assert m.meta["tref"] == 1.0
    assert all("viscoelastic_moduli" in lay for lay in m.layers)
    assert m.layer(3)["viscoelastic_moduli"].law.law == "constant_q"
    assert m.layer(3)["viscoelastic_moduli"].law.constants == {"reference_period": 1.0}
    r = np.linspace(1.0e5, 6.3e6, 200)
    at_ref = m.moduli_at(2.0 * np.pi / 1.0, part="real")
    assert np.allclose(at_ref.evaluate(r), m.elastic_moduli.evaluate(r), rtol=1e-14)
    diag = np.einsum("nii->ni", m.viscoelastic_moduli.evaluate(r, omega=2 * np.pi))
    assert np.all(np.imag(diag) >= 0.0)                    # loss on the diagonal


def test_the_readers_are_factories(deck):
    from planetmodel.io.deck import read_isotropic_deck, read_mineos_deck
    assert type(deck) is ElasticModel                  # Q columns, no period
    assert "tref" not in deck.meta and "viscoelastic_moduli" not in deck
    calibrated = read_isotropic_deck("tests/data/prem.nocrust",
                                         reference_period=1.0)
    assert type(calibrated) is ViscoelasticModel
    assert calibrated.meta["tref"] == 1.0
    assert calibrated.layer(2)["viscoelastic_moduli"].law.law == "constant_q"
    mineos = read_mineos_deck("examples/prem.200")
    assert type(mineos) is ViscoelasticModel and mineos.meta["tref"] == 1.0
    assert type(read_mineos_deck("examples/prem.200",
                                     reference_period=100.0)).__name__ \
        == "ViscoelasticModel"


def test_a_factory_model_rescales_with_its_law():
    m = PREM(ocean=False)
    nd = m.nondimensionalised()
    assert type(nd) is ViscoelasticModel
    s = nd.scales
    law = nd.layer(3)["viscoelastic_moduli"].law
    assert law.constants["reference_period"] == pytest.approx(1.0 / s.time)
    r_si = 4.0e6
    omega_si = 2.0 * np.pi / 30.0
    want = m.viscoelastic_moduli.evaluate(r_si, omega=omega_si)
    got = nd.viscoelastic_moduli.evaluate(r_si / s.length, omega=omega_si * s.time)
    assert np.allclose(got * s.modulus, want, rtol=1e-12)
    back = nd.redimensionalised()
    assert np.allclose(back.viscoelastic_moduli.evaluate(r_si, omega=omega_si), want,
                       rtol=1e-12)


def test_the_sampler_skips_dynamic_fields_by_default():
    from planetmodel.sampling import AngularGrid
    m = PREM(ocean=False)
    sample = m.sample(AngularGrid.gauss_legendre(2), fields=["rho"])
    assert set(sample.fields) == {"rho"}
    every = m.sample(AngularGrid.gauss_legendre(2), drmax=2.0e6)
    assert "viscoelastic_moduli" not in every.fields
    assert "elastic_moduli" in every.fields


def test_the_exporter_skips_dynamic_fields_by_default():
    pytest.importorskip("gmsh")                 # mesh3d imports it at package level
    from planetmodel.mesh3d.export import _chosen_fields
    m = PREM(ocean=False)
    chosen = _chosen_fields(m, None)
    assert "viscoelastic_moduli" not in chosen and "elastic_moduli" in chosen
