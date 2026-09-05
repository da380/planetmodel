"""The model: construction and its refusals, access, surgery, units."""
import numpy as np
import pytest

from planetmodel import Geometry, Skeleton
from planetmodel.character import DENSITY, SCALAR, VECTOR, Character
from planetmodel.displacement import flattening
from planetmodel.fields import AnalyticField, RadialField, constant_field
from planetmodel.layerfunction import polynomial_layer
from planetmodel.model import Layer, Model
from planetmodel.testing import check_model
from planetmodel.units import DIMENSIONLESS, LENGTH, MASS, TIME, G_SI, Scales
from planetmodel.vocabulary import Constant, FieldSpec

SK = Skeleton([0.0, 0.5, 1.0])


def geometry():
    return Geometry(SK, layer_names=["core", "mantle"],
                    interface_names=["cmb", "surface"])


def fields(i, *, extra=()):
    iv = SK.interval(i)
    out = {"rho": RadialField(iv, polynomial_layer([2.0 + i, -0.5], iv),
                              character=DENSITY, name="rho"),
           "vs": constant_field(0.0 if i == 0 else 1.0, iv, name="vs")}
    for name, f in extra:
        out[name] = f
    return out


def model():
    return Model(geometry(), [fields(0), fields(1)])


# ------------------------------------------------------------ construction

def test_construction_and_access():
    m = model()
    assert m.nlayers == 2 and m.skeleton is SK and m.scales.is_si
    assert m.layer("mantle") is m.layer(1) is m.layer(-1)
    assert isinstance(m.layer(0), Layer)
    assert m.layer(0).names == ("rho", "vs") and "rho" in m.layer(0)
    assert m.layer(0)["rho"](0.25) == 2.0 - 0.125
    assert m.field_names() == ("rho", "vs") == m.common_names()
    assert m.layers_with("rho") == (0, 1)
    assert m.layer(0).interval == (0.0, 0.5) and m.layer(1).name == "mantle"
    assert list(m.layer(1)) == ["rho", "vs"] and len(m.layer(1)) == 2
    with pytest.raises(KeyError, match="holds no field"):
        m.layer(0)["mu"]
    assert "Model(2 layers" in repr(m) and "mantle" in repr(m.layer(1))


def test_refusals_by_name():
    with pytest.raises(TypeError, match="Geometry"):
        Model(SK, [fields(0), fields(1)])
    with pytest.raises(ValueError, match="fields for 1 layers"):
        Model(geometry(), [fields(0)])
    bad = fields(1)
    bad["rho"] = fields(0)["rho"]
    with pytest.raises(ValueError, match="lives on"):
        Model(geometry(), [fields(0), bad])
    wrong = fields(1)
    wrong["rho"] = constant_field(1.0, SK.interval(1))            # SCALAR, not DENSITY
    with pytest.raises(ValueError, match="spec says"):
        Model(geometry(), [fields(0), wrong])
    notfield = fields(1)
    notfield["rho"] = np.sin
    with pytest.raises(TypeError, match="not a Field"):
        Model(geometry(), [fields(0), notfield])
    with pytest.raises(TypeError, match="Scales"):
        Model(geometry(), [fields(0), fields(1)], scales=1.0)
    with pytest.raises(TypeError, match="FieldSpec"):
        Model(geometry(), [fields(0), fields(1)], specs={"x": 1})
    Model(geometry(), [fields(0), wrong], check=False)


def test_names_outside_the_vocabulary():
    iv = SK.interval(1)
    m = Model(geometry(), [fields(0), fields(1, extra=[
        ("visc", constant_field(1e21, iv, name="visc"))])])
    assert m.spec("visc") is None and m.spec("rho").character == DENSITY
    with pytest.raises(ValueError, match="no dimensions"):
        m.converted(Scales(length=2.0))
    spec = FieldSpec(SCALAR, MASS / LENGTH / TIME)
    m2 = Model(geometry(), [fields(0), fields(1, extra=[
        ("visc", constant_field(1e21, iv, name="visc"))])], specs={"visc": spec})
    assert m2.spec("visc") is spec
    assert m2.converted(Scales(length=2.0)).layer(1)["visc"](0.4) == 1e21 / 0.5
    with pytest.raises(ValueError, match="spec says"):
        Model(geometry(), [fields(0), fields(1, extra=[
            ("visc", constant_field([1.0, 0.0, 0.0], iv, character=VECTOR))])],
              specs={"visc": spec})


def test_constants_in_the_models_units():
    m = model()
    assert m.G == G_SI and m.constant("G") == G_SI
    nd = m.nondimensionalised()
    assert np.isclose(nd.G, 1.0, rtol=1e-14)
    two = Model(geometry(), [fields(0), fields(1)],
                constants={"two": Constant(2.0, DIMENSIONLESS)})
    assert two.constant("two") == 2.0
    with pytest.raises(KeyError, match="no constant"):
        m.constant("c")


# ------------------------------------------------------------------ copies

def test_with_and_without_field():
    m = model()
    mu = constant_field(3.0, SK.interval(1), character=Character(0, 1), name="mu")
    m2 = m.with_field("mantle", "mu", mu)
    assert "mu" in m2.layer(1) and "mu" not in m2.layer(0)
    assert "mu" not in m.layer(1)
    with pytest.raises(ValueError, match="replace=True"):
        m2.with_field(1, "mu", mu)
    assert m2.with_field(1, "mu", 2.0 * mu, replace=True).layer(1)["mu"](0.7) == 6.0
    assert "mu" not in m2.without_field("mu").layer(1)
    assert m2.without_field("rho", layers=["core"]).layers_with("rho") == (1,)


def test_geometry_changes_keep_the_fields():
    m = model()
    g = geometry().renamed(layers=["inner", "outer"])
    assert m.with_geometry(g).layer("outer")["rho"](0.7) == m.layer(1)["rho"](0.7)
    with pytest.raises(ValueError, match="another skeleton"):
        m.with_geometry(Geometry(Skeleton([0.0, 0.4, 1.0])))
    assert m.renamed(layers={"core": "ic"}).layer("ic").index == 0
    s = m.stretched(flattening(0.01, rmax=1.0))
    assert not s.geometry.is_identity
    assert s.layer(1)["rho"](0.7) == m.layer(1)["rho"](0.7)
    assert s.with_mapping(m.geometry.mapping).geometry.is_identity


# ----------------------------------------------------------------- surgery

def test_refined_re_states_the_split_layer_exactly():
    m = model().refined([0.75], names=["mid"])
    assert m.nlayers == 3 and m.layer(1).name is None and m.layer(2).name is None
    assert m.geometry.interfaces[1].name == "mid"
    for i in (1, 2):
        lo, hi = m.layer(i).interval
        r = np.linspace(lo, hi, 5)
        assert np.allclose(m.layer(i)["rho"](r), model().layer(1)["rho"](r))
    assert m.layer(1)["rho"].function.ppoly.c.shape == (2, 1)


def test_truncated_and_hollowed():
    m = model()
    t = m.truncated(0.8)
    assert t.nlayers == 2 and t.layer(1).interval == (0.5, 0.8)
    assert t.layer(1)["rho"](0.6) == m.layer(1)["rho"](0.6)
    h = m.hollowed(0.25)
    assert h.nlayers == 2 and h.layer(0).interval == (0.25, 0.5)
    assert h.skeleton.is_hollow
    assert h.layer(0)["vs"](0.3) == 0.0
    c = m.truncated(0.5)
    assert c.nlayers == 1 and c.layer(0).name == "core"


def test_extended_shells_hold_what_they_are_given():
    m = model()
    empty = m.extended([1.2])
    assert empty.nlayers == 3 and empty.layer(2).names == ()
    assert empty.common_names() == ()
    ext = m.extended([1.2, 1.5], fields="extrapolate", names=["a", "b"])
    assert ext.layer("b")["rho"](1.4) == m.layer(1)["rho"].on_interval(1.2, 1.5)(1.4)
    given = m.extended([1.2], fields=[{"rho": constant_field(0.5, (1.0, 1.2),
                                                             character=DENSITY)}])
    assert given.layer(2)["rho"](1.1) == 0.5
    with pytest.raises(ValueError, match="fields for 2 shells"):
        m.extended([1.2], fields=[{}, {}])
    with pytest.raises(ValueError, match="extrapolate"):
        m.extended([1.2], fields="guess")


def test_surgery_needs_an_identity_mapping_where_the_geometry_does():
    s = model().stretched(flattening(0.01, rmax=1.0))
    with pytest.raises(ValueError, match="identity"):
        s.extended([1.2])
    assert s.refined([0.75]).nlayers == 3


# ------------------------------------------------------------------- units

def test_conversion_is_by_name_and_exact():
    m = model()
    nd = m.nondimensionalised(density=1000.0)
    assert nd.scales.length == 1.0 and not nd.scales.is_si
    assert np.allclose(nd.skeleton.boundaries, SK.boundaries)
    assert np.isclose(nd.layer(1)["rho"](0.7), m.layer(1)["rho"](0.7) / 1000.0)
    big = m.converted(Scales(length=0.001, mass=2.0, time=3.0))
    assert np.allclose(big.skeleton.boundaries, SK.boundaries * 1000.0)
    r = np.linspace(500.0, 1000.0, 7)
    factor = 2.0 / 0.001 ** 3
    assert np.allclose(big.layer(1)["rho"](r), m.layer(1)["rho"](r / 1000.0) / factor)
    assert np.allclose(big.layer(1)["vs"](r), m.layer(1)["vs"](r / 1000.0) * 3000.0)
    back = big.in_si()
    assert np.allclose(back.layer(1)["rho"](r / 1000.0), m.layer(1)["rho"](r / 1000.0),
                       rtol=1e-14)
    with pytest.raises(ValueError, match="in SI"):
        big.nondimensionalised()
    with pytest.raises(TypeError, match="Scales"):
        m.converted(2.0)


def test_conversion_carries_the_mapping():
    s = model().stretched(flattening(0.01, rmax=1.0))
    big = s.converted(Scales(length=0.5))
    assert np.allclose(big.skeleton.boundaries, SK.boundaries * 2.0)
    X = np.array([[0.0, 0.0, 1.4]])
    assert np.allclose(big.geometry.mapping(X), 2.0 * s.geometry.mapping(X / 2.0))


# --------------------------------------------------------------- contracts

@pytest.mark.parametrize("m", [
    model(),
    model().stretched(flattening(0.01, rmax=1.0)),
    model().with_field(1, "a", AnalyticField(SK.interval(1),
                                            lambda r, t, p: r * np.cos(t), name="a"),
                       ).with_field(1, "v", RadialField(SK.interval(1), [1.0, 0.0, 0.5],
                                                       character=VECTOR)),
])
def test_shipped_models_pass_the_contract(m):
    check_model(m)


def test_a_subclass_keeps_its_class_through_surgery():
    class Mine(Model):
        def density_at(self, r):
            return self.layer(self.skeleton.locate(r).layer)["rho"](r)

    m = Mine(geometry(), [fields(0), fields(1)])
    assert isinstance(m.refined([0.7]), Mine) and isinstance(m.in_si(), Mine)
    assert m.density_at(0.7) == m.layer(1)["rho"](0.7)
    check_model(m)
