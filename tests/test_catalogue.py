"""The catalogue: PREM from its polynomials, and the simple models."""
import numpy as np
import pytest

from planetmodel.catalogue import PREM_RADIUS, LayeredIsotropicElastic, PREM
from planetmodel.character import DENSITY, SCALAR
from planetmodel.layerfunction import PolynomialLayer
from planetmodel import testing
from planetmodel.materials import elastic_moduli, is_fluid, kappa_mu, moduli
from planetmodel.mesh1d.gravity import gravity, mass
from planetmodel.testing import check_model
from planetmodel.units import Scales


@pytest.fixture(scope="module")
def model():
    return PREM()


def test_structure(model):
    assert model.nlayers == 13 and model.scales.is_si
    assert model.layer(0).name == "inner_core" and model.layer(-1).name == "ocean"
    faces = [f.name for f in model.geometry.interfaces]
    assert faces[:2] == ["icb", "cmb"] and faces[-1] == "surface"
    assert model.geometry.interface("cmb").radius == 3480e3
    assert model.geometry.interface("moho").radius == 6346.6e3
    assert model.common_names() == ("rho", "vpv", "vsv", "vph", "vsh", "eta", "qkappa")
    assert model.layers_with("qmu") == tuple(i for i in range(13) if i not in (1, 12))
    assert model.spec("rho").character == DENSITY
    assert model.spec("vpv").character == SCALAR
    for layer in model.layers:
        for f in layer.fields.values():
            assert isinstance(f.function, PolynomialLayer)


# Table I values, km/s and g/cm^3, at both sides of the main boundaries.
TABLE = [
    ("icb", "inner_core", "outer_core", {"rho": (12.7636, 12.1663),
                                         "vpv": (11.0283, 10.3557),
                                         "vsv": (3.5043, 0.0)}),
    ("cmb", "outer_core", "lowermost_mantle", {"rho": (9.9034, 5.5665),
                                               "vpv": (8.0648, 13.7166),
                                               "vsv": (0.0, 7.2647)}),
    ("d670", "upper_lower_mantle", "transition_zone_lower",
     {"rho": (4.3807, 3.9921), "vpv": (10.7513, 10.2662), "vsv": (5.9451, 5.5702)}),
    ("d400", "transition_zone_middle", "transition_zone_upper",
     {"rho": (3.7238, 3.5432), "vpv": (9.1340, 8.9052), "vsv": (4.9325, 4.7699)}),
    ("moho", "lid", "lower_crust", {"rho": (3.3808, 2.9), "vpv": (8.0209, 6.8),
                                    "vsv": (4.3960, 3.9)}),
]


@pytest.mark.parametrize("face, below, above, values", TABLE)
def test_tabulated_values_on_both_sides(model, face, below, above, values):
    r = model.geometry.interface(face).radius
    for name, (lo, hi) in values.items():
        assert np.isclose(model.layer(below)[name](r), lo * 1e3, rtol=2e-4)
        assert np.isclose(model.layer(above)[name](r), hi * 1e3, rtol=2e-4)


def test_transverse_isotropy_only_between_80_and_220_km(model):
    for layer in model.layers:
        lo, hi = layer.interval
        r = np.linspace(lo, hi, 5)
        ti = layer.name in ("low_velocity_zone", "lid")
        same = np.allclose(layer["vph"](r), layer["vpv"](r))
        assert same is not ti
        assert np.allclose(layer["eta"](r), 1.0) is not ti


def test_fluid_layers_and_moduli(model):
    assert [is_fluid(layer) for layer in model.layers] == \
        [i in (1, 12) for i in range(13)]
    A = moduli(model.layer("lower_mantle"))["A"]
    assert isinstance(A.function, PolynomialLayer) and A.function.degree == 9
    lm = model.layer("lower_mantle")
    r = np.linspace(*lm.interval, 7)
    assert np.allclose(A(r), lm["rho"](r) * lm["vph"](r) ** 2, rtol=1e-13)
    e = elastic_moduli(model.layer("lid"))
    assert e.symmetry.name == "VTI" and e(6300e3, 0.3, 0.2).shape == (6, 6)


def test_oceanless(model):
    dry = PREM(ocean=False)
    assert dry.nlayers == 12 and dry.skeleton.boundaries[-1] == 6368e3
    assert dry.geometry.interfaces[-1].name == "surface"
    assert dry.layer(-1).name == "upper_crust"
    assert dry.layer(3)["rho"](5e6) == model.layer(3)["rho"](5e6)
    assert PREM_RADIUS == 6371e3


def test_non_dimensional_prem(model):
    nd = model.nondimensionalised()
    assert np.isclose(nd.G, 1.0) and nd.skeleton.boundaries[-1] == 1.0
    lm, nl = model.layer("lower_mantle"), nd.layer("lower_mantle")
    assert np.isclose(nl["rho"](0.7), lm["rho"](0.7 * 6371e3) / 5515.0)
    assert np.isclose(nl["vpv"](0.7), lm["vpv"](0.7 * 6371e3) / nd.scales.factor(
        model.spec("vpv").dimensions))


def test_simple_models():
    h = LayeredIsotropicElastic.homogeneous(1.0, rho=3.0, vp=2.0, vs=1.0, name="ball")
    assert h.nlayers == 1 and h.layer("ball")["rho"](0.3) == 3.0
    assert not is_fluid(h.layer(0))
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0],
                                vs=[0.0, 1.0], layer_names=["core", "shell"],
                                interface_names=["cmb", "top"],
                                scales=Scales(length=1e6))
    assert is_fluid(m.layer("core")) and not is_fluid(m.layer("shell"))
    assert m.geometry.interface("top").radius == 1.0 and not m.scales.is_si
    assert elastic_moduli(m.layer("shell")).symmetry.name == "ISOTROPIC"
    with pytest.raises(ValueError, match="needs 2 values"):
        LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[1.0], vp=[1.0, 1.0],
                                vs=[0.0, 0.0])


@pytest.mark.parametrize("make", [
    PREM, lambda: PREM(ocean=False),
    lambda: LayeredIsotropicElastic.homogeneous(1.0, rho=3.0, vp=2.0, vs=1.0),
    lambda: LayeredIsotropicElastic([0.2, 0.5, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0],
                                    vs=[0.0, 1.0]),
])
def test_catalogue_models_pass_the_contract(make):
    check_model(make())


# -- model types: the behaviours as methods, copies keeping the class --------

def test_prem_exposes_the_free_functions_as_methods():
    m = PREM()
    assert m.is_fluid("outer_core") and not m.is_fluid("lower_mantle")
    A = m.moduli("lid")["A"]
    assert np.allclose(A(6300e3), moduli(m.layer("lid"))["A"](6300e3))
    assert m.elastic_moduli(-1).symmetry.name == "VTI"      # by description
    kappa, mu = m.kappa_mu(3)
    assert np.allclose(mu(5000e3), kappa_mu(m.layer(3))[1](5000e3))
    assert np.allclose(m.gravity(6371e3), gravity(m, 6371e3))
    assert np.isclose(m.mass(), mass(m))
    assert m.is_viscoelastic("lower_mantle")
    assert PREM.moduli.__doc__ == moduli.__doc__


def test_copies_keep_the_class_and_the_constructor_is_free():
    m = PREM(ocean=False)
    for out in (m.refined([5000e3]), m.truncated(6000e3), m.hollowed(1000e3),
                m.nondimensionalised(), m.renamed(), m.without_field("qmu"),
                m.replaced()):
        assert type(out) is PREM
        lo, hi = out.layer(0).interval
        assert out.moduli(0)["A"](0.5 * (lo + hi)) > 0.0
    with pytest.raises(TypeError):
        PREM(m.geometry, [layer.fields for layer in m.layers])


def test_frozen_keeps_the_class_and_elastic_drops_the_rheology():
    m = PREM(ocean=False)
    f = m.frozen(2 * np.pi / 43200.0)
    assert type(f) is PREM and "omega" in f.constants
    assert f.moduli("lower_mantle")["L"].dtype == np.complex128
    e = m.elastic()
    assert type(e) is PREM and not any(e.is_viscoelastic(i) for i in range(e.nlayers))
    assert "qkappa" not in e.layer(0) and "qmu" not in e.layer(0)
    assert np.allclose(e.moduli(3)["A"](5000e3), m.moduli(3)["A"](5000e3))
    fe = f.elastic()
    assert fe.moduli("lower_mantle")["L"].dtype == np.complex128


def test_isotropic_is_the_voigt_average_and_keeps_the_rest():
    m = PREM()
    iso = m.isotropic()
    assert type(iso) is PREM
    lid = iso.layer("lid")
    assert set(lid.names) == {"rho", "kappa", "mu", "qkappa", "qmu"}
    kappa, mu = kappa_mu(m.layer("lid"))
    assert np.allclose(lid["kappa"](6300e3), kappa(6300e3))
    assert np.allclose(lid["mu"](6300e3), mu(6300e3))
    assert iso.elastic_moduli("lid").symmetry.name == "ISOTROPIC"
    assert iso.is_fluid("outer_core") and iso.is_fluid("ocean")
    assert np.isclose(iso.mass(), m.mass())
    testing.check_model(iso)
    # a layer without an elastic description is left alone
    shell = m.extended([6500e3]).isotropic()
    assert shell.layer(-1).names == ()


def test_a_model_type_of_ones_own_wraps_a_layer_function():
    from planetmodel import Elastic, Model, layer_method

    def shear_velocity(layer):
        rho, mu = layer["rho"], kappa_mu(layer)[1]
        return lambda r: np.sqrt(mu(r) / rho(r))

    class Planet(Elastic, Model):
        vs_of = layer_method(shear_velocity)

    base = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0],
                                   vs=[0.0, 1.0])
    p = Planet(base.geometry, [layer.fields for layer in base.layers])
    assert np.isclose(p.vs_of(1)(0.7), 1.0)
    assert np.isclose(p.vs_of(-1)(0.7), 1.0)
    assert type(p.refined([0.25])) is Planet
    assert p.refined([0.25]).is_fluid(0)
