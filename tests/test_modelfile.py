"""The model file: written, validated, read back.

PREM and the model of a deck come back as the same polynomials, with no grid
given where a model is radial, and with the behaviours the file names
for them; a perturbed
PREM as a reference and a perturbation; a fully 3D model, a model in
three parts on three grids, and models placed by boundary reliefs and
by another mapping, all to the file's own nodes exactly and between
them by the stated rules; xarray opens the file as CF tools do; a
malformed file is refused naming what is wrong.  Fields on interfaces, on
a grid with poles and on one without, come back as gridded functions by
interface, a file without them reads as one with none, and what the
group cannot hold is refused.
"""
from pathlib import Path

import warnings

import numpy as np
import pytest

from planetmodel import (MINEOS, PREM, AnalyticField, ConstantQ, Elastic,
                         GriddedField, GriddedFunction, HarmonicField,
                         LayeredIsotropicElastic, Model, PerturbedField, Scales,
                         SelfGravitating, Viscoelastic, driscoll_healy, equiangular,
                         flattening, gauss_legendre, layer_linear, model_from_deck,
                         perturbed, read_deck, unperturbed)
from planetmodel.behaviours import BEHAVIOURS, model_type
from planetmodel.deck import mineos_constants, mineos_header, mineos_names
from planetmodel.character import DENSITY, SCALAR, VECTOR
from planetmodel.harmonics import packing
from planetmodel.testing import check_geometry, check_model, check_model_file
from planetmodel.units import Dimensions
from planetmodel.vocabulary import FieldSpec

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

from planetmodel.modelfile import (SCHEMA, radial_grid, read_model,  # noqa: E402
                                   read_model_file, read_surfaces,
                                   validate_structure, write_model)

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def prem_deck_model():
    """The model of PREM's own tabulation, `prem.200`, as its header
    describes it: named, with its reference frequency and every behaviour."""
    deck = read_deck(DATA / "prem.200", MINEOS)
    layers, faces = mineos_names(deck)
    return model_from_deck(deck, behaviours=BEHAVIOURS, layer_names=layers,
                           interface_names=faces, constants=mineos_constants(deck),
                           name=mineos_header(deck)["name"])


def at_random_radii(model, rng, n=12):
    """(layer, radii) pairs across every layer, ends kept clear."""
    out = []
    for layer in model.layers:
        lo, hi = layer.interval
        out.append((layer.index, lo + (hi - lo) * rng.uniform(0.01, 0.99, n)))
    return out


def same_radial_fields(a, b, rng, rtol=1e-12):
    for i, r in at_random_radii(a, rng):
        assert set(a.layer(i).names) == set(b.layer(i).names), i
        for name in a.layer(i).names:
            fa, fb = a.layer(i)[name], b.layer(i)[name]
            assert np.allclose(fb(r), fa(r), rtol=rtol, atol=0), (i, name)


@pytest.fixture(scope="module")
def prem():
    return PREM()


def test_prem_comes_back_as_the_same_polynomials(prem, tmp_path):
    path = write_model(tmp_path / "prem.nc", prem, grid=gauss_legendre(4),
                       meta={"source": "PREM", "run": 3})
    check_model_file(path, prem)
    back = read_model(path)
    same_radial_fields(prem, back, np.random.default_rng(0))
    assert [l.name for l in back.layers] == [l.name for l in prem.layers]
    data = read_model_file(path)
    assert data.schema == SCHEMA and data.name == prem.name
    assert dict(data.meta) == {"source": "PREM", "run": 3}
    assert data.constants["G"] == pytest.approx(prem.constant("G"))


def test_a_model_s_citation_is_written_and_read_back(prem, tmp_path):
    cites = prem.with_citation("A. Author (2026).  Something on top.")
    path = write_model(tmp_path / "cites.nc", cites, grid=gauss_legendre(2))
    check_model_file(path, cites)
    with netCDF4.Dataset(path) as ds:
        assert ds.getncattr("citation") == cites.cite()
    assert read_model_file(path).citation == cites.citation
    assert read_model(path).citation == cites.citation
    bare = prem.replaced(citation=())
    path = write_model(tmp_path / "bare.nc", bare, grid=gauss_legendre(2))
    with netCDF4.Dataset(path) as ds:
        assert "citation" not in ds.ncattrs()
    assert read_model(path).citation == ()


def test_a_model_in_other_scales_is_read_back_in_them(prem, tmp_path):
    nd = prem.nondimensionalised()
    path = write_model(tmp_path / "nd.nc", nd, grid=gauss_legendre(2))
    check_model_file(path, nd)
    back = read_model(path)
    assert back.scales == nd.scales
    si = read_model(path, scales=Scales.SI)
    same_radial_fields(prem, si, np.random.default_rng(1), rtol=1e-11)
    with netCDF4.Dataset(path) as ds:
        assert ds.groups["part_0"].variables["vsv"].units.endswith(" m s-1")


def test_a_deck_model_comes_back_as_the_same_polynomials(tmp_path):
    m = prem_deck_model()
    path = write_model(tmp_path / "deck.nc", m, grid=gauss_legendre(2))
    check_model_file(path, m)
    back = read_model(path)
    same_radial_fields(m, back, np.random.default_rng(2))
    assert type(back) is type(m) and back.behaviours == m.behaviours
    assert back.name == "prem.200" and back.layer(1).name == "outer_core"


LO, HI = 3480e3, 5701e3


def dlnvs(r, t, p):
    return 0.02 * np.sin(t) ** 2 * np.cos(2 * p) * (r - LO) / (HI - LO)


# -- a radial model, with no grid given -------------------------------------------

def only_radial_fields():
    """A plain model of two layers holding fields of the radius alone: a
    density, a viscosity graded in its logarithm on one layer, and a name
    outside the vocabulary with a spec of its own."""
    from planetmodel import Geometry, RadialField, Skeleton, constant_field
    from planetmodel.layerfunction import polynomial_layer
    geometry = Geometry(Skeleton([0.0, 0.5, 1.0]), layer_names=["core", "mantle"],
                        interface_names=["cmb", "surface"])
    layers = []
    for layer in geometry.layers:
        iv = layer.interval
        layers.append({
            "rho": RadialField(iv, polynomial_layer(iv, (9.0 - 4.0 * layer.index,
                                                         -1.0, 0.25)),
                               character=DENSITY, name="rho"),
            "porosity": constant_field(iv, 0.1 * (1 + layer.index), name="porosity"),
        })
    return Model(geometry, layers, name="two radial layers",
                 specs={"porosity": FieldSpec(SCALAR, Dimensions(),
                                              meaning="pore fraction")})


@pytest.mark.parametrize("make", [PREM, only_radial_fields,
                                  lambda: PREM().with_gravity()],
                         ids=["PREM", "radial fields", "PREM with gravity"])
def test_a_radial_model_is_written_with_no_grid(tmp_path, same_netcdf, make):
    """A model whose fields are all of the radius alone, under the
    identity, needs no grid: the file is the one the grid of one node
    writes, to the bit, and reads back to rounding with its names and
    constants."""
    model = make()
    path = write_model(tmp_path / "radial.nc", model)
    check_model_file(path, model)
    assert radial_grid(model) == gauss_legendre(0)
    same_netcdf(path, write_model(tmp_path / "given.nc", model,
                                  grid=gauss_legendre(0)))
    data = read_model_file(path)
    assert len(data.parts) == 1 and data.parts[0].grid == gauss_legendre(0)
    assert all(var.values.ndim == 1 for var in data.parts[0].variables.values())
    back = read_model(path)
    rtol = 1e-10 if "g" in model.field_names() else 1e-12
    same_radial_fields(model, back, np.random.default_rng(0), rtol=rtol)
    assert [lay.name for lay in back.layers] == [lay.name for lay in model.layers]
    assert ([f.name for f in back.geometry.interfaces]
            == [f.name for f in model.geometry.interfaces])
    assert back.name == model.name and back.scales == model.scales
    assert back.constant("G") == model.constant("G")
    assert back.spec("porosity") == model.spec("porosity")


def test_a_model_that_is_not_radial_is_refused_without_a_grid(prem, tmp_path):
    """With no grid given, a field that varies with direction and a mapping
    that is not the identity are refused by name, and nothing is written;
    the radial fields of such a model are written alone."""
    mp = perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="relative")
    with pytest.raises(ValueError, match=r"'vsv' on layer 2 \('lowermost_mantle'\) "
                                         r"depends on \(theta, phi\).*give the grid"):
        write_model(tmp_path / "x.nc", mp)
    with pytest.raises(ValueError, match="'vsv' on layer 2"):
        radial_grid(mp)
    placed = prem.stretched(flattening(1.0 / 300.0))
    with pytest.raises(ValueError, match="the model is placed by RadialStretch.*"
                                         "give the grid"):
        write_model(tmp_path / "x.nc", placed)
    assert not (tmp_path / "x.nc").exists()
    path = write_model(tmp_path / "rho.nc", mp, fields=["rho", "vpv"])
    check_model_file(path, mp)
    assert read_model(path).field_names() == ("rho", "vpv")
    # a function on an interface is of (theta, phi): it is held on a grid
    # of its own, or on the one named for it
    kernel = {"kernel": {"cmb": lambda t, p: np.cos(t) + 0.0 * p}}
    with pytest.raises(ValueError, match=r"surface fields \['kernel'\].*surface_grid="):
        write_model(tmp_path / "x.nc", prem, surfaces=kernel)
    grid = gauss_legendre(4)
    path = write_model(tmp_path / "k.nc", prem, surfaces=kernel, surface_grid=grid)
    check_model_file(path, prem, surfaces=kernel)
    assert read_model_file(path).parts[0].grid == gauss_legendre(0)
    held = {"kernel": {"cmb": GriddedFunction.sampled(kernel["kernel"]["cmb"], grid)}}
    path = write_model(tmp_path / "held.nc", prem, surfaces=held)
    assert same_grid(read_model_file(path).surfaces.grid, grid)


# -- what the model is: the behaviours a file names ---------------------------------

def test_prem_read_back_has_its_behaviours(prem, tmp_path):
    """The file names the behaviours of the model written, and the model
    read back has them: the methods are called on it as on the original
    and agree, the Love moduli the file's own fields."""
    path = write_model(tmp_path / "prem.nc", prem)
    names = ("Elastic", "ConstantQ", "SelfGravitating", "Viscoelastic")
    assert prem.behaviours == names
    with netCDF4.Dataset(path) as ds:
        assert ds.getncattr("behaviours") == " ".join(names)
    assert read_model_file(path).behaviours == names
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_model(path)
    assert type(back) is model_type(Model, BEHAVIOURS)
    assert all(isinstance(back, b) for b in (Elastic, ConstantQ, SelfGravitating,
                                             Viscoelastic))
    assert back.behaviours == names
    check_model(back)
    rng = np.random.default_rng(5)
    same_radial_fields(prem, back, rng)
    r = np.linspace(0.0, prem.skeleton.boundaries[-1], 41)
    assert np.allclose(back.gravity(r), prem.gravity(r), rtol=1e-12)
    assert np.allclose(back.pressure(r), prem.pressure(r), rtol=1e-10)
    assert back.mass() == pytest.approx(prem.mass(), rel=1e-13)
    assert back.reference_omega() == prem.reference_omega()
    omega = 2.0 * np.pi / 200.0
    frozen, want = back.frozen(omega), prem.frozen(omega)
    assert type(frozen) is type(back) and frozen.constant("omega") == omega
    for i, radii in at_random_radii(prem, rng):
        assert back.is_fluid(i) == prem.is_fluid(i)
        assert back.is_viscoelastic(i) == prem.is_viscoelastic(i)
        for a, b in ((back.moduli(i), prem.moduli(i)),
                     (back.moduli_at(i, omega), prem.moduli_at(i, omega)),
                     (dict(zip("km", back.kappa_mu(i))),
                      dict(zip("km", prem.kappa_mu(i)))),
                     (frozen.layer(i).fields, want.layer(i).fields)):
            assert set(a) == set(b)
            for name in a:
                assert np.allclose(a[name](radii), b[name](radii), rtol=1e-12), (
                    i, name)
        assert np.allclose(back.elastic_moduli(i).evaluate(radii, 0.3, 0.2),
                           prem.elastic_moduli(i).evaluate(radii, 0.3, 0.2),
                           rtol=1e-12)
    assert back.elastic().field_names() == prem.elastic().field_names()
    assert type(back.with_gravity()) is type(back)
    # written again, the file names the same behaviours
    again = write_model(tmp_path / "again.nc", back)
    assert read_model_file(again).behaviours == names


def test_reading_derives_nothing_the_file_does_not_hold(prem, tmp_path):
    """A model is read without its type's constructor: an elastic model
    written with some of its fields comes back elastic, holding those
    fields and no others, where a constructor would complete the Love
    moduli or refuse a description too short to give them."""
    velocities = ("rho", "vpv", "vsv", "vph", "vsh", "eta")
    path = write_model(tmp_path / "v.nc", prem, fields=velocities)
    back = read_model(path)
    assert isinstance(back, Elastic) and back.field_names() == velocities
    r = np.linspace(*prem.layer(3).interval, 9)
    for name, field in prem.moduli(3).items():
        assert np.allclose(back.moduli(3)[name](r), field(r), rtol=1e-12)
    path = write_model(tmp_path / "short.nc", prem, fields=["rho", "vsv"])
    back = read_model(path)
    assert isinstance(back, Elastic) and back.field_names() == ("rho", "vsv")
    with pytest.raises(KeyError, match="the moduli need"):
        model_type(Model, BEHAVIOURS)(back.geometry,
                                      [layer.fields for layer in back.layers])
    whole = read_model(write_model(tmp_path / "whole.nc", prem))
    assert whole.layer(3).names == prem.layer(3).names
    assert whole.layer(3)["A"].radial_pieces() is not None


class Tidal:
    """A behaviour of a program's own: a method over the model's fields."""

    def love_number(self):
        return 0.3 * self.nlayers


class TidalPlanet(Elastic, Tidal, SelfGravitating, Model):
    """A model type of a program's own, with a constructor of its own."""

    def __init__(self, radius):
        base = LayeredIsotropicElastic([0.0, 0.5 * radius, radius], rho=[10.0, 4.0],
                                       vp=[9.0, 7.0], vs=[0.0, 3.0], name="tidal")
        self.radius = radius
        super().__init__(base.geometry, [layer.fields for layer in base.layers],
                         name="tidal")


def test_a_behaviour_the_program_lacks_is_named_kept_and_written_again(tmp_path):
    """A file names a behaviour and cannot carry its code: a reader without
    it warns, naming it, gives the library's behaviours, and keeps the
    name in its place through every copy and into the file it writes."""
    planet = TidalPlanet(2.0)
    names = ("Elastic", "Tidal", "SelfGravitating")
    assert planet.behaviours == names
    path = write_model(tmp_path / "tidal.nc", planet)
    check_model_file(path, planet)
    with pytest.warns(UserWarning, match="names the behaviour 'Tidal'") as caught:
        back = read_model(path)
    assert len(caught) == 1
    assert type(back) is model_type(Model, (Elastic, SelfGravitating))
    assert not isinstance(back, Tidal) and not hasattr(back, "love_number")
    assert back.behaviours == names
    check_model(back)
    same_radial_fields(planet, back, np.random.default_rng(6))
    assert back.mass() == pytest.approx(planet.mass(), rel=1e-13)
    for copy in (back.replaced(), back.refined([0.25]), back.named("other"),
                 back.converted(Scales(length=2.0, mass=3.0))):
        assert copy.behaviours == names
    again = write_model(tmp_path / "again.nc", back)
    with netCDF4.Dataset(again) as ds:
        assert ds.getncattr("behaviours") == "Elastic Tidal SelfGravitating"
    with pytest.warns(UserWarning, match="'Tidal'"):
        check_model_file(again, back)
        assert read_model(again).behaviours == names


def test_the_caller_says_what_a_file_is_read_as(tmp_path):
    """`model_type` gives back a model type of the caller's own, made
    without its constructor, or a model of the behaviours given, whatever
    the file names, and with no warning."""
    planet = TidalPlanet(2.0)
    path = write_model(tmp_path / "tidal.nc", planet)
    rng = np.random.default_rng(7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        own = read_model(path, model_type=TidalPlanet)
        mixed = read_model(path, model_type=(Elastic, Tidal))
        plain = read_model(path, model_type=())
        leaf = read_model(path, model_type=Model)
    assert type(own) is TidalPlanet and own.behaviours == planet.behaviours
    assert own.love_number() == planet.love_number()
    assert not hasattr(own, "radius")         # its constructor was not called
    assert type(own.refined([0.25])) is TidalPlanet
    same_radial_fields(planet, own, rng)
    check_model(own)
    assert type(mixed) is model_type(Model, (Elastic, Tidal))
    assert mixed.behaviours == ("Elastic", "Tidal") and mixed.love_number() == 0.6
    assert not isinstance(mixed, SelfGravitating)
    assert type(plain) is Model and type(leaf) is Model
    assert plain.behaviours == () == leaf.behaviours
    same_radial_fields(planet, plain, rng)
    with netCDF4.Dataset(write_model(tmp_path / "plain.nc", plain)) as ds:
        assert "behaviours" not in ds.ncattrs()
    with pytest.raises(TypeError, match="Elastic is not a model type over it"):
        read_model(path, model_type=Elastic)
    with pytest.raises(TypeError, match="a behaviour is a mixin class"):
        read_model(path, model_type=["Elastic"])
    from planetmodel import cartesian
    with pytest.raises(TypeError, match=r"cartesian\.model\.Model is not a model "
                                        "type over it"):
        read_model(path, model_type=cartesian.Model)


def test_a_file_that_names_no_behaviours_reads_as_a_plain_model(prem, tmp_path):
    """A file without the attribute, as every file from before it is, reads
    as a plain `Model` holding the same fields, with no warning."""
    path = write_model(tmp_path / "prem.nc", prem)
    corrupt(path, lambda ds: ds.delncattr("behaviours"))
    validate_structure(path)
    assert read_model_file(path).behaviours == ()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_model(path)
    assert type(back) is Model and back.behaviours == ()
    same_radial_fields(prem, back, np.random.default_rng(8))
    with pytest.raises(AssertionError, match="names the behaviours"):
        check_model_file(path, prem)
    corrupt(path, lambda ds: ds.setncattr("behaviours", ""))
    assert type(read_model(path)) is Model


def test_a_perturbed_prem_comes_back_as_a_reference_and_a_perturbation(prem,
                                                                          tmp_path):
    mp = perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="relative")
    names = ["rho", "vpv", "vsv", "vph", "vsh", "eta"]
    path = write_model(tmp_path / "pert.nc", mp, grid=gauss_legendre(8), fields=names,
                       interpolation={"vsv": ("polynomial", "cubic")})
    check_model_file(path, mp)
    back = read_model(path)
    f = back.layer(3)["vsv"]
    assert isinstance(f, PerturbedField) and f.kind == "relative"
    assert isinstance(f.perturbation, GriddedField)
    assert not isinstance(back.layer(6)["vsv"], PerturbedField)
    rng = np.random.default_rng(3)
    r = rng.uniform(3.7e6, 5.5e6, 20)
    t, p = rng.uniform(0, np.pi, 20), rng.uniform(0, 2 * np.pi, 20)
    assert np.allclose(f(r, t, p), mp.layer(3)["vsv"](r, t, p), rtol=1e-4)
    base = unperturbed(back)
    assert np.allclose(base.layer(3)["vsv"](r), prem.layer(3)["vsv"](r), rtol=1e-12)


def smooth(r, t, p):
    st = np.sin(t)
    return (1.0 + r ** 2) * np.exp(0.3 * st * np.cos(p) - 0.2 * st * np.sin(p)
                                   + 0.1 * np.cos(t))


def wind(r, t, p):
    r, t, p = np.broadcast_arrays(r, t, p)
    st = np.sin(t)
    return np.stack([r * st * np.cos(p), r * st * np.sin(p), r * np.cos(t) + 1.0], -1)


def three_d_model():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    grid = gauss_legendre(8)
    m = m.with_field(1, "rho", GriddedField.sampled(
        AnalyticField((0.5, 1.0), smooth, character=DENSITY), grid,
        breaks=[0.5, 0.75, 1.0], ngll=4, lateral="cubic"), replace=True)
    m = m.with_field(1, "flow", GriddedField.sampled(
        AnalyticField((0.5, 1.0), wind, character=VECTOR, frame="cartesian"), grid,
        breaks=[0.5, 1.0], ngll=3, lateral="cubic"))
    L = 8
    c = np.zeros((4, 2, L + 1, L + 1))
    s, l, mm = packing(L)
    c[..., s, l, mm] = np.random.default_rng(4).standard_normal((4, s.size)) / (l + 1)
    from scipy.interpolate import CubicSpline
    return m.with_field(0, "q", HarmonicField(
        (0.0, 0.5), CubicSpline(np.linspace(0.0, 0.5, 4), c, axis=0)))


def test_a_fully_3d_model_comes_back(tmp_path):
    m = three_d_model()
    path = write_model(tmp_path / "3d.nc", m, grid=gauss_legendre(8))
    check_model_file(path, m)
    back = read_model(path)
    rng = np.random.default_rng(5)
    r, t, p = (rng.uniform(0.55, 0.95, 30), rng.uniform(0, np.pi, 30),
               rng.uniform(0, 2 * np.pi, 30))
    for name in ("rho", "flow"):
        f, g = m.layer(1)[name], back.layer(1)[name]
        assert isinstance(g, GriddedField) and g.lateral == "cubic"
        assert np.allclose(g(r, t, p), f(r, t, p), rtol=1e-12, atol=1e-12)
    q, qb = m.layer(0)["q"], back.layer(0)["q"]
    assert isinstance(qb, GriddedField) and qb.lateral == "harmonic"
    rq = rng.uniform(0.05, 0.45, 30)
    assert np.allclose(qb(rq, t, p), q(rq, t, p), rtol=0, atol=1e-11)


def test_a_field_without_pieces_subdivides_its_own_layer(tmp_path):
    """The outer layer's density is analytic, so its elements take a
    sample's spacing, 0.1 / (lmax + 1) of the outer radius; the core keeps
    one element and comes back exactly."""
    from planetmodel import RadialField
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[(2.0, -1.0), 1.0],
                                vp=[1.0, 2.0], vs=[0.0, 1.0])
    rho = RadialField((0.5, 1.0), lambda r: 1.0 + 0.1 * np.sin(5.0 * r),
                      character=DENSITY, name="rho")
    m = m.with_field(1, "rho", rho, replace=True)
    path = write_model(tmp_path / "m.nc", m, grid=gauss_legendre(8))
    with netCDF4.Dataset(path) as ds:
        part = ds["part_0"]
        layer = np.asarray(part["element_layer"][:])
        assert int(part.getncattr("ngll")) == 5
    assert np.bincount(layer).tolist() == [1, int(np.ceil(0.5 / (0.1 / 9)))]
    back = read_model(path)
    r = np.linspace(0.01, 0.49, 7)
    assert np.allclose(back.layer(0)["rho"](r), m.layer(0)["rho"](r), rtol=1e-14)
    r = np.linspace(0.51, 0.99, 7)
    assert np.allclose(back.layer(1)["rho"](r), rho(r), rtol=1e-10)


def test_a_stratified_layer_comes_back_through_its_breakpoints(prem, tmp_path):
    """The outer core made stable holds its Bullen excess, zero on the
    stretches made neutral, and the density that derives: both are
    piecewise polynomials, their breakpoints element edges of the file,
    and both come back as they were."""
    from planetmodel.mesh1d.stratification import BULLEN_EXCESS
    stable, report = prem.stabilised(layers=["outer_core"], inertia=False)
    core = stable.layer("outer_core")
    path = write_model(tmp_path / "stable.nc", stable, grid=gauss_legendre(2))
    check_model_file(path, stable)
    back = read_model(path)
    assert BULLEN_EXCESS in back.layer("outer_core")
    assert BULLEN_EXCESS not in back.layer("inner_core")
    lo, hi = core.interval
    r = np.linspace(lo, hi, 401)
    excess, density = back.layer("outer_core")[BULLEN_EXCESS], back.layer(1)["rho"]
    assert np.allclose(excess(r), core[BULLEN_EXCESS](r), atol=1e-13)
    assert np.allclose(density(r), core["rho"](r), rtol=1e-13)
    # neutral where it was made neutral, to the last node
    for a, b in report.layers[core.index].neutral:
        inside = np.linspace(a, b, 9)[1:-1]
        assert np.all(excess(inside) == 0.0)
    breaks = excess.radial_pieces()[0]
    for edge in (x for interval in report.layers[core.index].neutral
                 for x in interval):
        assert np.min(np.abs(breaks - edge)) <= 1e-9 * hi


def test_a_model_in_three_parts_on_three_grids(prem, tmp_path):
    grids = {i: gauss_legendre(4) for i in range(2)}
    dh = driscoll_healy(5)
    grids.update({i: dh for i in range(2, 9)})
    eq = equiangular(9, 16, poles=True)
    grids.update({i: eq for i in range(9, 13)})
    names = ["rho", "vpv", "vsv"]
    mp = perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="relative")
    path = write_model(tmp_path / "parts.nc", mp, grid=grids, fields=names)
    check_model_file(path, mp)
    data = read_model_file(path)
    assert [p.layers for p in data.parts] == [(0, 1), tuple(range(2, 9)),
                                              tuple(range(9, 13))]
    assert data.parts[1].grid.kind == "driscoll_healy"
    assert data.parts[2].grid.pole_nodes == (True, True)
    with netCDF4.Dataset(path) as ds:
        assert "pole" in ds.groups["part_1"].dimensions     # DH lacks the south
        assert "pole" not in ds.groups["part_2"].dimensions
    bad = dict(grids)
    bad[5] = gauss_legendre(4)          # the grid of layers 0 and 1, apart from them
    with pytest.raises(ValueError, match="contiguous"):
        write_model(tmp_path / "bad.nc", prem, grid=bad, fields=names)
    with pytest.raises(ValueError, match="no grid given"):
        write_model(tmp_path / "bad.nc", prem, grid={0: dh}, fields=names)


def bump(t, p):
    return 0.02 * np.sin(t) ** 3 * np.cos(3 * p) + 0.01 * np.cos(t)


def test_a_model_placed_by_boundary_reliefs_keeps_them_as_a_stack(tmp_path):
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    gf = GriddedFunction.sampled(bump, gauss_legendre(16))
    m = m.stretched(layer_linear(m.skeleton, [None, gf, gf]))
    path = write_model(tmp_path / "stack.nc", m, grid=gauss_legendre(4))
    check_model_file(path, m)
    with netCDF4.Dataset(path) as ds:
        assert "geometry" in ds.groups
        assert "displacement" not in ds.groups["part_0"].variables
    back = read_model(path)
    rng = np.random.default_rng(6)
    from planetmodel.frames import cartesian_points
    X = cartesian_points(rng.uniform(0.05, 1.0, 100), rng.uniform(0, np.pi, 100),
                         rng.uniform(0, 2 * np.pi, 100))
    assert np.allclose(back.geometry.mapping(X), m.geometry.mapping(X), atol=1e-14)
    check_geometry(back.geometry)


def test_a_converted_model_keeps_its_stack(tmp_path):
    """Nondimensionalised, PREM placed by gridded reliefs is still written
    as the stack, on grids that differ between parts, and read back."""
    m = PREM(ocean=False)
    g = gauss_legendre(16)
    rel = [None] * m.skeleton.boundaries.size
    rel[2] = GriddedFunction.sampled(lambda t, p: 5e3 * np.cos(t) + 0 * p, g)
    rel[-1] = GriddedFunction.sampled(bump, g)
    nd = m.stretched(layer_linear(m.skeleton, rel)).nondimensionalised()
    grids = {i: gauss_legendre(4) if i < 2 else gauss_legendre(8)
             for i in range(nd.nlayers)}
    path = write_model(tmp_path / "nd.nc", nd, grid=grids)
    with netCDF4.Dataset(path) as ds:
        assert "geometry" in ds.groups
        assert np.allclose(ds["geometry"]["boundary_radius"][:],
                           nd.skeleton.boundaries)
    back = read_model(path)
    from planetmodel.frames import cartesian_points
    rng = np.random.default_rng(8)
    X = cartesian_points(rng.uniform(0.05, 1.0, 100), rng.uniform(0, np.pi, 100),
                         rng.uniform(0, 2 * np.pi, 100))
    assert np.allclose(back.geometry.mapping(X), nd.geometry.mapping(X), rtol=0,
                       atol=1e-14)


def test_a_model_placed_by_another_mapping_keeps_its_displacement(tmp_path):
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    m = m.stretched(flattening(0.01))
    path = write_model(tmp_path / "field.nc", m, grid=gauss_legendre(8))
    check_model_file(path, m)
    back = read_model(path)
    from planetmodel import GriddedDisplacement
    from planetmodel.frames import cartesian_points
    assert isinstance(back.geometry.mapping, GriddedDisplacement)
    rng = np.random.default_rng(7)
    X = cartesian_points(rng.uniform(0.05, 1.0, 100), rng.uniform(0, np.pi, 100),
                         rng.uniform(0, 2 * np.pi, 100))
    # flattening is of degree two in angle: the bicubic rule on a grid of
    # band 8 holds it to about 1e-5 of the radius
    assert np.abs(back.geometry.mapping(X) - m.geometry.mapping(X)).max() < 5e-5


def test_xarray_opens_the_parts_with_their_coordinates_and_units(prem, tmp_path):
    xr = pytest.importorskip("xarray")
    path = write_model(tmp_path / "x.nc", three_d_model(), grid=gauss_legendre(6))
    tree = xr.open_datatree(path)
    part = tree["part_0"]
    assert "latitude" in part.coords and "longitude" in part.coords
    assert part["latitude"].attrs["units"] == "degrees_north"
    assert part["rho"].attrs["units"] == "kg m-3"
    assert part["rho"].dims == ("node", "latitude", "longitude")
    assert tree.attrs["schema"] == SCHEMA
    one = xr.open_dataset(path, group="part_0")
    assert float(one["latitude"][0]) > float(one["latitude"][-1])   # decreasing


def test_a_radial_layer_of_a_lateral_variable_comes_back_radial(prem, tmp_path):
    """L varies laterally where vsv does, in the lower mantle, so L is a
    lateral variable; its other layers are named in radial_layers and come
    back as radial fields, exactly; a perturbation radial in one layer and
    lateral in another does the same."""
    from planetmodel import RadialField
    lower = prem.layer("lower_mantle").interval
    above = prem.layer("upper_lower_mantle").interval
    m = perturbed(prem, "vsv", AnalyticField(lower, lambda r, t, p: 0.02 * np.sin(t)
                                             ** 2 * np.cos(2 * p)), kind="relative")
    m = perturbed(m, "vsv", RadialField(above, 0.01), kind="relative")
    path = write_model(tmp_path / "m.nc", m, grid=gauss_legendre(8),
                       fields=["rho", "vsv", "L"])
    check_model_file(path, m)
    i = m.layer("lower_mantle").index
    j = m.layer("upper_lower_mantle").index
    with netCDF4.Dataset(path) as ds:
        part = ds.groups["part_0"]
        assert part["L"].dimensions[1:3] == ("latitude", "longitude")
        assert sorted(np.atleast_1d(part["L"].radial_layers)) == [
            k for k in range(m.nlayers) if k != i]
        assert np.atleast_1d(part["vsv_perturbation"].radial_layers).tolist() == [j]
        assert "radial_layers" not in part["rho"].ncattrs()
    back = read_model(path)
    for k in (0, 6, j):
        f = back.layer(k)["L"]
        assert f.is_radial
        r = np.linspace(*f.interval, 9)
        assert np.allclose(f(r), m.layer(k)["L"](r), rtol=1e-12)
    assert not back.layer(i)["L"].is_radial
    assert back.layer(j)["vsv"].is_radial and not back.layer(i)["vsv"].is_radial
    r = np.linspace(*above, 5)
    assert np.allclose(back.layer(j)["vsv"](r), m.layer(j)["vsv"](r), rtol=1e-12)


def test_the_stated_interpolation_can_be_chosen(tmp_path):
    m = three_d_model()
    path = write_model(tmp_path / "rules.nc", m, grid=gauss_legendre(8),
                       interpolation={"rho": ("linear", "linear")})
    with netCDF4.Dataset(path) as ds:
        rho = ds.groups["part_0"].variables["rho"]
        assert rho.radial_interpolation == "linear"
        assert rho.lateral_interpolation == "linear"
    back = read_model(path, interpolation={"rho": ("polynomial", "cubic")})
    assert back.layer(1)["rho"].lateral == "cubic"
    with pytest.raises(ValueError, match="rules"):
        write_model(tmp_path / "x.nc", m, grid=gauss_legendre(8),
                    interpolation={"rho": ("spline", "linear")})


def test_writing_refuses_what_the_file_cannot_hold(prem, tmp_path):
    from planetmodel import AngularGrid
    with pytest.raises(KeyError, match="no layer holds"):
        write_model(tmp_path / "x.nc", prem, grid=gauss_legendre(2), fields=["vs3d"])
    m = prem.with_field(3, "radius", prem.layer(3)["rho"])
    with pytest.raises(ValueError, match="keeps for itself"):
        write_model(tmp_path / "x.nc", m, grid=gauss_legendre(2))
    mp = perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="relative")
    g = gauss_legendre(4)
    custom = AngularGrid(g.colatitudes, g.longitudes)
    with pytest.raises(ValueError, match="radial="):
        write_model(tmp_path / "x.nc", mp, grid=custom, fields=["vsv"])


def corrupt(path, edit):
    with netCDF4.Dataset(path, "a") as ds:
        edit(ds)


def nudge_the_top(ds):
    """The last element's top moved by 1e-7 of itself, beyond the tiling's
    tolerance."""
    eb = ds.groups["part_0"].variables["element_bounds"]
    eb[-1, 1] = float(eb[-1, 1]) * (1.0 + 1e-7)


def harmonic_without_a_band(ds):
    """A variable stating the harmonic rule in a part whose grid states no
    band."""
    part = ds.groups["part_0"]
    part.variables["rho"].setncattr("lateral_interpolation", "harmonic")
    part.delncattr("grid_lmax")


def relabel_an_element(ds):
    """The first element of a layer of several given to the layer below it:
    `element_layer` holds the part's layers still, and gives that element a
    layer its bounds lie outside."""
    el = ds.groups["part_0"].variables["element_layer"]
    layers = np.asarray(el[:], dtype=int)
    e = next(e for e in range(1, layers.size) if layers[e] != layers[e - 1]
             and np.sum(layers == layers[e]) > 1)
    el[e] = layers[e - 1]


@pytest.mark.parametrize("edit, message", [
    (lambda ds: ds.setncattr("schema", "planetmodel.model/0"), "schema is"),
    (lambda ds: ds.setncattr("rtol", -1.0), "rtol is"),
    (lambda ds: ds.setncattr("rtol", "loose"), "rtol is"),
    (lambda ds: ds.setncattr("behaviours", np.int32(4)),
     r"behaviours is np.int32\(4\), not"),
    (lambda ds: ds.setncattr("behaviours", "Elastic Tidal Elastic"),
     "behaviours is 'Elastic Tidal Elastic', not the names"),
    (lambda ds: ds.groups["part_0"].variables["rho"].setncattr(
        "radial_layers", np.int32([5])), "radial_layers"),
    (lambda ds: ds.groups["part_0"].variables["vp"].setncattr(
        "radial_layers", np.int32([0])), "radial_layers"),
    (lambda ds: ds.delncattr("scale_mass"), "scale_mass"),
    (lambda ds: ds.groups["part_0"].variables["rho"].delncattr("radial_interpolation"),
     "lacks the attributes"),
    (lambda ds: ds.groups["part_0"].variables["rho"].setncattr(
        "lateral_interpolation", "quintic"), "lateral rule"),
    (lambda ds: ds.groups["part_0"].setncattr("ngll", np.int32(7)), "ngll"),
    (lambda ds: ds.variables["layer_part"].__setitem__(0, 3), "part_3"),
    (lambda ds: ds.groups["part_0"].variables["rho"].setncattr("rank", np.int32(1)),
     "trailing axes"),
    (lambda ds: ds.groups["part_0"].setncattr("grid_kind", "healpix"), "grid_kind"),
    (nudge_the_top, "do not tile"),
    (lambda ds: ds.setncattr("scale_length", np.inf), "scale_length is missing or"),
    (lambda ds: ds.setncattr("scale_length", "big"), "scale_length is missing or"),
    (lambda ds: ds.setncattr("scale_time", np.array([1.0, 2.0])),
     "scale_time is missing or"),
    (lambda ds: ds.groups["part_0"].setncattr("ngll", "five"), "ngll is 'five'"),
    (lambda ds: ds.variables["interface_radius"].__setitem__(0, 0.25),
     "interface_radius does not hold the 2 interfaces"),
    (lambda ds: ds.groups["part_0"].variables["rho"].setncattr(
        "lateral_interpolation", np.int32([1, 2])), "lateral rule"),
    (harmonic_without_a_band, "part_0/rho states the lateral rule 'harmonic'"),
    (relabel_an_element, "part_0/element_layer gives element 3 to layer 0, which "
                         "its bounds do not lie in"),
])
def test_a_malformed_file_is_refused_by_name(tmp_path, edit, message):
    path = write_model(tmp_path / "m.nc", three_d_model(), grid=gauss_legendre(6))
    validate_structure(path)
    corrupt(path, edit)
    with pytest.raises(ValueError, match=f"malformed model file .*{message}"):
        read_model_file(path)


def test_a_variable_states_the_rule_its_layers_agree_on(prem, tmp_path):
    """A radial layer has no lateral rule to impose; layers that disagree are
    refused; a split reference states its own radial rule."""
    two = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                  vs=[4.0, 3.0])
    rho3d = AnalyticField((0.0, 0.5), lambda r, t, p: 10.0 + 0.2 * r * np.cos(t),
                          character=DENSITY)
    grid = gauss_legendre(8)
    cubic = two.with_field(0, "rho", GriddedField.sampled(rho3d, grid, ngll=4,
                                                          lateral="cubic"),
                           replace=True)
    path = write_model(tmp_path / "c.nc", cubic, grid=grid, fields=["rho"])
    rho = read_model_file(path).parts[0].variables["rho"]
    assert rho.attributes["lateral_interpolation"] == "cubic"
    flat = AnalyticField((0.5, 1.0), lambda r, t, p: 4.0 + 0.0 * t, character=DENSITY)
    linear = GriddedField.sampled(flat, grid, ngll=4)
    mixed = cubic.with_field(1, "rho", linear, replace=True)
    with pytest.raises(ValueError, match="different rules"):
        write_model(tmp_path / "x.nc", mixed, grid=grid, fields=["rho"])
    write_model(tmp_path / "y.nc", mixed, grid=grid, fields=["rho"],
                interpolation={"rho": ("polynomial", "cubic")})

    lo, hi = 3480e3, 5701e3
    dlnvs = GriddedField.sampled(AnalyticField(
        (lo, hi), lambda r, t, p: 0.01 * np.cos(t) * (r - lo) / (hi - lo)),
        gauss_legendre(6), ngll=4, radial="linear")
    mp = perturbed(prem, "vsv", dlnvs, kind="relative")
    path = write_model(tmp_path / "p.nc", mp, grid=gauss_legendre(6), fields=["vsv"])
    part = read_model_file(path).parts[0]
    assert part.variables["vsv"].attributes["radial_interpolation"] == "polynomial"
    assert part.variables["vsv_perturbation"].attributes[
        "radial_interpolation"] == "linear"


def test_a_harmonic_rule_is_stated_only_where_the_grid_reproduces_it(tmp_path):
    L = 8
    s, l, m = packing(L)
    c = np.zeros((2, L + 1, L + 1))
    c[s, l, m] = np.random.default_rng(1).standard_normal(s.size) / (1.0 + l)
    h = HarmonicField((0.5, 1.0), c, character=DENSITY)
    two = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                  vs=[4.0, 3.0]).with_field(1, "dens", h)
    for grid, rule in ((gauss_legendre(16), "harmonic"), (gauss_legendre(4), "linear")):
        path = write_model(tmp_path / f"h{grid.lmax}.nc", two, grid=grid,
                           fields=["dens"])
        var = read_model_file(path).parts[0].variables["dens"]
        assert var.attributes["lateral_interpolation"] == rule
        check_model_file(path, two)


def band_limited(L=8):
    """Two layers, the outer holding a density of band L."""
    s, l, m = packing(L)
    c = np.zeros((2, L + 1, L + 1))
    c[s, l, m] = np.random.default_rng(1).standard_normal(s.size) / (1.0 + l)
    h = HarmonicField((0.5, 1.0), c, character=DENSITY)
    return LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                   vs=[4.0, 3.0]).with_field(1, "dens", h)


def test_a_harmonic_rule_read_from_a_file_is_stated_again(tmp_path):
    """A field read from a file by the harmonic rule is of the band of the
    file's grid: written again on a grid of at least that band it states
    the rule and is the same function, and on a coarser one the default."""
    two = band_limited()
    path = write_model(tmp_path / "h.nc", two, grid=gauss_legendre(8), fields=["dens"])
    back = read_model(path)
    assert back.layer(1)["dens"].lateral == "harmonic"
    rng = np.random.default_rng(0)
    r, t, p = (rng.uniform(0.55, 0.95, 200), rng.uniform(0, np.pi, 200),
               rng.uniform(0, 2 * np.pi, 200))
    want = two.layer(1)["dens"](r, t, p)
    own = read_model_file(path).parts[0].grid
    for label, grid, rule in (("own", own, "harmonic"),
                              ("finer", gauss_legendre(16), "harmonic"),
                              ("coarser", gauss_legendre(4), "linear")):
        again = write_model(tmp_path / f"{label}.nc", back, grid=grid,
                            fields=["dens"])
        check_model_file(again, back)
        var = read_model_file(again).parts[0].variables["dens"]
        assert var.attributes["lateral_interpolation"] == rule, label
        if rule == "harmonic":
            got = read_model(again).layer(1)["dens"](r, t, p)
            assert np.allclose(got, want, rtol=0, atol=1e-12), label


def test_a_harmonic_rule_is_refused_on_a_grid_without_its_band(tmp_path):
    """A relief and a function on an interface held by the harmonic rule
    state it, and a file in which either's group states no band is
    refused."""
    two = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                  vs=[4.0, 3.0])
    g = gauss_legendre(8)
    relief = GriddedFunction(g, 0.01 * np.cos(g.colatitudes)[:, None]
                             + 0.0 * g.longitudes[None, :], lateral="harmonic")
    m = two.stretched(layer_linear(two.skeleton, [None, None, relief]))
    path = write_model(tmp_path / "h.nc", m, grid=gauss_legendre(4), fields=["rho"],
                       surfaces={"K": {1: relief}})
    data = read_model_file(path)
    assert data.geometry.lateral == "harmonic"
    assert data.surfaces.variables["K"].attributes[
        "lateral_interpolation"] == "harmonic"
    assert read_surfaces(path)["K"][1].lateral == "harmonic"
    for group, edit in (("geometry", lambda g: g.delncattr("grid_kind")),
                        ("geometry", lambda g: g.delncattr("grid_lmax")),
                        ("surfaces", lambda g: g.delncattr("grid_lmax")),
                        ("surfaces", lambda g: g.setncattr("grid_kind", "custom"))):
        bad = tmp_path / "bad.nc"
        bad.write_bytes(path.read_bytes())
        corrupt(bad, lambda ds: edit(ds.groups[group]))
        with pytest.raises(ValueError, match=f"malformed model file .*{group}/.* "
                                             "states the lateral rule 'harmonic'"):
            read_model(bad)


def test_a_displacement_states_its_radial_rule(tmp_path):
    two = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                  vs=[4.0, 3.0])
    path = write_model(tmp_path / "f.nc", two.stretched(flattening(0.01)),
                       grid=gauss_legendre(6))
    corrupt(path, lambda ds: ds.groups["part_0"].variables["displacement"].delncattr(
        "radial_interpolation"))
    with pytest.raises(ValueError, match="displacement lacks the attributes"):
        read_model_file(path)


def test_what_the_file_cannot_hold_faithfully_is_refused(prem, tmp_path):
    two = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                  vs=[4.0, 3.0])
    flat = two.stretched(flattening(0.01))
    with pytest.raises(ValueError, match="every layer one grid.*layer_linear"):
        write_model(tmp_path / "f.nc", flat,
                    grid={0: gauss_legendre(12), 1: driscoll_healy(12)})
    def bump(t, p):
        return 0.01 * np.cos(t)

    bumpy = two.stretched(layer_linear(two.skeleton, [None, bump, None]))
    with pytest.raises(ValueError, match="one grid of their own"):
        write_model(tmp_path / "b.nc", bumpy, grid=gauss_legendre(4))
    from planetmodel import frozen
    with pytest.raises(TypeError, match="complex"):
        write_model(tmp_path / "z.nc", frozen(prem, 1e-3), grid=gauss_legendre(2),
                    fields=["A"])
    good = write_model(tmp_path / "good.nc", two, grid=gauss_legendre(2))
    before = good.read_bytes()
    with pytest.raises(TypeError, match="meta"):
        write_model(good, two, grid=gauss_legendre(2), meta={"flag": True})
    assert good.read_bytes() == before                  # the good file survives
    assert sorted(p.name for p in tmp_path.iterdir()) == ["good.nc"]


def test_the_geometry_s_tolerance_comes_back(tmp_path):
    """A geometry of its own rtol is read back with it; a file that states
    none, as one written before it was recorded, gives the default."""
    from planetmodel import Geometry, Skeleton
    from planetmodel.skeleton import DEFAULT_RTOL
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    loose = m.with_geometry(Geometry(Skeleton([0.0, 0.5, 1.0]), rtol=1e-6))
    path = write_model(tmp_path / "loose.nc", loose, grid=gauss_legendre(2))
    check_model_file(path, loose)
    assert read_model_file(path).rtol == 1e-6
    assert read_model(path).geometry.rtol == 1e-6
    with netCDF4.Dataset(path, "a") as ds:
        ds.delncattr("rtol")
    assert read_model(path).geometry.rtol == DEFAULT_RTOL


def test_constants_come_back_with_their_dimensions(tmp_path):
    deck = prem_deck_model()
    path = write_model(tmp_path / "d.nc", deck, grid=gauss_legendre(0),
                       fields=["rho"])
    back = read_model(path)
    assert back.constant("omega_ref") == pytest.approx(deck.constant("omega_ref"))
    dims = back.constants["omega_ref"].dimensions
    assert dims == deck.constants["omega_ref"].dimensions
    assert read_model(path, scales=Scales.geophysical(6371e3)).constant(
        "omega_ref") == pytest.approx(deck.nondimensionalised().constant("omega_ref"))
    for prem in (PREM(), PREM(ocean=False), PREM().nondimensionalised()):
        path = write_model(tmp_path / "prem.nc", prem, grid=gauss_legendre(0),
                           fields=["rho"])
        back = read_model(path)
        assert back.constants["omega_ref"].dimensions == (
            prem.constants["omega_ref"].dimensions)
        assert back.constants["omega_ref"].value_si == pytest.approx(
            2.0 * np.pi, rel=1e-14)
        assert back.constants["omega_ref"].meaning == (
            prem.constants["omega_ref"].meaning)
        assert read_model(path, scales=prem.scales).constant(
            "omega_ref") == pytest.approx(prem.constant("omega_ref"), rel=1e-14)


def test_a_missing_perturbation_kind_is_refused(prem, tmp_path):
    lo, hi = 3480e3, 5701e3
    dlnvs = AnalyticField((lo, hi), lambda r, t, p: 0.01 * np.cos(t))
    mp = perturbed(prem, "vsv", dlnvs, kind="relative")
    path = write_model(tmp_path / "p.nc", mp, grid=gauss_legendre(4), fields=["vsv"])
    corrupt(path, lambda ds: ds.groups["part_0"].variables["vsv"].delncattr(
        "perturbation_kind"))
    with pytest.raises(ValueError, match="perturbation_kind"):
        validate_structure(path)


# -- fields on interfaces -------------------------------------------------------

KERNEL = FieldSpec(SCALAR, Dimensions(length=-2), meaning="a shape kernel")


def shape(t, p):
    return np.sin(t) ** 2 * np.cos(2 * p) + 0.3 * np.cos(t)


def shift(t, p):
    return 1e3 * (np.cos(t) ** 2 - 1.0 / 3.0) + 0.0 * p


def with_kernel(model):
    return model.replaced(specs={**model.specs, "K_shape": KERNEL})


def same_grid(a, b):
    """The same grid, its angles through degrees both ways."""
    return (a.kind == b.kind and a.lmax == b.lmax
            and np.allclose(a.colatitudes, b.colatitudes, rtol=0, atol=1e-15)
            and np.allclose(a.longitudes, b.longitudes, rtol=0, atol=1e-15))


@pytest.mark.parametrize("grid", [gauss_legendre(8), equiangular(9, 16, poles=True)],
                         ids=["without poles", "with poles"])
def test_fields_on_interfaces_come_back_as_gridded_functions(prem, tmp_path, grid):
    m = with_kernel(prem)
    surfaces = {"K_shape": {"cmb": shape, 12: shift}, "shift": {"moho": shift}}
    path = write_model(tmp_path / "s.nc", m, grid=gauss_legendre(2), fields=["rho"],
                       surfaces=surfaces, surface_grid=grid)
    check_model_file(path, m, surfaces=surfaces)
    data = read_model_file(path)
    assert data.surfaces.interfaces == (1, 9, 12)
    assert ("K_shape_poles" in data.surfaces.variables) == (grid.kind != "equiangular")
    back = read_surfaces(path)
    assert {n: sorted(on) for n, on in back.items()} == {
        "K_shape": ["cmb", "surface"], "shift": ["moho"]}
    rng = np.random.default_rng(3)
    t, p = rng.uniform(0, np.pi, 50), rng.uniform(-np.pi, 3 * np.pi, 50)
    f = back["K_shape"]["cmb"]
    assert isinstance(f, GriddedFunction) and f.lateral == "linear"
    assert np.allclose(f(np.array([0.0, np.pi]), np.zeros(2)),
                       shape(np.array([0.0, np.pi]), np.zeros(2)), atol=1e-14)
    assert np.allclose(f(t, p), shape(t, p), atol=0.1)      # bilinear between nodes
    spec = read_model(path).spec("K_shape")
    assert spec.dimensions == KERNEL.dimensions and spec.meaning == KERNEL.meaning
    assert read_model(path).spec("shift").dimensions is None
    # what is read back writes the same numbers again
    again = write_model(tmp_path / "again.nc", read_model(path), grid=gauss_legendre(2),
                        surfaces=back)
    second = read_model_file(again).surfaces
    assert second.interfaces == (1, 9, 12) and same_grid(second.grid, grid)
    for name, var in data.surfaces.variables.items():
        scale = np.nanmax(np.abs(var.values))
        assert np.allclose(second.variables[name].values, var.values, rtol=0,
                           atol=1e-14 * scale, equal_nan=True)


def test_a_surface_field_states_its_rule_and_takes_its_grid(tmp_path):
    two = with_kernel(LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0],
                                              vp=[9.0, 7.0], vs=[4.0, 3.0]))
    gf = GriddedFunction.sampled(shape, driscoll_healy(6))
    path = write_model(tmp_path / "a.nc", two, grid=gauss_legendre(4),
                       surfaces={"K_shape": {0: gf, 1: gf}})
    data = read_model_file(path)
    assert same_grid(data.surfaces.grid, driscoll_healy(6))
    assert data.surfaces.variables["K_shape"].attributes[
        "lateral_interpolation"] == "cubic"
    back = read_surfaces(path)["K_shape"]
    assert sorted(back) == [0, 1]                       # unnamed: by index
    assert back[1].lateral == "cubic"
    path = write_model(tmp_path / "b.nc", two, grid=gauss_legendre(4),
                       surfaces={"K_shape": {0: gf, 1: shape}},
                       interpolation={"K_shape": ("polynomial", "cubic")})
    data = read_model_file(path)
    assert same_grid(data.surfaces.grid, gauss_legendre(4))   # the outermost part's
    assert data.surfaces.variables["K_shape"].attributes[
        "lateral_interpolation"] == "cubic"
    assert read_surfaces(path, interpolation={"K_shape": ("polynomial", "linear")})[
        "K_shape"][0].lateral == "linear"


def test_a_file_without_surfaces_reads_as_none(prem, tmp_path):
    path = write_model(tmp_path / "p.nc", prem, grid=gauss_legendre(2), fields=["rho"])
    assert read_model_file(path).surfaces is None
    assert read_surfaces(path) == {}
    with netCDF4.Dataset(path) as ds:
        assert "surfaces" not in ds.groups


def test_writing_refuses_surface_fields_the_file_cannot_hold(prem, tmp_path):
    from planetmodel.character import VECTOR as V
    out = tmp_path / "x.nc"

    def write(surfaces, **kw):
        return write_model(out, prem, grid=gauss_legendre(2), fields=["rho"],
                           surfaces=surfaces, **kw)

    with pytest.raises(ValueError, match="keeps for itself"):
        write({"surface_interface": {"cmb": shape}})
    with pytest.raises(ValueError, match="never both"):
        write({"rho": {"cmb": shape}})
    with pytest.raises(ValueError, match="twice"):
        write({"K": {"cmb": shape, 1: shape}})
    with pytest.raises(KeyError, match="no interface named"):
        write({"K": {"core": shape}})
    with pytest.raises(ValueError, match="no interface"):
        write({"K": {}})
    with pytest.raises(TypeError, match="not a function"):
        write({"K": {"cmb": 3.0}})
    with pytest.raises(TypeError, match="complex"):
        write({"K": {"cmb": lambda t, p: (1 + 1j) * np.cos(t)}})
    with pytest.raises(ValueError, match="not finite"):
        write({"K": {"cmb": lambda t, p: np.where(t > 1.0, np.nan, 1.0)}})
    with pytest.raises(ValueError, match="no surfaces"):
        write(None, surface_grid=gauss_legendre(4))
    vector = prem.replaced(specs={**prem.specs, "wind": FieldSpec(V, None)})
    with pytest.raises(ValueError, match="number at each point"):
        write_model(out, vector, grid=gauss_legendre(2), fields=["rho"],
                    surfaces={"wind": {"cmb": shape}})
    assert not out.exists()


def surface_file(tmp_path):
    two = with_kernel(LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0],
                                              vp=[9.0, 7.0], vs=[4.0, 3.0]))
    return write_model(tmp_path / "s.nc", two, grid=gauss_legendre(4),
                       surfaces={"K_shape": {0: shape, 1: shift}, "shift": {1: shift}},
                       surface_grid=gauss_legendre(6))


@pytest.mark.parametrize("edit, message", [
    (lambda g: g.variables["surface_interface"].__setitem__(1, 2),
     "not interfaces of the file"),
    (lambda g: g.variables["surface_interface"].__setitem__(0, 1),
     "not interfaces of the file"),
    (lambda g: g.variables["K_shape"].setncattr("lateral_interpolation", "quintic"),
     "lateral rule"),
    (lambda g: g.renameVariable("shift_poles", "shift_p"), "shift_poles is missing"),
    (lambda g: g.renameVariable("shift", "rho"), "never both"),
    (lambda g: g.variables["shift"].setncattr("interfaces", np.int32([0, 3])),
     "not surfaces of the group"),
    (lambda g: g.setncattr("grid_kind", "healpix"), "grid_kind"),
])
def test_a_malformed_surfaces_group_is_refused_by_name(tmp_path, edit, message):
    path = surface_file(tmp_path)
    validate_structure(path)
    corrupt(path, lambda ds: edit(ds.groups["surfaces"]))
    with pytest.raises(ValueError, match=message):
        read_model_file(path)


def test_the_writers_stamp_their_provenance(tmp_path):
    from datetime import datetime

    import planetmodel
    from planetmodel.modelfile import stamped_source

    model = LayeredIsotropicElastic([0.0, 1.0], rho=[3.0], vp=[2.0], vs=[1.0])
    path = write_model(tmp_path / "stamped.nc", model, grid=gauss_legendre(4))
    data = read_model_file(path)
    assert data.source == f"planetmodel {planetmodel.__version__}"
    assert data.source == stamped_source(None)
    datetime.fromisoformat(data.created)
    path = write_model(tmp_path / "named.nc", model, grid=gauss_legendre(4),
                       source="my code 1.0")
    assert read_model_file(path).source == "my code 1.0"
    # a source that is not text, or is blank, is refused before a file is made
    for source in (3, b"my code", "", "  "):
        with pytest.raises(ValueError, match="source names the writing code"):
            write_model(tmp_path / "refused.nc", model, grid=gauss_legendre(4),
                        source=source)
    assert not (tmp_path / "refused.nc").exists()


# -- what a reader holds a file to, beyond what an edit in place can break ---------

def rebuilt(source, target, *without, sizes=None, then=None):
    """A copy of a netCDF file without the variables and dimensions
    `without` names (`<group>/<name>`, or `<name>` at the root), each
    dimension of `sizes` made that long and the variables over it cut to
    fit, and the edit `then` applied to the copy."""
    sizes = sizes or {}

    def copy(old, new, prefix):
        new.setncatts({key: old.getncattr(key) for key in old.ncattrs()})
        for name, dim in old.dimensions.items():
            if prefix + name not in without:
                new.createDimension(name, sizes.get(prefix + name, len(dim)))
        for name, var in old.variables.items():
            if prefix + name in without:
                continue
            var.set_auto_mask(False)
            attrs = {key: var.getncattr(key) for key in var.ncattrs()}
            out = new.createVariable(name, var.dtype, var.dimensions,
                                     fill_value=attrs.pop("_FillValue", None))
            out.set_auto_mask(False)
            out[...] = var[...][tuple(slice(0, sizes.get(prefix + d, None))
                                      for d in var.dimensions)]
            out.setncatts(attrs)
        for name, sub in old.groups.items():
            copy(sub, new.createGroup(name), f"{prefix}{name}/")

    with netCDF4.Dataset(source) as old, netCDF4.Dataset(target, "w") as new:
        copy(old, new, "")
        if then is not None:
            then(new)
    return target


def two_layers():
    return LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                   vs=[4.0, 3.0])


def lateral_file(tmp_path, grid):
    """A two-layer model whose density varies with direction, on `grid`."""
    rho = GriddedField.sampled(AnalyticField((0.5, 1.0), smooth, character=DENSITY),
                               grid, ngll=3)
    return write_model(tmp_path / "lateral.nc",
                       two_layers().with_field(1, "rho", rho, replace=True),
                       grid=grid, fields=["rho"])


def stacked_file(tmp_path, grid):
    """A two-layer model placed by boundary reliefs held on `grid`."""
    gf = GriddedFunction.sampled(bump, grid)
    m = two_layers()
    return write_model(tmp_path / "stacked.nc",
                       m.stretched(layer_linear(m.skeleton, [None, gf, gf])),
                       grid=grid)


def flattened_file(tmp_path, grid):
    """A two-layer model placed by a flattening, held as a displacement."""
    return write_model(tmp_path / "flattened.nc",
                       two_layers().stretched(flattening(0.01)), grid=grid)


def kernel_file(tmp_path, grid):
    """A two-layer model with a field on an interface, on `grid`."""
    return write_model(tmp_path / "kernel.nc", with_kernel(two_layers()), grid=grid,
                       fields=["rho"], surfaces={"K_shape": {1: shift}})


def poles_over(group, name, dims, **sizes):
    """An edit creating the polar values `name` of a group over `dims`,
    the dimensions of `sizes` created first."""
    def edit(ds):
        g = ds.groups[group]
        for dim, n in sizes.items():
            g.createDimension(dim, n)
        g.createVariable(name, "f8", dims)[...] = 0.0
    return edit


#: A file, what is left out of its copy or done to it, and the refusal: the
#: polar values a grid lacking a pole asks for, absent with their dimension
#: or of the wrong size or dimensions.
WITHOUT_POLES = {
    "a part's variable": (
        lateral_file, ("part_0/rho_poles", "part_0/pole"), None,
        "part_0/rho varies with direction on a grid without its poles, and "
        "rho_poles is missing"),
    "a part's variable, a pole short": (
        lateral_file, ("part_0/rho_poles", "part_0/pole"),
        poles_over("part_0", "rho_poles", ("node", "pole"), pole=1),
        "part_0/rho_poles does not hold both poles at every node"),
    "a part's variable, over other dimensions": (
        lateral_file, ("part_0/rho_poles",),
        poles_over("part_0", "rho_poles", ("pole", "node")),
        "part_0/rho_poles does not hold both poles at every node"),
    "a displacement": (
        flattened_file, ("part_0/displacement_poles", "part_0/pole"), None,
        "part_0/displacement varies with direction on a grid without its poles, "
        "and displacement_poles is missing"),
    "a displacement, without its components": (
        flattened_file, ("part_0/displacement_poles",),
        poles_over("part_0", "displacement_poles", ("node", "pole")),
        "part_0/displacement_poles does not hold both poles at every node"),
    "the stack": (
        stacked_file, ("geometry/boundary_displacement_poles", "geometry/pole"), None,
        "geometry/boundary_displacement varies with direction on a grid without "
        "its poles, and boundary_displacement_poles is missing"),
    "the stack, a pole short": (
        stacked_file, ("geometry/boundary_displacement_poles", "geometry/pole"),
        poles_over("geometry", "boundary_displacement_poles", ("boundary", "pole"),
                   pole=1),
        "geometry/boundary_displacement varies with direction on a grid without "
        "its poles, and boundary_displacement_poles is missing"),
    "the stack, over other dimensions": (
        stacked_file, ("geometry/boundary_displacement_poles",),
        poles_over("geometry", "boundary_displacement_poles", ("pole", "boundary")),
        "geometry/boundary_displacement_poles has the wrong dimensions"),
    "a field on an interface": (
        kernel_file, ("surfaces/K_shape_poles", "surfaces/pole"), None,
        "surfaces/K_shape is on a grid without its poles, and K_shape_poles is "
        "missing"),
    "a field on an interface, a pole short": (
        kernel_file, ("surfaces/K_shape_poles", "surfaces/pole"),
        poles_over("surfaces", "K_shape_poles", ("surface", "pole"), pole=1),
        "surfaces/K_shape is on a grid without its poles, and K_shape_poles is "
        "missing"),
}


@pytest.mark.parametrize("case", list(WITHOUT_POLES))
def test_a_grid_lacking_a_pole_asks_for_the_polar_values(tmp_path, case):
    """Whether a file needs polar values is the grid's to say, its first
    latitude short of 90 or its last of -90, not a `pole` dimension's: a
    file without them, or with them of the wrong size, is malformed to
    every reader."""
    make, without, then, message = WITHOUT_POLES[case]
    good = make(tmp_path, gauss_legendre(6))
    validate_structure(good)
    bad = rebuilt(good, tmp_path / "bad.nc", *without, then=then)
    for read in (validate_structure, read_model_file, read_model):
        with pytest.raises(ValueError, match="malformed model file") as refusal:
            read(bad)
        assert str(refusal.value).endswith(": " + message), case


@pytest.mark.parametrize("make", [lateral_file, stacked_file, flattened_file,
                                  kernel_file])
def test_a_grid_holding_both_poles_asks_for_no_polar_values(tmp_path, make):
    """On a grid whose first latitude is 90 and whose last is -90 a file
    holds no polar values and no `pole` dimension, and reads."""
    path = make(tmp_path, equiangular(7, 12, poles=True))
    with netCDF4.Dataset(path) as ds:
        groups = [ds, *ds.groups.values()]
        assert not any("pole" in g.dimensions for g in groups)
        assert not any(name.endswith("_poles") for g in groups for name in g.variables)
    validate_structure(path)
    read_model(path)


def test_a_grid_one_pole_short_asks_for_the_polar_values(tmp_path):
    """A grid holding one pole and not the other lacks a pole: the last
    latitude decides it as the first does."""
    grid = driscoll_healy(5)
    assert grid.pole_nodes == (True, False)
    good = lateral_file(tmp_path, grid)
    validate_structure(good)
    bad = rebuilt(good, tmp_path / "bad.nc", "part_0/rho_poles", "part_0/pole")
    with pytest.raises(ValueError, match="rho_poles is missing"):
        read_model_file(bad)


@pytest.mark.parametrize("change, message", [
    (dict(then=lambda ds: ds["geometry"]["boundary_radius"].__setitem__(1, 1.5)),
     "geometry/boundary_radius does not increase strictly"),
    (dict(then=lambda ds: ds["geometry"]["boundary_radius"].__setitem__(1, 0.0)),
     "geometry/boundary_radius does not increase strictly"),
    (dict(sizes={"geometry/boundary": 1}),
     "geometry/boundary_radius is fewer than two surfaces"),
])
def test_a_stack_is_two_surfaces_or_more_at_increasing_radii(tmp_path, change,
                                                             message):
    good = stacked_file(tmp_path, gauss_legendre(6))
    bad = rebuilt(good, tmp_path / "bad.nc", **change)
    for read in (validate_structure, read_model_file, read_model):
        with pytest.raises(ValueError, match="malformed model file") as refusal:
            read(bad)
        assert str(refusal.value).endswith(": " + message)


@pytest.mark.parametrize("change, message", [
    (dict(sizes={"layer": 0}),
     "layer_bounds do not make a skeleton of contiguous layers"),
    (dict(then=lambda ds: ds["layer_bounds"].__setitem__((0, 0), -0.1)),
     "layer_bounds do not make a skeleton: boundary radii must be non-negative"),
    (dict(then=lambda ds: ds["layer_bounds"].__setitem__((1, 1), np.inf)),
     "layer_bounds do not make a skeleton: boundary radii must be finite"),
])
def test_layer_bounds_that_make_no_skeleton_are_refused_by_name(tmp_path, change,
                                                                message):
    good = lateral_file(tmp_path, gauss_legendre(6))
    bad = rebuilt(good, tmp_path / "bad.nc", **change)
    with pytest.raises(ValueError, match="malformed model file") as refusal:
        read_model_file(bad)
    assert str(refusal.value).endswith(": " + message)


# -- the check ------------------------------------------------------------------

def scale_the_poles(group, name, factor=1.0 + 1e-6):
    """An edit multiplying the polar values `name` of a group by `factor`."""
    def edit(ds):
        var = ds.groups[group].variables[name]
        var[:] = np.asarray(var[:]) * factor
    return edit


def test_the_check_holds_the_root_and_the_polar_values(prem, tmp_path):
    """`check_model_file` refuses a file whose name or constants, or the
    polar values of whose perturbation, reliefs or displacement, are not
    the model's."""
    def refused(good, model, edit, message):
        check_model_file(good, model)
        bad = tmp_path / "bad.nc"
        bad.write_bytes(good.read_bytes())
        corrupt(bad, edit)
        with pytest.raises(AssertionError, match=message):
            check_model_file(bad, model)

    mp = perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="relative")
    good = write_model(tmp_path / "p.nc", mp, grid=gauss_legendre(4),
                       fields=["rho", "vsv"])
    refused(good, mp, lambda ds: ds.setncattr("name", "other"), "names its model")
    refused(good, mp, lambda ds: ds.setncattr("constant_G", 1.0), "constant G")
    refused(good, mp, lambda ds: ds.delncattr("constant_G_dimensions"),
            "dimensions of the model's constant G")

    def raise_the_poles(ds):
        ds.groups["part_0"].variables["vsv_perturbation_poles"][:] = 1e-3

    refused(good, mp, raise_the_poles, "part_0/vsv_perturbation_poles")

    two = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                  vs=[4.0, 3.0])
    gf = GriddedFunction.sampled(bump, gauss_legendre(16))
    stacked = two.stretched(layer_linear(two.skeleton, [None, gf, gf]))
    good = write_model(tmp_path / "s.nc", stacked, grid=gauss_legendre(4))
    refused(good, stacked, scale_the_poles("geometry", "boundary_displacement_poles"),
            r"boundary_displacement\[1\], its poles")
    flat = two.stretched(flattening(0.01))
    good = write_model(tmp_path / "f.nc", flat, grid=gauss_legendre(8))
    refused(good, flat, scale_the_poles("part_0", "displacement_poles"),
            "part_0/displacement_poles")
