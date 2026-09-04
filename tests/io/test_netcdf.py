"""The `planetmodel.model/1` file: what it says, and what comes back from it.

Three things are checked here.  The *header* is read with netCDF4
alone: the domain of every field as its `layers` attribute and the fill
value on the nodes outside it, `layer_fields` and `layer_is_vacuum` in `/skeleton`, the
`model_class`, `/rheology`, and `reference_period` in the file's own
time unit.  The *reader* is checked by round
trip: the sample comes back whole, the body comes back with its radial
scalar fields, PREM -- whose layer functions are polynomials of
degree at most three, below the GLL order -- comes back as the same
polynomials rather than as an interpolation of them, and it comes back
as the same `ViscoelasticModel`, its `viscoelastic_moduli`
rebuilt from the `LawRecord` the file records.  And the *layout*
is checked from a fresh process that has never imported planetmodel, which
is the only honest test of the promise that a consumer needs netCDF4
and numpy and nothing else.

The unit body is four layers across: a density-only core, a shell
carrying an analytic field, an empty solid shell and a vacuum buffer;
the unit *model* is a three-layer isotropic viscoelastic body, elastic
above and below a Maxwell shell, which is where the laws are checked
exactly.  The Earth-sized objects are the PREM file of the review note
and one written at GLL order 7, which are meshless and cost a tenth of
a second each: PREM's moduli are products of its cubics and are
degree 7, so order 4 interpolates them and only order 7 reproduces
them, which is what the `1e-12` round trip of the rheology needs.
"""
import json
import subprocess
import sys

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from planetmodel import (DENSITY, AnalyticField, AnalyticTopography,  # noqa: E402
                    Dimensions, PREM, RadialField, RadialMesh, ReferenceBody,
                    Skeleton, Symmetry, layer_linear)
from planetmodel.io.deck import read_isotropic_deck                         # noqa: E402
from planetmodel.io.netcdf import read, write                         # noqa: E402
from planetmodel.model.materials import ElasticField                  # noqa: E402
from planetmodel.model.rheology import maxwell                        # noqa: E402
from planetmodel.model.classes import ElasticModel, ViscoelasticModel  # noqa: E402
from planetmodel.sampling import AngularGrid                          # noqa: E402
from planetmodel.testing import check_sample                          # noqa: E402

pytestmark = pytest.mark.netcdf

Dataset = netCDF4.Dataset

B = 1.0
SK = Skeleton([0.0, 0.5 * B, B])


def body(*, scales=None, meta=None):
    """A density-only core, an analytic field above it, then two holes."""
    rho = RadialField(SK, [lambda r: 5.0e3 - 2.0e3 * (np.asarray(r) / B) ** 2,
                           None],
                      name="rho", character=DENSITY,
                      dimensions=Dimensions.DENSITY)
    scal = AnalyticField(lambda r, t, p: 1.0 + r / B + 0.1 * np.sin(t)
                         * np.cos(p), SK, character=DENSITY, name="scal")
    b = ReferenceBody.from_fields(SK, {"rho": rho, "scal": scal},
                      scales=scales, meta=meta)
    return (b.extended([1.4 * B], fields=None, interface_names=["crust"])
            .with_buffer(ratio=0.2))


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    b = body(meta={"name": "holes", "tref": 1.0})
    sample = b.sample(AngularGrid.gauss_legendre(2), drmax=0.3 * B)
    path = tmp_path_factory.mktemp("nc") / "holes.nc"
    write(b, sample, path)
    return b, sample, path


def prem_body():
    """The review note's recipe: PREM, four interfaces named, a bulge."""
    prem = (PREM().classify_states()
            .name_interface(0, "icb").name_interface(1, "cmb")
            .name_interface(-3, "moho").name_interface(-1, "surface"))
    relief = AnalyticTopography(
        lambda t, p: 8.0e2 * np.sin(t) ** 2 * np.cos(2.0 * p)
        + 3.0e2 * np.cos(t), name="degree-2 bulge plus a degree-1 tilt")
    return prem.with_surface("surface", relief)


#: The 50 s wave the review file's complex sample is taken at: away from
#: PREM's 1 s reference period, so the dispersion is visible.
OMEGA = 2.0 * np.pi / 50.0


@pytest.fixture(scope="module")
def prem_file(tmp_path_factory):
    """PREM plus an analytic surface, mapped by `layer_linear`."""
    b = prem_body()
    m = b.mapping(rule=layer_linear())
    sample = b.sample(AngularGrid.gauss_legendre(8), mapping=m, omega=OMEGA)
    path = tmp_path_factory.mktemp("nc") / "prem_bulge.nc"
    write(b, sample, path)
    return b, sample, path


@pytest.fixture(scope="module")
def prem_exact_file(tmp_path_factory):
    """The same PREM at GLL order 7, where its moduli are exact.

    A modulus is rho v^2, degree 7 in PREM's cubics, so the order-4
    elements of the review file interpolate it and the rebuilt rheology
    inherits that error; order 7 reproduces it and the round trip of
    `viscoelastic_moduli` is round-off.
    """
    b = prem_body()
    sample = b.sample(AngularGrid.gauss_legendre(4),
                      radial=RadialMesh(b, ngll=8, lmax=4), omega=OMEGA)
    path = tmp_path_factory.mktemp("nc") / "prem_order7.nc"
    write(b, sample, path)
    return b, sample, path


def viscoelastic_body():
    """Three isotropic layers, Maxwell in the middle and elastic outside.

    Constant moduli, so every layer function is exact at any GLL order
    and the round trip of the law is bit-exact rather than merely close.
    """
    sk = Skeleton([0.0, 0.4 * B, 0.7 * B, B])

    def rf(name, values, dimensions):
        return RadialField(
            sk, [(lambda v: lambda r: v + 0.0 * np.asarray(r))(v)
                 for v in values], name=name, dimensions=dimensions)

    kappa = rf("kappa", [2.0e11, 1.3e11, 8.0e10], Dimensions.MODULUS)
    mu = rf("mu", [1.0e11, 6.0e10, 4.0e10], Dimensions.MODULUS)
    fields = {
        "rho": rf("rho", [8.0e3, 5.0e3, 3.0e3], Dimensions.DENSITY),
        "kappa": kappa, "mu": mu,
        "viscosity": rf("viscosity", [1.0e21] * 3, Dimensions.VISCOSITY),
    }
    fields["elastic_moduli"] = ElasticField(Symmetry.ISOTROPIC,
                                     {"kappa": kappa, "mu": mu},
                                     name="elastic_moduli")
    b = ReferenceBody.from_fields(sk, fields, meta={"name": "maxwell shell"})
    b.add_field("viscoelastic_moduli",
                maxwell(b["elastic_moduli"], b["viscosity"]).restricted(1))
    return b.as_class(ViscoelasticModel)


# ------------------------------------------------------------------ the header

def test_the_domain_of_every_field_is_in_the_file(written):
    b, sample, path = written
    with Dataset(str(path)) as ds:
        g = ds.groups["fields"]
        assert json.loads(g.variables["rho"].layers) == [0]
        assert json.loads(g.variables["scal"].layers) == [0, 1]
        for name in ("rho", "scal"):
            v = g.variables[name]
            assert np.isnan(v._FillValue)
            arr = v[...]
            node_layer = np.repeat(sample.element_layer, sample.radial.ngll)
            inside = np.isin(node_layer, sample.metadata.domains[name])
            flat = np.ma.getmaskarray(arr).reshape(arr.shape[0], -1)
            assert np.all(flat[~inside]), (
                f"{name} is not fill outside its domain")
            assert not flat[inside].any(), f"{name} is fill inside its domain"
            raw = np.asarray(arr.filled(np.nan)).reshape(arr.shape[0], -1)
            assert np.all(np.isfinite(raw[inside]))


def test_the_skeleton_says_what_each_layer_holds(written):
    b, _, path = written
    with Dataset(str(path)) as ds:
        g = ds.groups["skeleton"]
        assert "layer_material" not in g.variables
        assert "layer_is_buffer" not in g.variables
        assert [json.loads(s) for s in g.variables["layer_fields"][:]] == \
            [["rho", "scal"], ["scal"], [], []]
        assert list(g.variables["layer_is_vacuum"][:]) == [0, 0, 0, 1]
        assert list(g.variables["layer_state"][:]) == \
            ["solid", "solid", "solid", "vacuum"]


def test_the_root_says_the_model_class(written):
    """A plain body guarantees nothing, and writes no rheology at all."""
    with Dataset(str(written[2])) as ds:
        assert ds.model_class == ""          # a plain ReferenceBody
        assert ds.index_base == 0
        # No law, no period: the root says only what the laws say.
        assert "reference_period" not in ds.ncattrs()
        assert "reserved_rheology" not in ds.ncattrs()
        assert "reserved_complex_sample" not in ds.ncattrs()
        assert "rheology" not in ds.groups


def test_default_names_are_zero_based(written):
    """An unnamed layer is `layer_0`, an unnamed interface `interface_0`."""
    with Dataset(str(written[2])) as ds:
        g = ds.groups["skeleton"]
        assert [str(s) for s in g.variables["layer_name"][:]] == \
            ["layer_0", "layer_1", "layer_2", "buffer"]
        assert [str(s) for s in g.variables["interface_name"][:]] == \
            ["interface_0", "interface_1", "crust", "buffer"]


def test_a_non_dimensional_body_writes_the_period_in_its_own_time_unit(
        tmp_path):
    """The root period comes from the laws, in the file's time unit, and
    a rescaled viscoelastic body reads back agreeing with itself."""
    m = PREM(ocean=False).nondimensionalised()
    assert "reference_period" not in m.meta
    sample = m.sample(AngularGrid.gauss_legendre(2),
                      drmax=0.2 * m.skeleton.boundaries[-1], omega=OMEGA)
    path = tmp_path / "nd.nc"
    write(m, sample, path)
    with Dataset(str(path)) as ds:
        assert ds.reference_period == pytest.approx(1.0 / m.scales.time)
        assert ds.scales_time_s == m.scales.time
        sub = ds.groups["rheology"].groups["viscoelastic_moduli"]
        row = json.loads(str(sub.variables["constants"][3]))
        assert row["reference_period"] == pytest.approx(1.0 / m.scales.time)
        dims = json.loads(str(sub.variables["constant_dimensions"][3]))
        assert dims["reference_period"] == [0, 0, 1]
    back, _ = read(path)
    assert "tref" not in back.meta
    rec = back.layer(3)["viscoelastic_moduli"].law
    assert rec.constants["reference_period"] == pytest.approx(1.0 / m.scales.time)
    r = np.array([0.6 * m.skeleton.boundaries[-1]])
    w = OMEGA
    assert np.allclose(back.layer(3)["viscoelastic_moduli"].evaluate(r, omega=w),
                       m.layer(3)["viscoelastic_moduli"].evaluate(r, omega=w),
                       rtol=1e-6)


def test_a_body_without_a_reference_period_writes_none(tmp_path):
    b = body(meta={"name": "plain"})
    sample = b.sample(AngularGrid.gauss_legendre(2), drmax=0.3 * B)
    path = tmp_path / "plain.nc"
    write(b, sample, path)
    with Dataset(str(path)) as ds:
        assert "reference_period" not in ds.ncattrs()


# ------------------------------------------------------------------ the reader

def _same(a, b, *, what):
    """Equal to 1e-12, NaN for NaN: the round trip the plan pins."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"{what}: {a.shape} became {b.shape}"
    assert np.array_equal(np.isnan(a), np.isnan(b)), (
        f"{what}: the holes moved")
    m = ~np.isnan(a)
    assert np.allclose(a[m], b[m], rtol=1e-12, atol=0.0), f"{what} differs"


def test_the_sample_comes_back_whole(written):
    b, sample, path = written
    with pytest.warns(UserWarning, match="scal"):
        _, back = read(path)
    check_sample(back)                       # source is None: layout only
    assert set(back.fields) == set(sample.fields)
    _same(sample.radius, back.radius, what="radius")
    assert np.array_equal(sample.element_start, back.element_start)
    assert np.array_equal(sample.element_layer, back.element_layer)
    assert back.radial.ngll == sample.radial.ngll
    for name in sample.fields:
        _same(sample.fields[name], back.fields[name], what=f"field {name}")
    grid, gback = sample.angular, back.angular
    _same(grid.colatitudes, gback.colatitudes, what="colatitudes")
    _same(grid.longitudes, gback.longitudes, what="longitudes")
    _same(grid.weights, gback.weights, what="angular weights")
    assert (gback.kind, gback.lmax) == (grid.kind, grid.lmax)
    meta, mback = sample.metadata, back.metadata
    assert mback.characters == meta.characters
    assert mback.dimensions == meta.dimensions
    assert mback.frames == meta.frames
    assert mback.domains == meta.domains
    assert mback.scales == meta.scales
    assert mback.skeleton == meta.skeleton
    assert back.source is None and back.mapping is None


def test_the_body_comes_back_with_its_radial_scalar_fields(written):
    b, sample, path = written
    with pytest.warns(UserWarning, match="scal"):
        back, back_sample = read(path)
    assert type(back) is ReferenceBody
    assert back.skeleton == b.skeleton
    # rho is radial and rank 0, so it is restored on its one layer; scal
    # depends on direction, so it lives in the sample alone.
    assert back.field_names == ("rho",)
    assert back["rho"].domain == (0,)
    assert "scal" not in back
    assert "scal" in back_sample.fields
    r = np.linspace(0.05, 0.45, 41)
    assert np.allclose(back["rho"][0](r), b["rho"][0](r), rtol=1e-12)
    assert back["rho"].character == b["rho"].character
    assert back["rho"].dimensions == b["rho"].dimensions
    # the annotations: default names are not names, and states survive
    assert [lay.name for lay in back.layers] == [None, None, None, "buffer"]
    assert [lay.state for lay in back.layers] == \
        ["solid", "solid", "solid", "vacuum"]
    assert [f.name for f in back.interfaces] == [None, None, "crust", "buffer"]
    assert [f.role for f in back.interfaces] == ["material"] * 4
    assert back.meta["name"] == "holes" and "tref" not in back.meta
    assert back.scales == b.scales
    assert not back.surfaces        # relief is not read back until M9


def test_prem_comes_back_as_the_same_polynomials(prem_file):
    """The GLL order is 4 and PREM's layer functions are cubics.

    Interpolation through five nodes reproduces a cubic exactly, so the
    restored `rho` is not an approximation of PREM's but PREM's own
    polynomial, to round-off.
    """
    b, sample, path = prem_file
    back, back_sample = read(path)
    rho, rho_back = b["rho"], back["rho"]
    assert rho_back.domain == rho.domain
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in rho.domain:
        lo, hi = b.skeleton.interval(i)
        r = rng.uniform(lo, hi, 200)
        want, got = rho[i](r), rho_back[i](r)
        worst = max(worst, float(np.max(np.abs(got - want) / np.abs(want))))
    assert worst < 1e-12, f"rho differs by {worst:.3e} relative"
    # the whole sample, arrays and displacement alike
    check_sample(back_sample)
    for name in sample.fields:
        _same(sample.fields[name], back_sample.fields[name],
              what=f"field {name}")
    _same(sample.displacement, back_sample.displacement, what="displacement")
    # elastic is rank 4, and it is rebuilt from the
    # components the file names beside it.
    assert "elastic_moduli" in back_sample.fields
    assert back["elastic_moduli"].symmetry is Symmetry.VTI
    assert back["elastic_moduli"].moduli_names == ("A", "C", "F", "L", "N")
    assert set(back.field_names) == set(b.field_names)


def test_a_second_write_is_byte_identical_in_the_fields(prem_file, tmp_path):
    b, sample, path = prem_file
    back, back_sample = read(path)
    again = tmp_path / "again.nc"
    write(back, back_sample, again)
    with Dataset(str(path)) as one, Dataset(str(again)) as two:
        one.set_auto_mask(False)
        two.set_auto_mask(False)
        first, second = one.groups["fields"], two.groups["fields"]
        assert set(first.variables) == set(second.variables)
        for name in first.variables:
            a = np.asarray(first.variables[name][...], dtype=float)
            c = np.asarray(second.variables[name][...], dtype=float)
            assert a.tobytes() == c.tobytes(), f"{name} is not byte-identical"
        assert np.array_equal(one.groups["radial"].variables["radius"][:],
                              two.groups["radial"].variables["radius"][:])




# ---------------------------------------------------------------- the rheology

def test_the_rheology_group_says_how_each_layer_was_built(prem_file):
    """PREM: constant Q on every layer, at the 1 s reference period."""
    b, _, path = prem_file
    with Dataset(str(path)) as ds:
        assert ds.model_class == "ViscoelasticModel"
        sub = ds.groups["rheology"].groups["viscoelastic_moduli"]
        assert sub.kind == "frequency" and sub.omega_domain == "real"
        assert (sub.character_rank, sub.character_weight) == (4, 1)
        n = b.skeleton.nlayers
        assert [str(s) for s in sub.variables["law"][:]] == ["constant_q"] * n
        assert [json.loads(s) for s in sub.variables["parameters"][:]] == \
            [["elastic_moduli", "qkappa", "qmu"]] * n
        assert [json.loads(s) for s in sub.variables["constants"][:]] == \
            [{"reference_period": 1.0}] * n
        assert [str(s) for s in sub.variables["convention"][:]] == \
            ["voigt_average"] * n


def test_prem_comes_back_as_a_viscoelastic_model(prem_exact_file):
    """The law rebuilt from its record."""
    b, _, path = prem_exact_file
    back, _ = read(path)
    assert isinstance(back, ViscoelasticModel)
    assert "tref" not in back.meta
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in range(b.skeleton.nlayers):
        lo, hi = b.skeleton.interval(i)
        r = rng.uniform(lo, hi, 50)
        for omega in (2.0e-3, 2.0 * np.pi, 25.0):
            want = b["viscoelastic_moduli"].evaluate(r, layer=i, omega=omega)
            got = back["viscoelastic_moduli"].evaluate(r, layer=i, omega=omega)
            worst = max(worst, float(np.max(np.abs(got - want))
                                     / np.max(np.abs(want))))
    assert worst < 1e-12, f"viscoelastic_moduli differs by {worst:.3e} relative"


def test_the_complex_sample_carries_its_omega_and_both_parts(prem_file):
    b, sample, path = prem_file
    with Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        v = ds.groups["fields"].variables["viscoelastic_moduli"]
        assert v.dimensions == ("node", "voigt_i", "voigt_j", "part")
        assert ds.dimensions["part"].size == 2
        assert v.getncattr("part") == "complex"
        assert v.getncattr("omega") == pytest.approx(OMEGA)
        arr = np.asarray(v[...], dtype=float)
    assert arr.tobytes() == sample.fields["viscoelastic_moduli"].tobytes()
    back, back_sample = read(path)
    assert back_sample.metadata.omegas == {"viscoelastic_moduli": OMEGA}
    assert (back_sample.fields["viscoelastic_moduli"].tobytes()
            == sample.fields["viscoelastic_moduli"].tobytes())
    check_sample(back_sample)
    # what the two halves are: the field itself at that frequency
    node_layer = np.repeat(sample.element_layer, sample.radial.ngll)
    node = int(np.flatnonzero(node_layer == 4)[3])
    got = sample.fields["viscoelastic_moduli"][node]
    want = b["viscoelastic_moduli"].evaluate(np.array([sample.radius[node]]),
                                        layer=4, omega=OMEGA)[0]
    assert np.allclose(got[..., 0] + 1j * got[..., 1], want, rtol=1e-14)
    assert np.any(np.abs(np.imag(want)) > 0.0)     # constant Q does attenuate


def test_a_lifted_layer_is_recorded_as_static_and_lifted_again(tmp_path):
    """An elastic layer beside a Maxwell one: `law = "static"`."""
    m = viscoelastic_body()
    omega = 1.0e-11
    sample = m.sample(AngularGrid.gauss_legendre(2), drmax=0.2 * B,
                      omega=omega)
    path = tmp_path / "maxwell.nc"
    write(m, sample, path)
    with Dataset(str(path)) as ds:
        sub = ds.groups["rheology"].groups["viscoelastic_moduli"]
        assert [str(s) for s in sub.variables["law"][:]] == \
            ["static", "maxwell", "static"]
        assert [json.loads(s) for s in sub.variables["parameters"][:]] == \
            [["elastic_moduli"], ["elastic_moduli", "viscosity"], ["elastic_moduli"]]
        assert [str(s) for s in sub.variables["convention"][:]] == [""] * 3
        assert sub.omega_domain == "complex"
    back, _ = read(path)
    assert isinstance(back, ViscoelasticModel)
    r = np.array([0.2 * B, 0.5 * B, 0.85 * B])
    for w in (1.0e-12, omega, 1.0e-9, 1.0e-12 + 5.0e-13j):
        assert np.array_equal(back["viscoelastic_moduli"].evaluate(r, omega=w),
                              m["viscoelastic_moduli"].evaluate(r, omega=w))
    # the lifted layers are the static tensor, unmoved by frequency
    assert np.all(np.imag(back["viscoelastic_moduli"].evaluate(
        np.array([0.2 * B]), omega=1.0e-9)) == 0.0)


def test_an_isotropic_deck_comes_back_as_an_elastic_model(tmp_path):
    m = read_isotropic_deck("tests/data/prem.nocrust")
    assert isinstance(m, ElasticModel)
    sample = m.sample(AngularGrid.gauss_legendre(2), drmax=3.0e5)
    path = tmp_path / "nocrust.nc"
    write(m, sample, path)
    with Dataset(str(path)) as ds:
        assert ds.model_class == "ElasticModel"
        assert "rheology" not in ds.groups      # no frequency-dependent field
        v = ds.groups["fields"].variables["elastic_moduli"]
        assert v.getncattr("symmetry") == "ISOTROPIC"
        assert json.loads(v.getncattr("components")) == ["kappa", "mu"]
    back, _ = read(path)
    assert isinstance(back, ElasticModel)
    assert back["elastic_moduli"].symmetry is Symmetry.ISOTROPIC
    assert back["elastic_moduli"].moduli_names == ("kappa", "mu")
    # A deck's layer functions are cubic splines through the knots, so an
    # element's one polynomial resamples them rather than reproducing
    # them (unlike PREM's polynomials): the tensor comes back as the same
    # tensor, to the resolution of the mesh it was sampled on.
    r = np.linspace(1.0e5, 6.3e6, 97)
    assert np.allclose(back["elastic_moduli"].evaluate(r),
                       m["elastic_moduli"].evaluate(r),
                       rtol=1e-5)


def test_a_model_class_this_build_does_not_know_is_refused_by_name(
        written, tmp_path):
    import shutil
    path = tmp_path / "typed.nc"
    shutil.copy(written[2], path)
    with Dataset(str(path), "a") as ds:
        ds.model_class = "elastic_earth"
    with pytest.raises(ValueError, match="elastic_earth"), \
            pytest.warns(UserWarning, match="scal"):
        read(path)


def test_a_law_whose_parameter_fields_are_missing_is_refused_by_name(tmp_path):
    """A file that names a law but does not carry what the law read.

    `qmu` is not sampled here, so constant Q cannot be rebuilt on any
    layer; the reader says so rather than handing back a body claiming
    less than its own `model_class` does.  (The tensor itself is rebuilt
    from its sample, so leaving the moduli components out is no longer
    a way to make the law fail.)
    """
    b = PREM()
    sample = b.sample(AngularGrid.gauss_legendre(2),
                      fields=["rho", "qkappa", "elastic_moduli"], drmax=1.0e6)
    path = tmp_path / "thin.nc"
    write(b, sample, path)
    with pytest.raises(ValueError, match="'qmu'"):
        read(path)


# ------------------------------------------------- the layout, in a fresh process

FRESH = r'''
"""gplspec's [element][node] arrays, from netCDF4 and numpy alone."""
import json
import sys

import numpy as np
from netCDF4 import Dataset

assert "planetmodel" not in sys.modules, "this process must not know planetmodel"

with Dataset(sys.argv[1]) as ds:
    assert ds.schema == "planetmodel.model/1"
    assert ds.index_base == 0

    radial = ds.groups["radial"]
    n_gll = int(radial.n_gll)
    start = radial.variables["element_start"][:]
    radius = radial.variables["radius"][:]
    rho = ds.groups["fields"].variables["rho"][:]
    nelement = start.size - 1

    # [element][node], the layout gplspec assembles on
    r = np.stack([radius[start[e]:start[e + 1]] for e in range(nelement)])
    d = np.stack([rho[start[e]:start[e + 1]] for e in range(nelement)])
    assert r.shape == (nelement, n_gll), r.shape
    assert d.shape == (nelement, n_gll), d.shape
    assert np.all(np.diff(r, axis=1) > 0.0), "nodes are not increasing"

    layer = radial.variables["element_layer"][:]
    assert layer.min() == 0, "element_layer is not 0-based"
    assert np.all(np.diff(layer) >= 0), "element_layer is not non-decreasing"

    # an element boundary is one radius held twice, one node each side
    assert np.allclose(r[1:, 0], r[:-1, -1], rtol=0.0, atol=0.0)
    same = layer[1:] == layer[:-1]
    assert np.allclose(d[1:, 0][same], d[:-1, -1][same], rtol=1e-9), (
        "rho is discontinuous inside a layer")
    jump = ~same
    assert jump.any() and np.any(
        np.abs(d[1:, 0][jump] - d[:-1, -1][jump]) > 1.0), (
        "no density jump survived at a layer boundary")

    u = ds.groups["mapping"].variables["displacement"]
    ntheta = ds.groups["angular"].variables["colatitude"].size
    nphi = ds.groups["angular"].variables["longitude"].size
    assert u.shape == (radius.size, ntheta, nphi, 3), u.shape

    names = [str(s) for s in ds.groups["skeleton"].variables["interface_name"][:]]
    surface = ds.groups["surfaces"].groups["surface"]
    assert surface.interface == names.index("surface"), (
        "the surface group points at the wrong interface")
    assert surface.variables["relief"].shape == (ntheta, nphi)

    # the rheology, as a consumer that cannot call planetmodel reads it
    assert ds.model_class == "ViscoelasticModel"
    sub = ds.groups["rheology"].groups["viscoelastic_moduli"]
    assert sub.kind == "frequency" and sub.omega_domain == "real"
    laws = [str(s) for s in sub.variables["law"][:]]
    params = [json.loads(s) for s in sub.variables["parameters"][:]]
    consts = [json.loads(s) for s in sub.variables["constants"][:]]
    nlayer = ds.dimensions["layer"].size
    assert len(laws) == nlayer and set(laws) == {"constant_q"}
    assert params[0] == ["elastic_moduli", "qkappa", "qmu"]
    assert consts[0]["reference_period"] == ds.reference_period

    # the complex sample: two real numbers per component, real first
    dyn = ds.groups["fields"].variables["viscoelastic_moduli"]
    assert dyn.dimensions[-1] == "part" and ds.dimensions["part"].size == 2
    assert dyn.part == "complex" and dyn.omega > 0.0
    # constant Q at omega != omega_0: the shear modulus has a loss part
    mu = dyn[:, 3, 3, :]
    assert np.any(np.abs(mu[:, 1]) > 0.0), "the imaginary part is all zero"

print("ok")
'''


def test_a_fresh_process_reads_the_gplspec_layout(prem_file, tmp_path):
    script = tmp_path / "fresh.py"
    script.write_text(FRESH)
    r = subprocess.run([sys.executable, str(script), str(prem_file[2])],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "ok"


# ----------------------------------------------------------- what the file says

def test_the_file_says_the_outer_core_is_fluid(prem_file):
    with Dataset(str(prem_file[2])) as ds:
        states = [str(s) for s in ds.groups["skeleton"].variables["layer_state"][:]]
    assert states[1] == "fluid" and states[0] == "solid"
    back, _ = read(prem_file[2])
    assert back.layers[1].state == "fluid"


def test_a_frequency_field_without_a_record_is_refused_by_name(tmp_path):
    from planetmodel.model.character import ELASTIC
    from planetmodel.model.fields.frequency import ComposedFrequencyField
    m = PREM(ocean=False)
    lay = m.layers[5]
    hand = ComposedFrequencyField(lambda v, omega: v * (1 + 0.1 * omega),
                                  [lay["elastic_moduli"]], character=ELASTIC,
                                  dimensions=lay["elastic_moduli"].dimensions,
                                  name="viscoelastic_moduli")
    v = m.with_layer(5, lay.with_field("viscoelastic_moduli", hand, replace=True))
    sample = v.sample(AngularGrid.gauss_legendre(2), drmax=1.0e6)
    with pytest.raises(ValueError, match="viscoelastic_moduli.*layer 5.*LawRecord"):
        write(v, sample, tmp_path / "hand.nc")


def test_an_elastic_layer_round_trips_as_one_and_is_never_lifted(tmp_path):
    """A layer holding static moduli alone is written as `static` and read
    back holding static moduli alone; the class lifts it at view time."""
    m = PREM(ocean=False)
    lay = m.layers[5].without_field("viscoelastic_moduli")
    v = m.with_layer(5, lay)
    assert "viscoelastic_moduli" not in v.layers[5].fields
    sample = v.sample(AngularGrid.gauss_legendre(2), drmax=1.0e6, omega=OMEGA)
    path = tmp_path / "elastic_layer.nc"
    write(v, sample, path)
    with Dataset(str(path)) as ds:
        laws = [str(s) for s in ds.groups["rheology"].groups["viscoelastic_moduli"]
                .variables["law"][:]]
    assert laws[5] == "static" and laws[4] == "constant_q"
    back, _ = read(path)
    assert type(back) is ViscoelasticModel
    assert "viscoelastic_moduli" not in back.layers[5].fields
    r = np.array([5.75e6])
    got = back.viscoelastic_moduli.evaluate(r, omega=OMEGA)
    assert np.allclose(got, m.layers[5]["elastic_moduli"].evaluate(r), rtol=1e-6)
    assert np.all(np.imag(got) == 0.0)


def test_fields_the_body_cannot_take_back_are_warned_about(written):
    """The analytic field depends on direction: it stays in the sample."""
    with pytest.warns(UserWarning, match="scal"):
        back, sample = read(written[2])
    assert "scal" in sample.fields and "scal" not in back.field_names


def test_a_user_field_type_rescales_and_is_warned_about_on_read(tmp_path):
    """Extensibility: a field written against FieldBase with its own
    `rescaled` survives nondimensionalisation and is named on read."""
    from planetmodel.model.fields.composite import FieldBase

    class Tabulated(FieldBase):
        def __init__(self, skeleton, r, v, *, name=None):
            self.skeleton = skeleton
            self.character = DENSITY
            self.dimensions = Dimensions.DENSITY
            self.name = name
            self._r, self._v = np.asarray(r, float), np.asarray(v, float)

        def evaluate(self, r, theta=None, phi=None, *, layer=None,
                     side="upper", frame="spherical"):
            return np.interp(np.asarray(r, float), self._r, self._v)

        def rescaled(self, convert, old, new):
            k = old.length / new.length
            factor = (old.mass / new.mass) * k ** -3      # a density's scale
            return Tabulated(Skeleton(self.skeleton.boundaries * k), self._r * k,
                             self._v * factor,
                             name=self.name)

    m = PREM(ocean=False)
    lay = m.layers[5]
    lo, hi = lay.interval
    tab = Tabulated(Skeleton([lo, hi]), np.linspace(lo, hi, 5),
                    np.linspace(1.0e3, 2.0e3, 5), name="porosity")
    b = m.with_layer(5, lay.with_field("porosity", tab))
    nd = b.nondimensionalised()
    assert type(nd.layers[5]["porosity"]) is Tabulated
    assert np.isclose(float(nd.layers[5]["porosity"](nd.layers[5].interval[0])),
                      1.0e3 / nd.scales.mass * nd.scales.length ** 3)
    sample = b.sample(AngularGrid.gauss_legendre(2), drmax=1.0e6)
    path = tmp_path / "user_field.nc"
    write(b, sample, path)
    # The type declares no radial dependence, so its sample is on the
    # angular grid too and the body cannot take it back: it says so.
    with pytest.warns(UserWarning, match="porosity"):
        back, sample = read(path)
    assert "porosity" in sample.fields and "porosity" not in back.field_names


def test_a_user_model_class_round_trips(tmp_path):
    from planetmodel.model.classes import HasDensity, ModelBase
    from planetmodel.registry import register, registered

    class HasPorosity:
        REQUIRES = ("porosity",)

        @property
        def porosity(self):
            return self["porosity"]

    if "PoroelasticModel" not in registered("model_class"):
        @register("model_class", "PoroelasticModel")
        class PoroelasticModel(HasDensity, HasPorosity, ModelBase):
            ASPECTS = (HasDensity, HasPorosity)
    else:
        from planetmodel.registry import lookup
        PoroelasticModel = lookup("model_class", "PoroelasticModel")

    m = PREM(ocean=False)
    b = ReferenceBody(m.layers)
    b.add_field("porosity", RadialField(m.skeleton, [lambda r: 0 * r + 0.1] * 12,
                                        name="porosity", character=DENSITY,
                                        dimensions=Dimensions.DIMENSIONLESS))
    pm = b.as_class(PoroelasticModel)
    sample = pm.sample(AngularGrid.gauss_legendre(2), drmax=1.0e6)
    path = tmp_path / "poro.nc"
    write(pm, sample, path)
    back, _ = read(path)
    assert type(back).__name__ == "PoroelasticModel"
    assert "porosity" in back.field_names


@pytest.mark.netcdf
def test_a_body_holding_the_tensor_alone_comes_back_holding_it(tmp_path):
    """A hand-built ElasticModel with no kappa or mu fields round-trips."""
    from planetmodel import (DENSITY, AngularGrid, ElasticField, ElasticModel,
                             Layer, ReferenceBody, Skeleton, Symmetry,
                             constant_field)
    from planetmodel.model.units import Dimensions
    sk = Skeleton([0.0, 0.5, 1.0])
    layers = []
    for i in range(2):
        one = Skeleton(sk.interval(i))
        rho = constant_field(one, 3000.0 + 1000.0 * i, name="rho",
                             character=DENSITY, dimensions=Dimensions.DENSITY)
        kappa = constant_field(one, 1.0e11 + 1.0e10 * i, dimensions=Dimensions.MODULUS)
        mu = constant_field(one, 4.0e10 + 1.0e10 * i, dimensions=Dimensions.MODULUS)
        el = ElasticField(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu},
                          name="elastic_moduli")
        layers.append(Layer(index=i, interval=sk.interval(i),
                            fields={"rho": rho, "elastic_moduli": el}))
    m = ReferenceBody(layers).as_class(ElasticModel)
    assert "kappa" not in m and "mu" not in m
    s = m.sample(AngularGrid.gauss_legendre(2), drmax=0.1)
    path = tmp_path / "tensor_alone.nc"
    write(m, s, path)
    back = read(path)
    back = back[0] if isinstance(back, tuple) else back
    assert type(back) is ElasticModel
    assert "kappa" not in back and "mu" not in back
    r = np.array([0.25, 0.75])
    assert np.allclose(back["elastic_moduli"].evaluate(r),
                       m["elastic_moduli"].evaluate(r), rtol=1e-10)
