"""The model file: written, validated, read back.

PREM and a deck model come back as the same polynomials; a perturbed
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

import numpy as np
import pytest

from planetmodel import (PREM, AnalyticField, GriddedField, GriddedFunction,
                         HarmonicField, LayeredIsotropicElastic, MineosModel,
                         PerturbedField, Scales, driscoll_healy, equiangular,
                         flattening, gauss_legendre, layer_linear, perturbed,
                         unperturbed)
from planetmodel.character import DENSITY, SCALAR, VECTOR
from planetmodel.harmonics import packing
from planetmodel.testing import check_geometry, check_model_file
from planetmodel.units import Dimensions
from planetmodel.vocabulary import FieldSpec

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

from planetmodel.modelfile import (SCHEMA, read_model, read_model_file,  # noqa: E402
                                   read_surfaces, validate_structure, write_model)

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


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
    m = MineosModel(DATA / "prem.200")
    path = write_model(tmp_path / "deck.nc", m, grid=gauss_legendre(2))
    check_model_file(path, m)
    same_radial_fields(m, read_model(path), np.random.default_rng(2))


LO, HI = 3480e3, 5701e3


def dlnvs(r, t, p):
    return 0.02 * np.sin(t) ** 2 * np.cos(2 * p) * (r - LO) / (HI - LO)


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


@pytest.mark.parametrize("edit, message", [
    (lambda ds: ds.setncattr("schema", "planetmodel.model/0"), "schema is"),
    (lambda ds: ds.setncattr("rtol", -1.0), "rtol is"),
    (lambda ds: ds.setncattr("rtol", "loose"), "rtol is"),
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
])
def test_a_malformed_file_is_refused_by_name(tmp_path, edit, message):
    path = write_model(tmp_path / "m.nc", three_d_model(), grid=gauss_legendre(6))
    validate_structure(path)
    corrupt(path, edit)
    with pytest.raises(ValueError, match=message):
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
    deck = MineosModel(DATA / "prem.200")
    path = write_model(tmp_path / "d.nc", deck, grid=gauss_legendre(0),
                       fields=["rho"])
    back = read_model(path)
    assert back.constant("omega_ref") == pytest.approx(deck.constant("omega_ref"))
    dims = back.constants["omega_ref"].dimensions
    assert dims == deck.constants["omega_ref"].dimensions
    assert read_model(path, scales=Scales.geophysical(6371e3)).constant(
        "omega_ref") == pytest.approx(deck.nondimensionalised().constant("omega_ref"))


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

    model = LayeredIsotropicElastic([0.0, 1.0], rho=[3.0], vp=[2.0], vs=[1.0])
    path = write_model(tmp_path / "stamped.nc", model, grid=gauss_legendre(4))
    data = read_model_file(path)
    assert data.source.startswith("planetmodel ")
    datetime.fromisoformat(data.created)
    path = write_model(tmp_path / "named.nc", model, grid=gauss_legendre(4),
                       source="my code 1.0")
    assert read_model_file(path).source == "my code 1.0"
