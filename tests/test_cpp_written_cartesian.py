"""The flat files the C++ writers wrote, read by the Python: each flat
model file validates, reads as a `cartesian.Model` with its fields on
interfaces, and holds the values the C++ reader reads from it, the
vertical stretch among them; a viscosity written in its logarithm reads
as its closed form; a flat fixture the C++ read and wrote back reads as
the fixture does; the flat surface files hold the fields they were
written from; and the decks of depths are, character for character, the
Python's own writing of them.

The C++ tests leave the files in their build tree; PLANETMODEL_CPP_WRITTEN
names that directory (`build/cpp/tests/written` for the build of
CLAUDE.md), and without it these tests are skipped.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from planetmodel import cartesian
from planetmodel.cartesian import depth_names
from planetmodel.deck import Deck, prony_branches, read_deck, write_deck

WRITTEN = os.environ.get("PLANETMODEL_CPP_WRITTEN")
FIXTURES = Path(__file__).resolve().parents[1] / "cpp" / "tests" / "fixtures"
pytestmark = pytest.mark.skipif(
    WRITTEN is None,
    reason="PLANETMODEL_CPP_WRITTEN names no directory of C++-written files")


def _written(pattern: str) -> list[Path]:
    return [] if WRITTEN is None else sorted(Path(WRITTEN).glob(pattern))


def _check_expect(model, expect: Path, *, rtol: float, surfaces=None) -> int:
    """The lines of a flat expectation file met by a model and the
    functions on its interfaces: values at (depth, x, y) in either frame,
    the stretch's displacement, and the fields on interfaces."""
    checked = 0
    base = float(model.skeleton.boundaries[-1])
    for line in expect.read_text().splitlines():
        words = line.split()
        if words[0] == "value":
            layer, name, frame = int(words[1]), words[2], words[3]
            d, x, y = (float(v) for v in words[4:7])
            want = np.array([float(v) for v in words[7:]])
            got = np.asarray(model.layer(layer)[name].evaluate(
                d, x, y, frame="local" if frame == "l" else "cartesian"),
                dtype=float).ravel()
            scale = max(1.0, float(np.max(np.abs(want))))
            assert np.allclose(got, want, rtol=0, atol=rtol * scale), line
            checked += 1
        elif words[0] == "displacement":
            X = np.array([float(v) for v in words[1:4]])
            want = np.array([float(v) for v in words[4:7]])
            got = np.asarray(model.geometry.mapping.displacement(X[None, :]),
                             dtype=float)[0]
            assert np.allclose(got, want, rtol=0, atol=1e-12 * base), line
            checked += 1
        elif words[0] in ("surface", "surface_gradient", "surface_absent"):
            k, name = int(words[1]), words[2]
            face = model.geometry.interface(k)
            on = surfaces.get(name, {})
            key = k if face.name is None else face.name
            if words[0] == "surface_absent":
                assert key not in on, line
                continue
            x, y = float(words[3]), float(words[4])
            want = np.array([float(v) for v in words[5:]])
            got = (np.atleast_1d(on[key](x, y)) if words[0] == "surface"
                   else np.array([float(v) for v in on[key].lateral_gradient(x, y)]))
            scale = max(1.0, float(np.max(np.abs(want))))
            tol = rtol if words[0] == "surface" else 1e-9
            assert np.allclose(got, want, rtol=0, atol=tol * scale), line
            checked += 1
    return checked


# -- model files -----------------------------------------------------------------

@pytest.mark.netcdf
def test_the_cpp_wrote_its_flat_files():
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    names = {p.name for p in _written("flat_*")}
    assert {"flat_written_box.nc", "flat_written_section.nc", "flat_written_log.nc",
            "flat_roundtrip_model_layered.nc", "flat_roundtrip_model_lateral.nc",
            "flat_roundtrip_model_section.nc", "flat_roundtrip_model_fourier.nc",
            "flat_roundtrip_model_stacked.nc", "flat_roundtrip_model_stretched.nc",
            "flat_roundtrip_model_surfaces.nc", "flat_roundtrip_model_held.nc",
            "flat_surface_written.nc", "flat_surface_written_section.nc",
            "flat_deck_elastic.txt", "flat_deck_maxwell_built.txt",
            "flat_example.nc"} <= names


@pytest.mark.netcdf
@pytest.mark.parametrize("path", _written("flat_written_*.nc"), ids=lambda p: p.name)
def test_a_flat_file_the_cpp_wrote_reads_as_the_cpp_reads_it(path):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.cartesian.modelfile import read_model_file
    from planetmodel.modelfile import (SCHEMA_FLAT, read_model, read_surfaces,
                                       schema_of, validate_structure)
    from planetmodel.modelfile import read_model_file as read_spherical
    validate_structure(path)
    assert schema_of(path) == SCHEMA_FLAT
    assert read_model_file(path).schema == SCHEMA_FLAT
    with pytest.raises(ValueError, match="cartesian.modelfile.read_model_file"):
        read_spherical(path)
    model = read_model(path)
    assert isinstance(model, cartesian.Model)
    assert _check_expect(model, path.with_suffix(".expect"), rtol=1e-10,
                         surfaces=read_surfaces(path)) > 10


@pytest.mark.netcdf
@pytest.mark.parametrize("path", _written("flat_roundtrip_*.nc"), ids=lambda p: p.name)
def test_a_flat_fixture_written_back_reads_as_the_fixture(path):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.cartesian.modelfile import read_model_file
    from planetmodel.modelfile import read_model, read_surfaces, validate_structure
    validate_structure(path)
    name = path.stem.removeprefix("flat_roundtrip_")
    fixture = FIXTURES / f"flat_{name}.nc"
    got, want = read_model_file(path), read_model_file(fixture)
    assert got.domain == want.domain
    assert got.source == want.source and got.created == want.created
    assert [p.grid for p in got.parts] == [p.grid for p in want.parts]
    for a, b in zip(got.parts, want.parts):
        assert set(a.variables) == set(b.variables)
        for key, var in b.variables.items():
            assert np.array_equal(a.variables[key].values, var.values, equal_nan=True)
    expect = fixture.with_suffix(".expect")
    if not expect.exists():
        return                              # the Fourier copy: its numbers alone
    model = read_model(path)
    assert _check_expect(model, expect, rtol=1e-11, surfaces=read_surfaces(path)) > 10


@pytest.mark.netcdf
def test_the_box_the_cpp_built_is_the_model_it_was_given():
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel import Dimensions
    from planetmodel.cartesian.modelfile import read_model_file
    from planetmodel.modelfile import read_model, read_surfaces
    path = Path(WRITTEN) / "flat_written_box.nc"
    data = read_model_file(path)
    box = cartesian.Box([(0.0, 400e3), (0.0, 300e3)], padding=(50e3, 0.0))
    assert data.domain == box
    assert data.source == "planetmodel C++ tests"
    assert [p.grid for p in data.parts] == [cartesian.Grid(box, (12, 8)),
                                            cartesian.Grid(box, (9, 6))]
    assert data.geometry.grid == cartesian.Grid(box, (16, 12))
    model = read_model(path)
    assert model.name == "a box built in C++"
    assert model.citation == ("A. Author (2026).  A written box.",
                              "B. Author (2026).  Its source.")
    assert [lay.name for lay in model.geometry.layers] == ["crust", None,
                                                          "asthenosphere"]
    assert [f.name for f in model.geometry.interfaces] == ["surface", "moho", None]
    assert np.array_equal(model.skeleton.boundaries, [0.0, 30e3, 100e3, 200e3])
    assert model.constant("g") == 9.8
    assert model.layers_with("flow") == (1,) and model.layers_with("stress") == (0, 2)

    def wave(x, y):
        return np.sin(2 * np.pi * x / 500e3) + 0.5 * np.cos(2 * np.pi * y / 300e3 + 0.3)

    # the fields the C++ was given, at their own nodes: x, y, z from (up, x, y)
    grid = cartesian.Grid(box, (12, 8))
    x, y = grid.nodes
    d = 47e3
    flow = model.layer(1)["flow"].evaluate(d, x, y, frame="cartesian")
    want = np.stack(np.broadcast_arrays(1.0 + wave(x, y), 2e-5 * d,
                                        np.cos(2 * np.pi * x / 500e3) - 1e-5 * d), -1)
    assert np.allclose(flow, want, rtol=0, atol=1e-6)        # between depth nodes
    local = model.layer(1)["flow"].evaluate(d, x, y)
    assert np.allclose(local[..., 0], flow[..., 2]) and np.allclose(local[..., 1:],
                                                                    flow[..., :2])
    # the stretch: heights, positive up; the surface to its relief
    reliefs = cartesian.Grid(box, (16, 12))
    x, y = (np.broadcast_to(v, reliefs.shape) for v in reliefs.nodes)
    X = cartesian.FLAT.points(np.zeros(reliefs.shape), x, y)
    moved = model.geometry.mapping(X)
    assert np.allclose(moved[..., 2], 1.5e3 * wave(x, y), rtol=0, atol=1e-9)
    assert np.array_equal(moved[..., :2], X[..., :2])
    surfaces = read_surfaces(path)
    assert sorted(surfaces) == ["K_shape", "shift"]
    assert sorted(surfaces["K_shape"], key=str) == [2, "surface"]
    assert model.spec("K_shape").dimensions == Dimensions(length=-2)
    assert float(surfaces["shift"]["moho"](0.0, 0.0)) == pytest.approx(
        30.0 * wave(0.0, 0.0), abs=1e-9)


@pytest.mark.netcdf
def test_the_section_the_cpp_built_is_a_section_in_its_units():
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.cartesian import GriddedDisplacement, VerticalStretch
    from planetmodel.modelfile import read_model
    model = read_model(Path(WRITTEN) / "flat_written_section.nc")
    assert model.geometry.domain == cartesian.Rectangle((0.0, 400.0), padding=50.0)
    assert model.scales.length == 1e3 and model.geometry.rtol == 1e-8
    assert np.array_equal(model.skeleton.boundaries, [5.0, 30.0, 200.0])
    mapping = model.geometry.mapping
    assert isinstance(mapping, VerticalStretch)
    assert isinstance(mapping.w, GriddedDisplacement)
    # the upward displacement at the nodes of the file's own grid
    x = cartesian.Grid(model.geometry.domain, (14,)).x
    w = np.asarray(mapping.w(30.0, x, 0.0))
    assert np.allclose(w, 0.8 * np.sin(2 * np.pi * x / 500.0) * 170.0 / 195.0, rtol=0,
                       atol=1e-12)


@pytest.mark.netcdf
def test_a_viscosity_the_cpp_wrote_in_its_logarithm_reads_exactly():
    """The C++'s exp(46 + 8 d - 5 d^2) on its lower layer, three nodes an
    element, read by the Python as an exponential layer of depth: exact
    between the nodes, with its derivative, and the constant above it to
    the bit."""
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.layerfunction import ExponentialLayer
    from planetmodel.modelfile import SCHEMA_FLAT, read_model, schema_of
    path = Path(WRITTEN) / "flat_written_log.nc"
    assert schema_of(path) == SCHEMA_FLAT
    model = read_model(path)
    eta = model.layer(1)["viscosity"]
    assert isinstance(eta, cartesian.DepthField)
    assert isinstance(eta.function, ExponentialLayer)
    d = np.linspace(0.3, 1.0, 101)
    want = np.exp(46.0 + 8.0 * d - 5.0 * d ** 2)
    assert np.allclose(eta(d), want, rtol=1e-13, atol=0)
    assert np.allclose(eta.derivative()(d), want * (8.0 - 10.0 * d), rtol=1e-12,
                       atol=1e-12 * 8.0 * want.max())
    above = model.layer(0)["viscosity"]
    d = np.linspace(0.0, 0.3, 13)
    assert np.array_equal(above(d), np.full(d.size, 2e21))


@pytest.mark.netcdf
def test_the_flat_example_reads_as_a_box_under_topography():
    """The file the example `write_flat_model` wrote: a crust over a mantle
    in a box, the mantle's viscosity lateral, the surface raised to a ridge
    with a root under it at the Moho."""
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.modelfile import read_model, validate_structure
    path = Path(WRITTEN) / "flat_example.nc"
    validate_structure(path)
    model = read_model(path)
    assert model.name == "example box"
    assert model.geometry.domain == cartesian.Box([(0.0, 400e3), (0.0, 300e3)])
    assert model.layers_with("viscosity") == (1,)
    assert model.layers_with("rho") == (0, 1)
    assert float(model.layer(1)["rho"].evaluate(130e3, 0.0, 0.0)) == pytest.approx(
        3350.0)
    x = np.linspace(0.0, 400e3, 9)
    ridge = 2e3 * np.cos(2 * np.pi * x / 400e3)
    for depth, height in ((0.0, ridge), (30e3, -5.0 * ridge), (200e3, 0.0 * ridge)):
        X = cartesian.FLAT.points(np.full(x.size, depth), x, 100e3)
        assert np.allclose(model.geometry.mapping(X)[:, 2] + depth, height, rtol=0,
                           atol=1e-8)


# -- surface files ---------------------------------------------------------------

@pytest.mark.netcdf
@pytest.mark.parametrize("box", [True, False], ids=["box", "section"])
def test_the_flat_surface_file_the_cpp_wrote_holds_what_it_was_given(box):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel import Scales
    from planetmodel.cartesian.surfacefile import read_surface_file
    from planetmodel.surfacefile import SCHEMA_FLAT, SurfaceVariable, read_surface
    from planetmodel.surfacefile import read_surface_file as read_spherical
    from planetmodel.testing import check_surface_file
    from planetmodel.units import LENGTH

    path = Path(WRITTEN) / ("flat_surface_written.nc" if box
                            else "flat_surface_written_section.nc")
    grid = (cartesian.Grid(cartesian.Box([(0.0, 4.0), (-1.0, 2.0)], padding=(0.5, 0.0)),
                           (6, 4)) if box
            else cartesian.Grid(cartesian.Rectangle((0.0, 4.0), padding=0.5), (6,)))
    assert read_surface_file(path).schema == SCHEMA_FLAT
    with pytest.raises(ValueError, match="cartesian.surfacefile.read_surface_file"):
        read_spherical(path)
    k = np.arange(3).astype(float).reshape((3,) + (1,) * grid.dimension)
    i = np.arange(6).astype(float).reshape((6,) + (1,) * (grid.dimension - 1))
    j = np.arange(4).astype(float)[None, :] if box else 0.0
    topography = np.broadcast_to(i + 0.01 * j, grid.shape).copy()
    topography.flat[1] = np.nan
    fields = {
        "topography": SurfaceVariable(topography, dimensions=LENGTH,
                                      meaning="present-day topography"),
        "ice_thickness": SurfaceVariable(
            np.broadcast_to(100.0 * k + i + 0.01 * j, (3,) + grid.shape),
            support="node", time_interpolation="linear", dimensions=LENGTH),
        "rate": SurfaceVariable(
            np.broadcast_to(10.0 * k + i - 0.1 * j, (3,) + grid.shape)[:2],
            support="interval"),
        "instantaneous": SurfaceVariable(
            np.broadcast_to(1.0 * k + 2.0 * i + 0.2 * j, (3,) + grid.shape)[:2],
            support="instant", lateral_interpolation="cubic"),
    }
    check_surface_file(path, grid, fields=fields, time=[0.0, 1.0, 2.5],
                       instant=[1.5, 2.5], depth=1.25 if box else None,
                       scales=Scales(length=2.0, mass=3.0, time=5.0),
                       meta={"epoch": "present", "count": 4}, citation=("one", "two"))
    # the history, and a field at a date as a function of (x, y)
    history = read_surface(path)
    assert isinstance(history, cartesian.surfacefile.SurfaceHistory)
    f = history.function("ice_thickness", date=0.5)
    assert np.allclose(f(grid.x[2], 0.0 if not box else grid.y[1]),
                       50.0 + 2.0 + (0.01 if box else 0.0))


# -- decks of depths --------------------------------------------------------------

#: Each flat fixture the C++ writes back, and its format.
DECKS = {"elastic": cartesian.ELASTIC, "isotropic": cartesian.ELASTIC,
         "maxwell": cartesian.MAXWELL, "prony": cartesian.PRONY,
         "burgers": cartesian.BURGERS}


@pytest.mark.parametrize("label", sorted(DECKS))
def test_a_deck_of_depths_written_back_by_the_cpp_is_the_pythons_writing(label,
                                                                        tmp_path):
    written = Path(WRITTEN) / f"flat_deck_{label}.txt"
    assert written.exists(), f"the C++ tests wrote no {written.name}"
    fmt = DECKS[label]
    ours = write_deck(tmp_path / written.name,
                      read_deck(FIXTURES / f"flat_deck_{label}.txt", fmt), fmt)
    assert written.read_text() == ours.read_text()


#: The knots and elastic columns of the boxes the C++ builds.
_DEPTH = [0.0, 15e3, 30e3, 30e3, 100e3, 200e3]
_ELASTIC = {"rho": [2.7e3, 2.8e3, 2.9e3, 3.3e3, 3.35e3, 3.4e3],
            "vpv": [6e3, 6.2e3, 6.4e3, 8e3, 8.1e3, 8.2e3],
            "vsv": [3.5e3, 3.6e3, 3.7e3, 4.5e3, 4.55e3, 4.6e3],
            "vph": [6.1e3, 6.3e3, 6.5e3, 8.1e3, 8.2e3, 8.3e3],
            "vsh": [3.6e3, 3.7e3, 3.8e3, 4.6e3, 4.65e3, 4.7e3],
            "eta": [1.0, 1.0, 1.0, 0.95, 0.97, 1.0]}
_NAMED = {"layers": ("crust", None), "interfaces": ("surface", "moho")}


def _built(name: str, columns: dict, header: dict, fmt, tmp_path) -> Deck:
    """A deck the C++ built, held to the Python's writing of the same deck,
    and read back."""
    written = Path(WRITTEN) / f"flat_deck_{name}_built.txt"
    assert written.exists(), f"the C++ tests wrote no {written.name}"
    ours = write_deck(tmp_path / written.name, Deck(_DEPTH, columns, header=header),
                      fmt)
    assert written.read_text() == ours.read_text()
    return read_deck(written, fmt)


def test_the_maxwell_deck_of_depths_the_cpp_built_is_the_pythons(tmp_path):
    nan = np.nan
    back = _built("maxwell", {**_ELASTIC,
                              "viscosity": [nan, nan, nan, 1e23, 1e22, 1e21]},
                  {"name": "a box built in C++", **_NAMED}, cartesian.MAXWELL,
                  tmp_path)
    assert back.header["columns"] == ("rho", "vpv", "vsv", "vph", "vsh", "eta",
                                      "viscosity")
    assert depth_names(back) == (["crust", None], ["surface", "moho"])
    model = cartesian.MaxwellDeckModel(back, cartesian.Box([(0.0, 1e5), (0.0, 1e5)]))
    assert model.layers_with("viscosity") == (1,)
    assert float(model.layer(1)["viscosity"](30e3)) == pytest.approx(1e23)


def test_the_prony_deck_of_depths_the_cpp_built_completes_alike(tmp_path):
    nan = np.nan
    back = _built("prony", {
        **_ELASTIC,
        "relaxation_strength_1": [nan, nan, nan, 1.0, 1.0, 1.0],
        "relaxation_time_1": [nan, nan, nan, 3e11, 2e11, 1e11],
        "relaxation_strength_2": [nan] * 6, "relaxation_time_2": [nan] * 6},
        {"name": "a box built in C++", **_NAMED}, cartesian.PRONY, tmp_path)
    assert prony_branches(back) == 2
    assert list(back["relaxation_strength_2"][3:]) == [0.0, 0.0, 0.0]
    assert list(back["relaxation_time_2"][3:]) == [3e11, 2e11, 1e11]


def test_the_burgers_and_elastic_decks_of_depths_the_cpp_built_are_the_pythons(
        tmp_path):
    nan = np.nan
    back = _built("burgers", {
        **_ELASTIC, "viscosity": [nan, nan, nan, 1e22, 3e21, 1e21],
        "kelvin_viscosity": [nan, nan, nan, 1e20, 5e19, 2e19],
        "kelvin_modulus_fraction": [nan, nan, nan, 0.5, 1.0, 1.5]},
        {"name": "a Burgers box built in C++", "layers": _NAMED["layers"]},
        cartesian.BURGERS, tmp_path)
    assert depth_names(back) == (["crust", None], ["surface", None])
    back = _built("elastic", {
        "rho": _ELASTIC["rho"], "vp": _ELASTIC["vpv"], "vs": _ELASTIC["vsv"],
        "qkappa": [nan, nan, nan, 5e4, 5e4, 5e4],
        "qmu": [600.0, 600.0, 600.0, 300.0, 250.0, 200.0]},
        {"name": "an elastic box built in C++", **_NAMED}, cartesian.ELASTIC,
        tmp_path)
    assert back.names == ("rho", "vp", "vs", "qkappa", "qmu")
    assert np.isnan(back["qkappa"][0]) and back["qmu"][5] == 200.0
