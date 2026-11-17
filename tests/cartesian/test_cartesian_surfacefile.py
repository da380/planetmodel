"""The flat surface file: fields on a horizontal plane of a flat body,
written, validated and read back.

Fields of every time support round-trip on a box's grid and a rectangle's,
padded, NaN kept, held to the file by `check_surface_file`; the history
serves them at dates by the surface file's own time rules, and as flat
gridded functions by their lateral rules, which `cartesian.slices.surface`
shows at their own nodes; what a flat file cannot hold is refused on
writing, and a damaged one on reading, by name.
"""
import numpy as np
import pytest

from planetmodel import Scales, gauss_legendre
from planetmodel import cartesian as flat
from planetmodel.surfacefile import SurfaceVariable
from planetmodel.testing import check_surface_file
from planetmodel.units import LENGTH, Dimensions

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

from planetmodel.cartesian.surfacefile import (SurfaceHistory,  # noqa: E402
                                               read_surface_file, write_surface)
from planetmodel.surfacefile import (SCHEMA_FLAT, read_surface,  # noqa: E402
                                     validate_structure)

BOX = flat.Box([(0.0, 400e3), (0.0, 300e3)], padding=(50e3, 25e3))
RECT = flat.Rectangle((-100e3, 100e3), padding=20e3)


def topography(grid):
    x, y = grid.nodes
    lx, ly = (grid.lengths + (1.0,))[:2]
    return np.broadcast_to(1e3 * np.cos(2 * np.pi * x / lx)
                           * np.cos(2 * np.pi * y / ly) + 200.0, grid.shape).copy()


def case(grid):
    """Fields of every support on `grid`, one with a NaN."""
    rng = np.random.default_rng(7)
    shape = grid.shape
    topo = topography(grid)
    topo.flat[0] = np.nan
    return (np.array([0.0, 1.0, 2.5, 4.0]), np.array([2.5, 4.0]), {
        "topography": SurfaceVariable(topo, dimensions=LENGTH, meaning="topography"),
        "ice_thickness": SurfaceVariable(
            np.stack([k * topography(grid) for k in range(4)]), support="node",
            time_interpolation="linear", dimensions=LENGTH,
            lateral_interpolation="fourier"),
        "rate": SurfaceVariable(rng.normal(size=(3,) + shape), support="interval"),
        "seen": SurfaceVariable(rng.normal(size=(2,) + shape), support="instant",
                                lateral_interpolation="cubic"),
    })


@pytest.mark.parametrize("grid", [flat.Grid(BOX, (16, 12)), flat.Grid(RECT, 20)],
                         ids=repr)
def test_every_support_comes_back(tmp_path, grid):
    time, instant, fields = case(grid)
    scales = Scales.geophysical(1e5)
    path = write_surface(tmp_path / "s.nc", grid, fields=fields, source="tests",
                         time=time, instant=instant, depth=0.0, scales=scales,
                         title="a plane", meta={"epoch": "present"},
                         citation=("one work",))
    check_surface_file(path, grid, fields=fields, time=time, instant=instant,
                       depth=0.0, scales=scales, meta={"epoch": "present"},
                       citation=("one work",))
    data = read_surface_file(path)
    assert data.schema == SCHEMA_FLAT and data.grid == grid and data.depth == 0.0
    assert data.grid.domain.bounds == grid.domain.bounds
    assert data.grid.domain.padding == grid.domain.padding
    with netCDF4.Dataset(path) as ds:
        assert ds.getncattr("coordinate_system") == "flat"
        assert ds["topography"].dimensions == grid_dims(grid)
        assert np.array_equal(ds["x"][:], grid.x)
        for name in ("latitude", "grid_kind", "radius", "pole"):
            assert name not in ds.variables and name not in ds.ncattrs()


def grid_dims(grid):
    return ("x", "y") if grid.dimension == 2 else ("x",)


def test_a_plane_below_the_surface_and_none(tmp_path):
    grid = flat.Grid(BOX, 4)
    fields = {"k": SurfaceVariable(np.ones(grid.shape))}
    path = write_surface(tmp_path / "deep.nc", grid, fields=fields, source="tests",
                         depth=35e3)
    check_surface_file(path, grid, fields=fields, depth=35e3)
    path = write_surface(tmp_path / "top.nc", grid, fields=fields, source="tests")
    check_surface_file(path, grid, fields=fields)
    assert read_surface_file(path).depth is None


def test_the_history_serves_dates_and_flat_functions(tmp_path):
    """The shared time rules serve the flat fields; a field at a date is a
    flat gridded function by its rule, which the surface slice shows at
    its own nodes."""
    from planetmodel.cartesian import slices
    grid = flat.Grid(BOX, (16, 12))
    time, instant, fields = case(grid)
    path = write_surface(tmp_path / "h.nc", grid, fields=fields, source="tests",
                         time=time, instant=instant)
    history = read_surface(path)
    assert isinstance(history, SurfaceHistory) and history.grid == grid
    assert set(history.names()) == set(fields)
    assert history.dimensions("ice_thickness") == Dimensions(length=1)
    base = topography(grid)
    # between the nodes at 1 and 2.5, holding 1 and 2 times the base
    assert np.allclose(history.values("ice_thickness", date=1.75), 1.5 * base,
                       rtol=1e-14)
    assert np.allclose(history.values("ice_thickness", date=9.0, extrapolation="hold"),
                       3.0 * base)
    with pytest.raises(ValueError, match="outside"):
        history.values("ice_thickness", date=9.0)
    with pytest.raises(ValueError, match="stored instants alone"):
        history.values("seen", date=3.0)
    fn = history.function("ice_thickness", date=2.5)
    assert isinstance(fn, flat.GriddedFunction) and fn.lateral == "fourier"
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-1e6, 1e6, 50), rng.uniform(-1e6, 1e6, 50)
    lx, ly = grid.lengths
    want = 2.0 * (1e3 * np.cos(2 * np.pi * x / lx) * np.cos(2 * np.pi * y / ly) + 200.0)
    assert np.abs(fn(x, y) - want).max() < 1e-9 * np.abs(want).max()
    with pytest.raises(ValueError, match="finite"):
        history.function("topography")       # its NaN
    model = flat.CrustOverMantle(BOX)
    shown = slices.surface(model, fn, "surface", name="ice_thickness")
    assert shown.grid == grid and np.array_equal(shown.values, fn.values)


def test_writing_refuses_what_a_flat_file_cannot_hold(tmp_path):
    grid = flat.Grid(BOX, 4)
    ones = np.ones(grid.shape)
    path = tmp_path / "refused.nc"

    def refuse(error, match, fields, **kw):
        with pytest.raises(error, match=match):
            write_surface(path, kw.pop("grid", grid), fields=fields, source="tests",
                          **kw)

    refuse(ValueError, "a flat grid has no poles",
           {"a": SurfaceVariable(ones, poles=[0.0, 0.0])})
    refuse(ValueError, "lateral_interpolation 'harmonic'",
           {"a": SurfaceVariable(ones, lateral_interpolation="harmonic")})
    refuse(ValueError, "keeps for itself", {"x": SurfaceVariable(ones)})
    refuse(ValueError, "depth must be finite and not negative",
           {"a": SurfaceVariable(ones)}, depth=-1.0)
    refuse(ValueError, "has shape", {"a": SurfaceVariable(np.ones((3, 3)))})
    refuse(TypeError, "expected a cartesian.Grid", {"a": SurfaceVariable(ones)},
           grid=gauss_legendre(2))
    assert not path.exists()


@pytest.mark.parametrize("edit, message", [
    (lambda ds: ds.delncattr("coordinate_system"), "coordinate_system is None"),
    (lambda ds: ds.setncattr("domain", "box"), "a box states 4 bounds"),
    (lambda ds: ds.setncattr("domain_padding", np.array([5.0, 1.0])),
     "a rectangle states 2 bounds and 1 padding"),
    (lambda ds: ds["x"].__setitem__(2, 0.0), "x is not the nodes"),
    (lambda ds: ds.setncattr("grid_kind", "custom"), "grid_kind is a spherical"),
    (lambda ds: ds.setncattr("radius", 1.0), "radius is a spherical"),
    (lambda ds: ds.setncattr("depth", -5.0), "depth is .*, not a depth"),
    (lambda ds: ds["topography"].setncattr("lateral_interpolation", "harmonic"),
     "no known lateral rule"),
    (lambda ds: ds.setncattr("schema", "planetmodel.surface/1"), "grid_kind"),
])
def test_a_damaged_flat_file_is_refused_by_name(tmp_path, edit, message):
    grid = flat.Grid(RECT, 10)
    time, instant, fields = case(grid)
    good = write_surface(tmp_path / "good.nc", grid, fields=fields, source="tests",
                         time=time, instant=instant)
    validate_structure(good)
    bad = tmp_path / "bad.nc"
    bad.write_bytes(good.read_bytes())
    with netCDF4.Dataset(bad, "a") as ds:
        edit(ds)
    with pytest.raises(ValueError, match=message):
        check_surface_file(bad, grid, fields=fields, time=time, instant=instant)
