"""The surface file: written, validated, read back.

Fields of every time support round-trip on a weighted grid and on one
without its poles, NaN and polar values preserved, held to the file by
`check_surface_file`; xarray opens the file as CF tools do; what the
format cannot hold is refused on writing, and a malformed file is
refused on reading naming what is wrong.
"""
from pathlib import Path

import numpy as np
import pytest

from planetmodel import AngularGrid, Scales, equiangular, gauss_legendre
from planetmodel.surfacefile import (SCHEMA, SurfaceVariable, read_surface,
                                     read_surface_file, validate_structure,
                                     write_surface)
from planetmodel.testing import check_surface_file
from planetmodel.units import LENGTH, Dimensions

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")


def _case(grid, *, poles: bool):
    """Fields of every support on `grid`, with a NaN and, where asked,
    polar values."""
    rng = np.random.default_rng(7)
    time = np.array([0.0, 1.0, 2.5, 4.0])
    instant = np.array([2.5, 4.0])
    shape = (grid.ntheta, grid.nphi)
    topo = rng.normal(size=shape)
    topo[0, 0] = np.nan
    fields = {
        "topography": SurfaceVariable(topo, dimensions=LENGTH,
                                      meaning="present-day topography",
                                      poles=[1.0, -2.0] if poles else None),
        "ice_thickness": SurfaceVariable(
            rng.normal(size=(4,) + shape), support="node",
            time_interpolation="linear", dimensions=LENGTH,
            poles=np.zeros((4, 2)) if poles else None),
        "rate": SurfaceVariable(rng.normal(size=(3,) + shape),
                                support="interval"),
        "instantaneous": SurfaceVariable(rng.normal(size=(2,) + shape),
                                         support="instant",
                                         lateral_interpolation="cubic"),
    }
    return time, instant, fields


def test_round_trip_every_support(tmp_path: Path):
    grid = gauss_legendre(8)
    time, instant, fields = _case(grid, poles=True)
    scales = Scales.geophysical(6.371e6)
    out = write_surface(tmp_path / "surface.nc", grid, fields=fields,
                        source="planetmodel tests", time=time, instant=instant,
                        radius=1.0, scales=scales, title="a test surface",
                        meta={"epoch": "present", "lmax": 8},
                        citation=("one work", "another"))
    check_surface_file(out, grid, fields=fields, time=time, instant=instant,
                       radius=1.0, scales=scales,
                       meta={"epoch": "present", "lmax": 8},
                       citation=("one work", "another"))
    data = read_surface_file(out)
    assert data.schema == SCHEMA and data.title == "a test surface"
    assert data.source == "planetmodel tests" and data.created
    assert data.radius == 1.0 and data.scales == scales


def test_round_trip_without_time_or_radius(tmp_path: Path):
    grid = equiangular(9, 16)
    fields = {"kernel": SurfaceVariable(np.ones((grid.ntheta, grid.nphi)))}
    out = write_surface(tmp_path / "static.nc", grid, fields=fields,
                        source="planetmodel tests")
    check_surface_file(out, grid, fields=fields)
    data = read_surface_file(out)
    assert data.time is None and data.instant is None and data.radius is None
    assert data.scales == Scales.SI and data.meta == {}


def test_xarray_opens_the_file(tmp_path: Path):
    xr = pytest.importorskip("xarray")
    grid = gauss_legendre(4)
    time, instant, fields = _case(grid, poles=False)
    out = write_surface(tmp_path / "cf.nc", grid, fields=fields,
                        source="planetmodel tests", time=time, instant=instant)
    with xr.open_dataset(out) as ds:
        assert ds["ice_thickness"].dims == ("time", "latitude", "longitude")
        assert ds["rate"].dims == ("interval", "latitude", "longitude")
        assert float(ds["time"][-1]) == time[-1]


def test_refusals_on_writing(tmp_path: Path):
    grid = gauss_legendre(4)
    shape = (grid.ntheta, grid.nphi)
    ones = np.ones(shape)
    path = tmp_path / "refused.nc"

    def refuse(match, fields, **kw):
        with pytest.raises((TypeError, ValueError), match=match):
            write_surface(path, grid, fields=fields,
                          source=kw.pop("source", "tests"), **kw)

    refuse("at least one", {})
    refuse("keeps for itself", {"latitude": SurfaceVariable(ones)})
    refuse("keeps for itself", {"a_poles": SurfaceVariable(ones)})
    refuse("time_support", {"a": SurfaceVariable(ones, support="sometimes")})
    refuse("time_interpolation",
           {"a": SurfaceVariable(ones, time_interpolation="spline")})
    refuse("is not sampled between times",
           {"a": SurfaceVariable(ones, time_interpolation="linear")})
    refuse("lateral_interpolation",
           {"a": SurfaceVariable(ones, lateral_interpolation="nearest")})
    refuse("no time is given", {"a": SurfaceVariable(ones, support="node")})
    refuse("one time makes none",
           {"a": SurfaceVariable(ones, support="interval")}, time=[1.0])
    refuse("no instant is given", {"a": SurfaceVariable(ones, support="instant")})
    refuse("has shape", {"a": SurfaceVariable(np.ones((2,) + shape))})
    refuse("is complex", {"a": SurfaceVariable(ones + 0j)})
    refuse("strictly increasing", {"a": SurfaceVariable(ones)}, time=[1.0, 1.0])
    refuse("radius must be positive", {"a": SurfaceVariable(ones)}, radius=-1.0)
    refuse("source names the writing code", {"a": SurfaceVariable(ones)},
           source="  ")
    refuse("a netCDF attribute", {"a": SurfaceVariable(ones)},
           meta={"flag": True})
    refuse("not one string", {"a": SurfaceVariable(ones)}, citation="Smith 2020")
    refuse("is one line", {"a": SurfaceVariable(ones)},
           citation=("Smith\n2020",))
    custom = AngularGrid(grid.colatitudes, grid.longitudes)
    with pytest.raises(ValueError, match="'a': the harmonic rule needs a grid"):
        write_surface(path, custom, source="tests", fields={"a": SurfaceVariable(
            ones, lateral_interpolation="harmonic", poles=[1.0, 1.0])})
    assert not path.exists(), "a refused write left a file behind"


def test_poles_against_a_grid_with_them(tmp_path: Path):
    grid = equiangular(9, 16, poles=True)
    with pytest.raises(ValueError, match="holds both poles"):
        write_surface(tmp_path / "p.nc", grid, source="tests",
                      fields={"a": SurfaceVariable(
                          np.ones((grid.ntheta, grid.nphi)), poles=[0.0, 0.0])})


def test_a_reader_refuses_polar_values_on_a_grid_holding_both_poles(tmp_path: Path):
    """What the writer refuses to write, a reader refuses to read: polar
    values beside a field on a grid whose poles are nodes."""
    grid = equiangular(9, 16, poles=True)
    theta, _ = grid.nodes
    good = write_surface(tmp_path / "good.nc", grid, fields={
        "a": SurfaceVariable(np.cos(theta) + np.zeros(grid.shape))})
    validate_structure(good)
    bad = tmp_path / "bad.nc"
    bad.write_bytes(good.read_bytes())
    with netCDF4.Dataset(bad, "a") as ds:
        ds.createDimension("pole", 2)
        ds.createVariable("a_poles", "f8", ("pole",))[:] = [1.0, -1.0]
    for read in (validate_structure, read_surface_file, read_surface):
        with pytest.raises(ValueError, match="malformed surface file .*a_poles: 'a' "
                                             "gives polar values, and the grid "
                                             "holds both poles"):
            read(bad)


def test_a_writer_stamps_this_library_as_its_source_unless_given_one(tmp_path: Path):
    """`source` is the model file's: this library and its version unless
    given, and refused where it is not text."""
    import planetmodel

    grid = gauss_legendre(4)
    fields = {"a": SurfaceVariable(np.ones(grid.shape))}
    path = write_surface(tmp_path / "stamped.nc", grid, fields=fields)
    assert read_surface_file(path).source == f"planetmodel {planetmodel.__version__}"
    path = write_surface(tmp_path / "named.nc", grid, fields=fields,
                         source="my code 1.0")
    assert read_surface_file(path).source == "my code 1.0"
    for source in (3, ""):
        with pytest.raises(ValueError, match="source names the writing code"):
            write_surface(tmp_path / "refused.nc", grid, fields=fields, source=source)
    assert not (tmp_path / "refused.nc").exists()


def test_malformed_files_are_refused(tmp_path: Path):
    grid = gauss_legendre(4)
    time, instant, fields = _case(grid, poles=False)
    out = write_surface(tmp_path / "good.nc", grid, fields=fields,
                        source="planetmodel tests", time=time, instant=instant)
    validate_structure(out)

    def corrupt(what, match):
        bad = tmp_path / "bad.nc"
        bad.write_bytes(out.read_bytes())
        with netCDF4.Dataset(bad, "a") as ds:
            what(ds)
        with pytest.raises(ValueError, match=match):
            read_surface_file(bad)

    corrupt(lambda ds: ds.setncattr("schema", "planetmodel.model/1"), "schema")
    corrupt(lambda ds: ds.delncattr("source"), "source")
    corrupt(lambda ds: ds.delncattr("created"), "created")
    corrupt(lambda ds: ds.setncattr("scale_time", -1.0), "scale_time")
    corrupt(lambda ds: ds.setncattr("radius", -2.0), "radius")
    corrupt(lambda ds: ds.setncattr("grid_kind", "odd"), "grid_kind")
    corrupt(lambda ds: ds.variables["ice_thickness"].setncattr(
        "time_support", "always"), "time support")
    corrupt(lambda ds: ds.variables["rate"].setncattr(
        "time_interpolation", "linear"), "time rule")
    corrupt(lambda ds: ds.variables["topography"].setncattr(
        "lateral_interpolation", "odd"), "lateral rule")
    corrupt(lambda ds: ds.variables["time"].__setitem__(
        slice(None), [3.0, 2.0, 1.0, 0.0]), "strictly increasing")
    for value in (np.inf, "big", np.array([1.0, 2.0])):
        corrupt(lambda ds: ds.setncattr("scale_length", value),
                "malformed surface file .*scale_length is missing or")
    corrupt(lambda ds: ds.setncattr("schema", np.int32([1, 2])),
            "malformed surface file .*schema is")
    corrupt(lambda ds: ds.variables["ice_thickness"].setncattr(
        "time_support", np.int32([1, 2])), "malformed surface file .*time support")
    corrupt(lambda ds: ds.variables["ice_thickness"].setncattr(
        "time_interpolation", np.int32([1, 2])), "malformed surface file .*time rule")
    corrupt(lambda ds: ds.variables["topography"].setncattr(
        "lateral_interpolation", np.int32([1, 2])),
        "malformed surface file .*lateral rule")
    corrupt(lambda ds: ds.variables["latitude"].__setitem__(
        slice(None), ds.variables["latitude"][::-1]),
        "malformed surface file .*the grid is malformed")
    corrupt(lambda ds: ds.variables["latitude"].__setitem__(0, 90.5),
            "the grid is malformed: latitudes must lie in \\[-90, 90\\]")
    # the stamps are text, and a field's dimensions four whole numbers
    for key in ("source", "created"):
        for value in (3.0, np.int32(2026), np.array([1.0, 2.0]), " "):
            corrupt(lambda ds: ds.setncattr(key, value),
                    f"malformed surface file .*: {key} is missing or empty")
    for value in (np.int32([0, 1, 0]), np.int32([0, 1, 0, 0, 0]), np.int32(1),
                  "length", np.array([0.0, 0.5, 0.0, 0.0])):
        corrupt(lambda ds: ds.variables["topography"].setncattr("dimensions", value),
                "malformed surface file .*: topography declares dimensions that "
                "are not four exponents")
    whole = tmp_path / "whole.nc"
    whole.write_bytes(out.read_bytes())
    with netCDF4.Dataset(whole, "a") as ds:
        ds.variables["topography"].setncattr("dimensions",
                                             np.array([0.0, 1.0, 0.0, 0.0]))
    assert read_surface(whole).dimensions("topography") == Dimensions(length=1)
    with pytest.raises(ValueError, match="not a planetmodel surface file"):
        bad = tmp_path / "empty.nc"
        with netCDF4.Dataset(bad, "w"):
            pass
        validate_structure(bad)


def _without(source: Path, target: Path, *names: str) -> Path:
    """A copy of a surface file without the variables `names`."""
    with netCDF4.Dataset(source) as old, netCDF4.Dataset(target, "w") as new:
        new.setncatts({key: old.getncattr(key) for key in old.ncattrs()})
        for name, dimension in old.dimensions.items():
            new.createDimension(name, len(dimension))
        for name, var in old.variables.items():
            if name in names:
                continue
            var.set_auto_mask(False)
            attributes = {key: var.getncattr(key) for key in var.ncattrs()}
            copy = new.createVariable(name, var.dtype, var.dimensions,
                                      fill_value=attributes.pop("_FillValue", None))
            copy.setncatts(attributes)
            copy[...] = var[...]
    return target


def test_a_field_in_time_needs_its_coordinate_and_poles_their_shape(tmp_path: Path):
    grid = gauss_legendre(4)
    time, instant, fields = _case(grid, poles=True)
    del fields["rate"]              # no interval, whose size is held to time's
    out = write_surface(tmp_path / "good.nc", grid, fields=fields,
                        source="planetmodel tests", time=time, instant=instant)
    for axis, field in (("time", "ice_thickness"), ("instant", "instantaneous")):
        bad = _without(out, tmp_path / f"no_{axis}.nc", axis)
        with pytest.raises(ValueError, match=f"malformed surface file .*{field} is "
                                             f".* no coordinate variable '{axis}'"):
            validate_structure(bad)
        with pytest.raises(ValueError, match="no coordinate variable"):
            read_surface(bad)
    bad = _without(out, tmp_path / "poles.nc", "ice_thickness_poles")
    with netCDF4.Dataset(bad, "a") as ds:
        ds.createVariable("ice_thickness_poles", "f8", ("pole",))[:] = 0.0
    with pytest.raises(ValueError, match="ice_thickness_poles does not run over "
                                         r"\('time', 'pole'\)"):
        validate_structure(bad)
    bad = _without(out, tmp_path / "orphan.nc", "topography")
    with pytest.raises(ValueError, match="topography_poles accompanies no field"):
        validate_structure(bad)


def test_a_harmonic_field_is_read_by_the_harmonic_rule(tmp_path: Path):
    """A field stating the harmonic rule comes back as a function read by
    it, exact for a field of the grid's band; a file stating it on a grid
    without a band is refused."""
    grid = gauss_legendre(4)
    theta = grid.colatitudes[:, None] + 0.0 * grid.longitudes[None, :]
    path = write_surface(tmp_path / "h.nc", grid, source="tests", fields={
        "zonal": SurfaceVariable(np.cos(theta) ** 2,
                                 lateral_interpolation="harmonic")})
    fn = read_surface(path).function("zonal")
    assert fn.lateral == "harmonic"
    t, p = np.linspace(0.0, np.pi, 9), np.linspace(0.0, 6.0, 9)
    assert np.allclose(fn(t, p), np.cos(t) ** 2, rtol=0, atol=1e-13)
    for edit in (lambda ds: ds.setncattr("grid_kind", "custom"),
                 lambda ds: ds.delncattr("grid_lmax")):
        bad = tmp_path / "bad.nc"
        bad.write_bytes(path.read_bytes())
        with netCDF4.Dataset(bad, "a") as ds:
            edit(ds)
        with pytest.raises(ValueError, match="malformed surface file .*zonal states "
                                             "the lateral rule 'harmonic'"):
            read_surface_file(bad)


def test_without_netcdf4_the_writer_names_the_extra(tmp_path: Path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "netCDF4", None)
    grid = gauss_legendre(4)
    with pytest.raises(ImportError, match=r"planetmodel\[netcdf\]") as refusal:
        write_surface(tmp_path / "s.nc", grid, source="tests", fields={
            "a": SurfaceVariable(np.ones((grid.ntheta, grid.nphi)))})
    assert "model file" not in str(refusal.value)


def test_the_history_serves_every_support(tmp_path: Path):
    grid = gauss_legendre(4)
    i = np.arange(grid.ntheta)[:, None].astype(float)
    j = np.arange(grid.nphi)[None, :].astype(float)
    base = i + 0.01 * j
    nodes = np.stack([k * 10.0 + base for k in range(3)])
    nodes[1, 0, 0] = np.nan
    fields = {
        "h": SurfaceVariable(nodes, support="node", time_interpolation="linear"),
        "p": SurfaceVariable(nodes, support="node", time_interpolation="previous"),
        "n": SurfaceVariable(nodes, support="node"),
        "r": SurfaceVariable(nodes[:2], support="interval"),
        "q": SurfaceVariable(nodes[:1], support="instant"),
        "s": SurfaceVariable(base, poles=[3.0, -4.0],
                             dimensions=Dimensions(length=1)),
    }
    path = write_surface(tmp_path / "history.nc", grid, fields=fields,
                         source="tests", time=[0.0, 1.0, 3.0], instant=[2.0])
    history = read_surface(path)
    assert set(history.names()) == set(fields)
    assert history.support("r") == "interval"
    assert history.dimensions("s") == Dimensions(length=1)
    assert history.dimensions("h") is None
    assert history.variable("h").attributes["time_interpolation"] == "linear"
    assert np.array_equal(history.variable("s").values, base)
    with pytest.raises(KeyError, match="no field"):
        history.variable("s_poles")

    # linear: between nodes, at a node (the NaN beside it kept out), held
    assert np.allclose(history.values("h", date=2.0)[1:], 15.0 + base[1:])
    assert np.isnan(history.values("h", date=0.5)[0, 0])
    assert history.values("h", date=0.0)[0, 0] == base[0, 0]
    assert np.allclose(history.values("h", date=9.0, extrapolation="hold")[1:],
                       20.0 + base[1:])
    with pytest.raises(ValueError, match="outside"):
        history.values("h", date=9.0)
    # previous: the latest node not after the date
    assert np.allclose(history.values("p", date=2.999)[1:], 10.0 + base[1:])
    assert np.allclose(history.values("p", date=3.0)[1:], 20.0 + base[1:])
    # none: the stored nodes alone
    assert np.allclose(history.values("n", date=1.0)[1:], 10.0 + base[1:])
    with pytest.raises(ValueError, match="stored nodes alone"):
        history.values("n", date=0.5)
    # interval: right-continuous, the last closed at the end
    assert np.allclose(history.values("r", date=0.5), base)
    assert np.allclose(history.values("r", date=1.0)[1:], 10.0 + base[1:])
    assert np.allclose(history.values("r", date=3.0)[1:], 10.0 + base[1:])
    # instant: the stored instants alone, never held
    assert np.allclose(history.values("q", date=2.0), base)
    with pytest.raises(ValueError, match="stored instants alone"):
        history.values("q", date=2.0001, extrapolation="hold")
    # static: any date, and a date needed otherwise
    assert np.allclose(history.values("s"), base)
    assert np.allclose(history.values("s", date=7.0), base)
    with pytest.raises(ValueError, match="needs a date"):
        history.values("h")
    with pytest.raises(KeyError):
        history.values("s_poles")
    with pytest.raises(ValueError, match="extrapolation"):
        history.values("h", date=1.0, extrapolation="clamp")

    # the gridded function carries the rule and the stored poles
    fn = history.function("s")
    assert np.allclose(fn.poles, [3.0, -4.0])
    theta = grid.colatitudes[2]
    assert np.isclose(fn(theta, grid.longitudes[3]), base[2, 3])


def test_dimensions_round_trip(tmp_path: Path):
    grid = gauss_legendre(4)
    dims = Dimensions(mass=1, time=-2)
    out = write_surface(tmp_path / "dims.nc", grid, source="tests",
                        fields={"load": SurfaceVariable(
                            np.ones((grid.ntheta, grid.nphi)),
                            dimensions=dims)})
    var = read_surface_file(out).variables["load"]
    assert list(var.attributes["dimensions"]) == [1, 0, -2, 0]
    assert var.attributes["units"] == dims.unit_string()


def test_a_single_time_node_reads_itself():
    from planetmodel.surfacefile import time_rows
    t = np.array([3.0])
    for rule in ("linear", "previous"):
        assert time_rows("node", rule, 3.0, time=t) == ((0, 1.0), (0, 0.0))
        assert time_rows("node", rule, 9.0, time=t,
                         extrapolation="hold") == ((0, 1.0), (0, 0.0))
    with pytest.raises(ValueError, match="outside"):
        time_rows("node", "linear", 2.0, time=t)
