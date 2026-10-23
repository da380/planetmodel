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

from planetmodel import Scales, equiangular, gauss_legendre
from planetmodel.surfacefile import (SCHEMA, SurfaceVariable,
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
    assert not path.exists(), "a refused write left a file behind"


def test_poles_against_a_grid_with_them(tmp_path: Path):
    grid = equiangular(9, 16, poles=True)
    with pytest.raises(ValueError, match="holds both poles"):
        write_surface(tmp_path / "p.nc", grid, source="tests",
                      fields={"a": SurfaceVariable(
                          np.ones((grid.ntheta, grid.nphi)), poles=[0.0, 0.0])})


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
    with pytest.raises(ValueError, match="not a planetmodel surface file"):
        bad = tmp_path / "empty.nc"
        with netCDF4.Dataset(bad, "w"):
            pass
        validate_structure(bad)


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
