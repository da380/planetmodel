"""The polar stereographic projection and the projected grids read on it:
the projection against EPSG's published values and its own closed-form
inverse, both variants and both poles; a grid written here on a projected
plane evaluating at (theta, phi) as the field it was sampled from; its
region's edge; and a projected grid merged into a model."""
import numpy as np
import pytest

from planetmodel import PREM, equiangular, from_geographic
from planetmodel.datasets import (GridReader, PolarStereographic, ProjectedData,
                                  RegionalGrid, merged, read_grid)
from planetmodel.testing import check_regional

EPSG_3031 = ("+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
             "+datum=WGS84 +units=m +no_defs")


def test_the_projection_has_epsg_s_published_values():
    # EPSG Guidance Note 7-2, Polar Stereographic variant B: WGS 84 /
    # Australian Antarctic Polar Stereographic (EPSG:3032), its worked example
    p = PolarStereographic(-1, standard_parallel=-71.0, central_meridian=70.0,
                           false_easting=6e6, false_northing=6e6)
    x, y = p.forward(-75.0, 120.0)
    assert abs(x - 7255380.79) < 0.01 and abs(y - 7053389.56) < 0.01
    # EPSG:3031: the pole at the origin, true scale on 71 S, x along 90 E
    q = PolarStereographic.from_proj4(EPSG_3031)
    assert q.forward(-90.0, 37.0) == (0.0, 0.0)
    assert np.isclose(q.scale_factor(-71.0), 1.0, rtol=1e-14)
    assert q.scale_factor(-89.0) < 1.0 < q.scale_factor(-50.0)
    x, y = q.forward(-71.0, 90.0)
    assert np.isclose(x, 2082760.1085, atol=1e-3) and abs(y) < 1e-6
    x, y = q.forward(-71.0, 0.0)
    assert abs(x) < 1e-6 and np.isclose(y, 2082760.1085, atol=1e-3)


@pytest.mark.parametrize("p", [
    PolarStereographic(-1, standard_parallel=-71.0),
    PolarStereographic(1, standard_parallel=70.0, central_meridian=-45.0),
    PolarStereographic(-1, scale=0.994, false_easting=2e6, false_northing=2e6),
    PolarStereographic(1, standard_parallel=90.0, semi_major_axis=6378137.0,
                       inverse_flattening=298.257222101)],
    ids=["3031", "3413", "UPS south", "north, true at the pole"])
def test_the_inverse_undoes_the_projection(p):
    rng = np.random.default_rng(3)
    lat = p.pole * rng.uniform(20.0, 90.0, 500)
    lon = rng.uniform(-180.0, 180.0, 500)
    x, y = p.forward(lat, lon)
    la, lo = p.inverse(x, y)
    assert np.allclose(la, lat, rtol=0, atol=1e-9)
    away = np.abs(lat) < 89.999
    assert np.allclose((lo - lon + 180.0)[away] % 360.0 - 180.0, 0.0, atol=1e-9)
    # the library's geocentric angles and back
    theta, phi = np.arccos(rng.uniform(-1, 1, 300)), rng.uniform(0, 2 * np.pi, 300)
    keep = p.pole * (0.5 * np.pi - theta) > 0.2
    t, f = p.to_angles(*p.from_angles(theta[keep], phi[keep]))
    assert np.allclose(t, theta[keep], atol=1e-11)
    assert np.allclose(np.mod(f - phi[keep] + np.pi, 2 * np.pi) - np.pi, 0.0,
                       atol=1e-9)
    # the geodetic latitude lies further from the equator than the geocentric
    g = p.geodetic_latitude(np.radians(40.0))
    assert 50.0 < g < 50.2


def test_the_inverse_gives_longitudes_from_minus_180_up_to_180():
    p = PolarStereographic(1, standard_parallel=70.0)
    assert p.inverse(0.0, 1e6)[1] == -180.0       # on the meridian 180 E
    lon = np.linspace(-360.0, 360.0, 49)
    back = p.inverse(*p.forward(np.full(lon.shape, 80.0), lon))[1]
    assert np.all((back >= -180.0) & (back < 180.0))
    assert np.allclose((back - lon + 180.0) % 360.0 - 180.0, 0.0, atol=1e-9)


def test_the_projection_is_pyproj_s():
    pyproj = pytest.importorskip("pyproj", reason="pyproj, a dependency of cartopy")
    p = PolarStereographic.from_proj4(EPSG_3031)
    rng = np.random.default_rng(4)
    lat, lon = rng.uniform(-90, -30, 300), rng.uniform(-180, 180, 300)
    want = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031",
                                       always_xy=True).transform(lon, lat)
    assert np.allclose(p.forward(lat, lon), want, rtol=0, atol=1e-6)


def test_a_proj4_string_is_read_or_refused_by_name():
    p = PolarStereographic.from_proj4("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 "
                                      "+ellps=GRS80 +x_0=1 +y_0=2")
    assert (p.pole, p.standard_parallel, p.central_meridian) == (1, 70.0, -45.0)
    assert p.inverse_flattening == 298.257222101 and p.false_northing == 2.0
    a = PolarStereographic.from_proj4("+proj=stere +lat_0=-90 +k=0.994 +a=6378137 "
                                      "+b=6356752.314245")
    assert a.scale == 0.994 and np.isclose(a.inverse_flattening, 298.257223563,
                                            rtol=1e-8)
    for text, words in (("+proj=merc +lat_ts=10", "stereographic"),
                        ("+proj=stere +lat_0=45 +lat_ts=60", "not polar"),
                        ("+proj=stere +lat_0=-90 +lat_ts=-71 +units=km", "metres"),
                        ("+proj=stere +lat_0=-90 +lat_ts=-71 +ellps=clrk66",
                         "ellipsoid")):
        with pytest.raises(ValueError, match=words):
            PolarStereographic.from_proj4(text)
    with pytest.raises(ValueError, match="one of"):
        PolarStereographic(-1)
    with pytest.raises(ValueError, match="hemisphere"):
        PolarStereographic(-1, standard_parallel=71.0)


# -- projected grids -----------------------------------------------------------------

R = 6371e3
P = PolarStereographic.from_proj4(EPSG_3031)


def linear(r, x, y):
    """vs in km/s, linear in r, x and y, which the readers' rules hold
    exactly between nodes."""
    return 4.5 + 2e-7 * x - 1e-7 * y + 1e-6 * (r - (R - 200e3))


def write_projected(path, *, proj4=EPSG_3031, footprint=True, km=True):
    netCDF4 = pytest.importorskip("netCDF4",
                                  reason="needs the planetmodel[netcdf] extra")
    xs = np.linspace(-3000e3, 3000e3, 61)
    ys = np.linspace(-3000e3, 3000e3, 61)
    depth = np.array([50.0, 150.0, 250.0, 400.0])
    X, Y = np.meshgrid(xs, ys)                         # (y, x)
    v = np.stack([linear(R - 1e3 * d, X, Y) for d in depth])
    if footprint:
        v[:, np.hypot(X, Y) > 2500e3] = 99999.0         # a disc: no corners
    with netCDF4.Dataset(path, "w") as ds:
        scale = 1e-3 if km else 1.0
        for name, c in (("depth", depth), ("Y", ys * scale), ("X", xs * scale)):
            ds.createDimension(name, c.size)
            var = ds.createVariable(name, "f8", (name,))
            var[:] = c
        ds["depth"].units, ds["depth"].positive = "km", "down"
        ds["X"].units = ds["Y"].units = "km" if km else "m"
        ds["X"].long_name = "X; positive grid north"           # as ANT-20 mislabels
        var = ds.createVariable("vs", "f4", ("depth", "Y", "X"),
                                fill_value=np.float32(99999.0))
        var.units = "km.s-1"
        var[:] = v
        if proj4:
            ds.proj4text = proj4
    return path


@pytest.mark.netcdf
@pytest.mark.parametrize("km", [True, False])
def test_a_projected_grid_evaluates_as_the_field_it_was_sampled_from(tmp_path, km):
    data = read_grid(write_projected(tmp_path / "p.nc", km=km))
    assert isinstance(data, ProjectedData) and not data.is_global
    assert data.projection == P and data.x[0] == -3000e3 and data.y[-1] == 3000e3
    g = RegionalGrid(data, {"vs": ("vs", None)})
    rng = np.random.default_rng(6)
    lat, lon = rng.uniform(-90.0, -66.0, 400), rng.uniform(-180, 180, 400)
    theta, phi = from_geographic(lat, lon)
    r = rng.uniform(R - 400e3, R - 50e3, 400)
    x, y = P.from_angles(theta, phi)
    inside = np.hypot(x, y) < 2400e3
    got = g("vs", r, theta, phi)
    assert np.allclose(got[inside], 1e3 * linear(r, x, y)[inside], rtol=1e-6)
    assert g.contains(theta, phi)[inside].all()
    check_regional(g)
    # outside the footprint, and north of it, nothing
    t, f = from_geographic(np.array([-50.0, 10.0]), np.array([30.0, 30.0]))
    assert np.all(np.isnan(g("vs", R - 100e3, t, f)))
    assert not g.contains(t, f).any()


@pytest.mark.netcdf
def test_a_projected_grid_s_edge_is_its_footprint_s(tmp_path):
    g = RegionalGrid(read_grid(write_projected(tmp_path / "p.nc")),
                     {"vs": ("vs", None)})
    # along the meridian 0, toward the footprint's edge 2500 km from the pole
    lat = np.array([-89.0, -80.0, -75.0, -70.0, -60.0])
    theta, phi = from_geographic(lat, np.zeros(5))
    d = g.edge_distance(theta, phi)
    y = P.from_angles(theta, phi)[1]
    assert np.all(np.diff(d[:4]) < 0.0) and d[-1] == 0.0
    # on the plane, its distance to the disc's edge divided by the scale
    k = P.scale_factor(P.geodetic_latitude(theta))
    want = (2500e3 - np.abs(y)) / (k * P.semi_major_axis)
    assert np.allclose(d[1:3], want[1:3], rtol=0.02)
    # a grid square to its corners: its edge is the square's
    whole = RegionalGrid(read_grid(write_projected(tmp_path / "w.nc", footprint=False)),
                         {"vs": ("vs", None)})
    corner = P.to_angles(np.array([2990e3]), np.array([2990e3]))
    assert whole.contains(*corner)[0] and not g.contains(*corner)[0]


@pytest.mark.netcdf
def test_a_projected_grid_needs_its_projection(tmp_path):
    path = write_projected(tmp_path / "p.nc", proj4=None)
    with pytest.raises(ValueError, match="proj4text"):
        read_grid(path)
    north = PolarStereographic(1, standard_parallel=70.0)
    assert read_grid(path, projection=north).projection == north
    reader = GridReader({"vs": ("vs", None)}, projection=north)
    from planetmodel.datasets import Dataset, Source
    entry = Dataset("p", (Source("https://example.org/p.nc", "0" * 64),), reader,
                    title="A test", citation=("A, 2026",), licence="CC0")
    assert reader.read((path,), entry).projection == north


@pytest.mark.netcdf
def test_a_projected_grid_merges_into_a_model(tmp_path):
    g = RegionalGrid(read_grid(write_projected(tmp_path / "p.nc")),
                     {"vs": ("vsv", None)})
    model = merged(PREM(), g, name="vsv", width=np.radians(4.0),
                   grid=equiangular(90, 180))
    lay = model.layer("low_velocity_zone")
    lo, hi = lay.interval
    r = 0.5 * (lo + hi)
    deep, edge = from_geographic(-85.0, 40.0), from_geographic(-30.0, 40.0)
    x, y = P.from_angles(*deep)
    assert np.isclose(lay["vsv"](r, *deep), 1e3 * linear(r, x, y), rtol=2e-3)
    prem = PREM().layer("low_velocity_zone")["vsv"](r)
    assert np.isclose(lay["vsv"](r, *edge), prem, rtol=1e-12)
