"""netCDF grids into the library's form, written here in every variant the
survey of EMC's files found, from one analytic field, so that each variant
must read to the same model."""
import hashlib

import numpy as np
import pytest

import planetmodel as pm
from planetmodel import datasets
from planetmodel.datasets import (Dataset, GridReader, RegionalGrid, Source, grid_model,
                                  read_grid)
from planetmodel.testing import check_dataset

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

R = 6371e3
DEPTHS = np.array([25.0, 100.0, 200.0, 410.0, 410.01, 600.0, 1000.0, 2000.0, 2890.0])


def field(depth_km, lat, lon):
    """vs in km/s at depths, latitudes and longitudes (degrees), broadcast,
    with a jump of 0.2 km/s across 410 km."""
    d, la, lo = np.broadcast_arrays(depth_km, np.radians(lat), np.radians(lon))
    jump = np.where(d > 410.005, 0.2, 0.0)
    return (4.4 + 2.5e-4 * d + jump + 0.05 * np.sin(la)
            + 0.03 * np.cos(la) * np.cos(lo) + 0.02 * np.cos(la) * np.sin(lo))


def write(path, *, depth=DEPTHS, lat=None, lon=None, order=("depth", "latitude",
          "longitude"), vertical="depth", vertical_units="km", units="km.s-1",
          values=None, fill=("_FillValue", 99999.0), valid_range=None, extra=None,
          name="vs", lat_units="degrees_north"):
    """A netCDF file of one variable on (depth, latitude, longitude) in the
    given dimension order, from `field` unless `values` are given."""
    lat = np.linspace(-90.0, 90.0, 46) if lat is None else np.asarray(lat, float)
    lon = np.linspace(-180.0, 180.0, 91) if lon is None else np.asarray(lon, float)
    v = (field(depth[:, None, None], lat[None, :, None], lon[None, None, :])
         if values is None else np.asarray(values, float))
    with netCDF4.Dataset(path, "w") as ds:
        axes = {"depth": depth, "latitude": lat, "longitude": lon}
        if vertical == "radius":
            axes = {"radius": R / 1e3 - depth, "latitude": lat, "longitude": lon}
            order = tuple("radius" if a == "depth" else a for a in order)
        for a, c in axes.items():
            ds.createDimension(a, c.size)
            var = ds.createVariable(a, "f8", (a,))
            var[:] = c
        ds["latitude"].units = lat_units
        ds["longitude"].units = "degrees_east"
        vname = "radius" if vertical == "radius" else "depth"
        ds[vname].units = vertical_units
        ds[vname].positive = "up" if vertical == "radius" else "down"
        kw = {}
        if fill and fill[0] == "_FillValue":
            kw["fill_value"] = np.float32(fill[1])
        var = ds.createVariable(name, "f4", order, **kw)
        var.set_auto_mask(False)
        if fill and fill[0] == "missing_value":
            var.missing_value = np.float32(fill[1])
        if valid_range is not None:
            var.valid_range = np.array(valid_range, dtype=np.float32)
        var.units = units
        canon = ("radius" if vertical == "radius" else "depth", "latitude", "longitude")
        var[:] = np.transpose(v, [canon.index(a) for a in order])
        for k, val in (extra or {}).items():
            ds.setncattr(k, val)
    return path


def vs_model(path, **kw):
    g = read_grid(path, **kw)
    return g, grid_model(g, {"vs": ("vs", None)}, name="t")


def at_random(model, rng, n=200):
    lo, hi = model.skeleton.boundaries[0], 6346e3
    r = rng.uniform(lo, hi, n)
    theta = np.arccos(rng.uniform(-1, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    out = np.full(n, np.nan)
    for i in range(n):
        loc = model.skeleton.locate(r[i])
        if loc.boundary is None and model.layer(loc.layer).names:
            out[i] = model.layer(loc.layer)["vs"](r[i], theta[i], phi[i])
    return out


def test_a_grid_reads_to_its_nodes_exactly(tmp_path):
    g, m = vs_model(write(tmp_path / "a.nc"))
    assert g.is_global and g.grid.kind == "equiangular" and g.grid.pole_nodes == (True,
                                                                                    True)
    assert g.grid.nphi == 90 and g.grid.longitudes[0] == 0.0     # the seam dropped
    pm.testing.check_model(m)
    lay = m.layer(m.skeleton.locate(R - 150e3).layer)
    theta, phi = g.grid.colatitudes[None, :, None], g.grid.longitudes[None, None, :]
    lat, lon = 90.0 - np.degrees(theta), np.degrees(phi)
    for d in (100.0, 200.0):
        got = lay["vs"](R - d * 1e3, theta, phi)
        want = 1e3 * field(d, lat, lon).astype(np.float32).astype(float)
        assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_a_near_repeated_depth_is_a_boundary(tmp_path):
    g, m = vs_model(write(tmp_path / "a.nc"))
    jump = R - 410.005e3
    assert np.any(np.isclose(m.skeleton.boundaries, jump, rtol=0.0, atol=1e-6))
    assert int(np.sum(np.diff(g.radius) == 0.0)) == 1
    above = m.layer(m.skeleton.locate(jump + 1.0).layer)["vs"](jump + 1.0, 1.0, 1.0)
    below = m.layer(m.skeleton.locate(jump - 1.0).layer)["vs"](jump - 1.0, 1.0, 1.0)
    assert above - below == pytest.approx(-200.0, abs=1.0)


VARIANTS = {
    "lat, lon, depth": dict(order=("latitude", "longitude", "depth")),
    "depth, lon, lat": dict(order=("depth", "longitude", "latitude")),
    "latitude descending": dict(lat=np.linspace(90.0, -90.0, 46)),
    "longitudes from 0, no seam": dict(lon=np.arange(0.0, 360.0, 4.0)),
    "spelled-out units": dict(units="kilometers/second", vertical_units="kilometer "),
    "a radius axis": dict(vertical="radius", vertical_units="kilometer"),
    "missing_value, not _FillValue": dict(fill=("missing_value", 9999.0)),
    "no fill at all": dict(fill=None),
    "a valid_range narrower than the data": dict(valid_range=(4.5, 4.6)),
    "a misleading latitude unit": dict(lat_units="degrees"),
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_reads_to_the_same_model(tmp_path, variant):
    _, canonical = vs_model(write(tmp_path / "a.nc"))
    _, other = vs_model(write(tmp_path / "b.nc", **VARIANTS[variant]))
    a = at_random(canonical, np.random.default_rng(1))
    b = at_random(other, np.random.default_rng(1))
    assert np.allclose(a, b, rtol=1e-12, atol=0.0, equal_nan=True)


def test_the_shipped_kinds_are_recognised_and_others_are_custom(tmp_path):
    cells = read_grid(write(tmp_path / "c.nc", lat=np.arange(-89.0, 90, 2.0),
                            lon=np.arange(-179.0, 180, 2.0)))
    assert cells.grid.kind == "equiangular" and cells.grid.pole_nodes == (False, False)
    assert np.degrees(cells.grid.longitudes[0]) == pytest.approx(1.0)
    fours = read_grid(write(tmp_path / "f.nc", lat=np.linspace(-88.0, 88.0, 45)))
    assert fours.grid.kind == "equiangular"                         # 4-degree cells
    odd = read_grid(write(tmp_path / "o.nc", lat=np.linspace(-88.0, 88.0, 89)))
    assert odd.grid.kind == "custom"


def test_a_grid_without_poles_takes_its_nearest_rings_mean(tmp_path):
    lat = np.linspace(-88.0, 88.0, 45)
    g, m = vs_model(write(tmp_path / "o.nc", lat=lat))
    lay = m.layer(m.skeleton.locate(R - 150e3).layer)
    r = R - 100e3
    ring = np.asarray(g.values["vs"])[np.argmin(np.abs(g.radius - r)), 0] * 1e3
    assert lay["vs"](r, 0.0, 0.3) == pytest.approx(np.mean(ring), rel=1e-12)


def test_pole_rows_and_seams_hold_one_value_or_are_refused(tmp_path):
    lat = np.linspace(-90.0, 90.0, 46)
    lon = np.linspace(-180.0, 180.0, 91)
    v = field(DEPTHS[:, None, None], lat[None, :, None], lon[None, None, :])
    noisy = v.copy()
    noisy[:, -1, :] += 0.01 * np.cos(np.radians(lon))              # the north row
    path = write(tmp_path / "p.nc", values=noisy)
    with pytest.raises(ValueError, match="along the pole row.*pole_rows='mean'"):
        read_grid(path)
    g = read_grid(path, pole_rows="mean")
    assert np.ptp(g.values["vs"][:, 0, :], axis=-1).max() == 0.0
    seam = v.copy()
    seam[:, 1:-1, -1] += 0.01                                        # off the poles
    path = write(tmp_path / "s.nc", values=seam)
    with pytest.raises(ValueError, match="seam repeated; seam='mean'"):
        read_grid(path)
    g = read_grid(path, seam="mean")
    assert g.grid.nphi == 90


def test_a_glitched_coordinate_is_rebuilt_when_the_entry_says_so(tmp_path):
    lat = np.arange(-89.5, 90.0, 1.0)
    glitched = lat.copy()
    glitched[[76, 103]] = [-13.6, 13.6]                             # LITHO1.0's kind
    lon = np.arange(-179.5, 180.0, 4.0)
    v = field(DEPTHS[:, None, None], lat[None, :, None], lon[None, None, :])
    path = write(tmp_path / "g.nc", lat=glitched, lon=lon, values=v)
    assert read_grid(path).grid.kind == "custom"
    fixed = read_grid(path, regular=True)
    assert fixed.grid.kind == "equiangular"
    far = lat.copy()
    far[76] = -12.0
    with pytest.raises(ValueError, match="half a step"):
        read_grid(write(tmp_path / "f.nc", lat=far, lon=lon, values=v), regular=True)


def test_missing_slices_and_values(tmp_path):
    lat, lon = np.linspace(-90.0, 90.0, 46), np.linspace(-180.0, 180.0, 91)
    v = field(DEPTHS[:, None, None], lat[None, :, None], lon[None, None, :])
    top = v.copy()
    top[0] = 99999.0                                                 # the 25 km slice
    g, m = vs_model(write(tmp_path / "t.nc", values=top))
    assert m.skeleton.boundaries[0] == pytest.approx(R - 2890e3)
    assert m.layers[-2].interval[1] == pytest.approx(R - 100e3)      # 25 km dropped
    inside = v.copy()
    inside[5] = np.nan
    with pytest.raises(ValueError, match="whole slice inside its support"):
        vs_model(write(tmp_path / "i.nc", values=inside))
    holes = v.copy()
    holes[4, 10, 10] = 99999.0
    with pytest.raises(ValueError, match="missing at 1 nodes.*fill="):
        vs_model(write(tmp_path / "h.nc", values=holes))
    g = read_grid(write(tmp_path / "h2.nc", values=holes))
    model = grid_model(g, {"vs": ("vs", None)}, fill=4500.0)
    pm.testing.check_model(model)


def test_units_are_checked_against_what_a_variable_is(tmp_path):
    g = read_grid(write(tmp_path / "u.nc"))
    with pytest.raises(ValueError, match="must be of dimensionless"):
        grid_model(g, {"vs": ("vs", "relative")}, reference=pm.PREM())
    pct = read_grid(write(tmp_path / "p.nc", units="percent", name="dvs",
                          values=np.full((DEPTHS.size, 46, 91), 1.5)))
    m = grid_model(pct, {"dvs": ("vs", "relative")}, reference=pm.PREM())
    lay = m.layer("lower_mantle")
    assert lay["vs"].perturbation(5e6, 1.0, 1.0) == pytest.approx(0.015)
    wrong = read_grid(write(tmp_path / "w.nc", units="g.cm-1"))
    with pytest.raises(ValueError, match="units= corrects a mislabel"):
        grid_model(wrong, {"vs": ("vs", None)})
    fixed = grid_model(wrong, {"vs": ("vs", None)}, units={"vs": "km/s"})
    pm.testing.check_model(fixed)
    with pytest.raises(ValueError, match="states no unit"):
        read_grid(write(tmp_path / "e.nc", vertical_units=""))
    assert read_grid(write(tmp_path / "e2.nc", vertical_units=""),
                     vertical_units="km").radius is not None


def test_a_regional_grid_evaluates_inside_its_box(tmp_path):
    lat, lon = np.arange(50.0, 70.01, 1.0), np.arange(-20.0, 20.01, 1.0)
    path = write(tmp_path / "r.nc", lat=lat, lon=lon, depth=DEPTHS[:6])
    g = read_grid(path)
    assert not g.is_global and g.grid is None
    rg = RegionalGrid(g, {"vs": ("vs", None)})
    (r0, r1), (t0, t1), (p0, p1) = rg.extent
    assert np.degrees([p0, p1]) == pytest.approx([340.0, 380.0])       # contiguous
    theta, phi = pm.from_geographic(lat[3], lon[30])                 # at a node
    got = rg("vs", R - 100e3, theta, phi)
    assert got == pytest.approx(1e3 * float(np.float32(field(100.0, lat[3], lon[30]))))
    assert rg.contains(*pm.from_geographic(60.0, 0.0))
    assert not rg.contains(*pm.from_geographic(60.0, 30.0))
    assert np.isnan(rg("vs", R - 100e3, *pm.from_geographic(60.0, 30.0)))
    assert np.isnan(rg("vs", R - 1000e3, theta, phi))                # below its support
    with pytest.raises(ValueError, match="global grid"):
        grid_model(g, {"vs": ("vs", None)})


def boundary_angle(theta, phi, lat, lon, n=20001):
    """The least great-circle angle from points to the sides of a box of
    latitudes and longitudes (degrees, each a low and a high): its two
    parallels and its two meridian arcs sampled at `n` points each, the
    angle from the dot products of unit vectors."""
    def unit(t, f):
        return np.stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)], -1)

    la, lo = np.radians(np.linspace(*lat, n)), np.radians(np.linspace(*lon, n))
    sides = [unit(0.5 * np.pi - np.full(n, np.radians(x)), lo) for x in lat]
    sides += [unit(0.5 * np.pi - la, np.full(n, np.radians(x))) for x in lon]
    edge = np.concatenate(sides)
    points = unit(np.asarray(theta), np.asarray(phi))
    return np.array([np.arccos(np.clip(np.max(edge @ p), -1.0, 1.0)) for p in points])


def box_grid(lat, lon, step=1.0):
    """A regional grid of constant vs over a box of latitudes and longitudes
    (degrees, each a low and a high), from 50 to 150 km depth."""
    lats = np.arange(lat[1], lat[0] - 1e-9, -step)
    lons = np.arange(lon[0], lon[1] + 1e-9, step)
    radius = R - np.array([150e3, 50e3])
    data = datasets.GridData(np.radians(90.0 - lats), np.radians(lons), radius,
                             {"vs": np.full((2, lats.size, lons.size), 4.5)},
                             {"vs": "km/s"})
    return RegionalGrid(data, {"vs": ("vs", None)})


@pytest.mark.parametrize("lat, lon", [
    ((50.0, 70.0), (-20.0, 20.0)), ((60.0, 80.0), (0.0, 200.0)),
    ((40.0, 90.0), (10.0, 260.0)), ((-90.0, -55.0), (-170.0, 100.0)),
    ((-30.0, 40.0), (-100.0, 170.0))],
    ids=["narrow", "wider than 180", "to the north pole", "to the south pole",
         "across the equator"])
def test_a_box_s_edge_distance_is_the_angle_to_its_nearest_side(lat, lon):
    """Against the box's sides sampled densely, the least angle to any of
    their points: 20001 points a side, less than 1e-4 radians apart."""
    rg = box_grid(lat, lon)
    rng = np.random.default_rng(12)
    la = rng.uniform(lat[0] + 0.01, lat[1] - 0.01, 300)
    lo = rng.uniform(lon[0] + 0.01, lon[1] - 0.01, 300)
    theta, phi = pm.from_geographic(la, lo)
    assert rg.contains(theta, phi).all()
    got = rg.edge_distance(theta, phi)
    assert np.allclose(got, boundary_angle(theta, phi, lat, lon), rtol=0.0, atol=1e-4)
    outside = pm.from_geographic(np.full(3, 0.5 * sum(lat)),
                                 lon[1] + np.array([1.0, 30.0, 60.0]))
    assert not rg.contains(*outside).any()
    assert np.all(rg.edge_distance(*outside) == 0.0)
    pm.testing.check_regional(rg)


def test_a_band_round_the_circle_has_no_sides_in_longitude(tmp_path):
    """Longitudes that close the circle between latitudes short of the
    poles: a regional grid with no seam, read across the join by the two
    columns either side of it, its edge the two parallels."""
    lat, lon = np.arange(60.0, 80.01, 1.0), np.arange(0.0, 360.0, 1.0)
    g = read_grid(write(tmp_path / "band.nc", lat=lat, lon=lon, depth=DEPTHS[:4]))
    assert not g.is_global and g.phi.size == 360
    rg = RegionalGrid(g, {"vs": ("vs", None)})
    (_, _), _, (p0, p1) = rg.extent
    assert p1 - p0 == pytest.approx(2.0 * np.pi, rel=1e-12)
    rng = np.random.default_rng(13)
    la, lo = rng.uniform(60.0, 80.0, 200), rng.uniform(-360.0, 720.0, 200)
    theta, phi = pm.from_geographic(la, lo)
    assert rg.contains(theta, phi).all()
    want = np.radians(np.minimum(la - 60.0, 80.0 - la))
    assert np.allclose(rg.edge_distance(theta, phi), want, rtol=0.0, atol=1e-12)
    # across the join, half way between the last column and the first
    for where in (359.5, -0.5, 719.5):
        got = rg("vs", R - 100e3, *pm.from_geographic(65.0, where))
        ends = np.float32(field(100.0, 65.0, np.array([359.0, 0.0])))
        assert got == pytest.approx(1e3 * float(np.mean(ends.astype(float))),
                                    rel=1e-12)
    # at a node of every column, the join's two among them
    nodes = pm.from_geographic(np.full(lon.size, 70.0), lon)
    assert np.allclose(rg("vs", R - 100e3, *nodes),
                       1e3 * np.float32(field(100.0, 70.0, lon)).astype(float),
                       rtol=1e-12)
    pm.testing.check_regional(rg)
    # a box short of the circle keeps its sides
    part = RegionalGrid(read_grid(write(tmp_path / "part.nc", lat=lat,
                                        lon=np.arange(0.0, 300.0, 1.0),
                                        depth=DEPTHS[:4])), {"vs": ("vs", None)})
    assert not part.contains(*pm.from_geographic(70.0, 330.0))
    assert np.isnan(part("vs", R - 100e3, *pm.from_geographic(70.0, 330.0)))
    # as do two columns, which have no median step to close a circle by
    two = box_grid((60.0, 80.0), (0.0, 100.0), step=100.0)
    assert not two.contains(*pm.from_geographic(70.0, 200.0))
    assert two.extent[2] == pytest.approx((0.0, np.radians(100.0)))


def test_a_packed_variable_is_unpacked_and_its_missing_values_found(tmp_path):
    """A variable stored as scaled integers: the values are the stored
    ones times `scale_factor` plus `add_offset`, a signed integer marked
    `_Unsigned` read as unsigned first, and a missing value is the stored
    integer the file names, not its scaled value."""
    lat, lon = np.arange(50.0, 60.01, 1.0), np.arange(0.0, 20.01, 1.0)
    depth = np.array([50.0, 100.0, 150.0])
    shape = (depth.size, lat.size, lon.size)
    rng = np.random.default_rng(14)
    packed = rng.integers(-2000, 2000, shape).astype("i2")
    packed[0, 0, 0] = packed[2, 5, 7] = -32768
    small = rng.integers(-128, 127, shape).astype("i1")
    small[1, 2, 3] = -1                                  # 255 unsigned: missing
    path = tmp_path / "packed.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for name, c, unit in (("depth", depth, "km"),
                              ("latitude", lat, "degrees_north"),
                              ("longitude", lon, "degrees_east")):
            ds.createDimension(name, c.size)
            var = ds.createVariable(name, "f8", (name,))
            var[:], var.units = c, unit
        dims = ("depth", "latitude", "longitude")
        vs = ds.createVariable("vs", "i2", dims, fill_value=-32768)
        vs.set_auto_maskandscale(False)
        vs.scale_factor, vs.add_offset, vs.units = 0.001, 4.0, "km/s"
        vs[:] = packed
        rho = ds.createVariable("rho", "i1", dims, fill_value=-1)
        rho.set_auto_maskandscale(False)
        rho.scale_factor, rho.units = 0.02, "g/cm3"
        rho.setncattr("_Unsigned", "true")
        rho[:] = small
    g = read_grid(path)
    order = np.argsort(-lat)                             # colatitude from the north
    for name, stored, missing, scale, offset in (
            ("vs", packed.astype(float), packed == -32768, 0.001, 4.0),
            ("rho", small.astype(float) % 256.0, small == -1, 0.02, 0.0)):
        want = np.where(missing, np.nan, stored * scale + offset)[::-1][:, order]
        got = g.values[name]
        assert got.dtype == np.float64
        assert np.array_equal(np.isnan(got), np.isnan(want)), name
        assert np.isnan(got).sum() == missing.sum() > 0
        assert np.allclose(got, want, rtol=1e-14, atol=0, equal_nan=True), name
    rg = RegionalGrid(g, {"vs": ("vs", None)})
    at = pm.from_geographic(lat[3], lon[4])
    assert rg("vs", R - 100e3, *at) == pytest.approx(
        1e3 * (float(packed[1, 3, 4]) * 0.001 + 4.0), rel=1e-12)


def test_a_reader_gives_a_model_over_its_reference_or_a_regional_grid(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setenv("PLANETMODEL_DATA", str(tmp_path / "cache"))
    pct = write(tmp_path / "p.nc", units="%", name="dvs",
                values=np.full((DEPTHS.size, 46, 91), -2.0))
    d = Dataset("pct", (Source(pct.as_uri(), hashlib.sha256(pct.read_bytes())
                               .hexdigest()),),
                GridReader({"dvs": ("vs", "relative")}), title="t", citation=("c",),
                licence="l", reference="PREM")
    m = datasets.load(d)
    lay = m.layer("lower_mantle")
    assert isinstance(lay["vs"], pm.PerturbedField)
    want = 0.98 * float(lay["vs"].reference(5e6))
    assert lay["vs"](5e6, 1.0, 1.0) == pytest.approx(want)
    check_dataset(d, paths=datasets.fetch(d))
    box = write(tmp_path / "r.nc", lat=np.arange(50.0, 70.01, 1.0),
                lon=np.arange(-20.0, 20.01, 1.0), depth=DEPTHS[:6])
    r = Dataset("box", (Source(box.as_uri(), hashlib.sha256(box.read_bytes())
                               .hexdigest()),),
                GridReader({"vs": ("vs", None)}), title="t", citation=("c",),
                licence="l")
    assert isinstance(datasets.load(r), RegionalGrid)
    check_dataset(r, paths=datasets.fetch(r))
    coarse = datasets.load(d, stride=3)
    assert coarse.layer("lower_mantle")["vs"].perturbation.grid.ntheta == 16


def test_refusals_of_what_is_not_a_grid_of_latitude_and_longitude(tmp_path):
    path = tmp_path / "xy.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for a in ("x", "y"):
            ds.createDimension(a, 3)
            ds.createVariable(a, "f8", (a,))[:] = np.arange(3.0)
        ds.createVariable("vs", "f4", ("y", "x"))[:] = np.ones((3, 3))
    with pytest.raises(ValueError, match="projected"):
        read_grid(path)
    with pytest.raises(ValueError, match="pole_rows must be"):
        read_grid(write(tmp_path / "a.nc"), pole_rows="drop")


EMC_ENTRIES = ("S362ANI+M", "SEMUCB-WM1", "SGLOBE-rani", "GyPSuM", "GLAD-M25",
               "CSEM-North-Atlantic")


@pytest.mark.parametrize("name", EMC_ENTRIES)
def test_the_shipped_emc_entries_hold_to_their_contract(name):
    d = datasets.REGISTRY[name]
    check_dataset(d)
    assert isinstance(d.reader, GridReader)
    assert d.sources[0].url.startswith(
        "https://data.earthscope.org/archive/seismology/products/emc/netcdf/")
    assert "CC BY 4.0" in d.licence and any("10.17611/" in c for c in d.citation)
    assert any("Trabant" in c for c in d.citation)


def test_a_log10_variable_and_a_floor_and_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANETMODEL_DATA", str(tmp_path / "cache"))
    rng = np.random.default_rng(2)
    log10 = rng.uniform(17.0, 40.0, (DEPTHS.size, 46, 91))
    log10[:, 0, :] = log10[:, 0, :1]                   # single-valued poles
    log10[:, -1, :] = log10[:, -1, :1]
    log10[:, :, -1] = log10[:, :, 0]                   # the seam repeated
    path = write(tmp_path / "eta.nc", units="", name="log10eta", values=log10)

    def entry(reader):
        return Dataset("eta", (Source(path.as_uri(), hashlib.sha256(
            path.read_bytes()).hexdigest()),), reader, title="t", citation=("c",),
            licence="l")

    plain = GridReader({"log10eta": ("viscosity", None)}, units={"log10eta": "Pa s"},
                       log10=("log10eta",))
    m = datasets.load(entry(plain))
    lay = m.layer(m.skeleton.locate(5e6).layer)
    r = R - 1e3 * DEPTHS[6]
    theta, phi = np.radians(90.0 - 30.0), np.radians(40.0)     # a node of the grid
    j, k = list(np.linspace(-90.0, 90.0, 46)).index(30.0), 55
    assert np.isclose(np.linspace(-180.0, 180.0, 91)[k], 40.0)
    assert lay["viscosity"](r, theta, phi) == pytest.approx(10.0 ** log10[6, j, k],
                                                            rel=1e-5)
    clipped = GridReader({"log10eta": ("viscosity", None)}, units={"log10eta": "Pa s"},
                         log10=("log10eta",), clip={"log10eta": (1e18, 1e30)})
    c = datasets.load(entry(clipped))
    lay = c.layer(c.skeleton.locate(5e6).layer)
    want = np.clip(10.0 ** log10[6, j, k], 1e18, 1e30)
    assert lay["viscosity"](r, theta, phi) == pytest.approx(want, rel=1e-5)
    check_dataset(entry(clipped), paths=datasets.fetch(entry(clipped)))
    with pytest.raises(ValueError, match="not below its cap"):
        datasets.load(entry(GridReader({"log10eta": ("viscosity", None)},
                                       units={"log10eta": "Pa s"}, log10=("log10eta",),
                                       clip={"log10eta": (1e30, 1e18)})))
