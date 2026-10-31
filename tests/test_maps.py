"""Maps of sphere slices on cartopy's projections: the cells around the
nodes, the seam closed and the poles reached on every kind of grid."""
import io

import numpy as np
import pytest

from planetmodel import (AnalyticField, AngularGrid, LayeredIsotropicElastic,
                         driscoll_healy, equiangular, gauss_legendre)
from planetmodel import slices
from planetmodel.frames import cartesian_points, spherical_coordinates
from planetmodel.plotting import DIVERGING, cell_edges, sphere_map

pytestmark = pytest.mark.maps

GRIDS = [gauss_legendre(6), driscoll_healy(5), equiangular(9, 16, poles=True),
         equiangular(8, 16),
         AngularGrid(np.array([0.3, 1.0, 1.9, 2.8]),
                     np.array([0.2, 1.5, 3.0, 4.4, 5.9]))]


@pytest.mark.parametrize("grid", GRIDS, ids=lambda g: g.kind)
def test_the_cells_close_the_seam_and_reach_the_poles(grid):
    lon, lat = cell_edges(grid)
    assert lon.size == grid.nphi + 1 and lat.size == grid.ntheta + 1
    assert np.isclose(lon[-1] - lon[0], 360.0)
    assert lat[0] == 90.0 and lat[-1] == -90.0
    assert np.all(np.diff(lon) > 0) and np.all(np.diff(lat) < 0)
    # each node inside its own cell
    phi = np.degrees(grid.longitudes)
    assert np.all((lon[:-1] < phi) & (phi < lon[1:]))
    nodes = 90.0 - np.degrees(grid.colatitudes)
    assert np.all((lat[:-1] >= nodes) & (nodes >= lat[1:]))


def _model():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    return m.with_field(1, "vs", AnalyticField(
        (0.5, 1.0), lambda r, t, p: 3.0 + 0.2 * r * np.sin(t) * np.cos(p),
        name="vs"),
        replace=True)


def test_a_map_draws_the_slice_on_a_projection():
    ccrs = pytest.importorskip("cartopy.crs",
                               reason="needs the planetmodel[maps] extra")
    plt = pytest.importorskip("matplotlib.pyplot")
    m = _model()
    grid = gauss_legendre(8)
    s = slices.sphere(m, "vs", radius=0.8, grid=grid, relative_to="mean")
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    mesh = sphere_map(ax, s, coastlines=True, colorbar=True)
    assert np.allclose(np.asarray(mesh.get_array()).reshape(s.values.shape), s.values)
    lo, hi = mesh.get_clim()
    assert lo == -hi and mesh.get_cmap().name == DIVERGING
    # coastlines asked for, not fetched: nothing is drawn here
    assert any(type(a).__name__ == "FeatureArtist" for a in ax.get_children())
    assert fig.axes[-1].get_xlabel() == s.label
    plt.close(fig)
    # without coastlines it draws, on any projection, offline
    for projection in (ccrs.Mollweide(central_longitude=180.0),
                       ccrs.Orthographic(30, 20), ccrs.PlateCarree()):
        fig, ax = plt.subplots(subplot_kw={"projection": projection})
        sphere_map(ax, slices.sphere(m, "vs", radius=0.8, grid=driscoll_healy(6)))
        fig.savefig(io.BytesIO(), format="png")
        plt.close(fig)


def test_a_kernel_s_map_is_symmetric_clipped_and_normalised():
    ccrs = pytest.importorskip("cartopy.crs",
                               reason="needs the planetmodel[maps] extra")
    plt = pytest.importorskip("matplotlib.pyplot")
    from planetmodel import GriddedFunction
    grid = gauss_legendre(16)
    def kernel(t, p):
        return np.cos(t) + 40.0 * np.exp(-((t - 1.0) ** 2 + (p - 2.0) ** 2) / 0.01)

    spike = GriddedFunction.sampled(kernel, grid, lateral="linear")
    s = slices.surface(_model(), spike, 1, name="K")
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    mesh = sphere_map(ax, s, diverging=True, clip=95, colorbar=True)
    assert np.isclose(mesh.get_clim()[1], np.percentile(np.abs(s.values), 95))
    assert mesh.get_clim()[0] == -mesh.get_clim()[1] and mesh.colorbar.extend == "max"
    plt.close(fig)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    mesh = sphere_map(ax, s, normalised=True, diverging=True, colorbar=True)
    assert np.allclose(np.asarray(mesh.get_array()).reshape(s.values.shape),
                       s.values / np.max(np.abs(s.values)))
    assert fig.axes[-1].get_xlabel() == "K, on interface 1, normalised"
    plt.close(fig)


def test_a_map_is_refused_on_axes_other_than_cartopys():
    plt = pytest.importorskip("matplotlib.pyplot")
    s = slices.sphere(_model(), "vs", radius=0.8, grid=gauss_legendre(4))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="cartopy GeoAxes"):
        sphere_map(ax, s)
    geo = fig.add_subplot(2, 1, 2, projection="mollweide")
    with pytest.raises(ValueError, match="maps extra"):
        sphere_map(geo, s)
    plt.close(fig)


def test_a_cutaway_of_the_earth_wears_its_relief_its_land_and_its_coastlines():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from planetmodel import PREM
    from planetmodel.plotting import LAND, SEA, cutaway
    from matplotlib.colors import to_rgb
    prem = PREM()
    c = slices.cutaway(prem, "vsv", inner=3480e3, n=60, longitudes=(-1.5, 0.0),
                       colatitudes=(0.5 * np.pi, np.pi))
    fig, ax = plt.subplots()
    relief = cutaway(ax, c, surface="image", coastlines=True, scale=1e-6).get_array()
    top = c.mask("surface")
    assert np.ptp(relief[top], axis=0).max() > 0.2       # not one colour
    lines = ax.get_lines()
    assert len(lines) == 2                                # the arcs and the coast
    x, y = lines[-1].get_data()
    seen = np.isfinite(x)
    assert seen.sum() > 100 and np.any(~seen)
    assert np.all(np.hypot(x[seen], y[seen]) <= c.outer * 1e-6 * (1 + 1e-9))
    # the coastline is on the surface that remains: in front, out of the cut
    X = x[seen][:, None] * c.view[0] + y[seen][:, None] * c.view[1]
    depth = np.sqrt(np.maximum((c.outer * 1e-6) ** 2 - x[seen] ** 2 - y[seen] ** 2,
                               0.0))
    X = (X + depth[:, None] * c.view[2]) / 1e-6
    r, theta, phi, _ = spherical_coordinates(X)
    assert np.allclose(r, c.outer)
    start, end = c.longitudes
    in_cut = ((np.mod(phi - start, 2 * np.pi) < np.mod(end - start, 2 * np.pi))
              & (theta > c.colatitudes[0]) & (theta < c.colatitudes[1]))
    assert not np.any(in_cut)
    # and every coastline vertex that is seen is drawn, none that is not
    from cartopy.feature import NaturalEarthFeature
    from planetmodel.frames import from_geographic
    drawn = {(round(a, 3), round(b, 3)) for a, b in zip(x[seen], y[seen])}
    count = 0
    for geometry in NaturalEarthFeature("physical", "coastline", "110m").geometries():
        for line in getattr(geometry, "geoms", [geometry]):
            lon, lat = np.asarray(line.coords).T
            P = cartesian_points(c.outer, *from_geographic(lat, lon))
            px, py = c.project(P)
            for k, ok in enumerate(c.visible(P)):
                key = (round(px[k] * 1e-6, 3), round(py[k] * 1e-6, 3))
                assert (key in drawn) == bool(ok)
                count += ok
    assert count == seen.sum()
    land = cutaway(ax, c, surface="land", interfaces=False).get_array()
    colours = {tuple(np.round(p[:3] / p[:3].max(), 3)) for p in land[top]}
    want = {tuple(np.round(np.array(to_rgb(k)) / max(to_rgb(k)), 3))
            for k in (LAND, SEA)}
    assert colours == want                                # both, shaded
    plt.close(fig)
