"""CRUST-1.0 read off disk: the values, the grid, and what they interpolate to.

The mechanics of GriddedTopography -- bilinear weights, pole handling,
the longitude seam, area-weighted means -- are covered on synthetic
grids in tests/model/test_topography.py, where the right answer can be
written down.  What only the real files can tell us is whether *these*
files are read correctly: whether a value lands on the node the file put
it on when the latitudes run backwards, whether kilometres become
metres, and whether the two grids combine into a surface elevation the
right way round.

The one physical assertion is isostasy.  Thick crust stands high and
thin crust sits low, so the correlation between crustal thickness and
the elevation built from it must be positive -- which it would not be if
the depth-to-Moho sign were flipped, the arithmetic most likely to be
wrong here and the one a shape check would never catch.
"""
import numpy as np
import pytest

from planetmodel.model.topography import GriddedTopography

pytestmark = pytest.mark.data

MOHO = "tests/data/crust-1.0/depthtomoho.xyz"
THICKNESS = "tests/data/crust-1.0/crsthk.xyz"


def at(lat, lon):
    """(colatitude, longitude) in radians, from degrees north and east."""
    return np.deg2rad(90.0 - lat), np.deg2rad(lon)


@pytest.fixture(scope="module")
def moho():
    return GriddedTopography.from_xyz(MOHO, scale=1e3)


@pytest.fixture(scope="module")
def thickness():
    return GriddedTopography.from_xyz(THICKNESS, scale=1e3)


@pytest.fixture(scope="module")
def elevation(moho, thickness):
    """Depth to the Moho plus the crust above it: the surface."""
    return thickness + moho


# ------------------------------------------------------------- the grid

def test_the_files_are_a_one_degree_global_grid(moho, thickness):
    for grid in (moho, thickness):
        assert grid.values.shape == (180, 360)
        assert grid.lons[0] == -179.5 and grid.lons[-1] == 179.5
        assert grid.lats[0] == -89.5 and grid.lats[-1] == 89.5
        assert np.allclose(np.diff(grid.lons), 1.0)
        assert np.allclose(np.diff(grid.lats), 1.0)


def test_values_land_on_the_nodes_the_file_puts_them_on(moho, thickness):
    """The files run latitude *downward*; a reader that assumed otherwise
    would mirror the planet without changing a single value."""
    # First line of each file: lon -179.5, lat 89.5.
    assert moho(*at(89.5, -179.5)) == pytest.approx(-11750.0)
    assert thickness(*at(89.5, -179.5)) == pytest.approx(8060.0)
    # Last line of depthtomoho.xyz: lon 179.5, lat -89.5, -36.14 km.
    assert moho(*at(-89.5, 179.5)) == pytest.approx(-36140.0)


def test_kilometres_become_metres(moho):
    """The model layer is SI; the files are not."""
    raw = GriddedTopography.from_xyz(MOHO)
    assert raw(*at(89.5, -179.5)) == pytest.approx(-11.75)
    assert moho(*at(89.5, -179.5)) == pytest.approx(-11.75 * 1e3)


def test_the_mean_is_area_weighted(moho):
    """And on this data it matters by a kilometre and a half.

    A lon-lat grid packs its cells together towards the poles, so an
    unweighted average counts polar cells far too heavily.  Greenland
    and Antarctica sit there under thick crust, which drags the
    unweighted mean 1.5 km deeper than the truth -- and that mean is
    the radius the Moho gets placed at.  This is the C++ behaviour the
    plan replaces, measured.
    """
    lat = np.deg2rad(moho.lats)[:, None]
    weighted = float((moho.values * np.cos(lat)).sum() / (np.cos(lat).sum() * 360))
    assert moho.mean() == pytest.approx(weighted, rel=1e-3)
    assert moho.mean() == pytest.approx(-21421.0, abs=50.0)

    unweighted = float(moho.values.mean())
    assert unweighted == pytest.approx(-22903.0, abs=50.0)
    assert moho.mean() - unweighted == pytest.approx(1482.0, abs=50.0)


# ---------------------------------------------------------- the values

def test_the_moho_is_everywhere_below_sea_level(moho):
    """Depth to the Moho, so negative everywhere, and deepest under Tibet."""
    assert moho.values.max() < 0.0
    assert moho.values.min() == pytest.approx(-74810.0, abs=10.0)
    deepest = np.unravel_index(np.argmin(moho.values), moho.values.shape)
    lat, lon = moho.lats[deepest[0]], moho.lons[deepest[1]]
    assert 25.0 < lat < 40.0 and 75.0 < lon < 100.0     # the Tibetan plateau


def test_thick_crust_stands_high(thickness, elevation):
    """Isostasy, and the check that the two grids combine the right way.

    A flipped sign on the Moho depth would leave every shape check happy
    and turn this correlation negative.
    """
    lat = np.deg2rad(thickness.lats)[:, None]
    w = np.broadcast_to(np.cos(lat), thickness.values.shape).ravel()
    x, y = thickness.values.ravel(), elevation.values.ravel()
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    r = cov / np.sqrt(np.average((x - mx) ** 2, weights=w)
                      * np.average((y - my) ** 2, weights=w))
    assert r > 0.5, f"crustal thickness and elevation correlate at {r:.3f}"


def test_the_elevation_spans_ocean_floor_to_plateau(elevation):
    assert -8.0e3 < elevation.values.min() < -3.0e3
    assert 3.0e3 < elevation.values.max() < 6.0e3
    assert elevation.mean() < 0.0                       # most of it is ocean


def test_the_crust_has_a_sensible_mean_thickness(moho, elevation):
    """19 km on average, and thicker than the ocean is deep everywhere."""
    assert (elevation.mean() - moho.mean()) == pytest.approx(19.0e3, abs=1.0e3)
    assert np.all(elevation.values - moho.values > 0.0)


# --------------------------------------------------------- interpolation

def test_interpolation_stays_inside_the_cell_it_came_from(moho):
    """Bilinear cannot overshoot, so a midpoint lies between its corners."""
    i, j = 100, 200
    lat0, lat1 = moho.lats[i], moho.lats[i + 1]
    lon0, lon1 = moho.lons[j], moho.lons[j + 1]
    corners = moho.values[i:i + 2, j:j + 2]
    middle = float(moho(*at(0.5 * (lat0 + lat1), 0.5 * (lon0 + lon1))))
    assert corners.min() <= middle <= corners.max()
    assert middle == pytest.approx(corners.mean(), rel=1e-9)


def test_the_field_is_finite_and_bounded_everywhere(moho):
    """Including the poles and the dateline, where a grid reader fails."""
    theta = np.linspace(0.0, np.pi, 181)
    phi = np.linspace(-np.pi, np.pi, 361)
    values = moho(theta[:, None], phi[None, :])
    assert np.all(np.isfinite(values))
    assert values.min() >= moho.values.min() - 1e-9
    assert values.max() <= moho.values.max() + 1e-9


def test_the_dateline_is_not_a_seam(moho):
    """The grid stops at 179.5 but the sphere does not."""
    east = moho(*at(10.0, 179.999))
    west = moho(*at(10.0, -179.999))
    assert float(east) == pytest.approx(float(west), rel=1e-3)


def test_the_poles_are_single_valued(moho):
    """One point on the sphere cannot hold 360 different depths."""
    north = [float(moho(*at(90.0, lon))) for lon in (-179.0, -60.0, 0.0, 120.0)]
    assert north == pytest.approx([north[0]] * 4)
