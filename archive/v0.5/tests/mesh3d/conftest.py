"""Every meshing test runs inside a session, and leaves none behind."""
import numpy as np
import pytest

pytest.importorskip("gmsh", reason="needs the planetmodel[meshing] extra")

from planetmodel.mesh3d._session import is_active  # noqa: E402


@pytest.fixture(autouse=True)
def no_leaked_gmsh_session():
    """Assert the process is left as it was found.

    gmsh has one global model per process, so a test that leaks a
    session makes the *next* test fail for reasons of its own.  Checking
    both before and after localises that to the test that caused it.
    """
    assert not is_active(), "a gmsh session was already active on entry"
    yield
    assert not is_active(), "this test leaked a gmsh session"


def _write_relief_xyz(path, *, offset_km=0.0, amplitude_km=5.0,
                      spacing_deg=10.0):
    """Write the grid and return the path."""
    lon = np.arange(-180.0, 180.0, spacing_deg)
    lat = np.arange(90.0, -90.0 - 0.5 * spacing_deg, -spacing_deg)
    L, A = np.meshgrid(lon, lat, indexing="xy")
    t, p = np.deg2rad(90.0 - A), np.deg2rad(L)
    value = offset_km + amplitude_km * (0.5 * (3.0 * np.cos(t) ** 2 - 1.0)
                                        + np.sin(t) ** 2 * np.cos(2.0 * p))
    np.savetxt(path, np.c_[L.ravel(), A.ravel(), value.ravel()], fmt="%.6f")
    return path


@pytest.fixture
def write_relief_xyz():
    """A writer of `lon lat value` relief grids, in the layout CRUST ships.

    Degree two, zonal plus sectoral -- `P_2(cos t) + sin^2 t cos 2p` --
    about an offset, in kilometres, on a ten-degree grid with latitude
    running downwards through the file.  Both harmonics integrate to
    zero over the sphere, so the offset is the mean the reader will
    find, up to the grid it is sampled on; the tests that need that mean
    exactly compute it from the file rather than assume it, which is
    what a recipe's author has to do too.
    """
    return _write_relief_xyz
