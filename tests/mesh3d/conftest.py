"""Every meshing test runs inside a session, and leaves none behind.

The geometries here are unit sized and a few elements across: the
tests ask whether the pipeline is right, and no such question is
answered better by a large mesh.
"""
import numpy as np
import pytest

pytest.importorskip("gmsh", reason="needs the planetmodel[meshing] extra")

from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton  # noqa: E402
from planetmodel.mesh3d import UniformInterfaces  # noqa: E402
from planetmodel.mesh3d._session import is_active  # noqa: E402


@pytest.fixture(autouse=True)
def no_leaked_gmsh_session():
    """The process is left as it was found: gmsh has one global model."""
    assert not is_active(), "a gmsh session was already active on entry"
    yield
    assert not is_active(), "this test leaked a gmsh session"


#: Coarse sizing for unit geometries: a few elements across.
COARSE = UniformInterfaces(0.15, 0.3, 0.3)


def full_geometry():
    """Three shells of a unit ball, named."""
    return Geometry(Skeleton([0.0, 0.4, 0.8, 1.0]),
                    layer_names=["core", "mantle", "crust"],
                    interface_names=["cmb", "moho", "surface"])


def hollow_geometry():
    """Two shells of a unit ball with a hole at the centre."""
    return Geometry(Skeleton([0.5, 0.8, 1.0]), layer_names=["lower", "upper"],
                    interface_names=["inner", "mid", "outer"])


def flattening(amplitude=0.05):
    """h = -f r P2(cos theta): a degree-2 flattening, not zero on the surface."""
    def h(r, theta, phi):
        return -amplitude * r * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    return RadialStretch(CallableDisplacement(h, name="flattening"), rmax=1.0)


def confined_flattening(amplitude=0.05, *, top=1.0, ramp=0.2):
    """The same flattening tapered to zero at r = top, with kinks declared.

    Zero on and above r = top, so it is the identity on the outer
    boundary of a domain whose shells begin there.
    """
    def h(r, theta, phi):
        taper = np.clip((top - r) / ramp, 0.0, 1.0)
        return -amplitude * r * taper * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    return RadialStretch(CallableDisplacement(h, knots=[top - ramp, top],
                                             name="confined"), rmax=2.0 * top)
