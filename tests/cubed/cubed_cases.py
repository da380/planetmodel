"""Geometries and sizings the cubed-sphere tests share: unit sized and a
few cells across."""
import numpy as np

from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton
from planetmodel.mesh3d import UniformInterfaces

#: A sizing asking for cells about 0.2 thick everywhere.
COARSE = UniformInterfaces(0.2, 0.2, 0.2)


def ball():
    """Three layers of a unit ball, named."""
    return Geometry(Skeleton([0.0, 0.4, 0.8, 1.0]),
                    layer_names=["core", "mantle", "crust"],
                    interface_names=["cmb", "moho", "surface"])


def shell():
    """Two layers of a unit ball with a hole at the centre."""
    return Geometry(Skeleton([0.5, 0.8, 1.0]), layer_names=["lower", "upper"],
                    interface_names=["inner", "mid", "outer"])


def flattening(amplitude=0.05, *, rmax=1.0):
    """h = -f r P2(cos theta): a degree-2 flattening, not zero on the surface."""
    def h(r, theta, phi):
        return -amplitude * r * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    return RadialStretch(CallableDisplacement(h, name="flattening"), rmax=rmax)


def equatorial_stretch(amplitude=0.05, *, rmax=1.0):
    """h = f r cos(2 phi) sin(theta)^2: a stretch that keeps the equatorial
    plane in itself, so a disc's mesh may carry it."""
    def h(r, theta, phi):
        return amplitude * r * np.cos(2.0 * phi) * np.sin(theta) ** 2
    return RadialStretch(CallableDisplacement(h, name="stretch"), rmax=rmax)
