"""_units.py -- the mesher's one conversion between body lengths and mesh lengths.

gmsh wants coordinates of order one: OCC's tolerances are absolute
(bounding boxes padded by 1e-7, geometric tolerance 1e-8), so a kernel
fed earth-scale coordinates is operating six orders of magnitude away
from where it is tuned.  The builder therefore normalises its
*geometry* -- and only its geometry -- and every division or
multiplication by the length scale goes through `MeshUnits`, so the
conversion is written once and read in one place.

The rule has two cases and no third:

* **SI body** (`body.scales.is_si`): `rref` is required, geometry is
  divided by it, and the mesh file's length unit is `rref` metres.
* **Non-dimensional body**: the body's own length scale is the answer,
  nothing is divided, and giving `rref` as well is refused -- two
  answers to one question.

Fields are never rescaled here.  A length-only rescale of a density is
an absurd hybrid, and the mesher reads no material fields anyway; a
consumer wanting a fully non-dimensional model calls
`body.nondimensionalised()` at the model layer.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["MeshUnits", "resolve_mesh_units", "GeometryScaledMapping"]

#: Resolved outer radii outside this range, in mesh units, put gmsh far
#: from the coordinate magnitudes its kernel tolerances are tuned for.
_GMSH_COMFORTABLE_RANGE = (1e-3, 1e4)


@dataclass(frozen=True)
class MeshUnits:
    """How mesh coordinates relate to the body's lengths and to metres.

    `divisor` is what the builder divides body lengths by (1.0 for an
    already non-dimensional body); `rref_m` is the SI size of one mesh
    length unit, which is what the manifest records.  `to_mesh` and
    `to_body` are the two directions of that one conversion, and are
    the only place it is written.
    """

    divisor: float
    rref_m: float

    def to_mesh(self, x):
        """A body length (or array of them) in mesh units."""
        return np.asarray(x, dtype=float) / self.divisor

    def to_body(self, x):
        """A mesh length (or array of them) in the body's units."""
        return np.asarray(x, dtype=float) * self.divisor

    @classmethod
    def identity(cls) -> "MeshUnits":
        """Mesh units that are the body's own: nothing is scaled."""
        return cls(divisor=1.0, rref_m=1.0)


def resolve_mesh_units(body, rref: float | None) -> MeshUnits:
    """Decide the mesh's length unit from the body's scales and `rref`."""
    L = body.scales.length
    if body.scales.is_si:
        if rref is None:
            raise ValueError(
                "the body is SI, so the mesher needs rref: the length, in "
                "metres, that becomes 1 in the mesh file")
        units = MeshUnits(divisor=float(rref), rref_m=float(rref))
    else:
        if rref is not None and not np.isclose(rref, L, rtol=1e-12):
            raise ValueError(
                f"the body is already non-dimensional (length scale {L:g} m) "
                f"and rref={rref:g} disagrees; give one answer, not two")
        units = MeshUnits(divisor=1.0, rref_m=L)

    outer = float(units.to_mesh(body.skeleton.boundaries[-1]))
    lo, hi = _GMSH_COMFORTABLE_RANGE
    if not lo <= outer <= hi:
        warnings.warn(
            f"the resolved outer radius is {outer:g} mesh units, outside "
            f"[{lo:g}, {hi:g}]; gmsh's kernel tolerances are absolute and it "
            "is tuned for coordinates of order one, so expect degraded "
            "robustness. Check rref, or non-dimensionalise the body.",
            stacklevel=2)
    return units


class GeometryScaledMapping:
    """A mapping conjugated into mesh coordinates: (1/L) o m o (L .).

    The adapter that lets an SI mapping displace non-dimensional mesh
    nodes.  F and J pass through *unchanged* because both are
    dimensionless -- the length scale cancels between numerator and
    denominator -- so validity checks and Jacobian guards on the scaled
    mapping are checks on the original.  Takes the `MeshUnits` of the
    build, or a bare divisor.
    """

    def __init__(self, mapping, units) -> None:
        self._m = mapping
        self._units = (units if isinstance(units, MeshUnits)
                       else MeshUnits(divisor=float(units), rref_m=float(units)))

    @property
    def units(self) -> MeshUnits:
        return self._units

    def __call__(self, X):
        return self._units.to_mesh(self._m(self._units.to_body(X)))

    def deformation_gradient(self, X, *, frame: str = "cartesian"):
        return self._m.deformation_gradient(self._units.to_body(X), frame=frame)

    def jacobian(self, X):
        return self._m.jacobian(self._units.to_body(X))

    def is_valid(self, *, X=None, sample=None):
        if X is not None:
            return self._m.is_valid(X=self._units.to_body(X))
        if sample is not None:
            r, theta, phi = sample
            return self._m.is_valid(
                sample=(self._units.to_body(r), theta, phi))
        return self._m.is_valid()

    @property
    def knots(self):
        """The underlying knots, in mesh units."""
        return tuple(float(self._units.to_mesh(k))
                     for k in getattr(getattr(self._m, "h", None), "knots", ()))

    def __repr__(self) -> str:
        return (f"GeometryScaledMapping({self._m!r}, "
                f"/{self._units.divisor:g})")
