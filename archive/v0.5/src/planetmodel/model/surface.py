"""surface.py -- a topography placed at a radius.

A Topography is a shape; a Surface is that shape *somewhere*.  The split
matters because the two are easy to confuse and the confusion is silent:
a crustal-thickness grid and a Moho radius are both "30 km fields", and
only one of them is a radius.  Keeping placement in a separate, explicit
step means the units question is asked once, where the answer is known.

    Surface(reference_radius, topography)
    surface.radius(theta, phi)   ->  reference_radius + topography

Arithmetic delegates to the topography, so `surface * 20` exaggerates
the relief while leaving the reference radius alone -- which is what a
visualisation exaggeration factor should do, and what scaling the radius
would not.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field as _dc_field

import numpy as np

from .topography import (AnalyticTopography, CentredTopography, Topography,
                         ZeroTopography, as_topography)

__all__ = ["Surface", "ellipsoid_surface", "spherical_surface"]


@dataclass(frozen=True)
class Surface:
    """A boundary shape: a reference radius plus relief about it.

    `reference_radius` is where the boundary sits on average, in the
    body's units; `topography` is the departure from it as a function
    of direction.
    The split is what the manifest records and what a blend rule reads.
    """

    reference_radius: float
    _: KW_ONLY
    topography: Topography = _dc_field(default_factory=ZeroTopography)
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate, and adapt a plain callable to the protocol."""
        if not np.isfinite(self.reference_radius):
            raise ValueError(
                f"reference_radius must be finite, got {self.reference_radius}")
        if self.reference_radius <= 0.0:
            raise ValueError(
                f"reference_radius must be positive, got {self.reference_radius}")
        object.__setattr__(self, "reference_radius", float(self.reference_radius))
        object.__setattr__(self, "topography", as_topography(self.topography))

    # -- evaluation ---------------------------------------------------------

    def radius(self, theta, phi):
        """The boundary radius in the given directions."""
        return self.reference_radius + np.asarray(
            self.topography(theta, phi), dtype=float)

    def height(self, theta, phi):
        """The relief alone, without the reference radius."""
        return np.asarray(self.topography(theta, phi), dtype=float)

    def gradient(self, theta, phi):
        """(d/dtheta, d/dphi) of the radius -- the topography's gradient.

        The reference radius is constant, so the two gradients coincide.
        """
        gt, gp = self.topography.gradient(theta, phi)
        return np.asarray(gt, dtype=float), np.asarray(gp, dtype=float)

    def mean_radius(self) -> float:
        """The area-weighted mean radius.

        Equal to reference_radius exactly when the relief has zero mean,
        which is the usual convention but not enforced: a surface built
        from raw topography may sit off-centre, and saying so is more
        useful than quietly recentring it.
        """
        return self.reference_radius + float(self.topography.mean())

    def is_centred(self, *, atol: float = 1.0) -> bool:
        """Whether the relief averages to zero, to within atol.

        `atol` is in the body's units; the default is a metre for an SI
        body and needs choosing for a non-dimensional one.
        """
        return abs(float(self.topography.mean())) <= atol

    def centred(self) -> "Surface":
        """The same boundary, with the relief's mean moved into the radius."""
        m = float(self.topography.mean())
        return Surface(self.reference_radius + m,
                       topography=CentredTopography(self.topography, m),
                       name=self.name)

    def bounds(self) -> tuple[float, float] | None:
        """The radial extent of the boundary, if the relief knows its own."""
        b = getattr(self.topography, "bounds", None)
        if b is None:
            return None
        lo, hi = b()
        return self.reference_radius + lo, self.reference_radius + hi

    # -- arithmetic ---------------------------------------------------------

    def __mul__(self, k) -> "Surface":
        """Exaggerate the relief, leaving the reference radius alone."""
        if isinstance(k, Surface) or callable(k):
            raise TypeError(
                "surface * surface is not defined: multiplying two boundary "
                "shapes has no geometric meaning. Scale by a number to "
                "exaggerate relief, or add them to combine.")
        return Surface(self.reference_radius, topography=self.topography * float(k),
                       name=self.name)

    __rmul__ = __mul__

    def __add__(self, other) -> "Surface":
        """Add relief: a Surface at the same radius, or a bare topography."""
        if isinstance(other, Surface):
            if other.reference_radius != self.reference_radius:
                raise ValueError(
                    "cannot add surfaces at different reference radii "
                    f"({self.reference_radius} and {other.reference_radius}); "
                    "combine the topographies and place the sum explicitly")
            return Surface(self.reference_radius,
                           topography=self.topography + other.topography,
                           name=self.name)
        if callable(other):
            return Surface(self.reference_radius, topography=self.topography + other,
                           name=self.name)
        return NotImplemented

    def with_topography(self, topography) -> "Surface":
        """The same placement, different relief."""
        return Surface(self.reference_radius, topography=topography, name=self.name)

    def at(self, reference_radius: float) -> "Surface":
        """The same relief, placed at a different radius."""
        return Surface(reference_radius, topography=self.topography, name=self.name)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return (f"Surface(r={self.reference_radius:.6g}{nm}, "
                f"{self.topography!r})")


def spherical_surface(radius: float, *, name: str | None = None) -> Surface:
    """A boundary with no relief."""
    return Surface(radius, topography=ZeroTopography(), name=name)


def ellipsoid_surface(a: float, b: float, c: float,
                      *, name: str | None = None) -> Surface:
    """A triaxial ellipsoid with semi-axes (a, b, c) along (x, y, z).

    The radius in the direction n is

        r = (n_x^2/a^2 + n_y^2/b^2 + n_z^2/c^2)^(-1/2),

    and the reference radius is the area-weighted mean of that, so the
    relief carried by the surface has zero mean -- an ellipsoid is
    described as a sphere plus a departure from it, which is what every
    consumer of a Surface expects.
    """
    for value, label in ((a, "a"), (b, "b"), (c, "c")):
        if value <= 0.0:
            raise ValueError(f"semi-axis {label} must be positive, got {value}")

    def radius_of(theta, phi):
        """The ellipsoid radius in the direction (theta, phi)."""
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        st, ct = np.sin(theta), np.cos(theta)
        nx, ny, nz = st * np.cos(phi), st * np.sin(phi), ct
        return 1.0 / np.sqrt((nx / a) ** 2 + (ny / b) ** 2 + (nz / c) ** 2)

    mean_r = AnalyticTopography(radius_of).mean()
    relief = AnalyticTopography(lambda t, p: radius_of(t, p) - mean_r,
                                name=name)
    return Surface(mean_r, topography=relief, name=name)
