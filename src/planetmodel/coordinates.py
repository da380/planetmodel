"""Coordinate systems: what a field's three coordinates mean in space.

A field, a geometry's checks and the contracts speak three coordinates
(c, a, b): c the layered coordinate, the one a skeleton's boundaries
and a field's interval are values of, and (a, b) the lateral ones.  A
`Coordinates` object says what they mean:

    points(c, a, b)      the Cartesian points X, of shape broadcast + (3,)
    split(X)             (c, a, b, R), the inverse, with the local frame there
    frame(a, b)          R, the local orthonormal frame, basis vectors as
                         columns in Cartesian components
    scaled(c, a, b, k)   the coordinates of the point k X, for k > 0
    lattice(skeleton)    a sample of (c, a, b) covering every layer, for
                         a mapping's validity and continuity
    probe(rng, n)        lateral coordinates at which the contracts probe

and `frames`, the names of the frames a field presents its components
in: its local frame first, whose basis is the columns of R, then
`CARTESIAN`; and `labels`, the names of (c, a, b) in messages.  Local
components become Cartesian by a factor of R on every slot.  The frame
depends on the lateral coordinates alone, as it does wherever a body is
layered along c, and its first vector points out of the body (up).
Nothing asks c to increase along that vector, a body to have a centre,
or the lateral coordinates to have poles: those are the sphere's.
`increases_outward` reads from the points and the frame which way c
runs, and `layer_samples` is the sample of c inside every layer that a
lattice is built on.

`SPHERICAL` is (r, theta, phi), theta the colatitude and phi the
longitude in radians, with the local frame (e_r, e_theta, e_phi), built
on `planetmodel.frames`; its frames are ("spherical", "cartesian"), its
lattice is `mapping.validity_lattice`'s and its probes keep 0.05 rad off
the poles.  It is the system of every field, geometry and model that
does not state another.  The flat system, (depth, x, y), is
`planetmodel.cartesian.FLAT`.

Fields of two systems are never combined: `common_coordinates` names
the systems of a mix and refuses it, and `coordinates_of` reads the
system of a field, a geometry or a model, spherical for one that states
none.  What is defined in one system alone (gravity from the mass inside
a radius, a mean over the sphere) refuses an object of another by
`require_coordinates`, naming what it is and both systems.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from .frames import cartesian_points, spherical_coordinates, spherical_frame

if TYPE_CHECKING:
    from .skeleton import Skeleton

__all__ = ["Coordinates", "SphericalCoordinates", "SPHERICAL", "CARTESIAN",
           "coordinates_of", "common_coordinates", "checked_coordinates",
           "require_coordinates", "resolve_frame", "increases_outward",
           "layer_samples"]

#: The name of the Cartesian frame, the second of every system's `frames`.
CARTESIAN = "cartesian"


@runtime_checkable
class Coordinates(Protocol):
    """What a coordinate system says; see the module docstring.

    `name` names the system in messages; `frames` is (the local frame's
    name, `CARTESIAN`); `labels` names (c, a, b) in messages.
    """

    name: str
    frames: tuple[str, str]
    labels: tuple[str, str, str]

    def points(self, c: ArrayLike, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """The Cartesian points of the broadcast coordinates, shape
        broadcast + (3,)."""
        ...

    def split(self, X: ArrayLike, *, near: tuple[ArrayLike, ArrayLike] | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(c, a, b, R) of Cartesian points X of shape (..., 3), R the local
        frame there; where X leaves the lateral coordinates undefined,
        those of `near`, broadcasting against the points, when given."""
        ...

    def frame(self, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """The local frame at the broadcast lateral coordinates, basis
        vectors as columns: shape broadcast + (3, 3)."""
        ...

    def scaled(self, c: ArrayLike, a: ArrayLike, b: ArrayLike, k: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The coordinates of the point k X, for X at (c, a, b) and k > 0."""
        ...

    def lattice(self, skeleton: Skeleton, *, n_c: int = 8, n_a: int = 25,
                n_b: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(c, a, b) broadcastable arrays covering every layer of `skeleton`,
        `n_c` values strictly inside each layer and `n_a` by `n_b` lateral
        ones, each system holding its own defaults."""
        ...

    def probe(self, rng: np.random.Generator, n: int
              ) -> tuple[np.ndarray, np.ndarray]:
        """`n` lateral coordinates (a, b) drawn from `rng`, where the
        contracts probe a field."""
        ...


@dataclass(frozen=True)
class SphericalCoordinates:
    """(r, theta, phi) with the frame (e_r, e_theta, e_phi): `SPHERICAL`.

    Every instance is the same system and compares equal to the others.
    """

    name: ClassVar[str] = "spherical"
    frames: ClassVar[tuple[str, str]] = ("spherical", CARTESIAN)
    labels: ClassVar[tuple[str, str, str]] = ("radius", "theta", "phi")

    def points(self, c: ArrayLike, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """X = r e_r(theta, phi): `frames.cartesian_points`."""
        return cartesian_points(c, a, b)

    def split(self, X: ArrayLike, *, near: tuple[ArrayLike, ArrayLike] | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(r, theta, phi, R): `frames.spherical_coordinates`.

        On the polar axis, where X fixes no longitude, and at the centre,
        where it fixes neither angle, the angles of `near` are taken when
        it is given, and the frame is theirs.
        """
        if near is None:
            return spherical_coordinates(X)
        X = np.asarray(X, dtype=float)
        r, theta, phi, _ = spherical_coordinates(X)
        on_axis = np.hypot(X[..., 0], X[..., 1]) == 0.0
        phi = np.where(on_axis, near[1], phi)
        theta = np.where(np.linalg.norm(X, axis=-1) == 0.0, near[0], theta)
        return r, theta, phi, spherical_frame(theta, phi)

    def frame(self, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """(e_r, e_theta, e_phi) as columns: `frames.spherical_frame`."""
        return spherical_frame(a, b)

    def scaled(self, c: ArrayLike, a: ArrayLike, b: ArrayLike, k: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(k r, theta, phi): the angles are those of the direction, which
        a change of length scale keeps."""
        return (np.asarray(c, dtype=float) * float(k), np.asarray(a, dtype=float),
                np.asarray(b, dtype=float))

    def lattice(self, skeleton: Skeleton, *, n_c: int = 8, n_a: int = 25,
                n_b: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A (r, theta, phi) sample covering a skeleton.

        `n_c` radii are laid strictly inside every layer of the skeleton,
        so a thin layer is sampled as densely as a thick one, and the
        centre is sampled where the skeleton holds it; `n_a` colatitudes
        cover [0, pi] including both poles and `n_b` longitudes cover
        [-pi, pi).  Returns broadcastable arrays of shapes (n, 1, 1),
        (1, n_a, 1) and (1, 1, n_b), n being nlayers * n_c, one more for
        the centre.
        """
        centre = [np.zeros(1)] if skeleton.boundaries[0] == 0.0 else []
        r = np.concatenate(centre + [layer_samples(skeleton, n_c)])
        theta = np.linspace(0.0, np.pi, n_a)
        phi = np.linspace(-np.pi, np.pi, n_b, endpoint=False)
        return (r[:, None, None], theta[None, :, None], phi[None, None, :])

    def probe(self, rng: np.random.Generator, n: int
              ) -> tuple[np.ndarray, np.ndarray]:
        """Colatitudes uniform in (0.05, pi - 0.05), then longitudes
        uniform in (-pi, pi), `n` of each, drawn in that order."""
        theta = rng.uniform(0.05, np.pi - 0.05, n)
        phi = rng.uniform(-np.pi, np.pi, n)
        return theta, phi

    def __repr__(self) -> str:
        return "SPHERICAL"


#: The spherical system, (r, theta, phi): the default of every field and
#: geometry.
SPHERICAL = SphericalCoordinates()


def coordinates_of(obj: object) -> Coordinates:
    """The coordinate system `obj` states as `coordinates`, `SPHERICAL`
    where it states none (a field written to the `Field` protocol alone);
    a `coordinates` that is not a `Coordinates` is refused."""
    system = getattr(obj, "coordinates", SPHERICAL)
    if system is SPHERICAL or isinstance(system, Coordinates):
        return system
    raise TypeError(f"{obj!r} states coordinates {system!r}, which are not a "
                    "Coordinates")


def common_coordinates(objects: Iterable[object]) -> Coordinates:
    """The one coordinate system of `objects` (at least one), each read by
    `coordinates_of`; a mix of systems is refused, naming them."""
    systems = [coordinates_of(o) for o in objects]
    if not systems:
        raise ValueError("no fields to take a coordinate system from")
    first = systems[0]
    for other in systems[1:]:
        if other != first:
            raise ValueError(
                f"fields in different coordinate systems are not combined: "
                f"{first.name!r} and {other.name!r}")
    return first


def require_coordinates(obj: object, system: Coordinates, what: str) -> None:
    """Refuse `obj` (a field, a geometry or a model, read by
    `coordinates_of`) unless it is in `system`, naming `what` is defined
    in that system alone and the system `obj` is in."""
    have = coordinates_of(obj)
    if have != system:
        raise ValueError(f"{what} is defined in the coordinates {system.name!r} "
                         f"alone; this {type(obj).__name__} is in {have.name!r}")


def checked_coordinates(coordinates: object) -> Coordinates:
    """`coordinates` itself when it is a `Coordinates`; TypeError otherwise."""
    if not isinstance(coordinates, Coordinates):
        raise TypeError(f"expected a Coordinates, got {type(coordinates).__name__}")
    return coordinates


def resolve_frame(coordinates: Coordinates, frame: str | None) -> str:
    """The frame `frame` names among the system's `frames`, the local one
    for None; any other name is refused, listing the system's frames."""
    if frame is None:
        return coordinates.frames[0]
    if frame not in coordinates.frames:
        raise ValueError(
            f"unknown frame {frame!r}: expected one of {coordinates.frames}")
    return frame


def increases_outward(coordinates: Coordinates) -> bool:
    """Whether the layered coordinate c increases along the first vector of
    the local frame, the one pointing out of the body: True on the sphere,
    whose c is the radius, False where c is a depth.  Read from the
    system's points and frame across c = 1 at its first probe."""
    a, b = coordinates.probe(np.random.default_rng(0), 1)
    step = (np.asarray(coordinates.points(1.0 + 1e-3, a, b), dtype=float)
            - np.asarray(coordinates.points(1.0 - 1e-3, a, b), dtype=float))
    first = np.asarray(coordinates.frame(a, b), dtype=float)[..., :, 0]
    return bool(np.sum(step * first) > 0.0)


def layer_samples(skeleton: Skeleton, n: int) -> np.ndarray:
    """`n` values of the layered coordinate strictly inside every layer of
    `skeleton`, inset by 1e-6 of the layer's width from its ends, in
    increasing order: a thin layer is sampled as densely as a thick one."""
    b = np.asarray(skeleton.boundaries, dtype=float)
    samples = []
    for lo, hi in zip(b[:-1], b[1:]):
        inset = 1e-6 * (hi - lo)
        samples.append(np.linspace(lo + inset, hi - inset, n))
    return np.concatenate(samples)
