"""The flat coordinate system: (depth, x, y), with the frame (e_up, e_x, e_y).

Points are right-handed Cartesian (x, y, z), z up and z = 0 the free
surface of the reference body; the layered coordinate is the depth
d = -z, positive downward, and (x, y) are the lateral ones, lengths
like it:

    points(d, x, y) = (x, y, -d),     split(X) = (-X_z, X_x, X_y, R).

The local frame is vertical first, R = [e_up | e_x | e_y], the same
everywhere (`LOCAL_FRAME`): a cyclic permutation of (e_x, e_y, e_z), so
right-handed, and like (e_r, e_theta, e_phi) on the sphere its first
vector points out of the body.  The frame (e_down, e_x, e_y) would be
left-handed.  The first vector therefore points up while depth
increases down (`coordinates.increases_outward` is False): a rank-1
field's first component is its upward component, minus its component
along increasing depth, and a derivative along the layered coordinate
is d/d depth = -d/dz, downward.  A transversely isotropic medium's axis,
the first local vector in `materials`, is the vertical, its sense
immaterial.

A change of length scale multiplies all three coordinates.  `FLAT` is
unbounded: its `lattice` covers a nominal square about the origin unless
it is given lateral `bounds`, which a lateral domain gives it
(`cartesian.Box.lattice`), and its contract probes lie in the unit
square, the contracts' checks being pointwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from ..coordinates import CARTESIAN, layer_samples

if TYPE_CHECKING:
    from ..skeleton import Skeleton

__all__ = ["FlatCoordinates", "FLAT", "LOCAL_FRAME"]

#: The local frame (e_up, e_x, e_y) as columns in Cartesian components,
#: read-only: vertical first, right-handed.
LOCAL_FRAME = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
LOCAL_FRAME.setflags(write=False)

#: Lateral bounds, ((x0, x1), (y0, y1)), each increasing.
type Bounds = tuple[tuple[float, float], tuple[float, float]]


def _floats(*arrays: ArrayLike) -> tuple[np.ndarray, ...]:
    """The arrays as float64, broadcast against one another."""
    return tuple(np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in arrays)))


@dataclass(frozen=True)
class FlatCoordinates:
    """(depth, x, y) with the frame (e_up, e_x, e_y): `FLAT`; see the module
    docstring.

    Every instance is the same system and compares equal to the others.
    """

    name: ClassVar[str] = "flat"
    frames: ClassVar[tuple[str, str]] = ("local", CARTESIAN)
    labels: ClassVar[tuple[str, str, str]] = ("depth", "x", "y")

    def points(self, c: ArrayLike, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """X = (x, y, -depth), of shape broadcast + (3,)."""
        depth, x, y = _floats(c, a, b)
        return np.stack([x, y, -depth], axis=-1)

    def split(self, X: ArrayLike, *, near: tuple[ArrayLike, ArrayLike] | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(depth, x, y, R) of points X of shape (..., 3); every point fixes
        its lateral coordinates, so `near` is never needed."""
        X = np.asarray(X, dtype=float)
        x, y = X[..., 0], X[..., 1]
        return -X[..., 2], x, y, self.frame(x, y)

    def frame(self, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """`LOCAL_FRAME` at every lateral point: shape broadcast + (3, 3)."""
        shape = np.broadcast(np.asarray(a), np.asarray(b)).shape
        return np.broadcast_to(LOCAL_FRAME, shape + (3, 3)).copy()

    def scaled(self, c: ArrayLike, a: ArrayLike, b: ArrayLike, k: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(k depth, k x, k y): all three are lengths."""
        k = float(k)
        return tuple(k * v for v in _floats(c, a, b))

    def lattice(self, skeleton: Skeleton, *, n_c: int = 8, n_a: int = 17,
                n_b: int = 17, bounds: Bounds | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A (depth, x, y) sample covering a skeleton over lateral `bounds`.

        `n_c` depths lie strictly inside every layer (`layer_samples`);
        `n_a` values of x and `n_b` of y cover the bounds, both ends
        included, or the square of half-width the skeleton's span about
        the origin where none are given.  Returns broadcastable arrays of
        shapes (n, 1, 1), (1, n_a, 1) and (1, 1, n_b), n being
        nlayers * n_c.
        """
        if bounds is None:
            s = skeleton.span
            bounds = ((-s, s), (-s, s))
        (x0, x1), (y0, y1) = bounds
        depth = layer_samples(skeleton, n_c)
        x = np.linspace(float(x0), float(x1), n_a)
        y = np.linspace(float(y0), float(y1), n_b)
        return depth[:, None, None], x[None, :, None], y[None, None, :]

    def probe(self, rng: np.random.Generator, n: int
              ) -> tuple[np.ndarray, np.ndarray]:
        """x then y uniform in (-1, 1), `n` of each, drawn in that order."""
        x = rng.uniform(-1.0, 1.0, n)
        y = rng.uniform(-1.0, 1.0, n)
        return x, y

    def __repr__(self) -> str:
        return "FLAT"


#: The flat system, (depth, x, y): the system of every field, geometry and
#: model of `planetmodel.cartesian`.
FLAT = FlatCoordinates()
