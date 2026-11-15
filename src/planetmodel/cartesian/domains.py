"""Lateral domains: the box's (x, y) and the rectangle's x.

A finite lateral domain stands for an unbounded one: it is taken large
enough that its sides do not matter, so it carries no boundary physics,
only its extent.  `bounds` is the domain proper and `padding` a
periodic margin added on each side of each axis, zero by default; the
domain with its padding, `extent`, is the period of the lateral
directions and the body a geometry over the domain holds, so for a
model, whose box is its period, the padding is none.  `lengths` are the
periods.  A point is inside where its lateral coordinates lie in the
bounds (`contains`).

`Box` bounds x and y: the lateral domain of a three-dimensional body.
`Rectangle` bounds x alone: in the flat system, whose points have three
coordinates, a geometry over it is the section of a body invariant
along y, every y inside it, and its mapping is held to that invariance.
Each samples its body for the checks of a geometry (`lattice`, the
depths `FLAT.lattice` lays) and is rescaled with it (`scaled`).
Nothing here knows about units: bounds are numbers and every tolerance
is relative to a length.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .coordinates import FLAT

if TYPE_CHECKING:
    from ..skeleton import Skeleton

__all__ = ["Box", "Rectangle", "Domain"]


def _interval(pair: Sequence[float], axis: str) -> tuple[float, float]:
    """A finite increasing pair of floats, refused otherwise, naming the axis."""
    try:
        lo, hi = (float(v) for v in pair)
    except (TypeError, ValueError):
        raise ValueError(f"the bounds of {axis} are a (lower, upper) pair, "
                         f"got {pair!r}") from None
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"the bounds of {axis} must be finite and increase, "
                         f"got ({lo:g}, {hi:g})")
    return lo, hi


def _margin(value: float, axis: str) -> float:
    """A finite non-negative padding, refused otherwise, naming the axis."""
    p = float(value)
    if not (np.isfinite(p) and p >= 0.0):
        raise ValueError(f"the padding of {axis} must be finite and non-negative, "
                         f"got {p:g}")
    return p


def _inside(v: np.ndarray, bounds: tuple[float, float], rtol: float) -> np.ndarray:
    """Whether each value lies in the bounds, to rtol of their width."""
    lo, hi = bounds
    tol = rtol * (hi - lo)
    return (v >= lo - tol) & (v <= hi + tol)


class Box:
    """x in `bounds[0]` and y in `bounds[1]`, with `padding` on each side:
    one length for both axes, or one each; see the module docstring.
    Two boxes are equal when their bounds and padding are."""

    #: The lateral axes the domain bounds; the body is invariant along any other.
    axes: ClassVar[tuple[str, ...]] = ("x", "y")

    def __init__(self, bounds: Sequence[Sequence[float]], *,
                 padding: float | Sequence[float] = 0.0) -> None:
        pairs = tuple(bounds)
        if len(pairs) != 2:
            raise ValueError(f"a box takes the bounds of x and of y, got {bounds!r}")
        self._bounds = (_interval(pairs[0], "x"), _interval(pairs[1], "y"))
        pads = (padding, padding) if np.ndim(padding) == 0 else tuple(padding)
        if len(pads) != 2:
            raise ValueError(f"a box takes one padding or two, got {padding!r}")
        self._padding = (_margin(pads[0], "x"), _margin(pads[1], "y"))

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """((x0, x1), (y0, y1)): the domain proper."""
        return self._bounds

    @property
    def padding(self) -> tuple[float, float]:
        """The periodic margin on each side, of x and of y."""
        return self._padding

    @property
    def extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The bounds widened by the padding on each side: the period."""
        return tuple((lo - p, hi + p) for (lo, hi), p in zip(self._bounds,
                                                           self._padding))

    @property
    def lengths(self) -> tuple[float, float]:
        """The periods of x and y: each extent's width."""
        return tuple(hi - lo for lo, hi in self.extent)

    def contains(self, x: ArrayLike, y: ArrayLike, *, rtol: float = 1e-9) -> np.ndarray:
        """Whether each broadcast point (x, y) lies in the bounds, to `rtol`
        of their widths; the padding is outside."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        return _inside(x, self._bounds[0], rtol) & _inside(y, self._bounds[1], rtol)

    def lattice(self, skeleton: Skeleton, *, n_c: int = 8, n_a: int = 17,
                n_b: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`FLAT.lattice` of the skeleton over the extent: `n_c` depths in
        every layer, `n_a` values of x and `n_b` of y."""
        return FLAT.lattice(skeleton, n_c=n_c, n_a=n_a, n_b=n_b, bounds=self.extent)

    def scaled(self, k: float) -> "Box":
        """The box with every length multiplied by k > 0."""
        k = float(k)
        if not k > 0.0:
            raise ValueError(f"the scale factor must be positive, got {k}")
        return Box([(k * lo, k * hi) for lo, hi in self._bounds],
                   padding=[k * p for p in self._padding])

    def __eq__(self, other: object) -> bool:
        return (type(other) is Box and self._bounds == other._bounds
                and self._padding == other._padding)

    def __hash__(self) -> int:
        return hash((Box, self._bounds, self._padding))

    def __repr__(self) -> str:
        (x0, x1), (y0, y1) = self._bounds
        pad = "" if self._padding == (0.0, 0.0) else (
            f", padding=({self._padding[0]:g}, {self._padding[1]:g})")
        return f"Box((({x0:g}, {x1:g}), ({y0:g}, {y1:g})){pad})"


class Rectangle:
    """x in `bounds`, with `padding` on each side, every y inside: the
    lateral domain of the section of a body invariant along y; see the
    module docstring.  Two rectangles are equal when their bounds and
    padding are."""

    #: The lateral axes the domain bounds; the body is invariant along y.
    axes: ClassVar[tuple[str, ...]] = ("x",)

    def __init__(self, bounds: Sequence[float], *, padding: float = 0.0) -> None:
        self._bounds = _interval(bounds, "x")
        self._padding = _margin(padding, "x")

    @property
    def bounds(self) -> tuple[float, float]:
        """(x0, x1): the domain proper."""
        return self._bounds

    @property
    def padding(self) -> float:
        """The periodic margin of x on each side."""
        return self._padding

    @property
    def extent(self) -> tuple[float, float]:
        """The bounds widened by the padding on each side: the period."""
        lo, hi = self._bounds
        return lo - self._padding, hi + self._padding

    @property
    def length(self) -> float:
        """The period of x: the extent's width."""
        lo, hi = self.extent
        return hi - lo

    def contains(self, x: ArrayLike, y: ArrayLike, *, rtol: float = 1e-9) -> np.ndarray:
        """Whether each broadcast point (x, y) has x in the bounds, to `rtol`
        of their width; every y is inside."""
        x, _ = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        return _inside(x, self._bounds, rtol)

    def lattice(self, skeleton: Skeleton, *, n_c: int = 8, n_a: int = 17,
                n_b: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`FLAT.lattice` of the skeleton: `n_c` depths in every layer, `n_a`
        values of x over the extent, and `n_b` of y over a stretch of the
        length about y = 0, the body being the same at every y."""
        half = 0.5 * self.length
        lo, hi = self.extent
        return FLAT.lattice(skeleton, n_c=n_c, n_a=n_a, n_b=n_b,
                            bounds=((lo, hi), (-half, half)))

    def scaled(self, k: float) -> "Rectangle":
        """The rectangle with every length multiplied by k > 0."""
        k = float(k)
        if not k > 0.0:
            raise ValueError(f"the scale factor must be positive, got {k}")
        lo, hi = self._bounds
        return Rectangle((k * lo, k * hi), padding=k * self._padding)

    def __eq__(self, other: object) -> bool:
        return (type(other) is Rectangle and self._bounds == other._bounds
                and self._padding == other._padding)

    def __hash__(self) -> int:
        return hash((Rectangle, self._bounds, self._padding))

    def __repr__(self) -> str:
        lo, hi = self._bounds
        pad = "" if self._padding == 0.0 else f", padding={self._padding:g}"
        return f"Rectangle(({lo:g}, {hi:g}){pad})"


#: A lateral domain: what a flat geometry is built over.
type Domain = Box | Rectangle
