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

Every one of these is held per axis, in the order of `axes`: `bounds`
and `extent` a (lower, upper) pair for each axis, `padding` and
`lengths` a number for each.  A domain of two axes and a domain of one
so answer in one form, and what is built over a domain runs over its
axes (`DomainBase`).

`Box` bounds x and y: the lateral domain of a three-dimensional body.
`Rectangle` bounds x alone: in the flat system, whose points have three
coordinates, a geometry over it is the cross-section of a body invariant
along y, every y inside it, and its mapping is held to that invariance.
Each samples its body for the checks of a geometry (`lattice`, the
depths `FLAT.lattice` lays) and is rescaled with it (`scaled`).
Nothing here knows about units: bounds are numbers and every tolerance
is relative to a length.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike

from .coordinates import FLAT

if TYPE_CHECKING:
    from ..skeleton import Skeleton

__all__ = ["DomainBase", "Box", "Rectangle", "Domain"]


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


class DomainBase:
    """What every lateral domain holds and answers, per axis of `axes`.

    `bounds` is one (lower, upper) pair for each axis and `padding` one
    length for every axis, or one each.  A leaf names its `axes`, among
    the lateral coordinates of `FLAT` and in their order, and lays its
    `lattice`; two domains are equal when they are of one kind and their
    bounds and padding are.
    """

    #: The lateral axes the domain bounds; the body is invariant along any other.
    axes: ClassVar[tuple[str, ...]]

    def __init__(self, bounds: Sequence[Sequence[float]], *,
                 padding: float | Sequence[float] = 0.0) -> None:
        axes = self.axes
        kind = type(self).__name__.lower()
        try:
            pairs = tuple(bounds)
        except TypeError:
            pairs = ()
        if len(pairs) != len(axes):
            raise ValueError(f"a {kind} takes the bounds of {' and of '.join(axes)}, "
                             f"got {bounds!r}")
        pads = (padding,) * len(axes) if np.ndim(padding) == 0 else tuple(padding)
        if len(pads) != len(axes):
            raise ValueError(f"a {kind} takes one padding, or one per axis {axes}, "
                             f"got {padding!r}")
        self._bounds = tuple(_interval(pair, axis) for pair, axis in zip(pairs, axes))
        self._padding = tuple(_margin(p, axis) for p, axis in zip(pads, axes))

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """The domain proper: one (lower, upper) pair per axis."""
        return self._bounds

    @property
    def padding(self) -> tuple[float, ...]:
        """The periodic margin on each side of each axis: one length per axis."""
        return self._padding

    @property
    def extent(self) -> tuple[tuple[float, float], ...]:
        """The bounds widened by the padding on each side, one (lower, upper)
        pair per axis: the period."""
        return tuple((lo - p, hi + p) for (lo, hi), p in zip(self._bounds,
                                                           self._padding))

    @property
    def lengths(self) -> tuple[float, ...]:
        """The periods, one per axis: each extent's width."""
        return tuple(hi - lo for lo, hi in self.extent)

    def contains(self, x: ArrayLike, y: ArrayLike, *, rtol: float = 1e-9) -> np.ndarray:
        """Whether each broadcast point (x, y) lies in the bounds, to `rtol`
        of their widths; the padding is outside, and every value of a
        coordinate the domain does not bound is inside."""
        lateral = dict(zip(FLAT.labels[1:], np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float))))
        return np.logical_and.reduce([_inside(lateral[axis], pair, rtol)
                                      for axis, pair in zip(self.axes, self._bounds)])

    def lattice(self, skeleton: Skeleton, *, n_c: int = 8, n_a: int = 17,
                n_b: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`FLAT.lattice` of the skeleton over the body: `n_c` depths in
        every layer, `n_a` values of x and `n_b` of y."""
        raise NotImplementedError

    def scaled(self, k: float) -> Self:
        """The domain with every length multiplied by k > 0."""
        k = float(k)
        if not k > 0.0:
            raise ValueError(f"the scale factor must be positive, got {k}")
        return type(self)([(k * lo, k * hi) for lo, hi in self._bounds],
                          padding=[k * p for p in self._padding])

    def __eq__(self, other: object) -> bool:
        return (type(other) is type(self) and self._bounds == other._bounds
                and self._padding == other._padding)

    def __hash__(self) -> int:
        return hash((type(self), self._bounds, self._padding))


class Box(DomainBase):
    """x in `bounds[0]` and y in `bounds[1]`, with `padding` on each side:
    one length for both axes, or one each; see the module docstring.
    Two boxes are equal when their bounds and padding are."""

    axes: ClassVar[tuple[str, ...]] = ("x", "y")

    def lattice(self, skeleton: Skeleton, *, n_c: int = 8, n_a: int = 17,
                n_b: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`FLAT.lattice` of the skeleton over the extent: `n_c` depths in
        every layer, `n_a` values of x and `n_b` of y."""
        return FLAT.lattice(skeleton, n_c=n_c, n_a=n_a, n_b=n_b, bounds=self.extent)

    def __repr__(self) -> str:
        (x0, x1), (y0, y1) = self._bounds
        pad = "" if self._padding == (0.0, 0.0) else (
            f", padding=({self._padding[0]:g}, {self._padding[1]:g})")
        return f"Box((({x0:g}, {x1:g}), ({y0:g}, {y1:g})){pad})"


class Rectangle(DomainBase):
    """x in `bounds[0]`, with `padding` on each side, every y inside: the
    lateral domain of the cross-section of a body invariant along y; see
    the module docstring.  It is given the pair (x0, x1) and one padding,
    or each as it holds them, ((x0, x1),) and (padding,).  Two rectangles
    are equal when their bounds and padding are."""

    axes: ClassVar[tuple[str, ...]] = ("x",)

    def __init__(self, bounds: Sequence[float] | Sequence[Sequence[float]], *,
                 padding: float | Sequence[float] = 0.0) -> None:
        try:
            held = len(bounds) == 1                  # ((x0, x1),): the one pair
        except TypeError:
            held = False
        super().__init__(bounds if held else (bounds,), padding=padding)

    def lattice(self, skeleton: Skeleton, *, n_c: int = 8, n_a: int = 17,
                n_b: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`FLAT.lattice` of the skeleton: `n_c` depths in every layer, `n_a`
        values of x over the extent, and `n_b` of y over a stretch of the
        period about y = 0, the body being the same at every y."""
        (x,), (length,) = self.extent, self.lengths
        return FLAT.lattice(skeleton, n_c=n_c, n_a=n_a, n_b=n_b,
                            bounds=(x, (-0.5 * length, 0.5 * length)))

    def __repr__(self) -> str:
        (lo, hi), (p,) = self._bounds[0], self._padding
        pad = "" if p == 0.0 else f", padding={p:g}"
        return f"Rectangle(({lo:g}, {hi:g}){pad})"


#: A lateral domain: what a flat geometry is built over.
type Domain = Box | Rectangle
