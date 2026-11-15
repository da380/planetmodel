"""The horizontal grid of a box or a section: the flat counterpart of the
sphere's angular grid.

A `Grid` is the product of equally spaced nodes over the period of a
lateral domain: over a `Box` of extent [x0, x0 + Lx] x [y0, y0 + Ly] and
`shape` (nx, ny) the nodes

    x_i = x0 + i Lx / nx,    y_j = y0 + j Ly / ny,

and over a `Rectangle` of extent [x0, x0 + Lx] and `shape` (nx,) the x_i
alone, the section's one lateral coordinate.  The extent is the domain's
bounds widened by its padding (none for a model, whose box is its
period), and the nodes are those of `fourier.Transform(shape, domain)`,
which the grid holds as `transform` and asks for them: a grid and the
Fourier series on it are one source.  The last node of an axis is one
step short of the far end, the period closing it on the first.

**The band** is the shape: the grid holds exactly the components of its
packing (`fourier.Packing`), wavenumbers |k_i| <= n_i / 2 along each
axis, the Nyquist cosine of an even axis once.  A series of that band is
recovered exactly by analysis on the grid, and a Fourier field is read
on another grid's nodes through the fast transform where the other grid
holds every one of its modes.

**The mean** of values at the nodes over the period is their arithmetic
mean (`mean`), the periodic trapezoid, exact for every series the grid
holds.  It is the mean of each lateral rule's interpolant as well
(`cartesian.lateral`), so that a gridded field's mean and its rule
agree.

**A section** (a grid over a `Rectangle`) has one axis: the body is the
same at every y, a value at node x_i is the body's at every y, and the
grid's `nodes` put y at zero.  Where the grid is asked for the points of
its transform (`points`), y is not read.

`nodes` are the two lateral coordinates of every node, broadcasting to
`shape`, as a sampler asks them of any lateral grid
(`sampling.LateralGrid`).  Nothing here knows about units: the nodes
are lengths in the domain's.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike

from ..fourier import Transform
from .domains import Box, Domain, Rectangle

__all__ = ["Grid"]


class Grid:
    """The nodes of a lateral domain's period; see the module docstring.

    `domain` is a `Box` or a `Rectangle`; `shape` is the number of nodes
    along each of its axes, (nx, ny) over a box and (nx,) over a
    rectangle, or one number for every axis.  Two grids are equal when
    their domains and shapes are.
    """

    def __init__(self, domain: Domain, shape: int | Sequence[int]) -> None:
        if not isinstance(domain, (Box, Rectangle)):
            raise TypeError(f"a grid lies over a Box or a Rectangle, got "
                            f"{type(domain).__name__}")
        d = len(domain.axes)
        sizes = (shape,) * d if np.ndim(shape) == 0 else tuple(shape)
        if len(sizes) != d:
            raise ValueError(f"a grid over a {type(domain).__name__} has {d} "
                             f"size{'s' if d > 1 else ''}, one per axis "
                             f"{domain.axes}; got {shape!r}")
        try:
            sizes = tuple(int(n) for n in sizes)
        except (TypeError, ValueError):
            raise ValueError(f"a grid's shape is a number of nodes per axis, got "
                             f"{shape!r}") from None
        if any(n < 1 for n in sizes):
            raise ValueError(f"a grid has at least one node along each axis, got "
                             f"{sizes}")
        self._domain = domain
        self._shape = sizes
        self._transform = Transform(sizes, domain)

    @property
    def domain(self) -> Domain:
        """The lateral domain whose period the nodes cover."""
        return self._domain

    @property
    def shape(self) -> tuple[int, ...]:
        """The nodes along each axis: (nx, ny) over a box, (nx,) over a
        rectangle; the trailing shape of values on the grid."""
        return self._shape

    @property
    def dimension(self) -> int:
        """The number of lateral axes: two over a box, one over a rectangle."""
        return len(self._shape)

    @property
    def transform(self) -> Transform:
        """The Fourier transforms of the grid (`fourier.Transform`), whose
        nodes and packing the grid's are."""
        return self._transform

    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        """The nodes along each axis, read-only: the transform's."""
        return self._transform.grid_axes

    @property
    def x(self) -> np.ndarray:
        """The nodes along x."""
        return self.axes[0]

    @property
    def y(self) -> np.ndarray | None:
        """The nodes along y over a box; None over a rectangle, whose body
        is the same at every y."""
        return self.axes[1] if self.dimension == 2 else None

    @property
    def lengths(self) -> tuple[float, ...]:
        """The periods along each axis."""
        return self._transform.lengths

    @property
    def spacing(self) -> tuple[float, ...]:
        """The step between neighbouring nodes along each axis, L_i / n_i."""
        return tuple(length / n for length, n in zip(self.lengths, self._shape))

    @property
    def area(self) -> float:
        """The measure of one period: the area of a box's, the length of a
        rectangle's."""
        return self._transform.volume

    @cached_property
    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """The lateral coordinates (x, y) of every node, broadcasting to
        `shape`: x as a column and y as a row over a box, and x with y = 0
        over a rectangle."""
        if self.dimension == 2:
            return self.axes[0][:, None], self.axes[1][None, :]
        y = np.zeros(self._shape)
        y.setflags(write=False)
        return self.axes[0], y

    def points(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """The points of the grid's transform at lateral coordinates (x, y),
        broadcast together: shape broadcast + (D,), (x, y) over a box and
        (x,) over a rectangle, where y is not read."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        if self.dimension == 2:
            return np.stack([x, y], axis=-1)
        return np.array(x)[..., None]

    def mean(self, values: ArrayLike) -> np.ndarray:
        """The mean over the period of values at the nodes, shape (...,) +
        `shape` to (...): their arithmetic mean, the periodic trapezoid."""
        v = np.asarray(values)
        d = self.dimension
        if v.ndim < d or v.shape[v.ndim - d:] != self._shape:
            raise ValueError(f"values on a grid of shape {self._shape} have that "
                             f"trailing shape; got {v.shape}")
        return np.mean(v, axis=tuple(range(-d, 0)))

    def scaled(self, k: float) -> "Grid":
        """The grid of the same shape over the domain scaled by k > 0."""
        return Grid(self._domain.scaled(k), self._shape)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Grid) and self._domain == other._domain
                and self._shape == other._shape)

    def __hash__(self) -> int:
        return hash((Grid, self._domain, self._shape))

    def __repr__(self) -> str:
        return f"Grid({' x '.join(map(str, self._shape))} over {self._domain!r})"
