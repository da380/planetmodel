"""Values between the nodes of a horizontal grid: the flat lateral rules.

A field that varies laterally in a box is stored at the nodes of a
`cartesian.Grid` and read between them by one of three rules, the
doubly periodic counterparts of the sphere's (`planetmodel.lateral`):

    linear     bilinear on the grid closed by its period (linear along
               the one axis of a section's grid)
    cubic      the tensor-product cubic spline, periodic along every axis
    fourier    analysis on the grid and synthesis at the point
               (`fourier.Transform`), exact for a field band-limited to
               the grid's shape, to the accuracy of finufft
               (`fourier.POINT_EPSILON`)

**The period.**  The grid covers one period of its domain along each
axis, its last node one step short of the far end, so each axis is
closed by the first node again one period on: the bilinear rule is the
tensor product of the linear splines through the nodes and that wrap
node, and the bicubic rule that of the periodic cubic splines through
the nodes (`lateral.periodic_cubic_spline`), continuous with their first
two derivatives across the seam.  A point is read at its coordinates
reduced into the period from the grid's first node, so every rule is
periodic, as a Fourier series is; there are no poles, and nothing at
the edge of the domain needs to be given.  A section's grid has one
axis, and y is never read.

**Components.**  The rules act on any array whose trailing axes are the
grid's, component by component.  The frame (e_up, e_x, e_y) is the
same everywhere in a box, so the local components of a vector or a
tensor are interpolated as they stand: there is no rotation into a
frame that is single-valued, as there is on the sphere.

**Accuracy.**  On equally spaced nodes the bilinear rule converges as
h^2 and the bicubic as h^4 in the node spacing h; the Fourier rule is
exact once the grid holds the field's band.  For a field of band 8 along
each axis of a box (wavenumbers |k_i| <= 8 over the period, coefficients
N(0, 1) / (1 + |k|)), the largest error at 4000 random points relative
to the field's largest value, on grids of 32, 64, 128 and 256 nodes a
side (bands 2, 4, 8 and 16 times the field's), is 1.6e-1, 4.5e-2,
1.1e-2 and 3.0e-3 for the bilinear rule and 8.1e-3, 4.0e-4, 2.3e-5 and
1.5e-6 for the bicubic: orders 1.9 and 4.1 fitted, 2 and 4 as the
spacing falls; the Fourier rule's is 5e-13 on all four, finufft's
accuracy, and 0.25 on 16 nodes, a grid that lacks the sines of the
field's highest wavenumber (the cases of
`tests/cartesian/test_cartesian_lateral.py`; along a section's one axis
the orders are the same).  A local rule wants an oversampled grid, as
on the sphere.  The bicubic rule is twice continuously differentiable
everywhere, the bilinear one continuous with a gradient piecewise
constant between the nodes.

`LateralInterpolant` builds a rule once over values on a grid and gives
its values and its two lateral derivatives (d/dx, d/dy) at any points,
and its mean over the period: on equally spaced periodic nodes the
integral of every rule's interpolant is the trapezoid's of its nodes,
so that the mean of the interpolant is the mean of the values (for the
bicubic rule because the periodic spline's coefficients sum to its
values' sum, each B-spline integrating to one step).  `interpolate`
builds one and evaluates it.  Everything is numbers.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import NdBSpline, make_interp_spline

from ..frames import wrapped
from ..lateral import periodic_cubic_spline
from .grid import Grid

__all__ = ["METHODS", "LateralInterpolant", "interpolate", "check_rule"]

#: The flat lateral rules, in the order of the module docstring.
METHODS = ("linear", "cubic", "fourier")


def check_rule(grid: Grid, method: str) -> None:
    """Refuse, by name, a rule the grid cannot carry: an unknown method,
    or the cubic rule on fewer than three nodes along an axis, the
    fewest a periodic cubic spline is determined by."""
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a cartesian.Grid, got {type(grid).__name__}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "cubic" and min(grid.shape) < 3:
        raise ValueError(f"the cubic rule needs at least 3 nodes along every axis; "
                         f"this grid has {grid.shape}")


def _grid_values(values: ArrayLike, grid: Grid) -> np.ndarray:
    """Finite real values with the grid's axes last, or the error."""
    if np.iscomplexobj(values):
        raise TypeError("the lateral rules take real values")
    v = np.asarray(values, dtype=float)
    d = grid.dimension
    if v.ndim < d or v.shape[v.ndim - d:] != grid.shape:
        raise ValueError(f"values must have trailing shape {grid.shape}, got "
                         f"{v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("the lateral rules take finite values")
    return v


def _along(V: np.ndarray, nodes: np.ndarray, period: float, method: str
           ) -> tuple[np.ndarray, np.ndarray]:
    """(knots, coefficients) of the periodic rule through V of shape (n, ...)
    along axis 0: the linear spline through the nodes and the wrap node,
    or the periodic cubic spline."""
    if method == "linear":
        closed = np.append(nodes, nodes[0] + period)
        spline = make_interp_spline(closed, np.concatenate([V, V[:1]]), k=1, axis=0)
        return spline.t, spline.c
    return periodic_cubic_spline(V, nodes, period)


class LateralInterpolant:
    """One flat lateral rule, built once over an array of values on a grid.

    `values` has shape (...,) + `grid.shape`, the grid's axes last, and
    must be finite and real; the leading shape is `shape`, and calling the
    interpolant at lateral coordinates (x, y) that broadcast to a shape P
    gives an array of shape `shape + P`.  `method` is one of `METHODS`.
    """

    def __init__(self, values: ArrayLike, grid: Grid, *, method: str) -> None:
        check_rule(grid, method)
        v = _grid_values(values, grid)
        d = grid.dimension
        self._grid, self._method = grid, method
        self._shape = v.shape[:v.ndim - d]
        self._mean = np.asarray(grid.mean(v))
        if method == "fourier":
            self._coeffs = grid.transform.analyse_grid(v)
            return
        # the grid's axes first, then one periodic spline along each in turn
        C = np.moveaxis(v, tuple(range(-d, 0)), tuple(range(d)))
        knots = []
        for axis, (nodes, period) in enumerate(zip(grid.axes, grid.lengths)):
            t, C = _along(np.moveaxis(C, axis, 0), nodes, period, method)
            C = np.moveaxis(C, 0, axis)
            knots.append(t)
        self._degree = 1 if method == "linear" else 3
        self._spline = NdBSpline(tuple(knots), C, self._degree)

    @property
    def grid(self) -> Grid:
        """The grid the values are on."""
        return self._grid

    @property
    def method(self) -> str:
        """The lateral rule, one of `METHODS`."""
        return self._method

    @property
    def shape(self) -> tuple[int, ...]:
        """The leading shape of the values, carried through to every result."""
        return self._shape

    def _points(self, x: ArrayLike, y: ArrayLike
                ) -> tuple[np.ndarray, tuple[int, ...]]:
        """The points as the grid's (P, D) coordinates, checked, with their
        broadcast shape."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("the points must be finite")
        return self._grid.points(x, y).reshape(-1, self._grid.dimension), x.shape

    def _local(self, P: np.ndarray, nu: tuple[int, ...]) -> np.ndarray:
        """The spline, or one of its derivatives, at points (P, D) reduced
        into the period: shape `shape + (P,)`."""
        reduced = np.stack([wrapped(P[:, i], start=float(nodes[0]), period=period)
                            for i, (nodes, period) in enumerate(
                                zip(self._grid.axes, self._grid.lengths))], axis=-1)
        return np.moveaxis(self._spline(reduced, nu=nu), 0, -1)

    def __call__(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """The values at lateral coordinates (x, y), any values, broadcast
        together: shape `shape` followed by theirs."""
        P, points = self._points(x, y)
        if self._method == "fourier":
            out = self._grid.transform.synthesise(self._coeffs, P)
        else:
            out = self._local(P, (0,) * self._grid.dimension)
        return out.reshape(self._shape + points)

    def gradient(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """(d/dx, d/dy) of the interpolant at the points, each of the shape
        `__call__` gives: the derivatives of the spline, piecewise constant
        between the nodes for the bilinear rule, or of the series; d/dy is
        zero on a section's grid."""
        P, points = self._points(x, y)
        d = self._grid.dimension
        if self._method == "fourier":
            g = self._grid.transform.synthesise_gradient(self._coeffs, P)
            parts = [g[..., i] for i in range(d)]
        else:
            parts = [self._local(P, tuple(int(i == j) for j in range(d)))
                     for i in range(d)]
        dx = parts[0].reshape(self._shape + points)
        dy = (parts[1].reshape(self._shape + points) if d == 2
              else np.zeros(self._shape + points))
        return dx, dy

    def mean(self) -> np.ndarray:
        """The mean over the period of the interpolant itself, of shape
        `shape`: the mean of the values, which on equally spaced periodic
        nodes every rule's interpolant has (the module docstring)."""
        return self._mean

    def __repr__(self) -> str:
        return f"cartesian.LateralInterpolant({self._method}, {self._shape} on " \
               f"{self._grid!r})"


def interpolate(values: ArrayLike, grid: Grid, x: ArrayLike, y: ArrayLike, *,
                method: str) -> np.ndarray:
    """Values on a grid read at points by a flat lateral rule:
    `LateralInterpolant(values, grid, method=method)` evaluated at (x, y).
    Build the interpolant itself to read the same values more than once."""
    return LateralInterpolant(values, grid, method=method)(x, y)
