"""Fields and surface functions given by Fourier series over a box.

A `FourierField` is a field on one interval of depth written as

    f(depth, x, y) = sum_j c_j(depth) phi_j(x, y),

with phi_j the real orthonormal Fourier basis of a `cartesian.Grid`'s
transform (`fourier.Transform`: over the period of the grid's domain,
pygeoinf's conventions, the packing of the grid's shape) and the
coefficients c_j(depth) one scipy piecewise polynomial of depth whose
values have shape (N,) + the stored shape of the field's character, N
the number of components of the grid: the counterpart of the sphere's
`HarmonicField`, its coefficients held the same way, exact in depth on
polynomial pieces and exact laterally to the accuracy of finufft.  It is
the oracle the gridded fields of a box are held to.  A `FourierFunction`
is the same for a function of (x, y) alone, a fixed array of N
coefficients, with its two lateral derivatives, so that it serves as a
relief of `cartesian.layer_linear` (a height, positive up) and as a
surface quantity.

**Components.**  A field of rank 1 or above holds a series for each of
its stored components in the local frame (e_up, e_x, e_y), which is the
same at every point of a box: the components are those of one fixed
basis across the domain, and nothing rotates between the nodes of a
series, as it would on the sphere.  The Cartesian components are the
local ones permuted (`FieldBase.evaluate` with `frame="cartesian"`).

**Evaluation.**  At arbitrary points through finufft's synthesis
(`Transform.synthesise`), to `fourier.POINT_EPSILON` of the field's
size, every point read periodically.  A field groups its points by the
polynomial piece of depth they lie in, and synthesises each piece's
points once for each distinct depth among them, or, where they have
more depths than the piece has terms, once for each term of the piece's
polynomial, which it then sums at each point's depth: a section through
many depths costs a few syntheses a piece.  On the nodes of a grid,
`on_grid` takes the fast transform (`Transform.synthesise_grid`, after
`Transform.transfer` onto that grid) where the grid lies over the same
domain and holds every mode of the series, and synthesis at its nodes
otherwise.

**Means and scaling.**  The mean over the period is the constant
component over sqrt(A), A the period's area (`Grid.area`): exact, for a
field of any rank, the frame being one.  A change of length scale by k
scales the depth, the domain and the period, the coefficients of a
field taking a factor k^(D / 2) (D the grid's dimension) so that the
values are kept, the basis being normalised over the period.

Over a `Rectangle` the grid has one axis: the series is in x alone and
the field is the same at every y.  Everything is numbers and the
coefficients are real.
"""
from __future__ import annotations

import copy

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import BSpline, PPoly

from ..character import SCALAR, Character
from ..expansions import ppoly_of, rescaled_ppoly
from ..fields import FieldBase, radial_pieces_of, stored_shape
from ..layerfunction import PolynomialLayer, as_interval
from .coordinates import FLAT
from .fields import DepthField
from .grid import Grid

__all__ = ["FourierField", "FourierFunction"]


def _grid(grid: Grid) -> Grid:
    """The grid, or the error."""
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a cartesian.Grid, got {type(grid).__name__}")
    return grid


def _holds_every_mode(source: Grid, target: Grid) -> bool:
    """Whether `target` lies over the domain of `source` and holds every one
    of its modes, so that a series moves onto it by `Transform.transfer`."""
    if target.domain != source.domain or target.dimension != source.dimension:
        return False
    here, _, _ = source.transform.packing.matching(target.transform.packing)
    return here.size == source.transform.size


class FourierField(FieldBase):
    """sum_j c_j(depth) phi_j(x, y) on one interval of depth; see the module
    docstring.

    `grid` is the `cartesian.Grid` whose transform's basis the series is
    in.  `coefficients` is a scipy `PPoly` or `BSpline` of depth whose
    values have shape (N,) + stored shape, N being `grid.transform.size`,
    or such an array for coefficients constant in depth; beyond its
    breakpoints a piecewise polynomial continues its end pieces.  Values
    are real.
    """

    _coordinates = FLAT

    #: The rules a Fourier field states for itself: exact pieces in depth
    #: and the series between nodes.
    radial = "polynomial"
    lateral = "fourier"

    def __init__(self, interval: tuple[float, float], grid: Grid,
                 coefficients: PPoly | BSpline | ArrayLike, *,
                 character: Character = SCALAR, name: str | None = None,
                 rtol: float = 1e-9) -> None:
        self._grid = _grid(grid)
        self._interval = as_interval(interval)
        self._character = character
        self._name = name
        self._rtol = float(rtol)
        if isinstance(coefficients, BSpline):
            coefficients = ppoly_of(coefficients)
        if isinstance(coefficients, PPoly):
            c = np.asarray(coefficients.c)
            x = np.asarray(coefficients.x, dtype=float)
        else:
            c = np.asarray(coefficients)
            lo, hi = self._interval
            c, x = c[None, None], np.array([lo, hi])
        if np.iscomplexobj(c):
            raise TypeError("a Fourier field holds real coefficients")
        want = (grid.transform.size,) + stored_shape(character)
        if c.shape[2:] != want:
            raise ValueError(
                f"a {character} Fourier field on {grid!r} has coefficients with "
                f"values of shape (N,) + stored shape = {want}, got {c.shape[2:]}")
        self._pp = PPoly(c.astype(float), x)

    @property
    def grid(self) -> Grid:
        """The grid whose transform's basis the series is in."""
        return self._grid

    @property
    def coefficients(self) -> PPoly:
        """The coefficients as a piecewise polynomial of depth, values of
        shape (N,) + stored shape."""
        return self._pp

    @property
    def ppoly(self) -> PPoly:
        """The coefficients' piecewise polynomial, as `radial_pieces_of`
        reads a layer function."""
        return self._pp

    def coefficients_at(self, depth: ArrayLike) -> np.ndarray:
        """c_j at the depths: shape (N,) + stored shape + depth's shape."""
        d = np.asarray(depth, dtype=float)
        lead = 1 + len(self.stored_shape)
        return np.moveaxis(self._pp(d), tuple(range(d.ndim)),
                           tuple(range(lead, lead + d.ndim)))

    def radial_pieces(self) -> tuple[np.ndarray, int]:
        """The coefficients' breakpoints in the interval, its ends included,
        and their degree."""
        return radial_pieces_of(self._interval, [self])

    def _series(self, c: np.ndarray) -> np.ndarray:
        """Coefficients (..., N) + stored as the transform's batch, (...,) +
        stored + (N,)."""
        n = 1 + len(self.stored_shape)
        return np.moveaxis(c, c.ndim - n, -1)

    def _synthesised(self, c: np.ndarray, P: np.ndarray) -> np.ndarray:
        """The series of coefficients (B, N) + stored at points (P, D): (B,
        P) + stored."""
        values = self._grid.transform.synthesise(self._series(c), P)  # (B, *c, P)
        return np.moveaxis(values, -1, 1)

    def _values(self, d: np.ndarray, x: np.ndarray | None,
                y: np.ndarray | None) -> np.ndarray:
        d, x, y = np.broadcast_arrays(d, np.asarray(x, dtype=float),
                                      np.asarray(y, dtype=float))
        df = d.ravel()
        P = self._grid.points(x, y).reshape(-1, self._grid.dimension)
        xb, c = self._pp.x, self._pp.c
        terms = c.shape[0]
        piece = np.clip(np.searchsorted(xb, df, side="right") - 1, 0, xb.size - 2)
        out = np.empty((df.size,) + self.stored_shape)
        for i in np.unique(piece):
            at = np.flatnonzero(piece == i)
            depths, which = np.unique(df[at], return_inverse=True)
            if depths.size <= terms:                   # a synthesis a depth
                held = self._pp(depths)                # (nd, N) + stored
                for u in range(depths.size):
                    one = at[which == u]
                    out[one] = self._synthesised(held[u:u + 1], P[one])[0]
                continue
            each = self._synthesised(c[:, i], P[at])  # a synthesis a term
            s = (df[at] - xb[i]).reshape((-1,) + (1,) * len(self.stored_shape))
            v = each[0]
            for k in range(1, terms):
                v = v * s + each[k]
            out[at] = v
        return out.reshape(d.shape + self.stored_shape)

    def on_grid(self, depth: ArrayLike, grid: Grid) -> np.ndarray:
        """The values at `depth` (within the interval) on the nodes of
        `grid`, (depth.size,) + grid.shape + stored shape: by the fast
        transform where `grid` lies over this field's domain and holds
        every mode of the series, by synthesis at its nodes otherwise."""
        grid = _grid(grid)
        d, _, _ = self._points(np.asarray(depth, dtype=float).ravel(), 0.0, 0.0)
        if _holds_every_mode(self._grid, grid):
            c = self._series(np.moveaxis(self.coefficients_at(d), -1, 0))
            moved = self._grid.transform.transfer(c, grid.transform)
            values = grid.transform.synthesise_grid(moved)      # (nd, *c, *shape)
            n = len(self.stored_shape)
            return np.moveaxis(values, tuple(range(1, 1 + n)),
                               tuple(range(-n, 0)) if n else ())
        x, y = grid.nodes
        return self.evaluate(d.reshape((-1,) + (1,) * grid.dimension), x[None],
                             y[None])

    def horizontal_mean(self, *, grid: Grid | None = None) -> DepthField:
        """The mean over the period at every depth, c_0(depth) / sqrt(A):
        exact polynomial pieces, of any rank, a `DepthField`; `grid` is not
        needed."""
        pp = self._pp
        scale = 1.0 / np.sqrt(self._grid.area)
        shape = self.stored_shape
        if not shape:
            function = PolynomialLayer(self._interval, PPoly(pp.c[:, :, 0] * scale,
                                                             pp.x))
        else:
            function = np.empty(shape, dtype=object)
            for idx in np.ndindex(shape):
                function[idx] = PolynomialLayer(
                    self._interval, PPoly(pp.c[(slice(None), slice(None), 0) + idx]
                                          * scale, pp.x))
        return DepthField(self._interval, function, character=self._character,
                          name=self._name, rtol=self._rtol)

    def on_interval(self, lo: float, hi: float) -> "FourierField":
        """The same series on [lo, hi], the end pieces continuing."""
        out = copy.copy(self)
        out._interval = as_interval((lo, hi))
        return out

    def rescaled(self, *, k: float, v: float) -> "FourierField":
        """v f(depth / k, x / k, y / k): the breakpoints, interval and domain
        scaled by k, each piece's coefficients by v k^(D / 2) / k^p for its
        power p, the basis being normalised over a period k^D times the
        area."""
        k, v = float(k), float(v)
        lo, hi = self._interval
        grid = self._grid.scaled(k)
        return FourierField((k * lo, k * hi), grid,
                            rescaled_ppoly(self._pp, k=k,
                                           v=v * k ** (0.5 * grid.dimension)),
                            character=self._character, name=self._name,
                            rtol=self._rtol)

    def renamed(self, name: str | None) -> "FourierField":
        """The same field under `name`."""
        out = copy.copy(self)
        out._name = name
        return out

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return (f"FourierField({nm}{self._character} on [{lo:g}, {hi:g}], "
                f"{self._pp.x.size - 1} pieces in depth, on {self._grid!r})")


class FourierFunction:
    """sum_j c_j phi_j(x, y): a function of the lateral coordinates alone.

    `grid` is the `cartesian.Grid` whose transform's basis the series is
    in and `coefficients` its N components (`grid.transform.size`); the
    components of values on the grid are `grid.transform.analyse_grid`'s.
    Calling it gives the values at points, every point read periodically;
    `lateral_gradient` gives (d/dx, d/dy), d/dy zero on a section's grid,
    so that it is a relief of `cartesian.layer_linear`; `on_grid` the
    values on a grid's nodes, `mean` the mean over the period, and
    `scaled(k)` k f(x / k, y / k), the relief of a body k times the size.
    Values are real.
    """

    def __init__(self, grid: Grid, coefficients: ArrayLike) -> None:
        self._grid = _grid(grid)
        if np.iscomplexobj(coefficients):
            raise TypeError("a Fourier function holds real coefficients")
        c = np.array(coefficients, dtype=float)
        if c.shape != (grid.transform.size,):
            raise ValueError(f"a Fourier function on {grid!r} has "
                             f"{grid.transform.size} coefficients, got shape "
                             f"{c.shape}")
        c.setflags(write=False)
        self._c = c

    @property
    def grid(self) -> Grid:
        """The grid whose transform's basis the series is in."""
        return self._grid

    @property
    def coefficients(self) -> np.ndarray:
        """The N components of the series."""
        return self._c

    def __call__(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """The values at (x, y), broadcast together."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        return self._grid.transform.synthesise(self._c, self._grid.points(x, y))

    def lateral_gradient(self, x: ArrayLike, y: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(d/dx, d/dy) at (x, y), broadcast together."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        g = self._grid.transform.synthesise_gradient(self._c, self._grid.points(x, y))
        if self._grid.dimension == 2:
            return g[..., 0], g[..., 1]
        return g[..., 0], np.zeros(x.shape)

    def on_grid(self, grid: Grid) -> np.ndarray:
        """The values on the nodes of `grid`, of its shape: by the fast
        transform where it lies over this function's domain and holds every
        mode, by synthesis at its nodes otherwise."""
        grid = _grid(grid)
        if _holds_every_mode(self._grid, grid):
            moved = self._grid.transform.transfer(self._c, grid.transform)
            return grid.transform.synthesise_grid(moved)
        x, y = grid.nodes
        return self(x, y)

    def mean(self) -> float:
        """The mean over the period, c_0 / sqrt(A)."""
        return float(self._c[0] / np.sqrt(self._grid.area))

    def scaled(self, k: float) -> "FourierFunction":
        """k f(x / k, y / k) on the grid scaled by k: the coefficients times
        k^(1 + D / 2), the basis being normalised over the larger period."""
        k = float(k)
        grid = self._grid.scaled(k)
        return FourierFunction(grid, self._c * k ** (1.0 + 0.5 * grid.dimension))

    def __repr__(self) -> str:
        return f"FourierFunction(on {self._grid!r})"
