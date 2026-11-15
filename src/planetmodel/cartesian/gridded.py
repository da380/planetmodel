"""Stored fields that vary laterally in a box: values on a grid at GLL nodes.

A flat `GriddedField` is the sphere's (`planetmodel.gridded`) on a
horizontal grid: the interval of depth is cut into elements at `breaks`,
each element carries `ngll` Gauss-Lobatto-Legendre nodes, and at every
node the field is given on one `cartesian.Grid`, so that `values` has
shape

    (nelem, ngll) + grid.shape + stored shape,

(nelem, ngll, nx, ny) + stored over a box and (nelem, ngll, nx) + stored
over a section, which is a flat `Sample`'s layout cut to one layer.  The
radial half is shared with the sphere (`gridded.GriddedBase`): between
the nodes it is read along depth by one of `mesh1d.RADIAL_METHODS`
within the element holding the depth, the end elements continuing
beyond the interval, and laterally by one of the flat rules of
`cartesian.lateral` at every node of that element.  A table of depth
slices joined by straight lines is `ngll = 2` with the breaks at the
slices.

What differs from the sphere is the grid, which has no poles, so no
polar values are stored or given, and the frame: the components are
stored in the local frame (e_up, e_x, e_y), the same at every point of a
box, and the lateral rules read them as they stand.  The mean over the
period at every depth is the mean of the rule's interpolant at every
node joined by the radial rule (`horizontal_mean`), exact as the rules
define the field, and of any rank.

A `GriddedFunction` is the same thing for a function of (x, y) alone:
values on a grid, a lateral rule, and the two lateral derivatives, so
that it serves as a relief of `cartesian.layer_linear` (a height,
positive up) and as a surface quantity, a digital elevation model on
its own grid say.  A `GriddedDisplacement` is the vertical displacement
of `cartesian.displacement` held the same way: w(depth, x, y) a gridded
scalar field on every layer of a skeleton of depths, its derivatives
the rules' derivatives, and every element boundary declared as a knot,
since the radial rule's derivative may jump there; under
`cartesian.VerticalStretch` it is the flat form of the sphere's gridded
mapping (`gridded.GriddedDisplacement`), the vertical stretch being the
mapping of a flat body.

Everything is numbers and every stored array is read-only.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..character import SCALAR, Character
from ..fields import Field
from ..gridded import GriddedBase
from ..mesh1d.mesh import Mesh1D, RadialMesh
from ..skeleton import Skeleton
from .coordinates import FLAT
from .fields import DepthField
from .grid import Grid
from .lateral import METHODS, LateralInterpolant, check_rule

__all__ = ["GriddedField", "GriddedFunction", "GriddedDisplacement"]

#: The names of a grid's axes in a refusal, by its dimension.
_AXES = {1: ("nx",), 2: ("nx", "ny")}


def _held(a: ArrayLike) -> np.ndarray:
    """A read-only float64 copy."""
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


def _lateral_rule(grid: Grid, lateral: str) -> None:
    """Refuse, by name, a grid that is no `cartesian.Grid` and a rule it
    cannot carry."""
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a cartesian.Grid, got {type(grid).__name__}")
    if lateral not in METHODS:
        raise ValueError(f"lateral must be one of {METHODS}, got {lateral!r}")
    check_rule(grid, lateral)


class GriddedField(GriddedBase):
    """A field held on a horizontal grid at the GLL nodes of elements of
    depth; see the module docstring.

    `breaks` are the element boundaries, strictly increasing and
    spanning `interval` (within `rtol` of its width at each end), and
    `values` has shape `(nelem, ngll) + grid.shape + stored shape`, the
    components in the local frame (e_up, e_x, e_y), with `ngll` at least
    two.  `radial` is one of `mesh1d.RADIAL_METHODS` and `lateral` one of
    `cartesian.lateral.METHODS`.  Values are real.
    """

    _coordinates = FLAT

    def __init__(self, interval: tuple[float, float], breaks: ArrayLike, grid: Grid,
                 values: ArrayLike, *, character: Character = SCALAR,
                 name: str | None = None, radial: str = "polynomial",
                 lateral: str = "linear", rtol: float = 1e-9) -> None:
        _lateral_rule(grid, lateral)
        v = self._hold(interval, breaks, values, grid.shape,
                       axes=_AXES[grid.dimension], character=character, name=name,
                       radial=radial, rtol=rtol)
        self._grid = grid
        self._lateral = lateral
        # the values laid out for the lateral rules: the grid's axes last
        d = grid.dimension
        self._laid = np.moveaxis(v, tuple(range(2, 2 + d)), tuple(range(-d, 0)))

    @classmethod
    def sampled(cls, field: Field, grid: Grid, *, breaks: ArrayLike | None = None,
                ngll: int = 5, radial: str = "polynomial", lateral: str = "linear",
                name: str | None = None) -> "GriddedField":
        """`field` evaluated on `grid` at the GLL nodes of elements with
        `breaks` (the ends of its interval by default) and `ngll` nodes
        each: a gridded field that equals `field` at every node.  The name
        is the field's unless `name` is given."""
        lo, hi = (float(x) for x in field.interval)
        b = np.array([lo, hi] if breaks is None else breaks, dtype=float)
        mesh = Mesh1D(b, ngll=int(ngll))
        x, y = grid.nodes
        depth = mesh.r.reshape(mesh.r.shape + (1,) * grid.dimension)
        values = field.evaluate(depth, x[None, None], y[None, None])
        return cls(field.interval, b, grid, values, character=field.character,
                   name=field.name if name is None else name, radial=radial,
                   lateral=lateral, rtol=getattr(field, "rtol", 1e-9))

    @property
    def grid(self) -> Grid:
        """The horizontal grid at every node."""
        return self._grid

    def _interpolant(self, e: int) -> LateralInterpolant:
        """The lateral rule over the nodes of element e, built once."""
        f = self._interpolants.get(e)
        if f is None:
            f = LateralInterpolant(self._laid[e], self._grid, method=self._lateral)
            self._interpolants[e] = f
        return f

    def _values(self, d: np.ndarray, x: np.ndarray | None,
                y: np.ndarray | None) -> np.ndarray:
        out = self._values_at(d.ravel(), x.ravel(), y.ravel())
        return out.reshape(d.shape + self.stored_shape)

    def horizontal_mean(self, *, grid: Grid | None = None) -> DepthField:
        """The mean over the period as exact polynomial pieces in depth: at
        every node the mean of the lateral rule's interpolant, joined by
        the radial rule, so the mean of the field as its rules define it,
        of any rank; `grid` is not needed."""
        return self._lateral_mean(DepthField)

    def rescaled(self, *, k: float, v: float) -> "GriddedField":
        """v f(depth / k, x / k, y / k): the breaks, interval and grid scaled
        by k, the values by v."""
        k, v = float(k), float(v)
        lo, hi = self._interval
        breaks = k * self._breaks
        out = GriddedField((breaks[0], breaks[-1]), breaks, self._grid.scaled(k),
                           v * self._data, character=self._character,
                           name=self._name, radial=self._radial,
                           lateral=self._lateral, rtol=self._rtol)
        return out.on_interval(k * lo, k * hi)

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return (f"cartesian.GriddedField({nm}{self._character} on [{lo:g}, {hi:g}]: "
                f"{self._breaks.size - 1} x {self.ngll} nodes, {self._radial} x "
                f"{self._lateral} on {self._grid!r})")


class GriddedFunction:
    """A function of (x, y) held on a horizontal grid; see the module
    docstring.

    `values` has the grid's shape; `lateral` is one of
    `cartesian.lateral.METHODS`, the bicubic one by default.  Calling it
    gives the values at points, read periodically; `lateral_gradient`
    gives (d/dx, d/dy), d/dy zero on a section's grid, so that it is a
    relief of `cartesian.layer_linear`.
    """

    def __init__(self, grid: Grid, values: ArrayLike, *,
                 lateral: str = "cubic") -> None:
        if np.iscomplexobj(values):
            raise TypeError("a gridded function holds real values")
        _lateral_rule(grid, lateral)
        v = np.asarray(values, dtype=float)
        if v.shape != grid.shape:
            raise ValueError(f"values must have the grid's shape "
                             f"({', '.join(_AXES[grid.dimension])}) = {grid.shape}, "
                             f"got {v.shape}")
        self._f = LateralInterpolant(v, grid, method=lateral)
        self._data = _held(v)

    @classmethod
    def sampled(cls, fn: Callable[[np.ndarray, np.ndarray], ArrayLike], grid: Grid, *,
                lateral: str = "cubic") -> "GriddedFunction":
        """`fn(x, y)` evaluated on the nodes of `grid`."""
        x, y = grid.nodes
        values = np.asarray(fn(x, y), dtype=float)
        return cls(grid, np.broadcast_to(values, grid.shape), lateral=lateral)

    @property
    def grid(self) -> Grid:
        """The grid the values are on."""
        return self._f.grid

    @property
    def values(self) -> np.ndarray:
        """The values at the nodes, of the grid's shape."""
        return self._data

    @property
    def lateral(self) -> str:
        """The lateral rule, one of `cartesian.lateral.METHODS`."""
        return self._f.method

    def __call__(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """The values at (x, y), broadcast together."""
        return self._f(x, y)

    def lateral_gradient(self, x: ArrayLike, y: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(d/dx, d/dy) at (x, y), broadcast together."""
        return self._f.gradient(x, y)

    def mean(self) -> float:
        """The mean over the period of the function as its rule defines it."""
        return float(self._f.mean())

    def scaled(self, k: float) -> "GriddedFunction":
        """k f(x / k, y / k): k times the values on the grid scaled by k, by
        the same rule: the relief of a body k times the size."""
        k = float(k)
        return GriddedFunction(self.grid.scaled(k), k * self._data,
                               lateral=self.lateral)

    def __repr__(self) -> str:
        return f"cartesian.GriddedFunction({self.lateral} on {self.grid!r})"


class GriddedDisplacement:
    """w(depth, x, y), the upward displacement, held on grids; see the
    module docstring.

    `fields` holds one scalar flat `GriddedField` per layer of
    `skeleton`, in order from the surface down, each on its layer's
    interval: the height a point of the layer rises by (heights,
    positive up, as `cartesian.displacement` states).  A depth on a
    boundary is read from the deeper layer, and beyond the skeleton the
    end layers continue.  `depth_derivative` and `lateral_gradient` are
    the rules' derivatives, exact for the field as its rules define it,
    and every element boundary of every field inside the skeleton is a
    knot.  `GriddedDisplacement.sampled` builds one from any vertical
    displacement.
    """

    def __init__(self, skeleton: Skeleton, fields: Sequence[GriddedField]) -> None:
        fields = tuple(fields)
        b = skeleton.boundaries
        if len(fields) != b.size - 1:
            raise ValueError(f"{b.size - 1} layers need {b.size - 1} gridded "
                             f"fields, got {len(fields)}")
        tol = 1e-9 * skeleton.span
        for i, f in enumerate(fields):
            if not isinstance(f, GriddedField) or f.character.rank:
                raise ValueError(f"field {i} is not a flat gridded scalar field")
            lo, hi = f.interval
            if abs(lo - b[i]) > tol or abs(hi - b[i + 1]) > tol:
                raise ValueError(f"field {i} lies on [{lo:g}, {hi:g}], its layer "
                                 f"on [{b[i]:g}, {b[i + 1]:g}]")
        self._sk, self._fields = skeleton, fields
        breaks = np.unique(np.concatenate([f.breaks for f in fields]))
        inside = breaks[(breaks > b[0] + tol) & (breaks < b[-1] - tol)]
        self.knots = tuple(float(k) for k in inside)
        self.name = "gridded_displacement"

    @classmethod
    def sampled(cls, w: Callable[[np.ndarray, np.ndarray, np.ndarray], ArrayLike],
                mesh: RadialMesh, grid: Grid, *,
                lateral: str = "cubic") -> "GriddedDisplacement":
        """The vertical displacement `w(depth, x, y)` sampled at the nodes of
        `mesh`'s elements on `grid`, layer by layer: a stretch's own `w`,
        a stack of reliefs, any callable of the three."""
        sk = mesh.skeleton
        x, y = grid.nodes
        fields = []
        for i in range(sk.nlayers):
            elems = np.flatnonzero(mesh.layer == i)
            depth = mesh.r[elems]                               # (ne, ngll)
            breaks = np.append(mesh.left[elems], mesh.right[elems[-1]])
            values = np.asarray(w(depth.reshape(depth.shape + (1,) * grid.dimension),
                                  x[None, None], y[None, None]), dtype=float)
            values = np.broadcast_to(values, depth.shape + grid.shape)
            fields.append(GriddedField(tuple(sk.boundaries[i:i + 2]), breaks, grid,
                                       values, lateral=lateral))
        return cls(sk, fields)

    @property
    def skeleton(self) -> Skeleton:
        """The skeleton of depths the displacement is held over."""
        return self._sk

    @property
    def fields(self) -> tuple[GriddedField, ...]:
        """The displacement's gridded field on each layer, from the top."""
        return self._fields

    def _derivatives(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike
                     ) -> tuple[tuple[int, ...], list[np.ndarray]]:
        """The points' broadcast shape and [w, dw/d depth, dw/dx, dw/dy] at
        them, flat, each layer's field asked for its points."""
        depth, x, y = np.broadcast_arrays(np.asarray(depth, dtype=float),
                                          np.asarray(x, dtype=float),
                                          np.asarray(y, dtype=float))
        shape = depth.shape
        d, x, y = depth.ravel(), x.ravel(), y.ravel()
        b = self._sk.boundaries
        layer = np.clip(np.searchsorted(b, d, side="right") - 1, 0, b.size - 2)
        parts = [np.empty(d.size) for _ in range(4)]
        for i in np.unique(layer):
            m = layer == i
            for out, got in zip(parts, self._fields[i].derivatives(d[m], x[m], y[m])):
                out[m] = got
        return shape, parts

    def __call__(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """The upward displacement at the broadcast coordinates."""
        shape, (w, _, _, _) = self._derivatives(depth, x, y)
        return w.reshape(shape)

    def depth_derivative(self, depth: ArrayLike, x: ArrayLike,
                         y: ArrayLike) -> np.ndarray:
        """dw/d depth: the radial rule's derivative against the lateral
        rule's values."""
        shape, (_, dd, _, _) = self._derivatives(depth, x, y)
        return dd.reshape(shape)

    def lateral_gradient(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(dw/dx, dw/dy): the radial rule's weights against the lateral
        rule's gradient."""
        shape, (_, _, gx, gy) = self._derivatives(depth, x, y)
        return gx.reshape(shape), gy.reshape(shape)

    def scaled(self, k: float) -> "GriddedDisplacement":
        """k w(depth / k, x / k, y / k): the skeleton, the fields' depths and
        grids multiplied by k, and the displacement with them."""
        k = float(k)
        return GriddedDisplacement(Skeleton(k * self._sk.boundaries),
                                   [f.rescaled(k=k, v=k) for f in self._fields])

    def __repr__(self) -> str:
        return (f"cartesian.GriddedDisplacement({len(self._fields)} layers, "
                f"{len(self.knots)} knots)")
