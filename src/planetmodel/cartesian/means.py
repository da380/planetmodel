"""Horizontal means: a flat field reduced to a function of depth.

The horizontal mean of a field f on its layer of a box is

    fbar(depth) = (1 / A) integral over the period of f(depth, x, y) dx dy,

A the area of the period (its length over a section), a `DepthField` of
the field's character: the counterpart of the sphere's spherical mean
(`planetmodel.means`).  The frame (e_up, e_x, e_y) is the same at every
point of a box, so a field of any rank has one, its components averaged
as they stand.

A field that can give it exactly is asked (`horizontal_mean`, a method):
a field of depth alone is its own mean; a Fourier field's is its
constant component over sqrt(A), exact polynomial pieces; a gridded
field gives at each of its nodes the mean of its lateral rule's
interpolant, the mean of the node values, joined by its radial rule; a
perturbed field combines its parts, exactly from the perturbation's
mean when the perturbation is absolute or relative, and from the mean
of its exponential, by quadrature, when it is logarithmic.  Any other
field needs `grid=`, a `cartesian.Grid`, and its mean is that grid's
quadrature (`Grid.mean`, the periodic trapezoid, exact for a field
band-limited to the grid) at each depth, a `NumericLayer`.

`horizontally_averaged(model)` puts the mean of every field that is not
of depth alone in its place, through `with_field(replace=True)` in each
layer's order, so that a model type's own rule runs, as
`means.spherically_averaged` does on the sphere.  A mean over a box is
defined in `FLAT` alone: a field or a model of another system is
refused by name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..coordinates import require_coordinates
from ..fields import Field, stored_shape
from ..layerfunction import NumericLayer
from .coordinates import FLAT
from .fields import DepthField
from .grid import Grid

if TYPE_CHECKING:
    from .model import Model

__all__ = ["horizontal_mean", "quadrature_mean", "horizontally_averaged"]

# The most field values a quadrature mean evaluates at once (32 MB of doubles).
_BLOCK_VALUES = 1 << 22


def quadrature_mean(field: Field, grid: Grid) -> DepthField:
    """The mean of a flat field by `grid`'s quadrature at every depth it is
    asked at: a `DepthField` over `NumericLayer`s, one per stored
    component."""
    require_coordinates(field, FLAT, "a horizontal mean")
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a cartesian.Grid, got {type(grid).__name__}")
    x, y = grid.nodes
    d = grid.dimension
    shape = stored_shape(field.character)
    per_block = max(1, _BLOCK_VALUES // int(np.prod(grid.shape + shape)))

    def means(depth: np.ndarray) -> np.ndarray:
        """The mean of every stored component at the depths, depth's shape +
        stored shape; each distinct depth evaluated once on the grid."""
        depth = np.asarray(depth, dtype=float)
        levels, inverse = np.unique(depth.ravel(), return_inverse=True)
        out = np.empty(levels.shape + shape)
        for start in range(0, levels.size, per_block):
            block = levels[start:start + per_block]
            values = field.evaluate(block.reshape((-1,) + (1,) * d), x[None], y[None])
            out[start:start + per_block] = grid.mean(
                np.moveaxis(values, tuple(range(1, 1 + d)), tuple(range(-d, 0))))
        return out[inverse.ravel()].reshape(depth.shape + shape)

    if not shape:
        function = NumericLayer(field.interval, means)
    else:
        function = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            function[idx] = NumericLayer(field.interval,
                                         lambda r, idx=idx: means(r)[(...,) + idx])
    return DepthField(field.interval, function, character=field.character,
                      name=field.name)


def horizontal_mean(field: Field, *, grid: Grid | None = None) -> DepthField:
    """The horizontal mean of a flat field, by the rule of the module
    docstring: the field's own when it has one, else quadrature on
    `grid`."""
    require_coordinates(field, FLAT, "a horizontal mean")
    own = getattr(field, "horizontal_mean", None)
    if own is not None:
        return own(grid=grid)
    if getattr(field, "is_radial", False):
        return DepthField(field.interval, _radial_functions(field),
                          character=field.character, name=field.name)
    if grid is None:
        raise ValueError(f"{field!r} has no horizontal mean of its own; pass "
                         "grid=, a cartesian.Grid, to take it by quadrature")
    return quadrature_mean(field, grid)


def _radial_functions(field: Field) -> NumericLayer | np.ndarray:
    """A field of depth alone without a mean of its own as layer functions,
    one per stored component, of the depth alone."""
    shape = stored_shape(field.character)
    if not shape:
        return NumericLayer(field.interval, lambda r: field.evaluate(r, 0.0, 0.0))
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        out[idx] = NumericLayer(field.interval, lambda r, idx=idx: field.evaluate(
            r, np.zeros_like(r), np.zeros_like(r))[(...,) + idx])
    return out


def horizontally_averaged(model: Model, *, grid: Grid | None = None) -> Model:
    """`model` with the horizontal mean of every field that is not of depth
    alone in its place, through `with_field(replace=True)`, layer by layer
    in each layer's order of names; a field made radial by an earlier
    replacement is left as it now is."""
    require_coordinates(model, FLAT, "a horizontal mean")
    out = model
    for layer in model.layers:
        for name in layer.names:
            current = out.layer(layer.index)
            if name not in current or getattr(current[name], "is_radial", False):
                continue
            out = out.with_field(layer.index, name,
                                 horizontal_mean(current[name], grid=grid),
                                 replace=True)
    return out
