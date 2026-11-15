"""Fields in the flat system: the classes that state `FLAT` as their own.

A field of `planetmodel.fields` is in whatever system it is built in;
the classes here are those whose constructors would otherwise default
to the sphere, with `FLAT` stated by the class, so that a field built
in `planetmodel.cartesian` is flat by construction and the algebra,
which keeps the class, keeps it flat.  `DepthField` is a field of depth
alone, the flat `RadialField` (and a `RadialField`: radial means of the
layered coordinate alone, and the exact algebra and `materials` treat it
so); `AnalyticField` is a formula of (depth, x, y); `constant_field` a
constant `DepthField`.  The kinds built from other fields take the
system of their sources and need no flat class: `ComposedField`,
`PerturbedField`, `ElasticField`, `PushedForwardField`, `HeldField`;
`PulledBackField` is given `coordinates=FLAT`.

Components are in the local frame (e_up, e_x, e_y) unless
`frame="cartesian"`: the first is the upward one.  The derivative of a
`DepthField` is along depth, downward, d/d depth = -d/dz, of every
component as stored.  A `DepthField` is its own horizontal mean
(`cartesian.horizontal_mean`), of any rank.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .. import fields
from ..character import SCALAR, Character
from ..frames import SphericalFunction
from ..layerfunction import LayerFunctionLike
from .coordinates import FLAT

if TYPE_CHECKING:
    from .grid import Grid

__all__ = ["DepthField", "AnalyticField", "constant_field"]


class DepthField(fields.RadialField):
    """A field of depth alone: one layer function of depth per stored
    component, in the local frame (e_up, e_x, e_y); `planetmodel.RadialField`
    in `FLAT`, which every copy and the algebra keep.

    `derivative` is along depth, downward.  A vector's first component
    is upward: a field pointing down with unit length has components
    (-1, 0, 0).
    """

    _coordinates = FLAT

    def __init__(self, interval: tuple[float, float],
                 function: LayerFunctionLike | Sequence | np.ndarray, *,
                 character: Character = SCALAR, name: str | None = None,
                 rtol: float = 1e-9) -> None:
        super().__init__(interval, function, character=character, name=name,
                         rtol=rtol)

    def horizontal_mean(self, *, grid: Grid | None = None) -> "DepthField":
        """A field of depth alone is its own horizontal mean, of any rank,
        the frame being the same everywhere; `grid` is not needed."""
        return self


class AnalyticField(fields.AnalyticField):
    """A formula `fn(depth, x, y)` returning components in `frame`, the local
    one (e_up, e_x, e_y) unless "cartesian": `planetmodel.AnalyticField`
    in `FLAT`, which every copy keeps."""

    _coordinates = FLAT

    def __init__(self, interval: tuple[float, float], fn: SphericalFunction, *,
                 character: Character = SCALAR, name: str | None = None,
                 frame: str | None = None, rtol: float = 1e-9) -> None:
        super().__init__(interval, fn, character=character, name=name, frame=frame,
                         rtol=rtol)


def constant_field(interval: tuple[float, float], value: ArrayLike, *,
                   character: Character = SCALAR,
                   name: str | None = None) -> DepthField:
    """The constant `value` (a number, or an array of the stored shape, in
    the local frame) as an exact `DepthField`."""
    constant = fields.constant_field(interval, value, character=character)
    return DepthField(interval, constant.functions, character=character, name=name)
