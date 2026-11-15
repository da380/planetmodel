"""A flat-layered model: a flat geometry with a bag of fields on every layer.

`Model` is the flat leaf of `model.ModelBase`: everything the spherical
model does (layers, specs, scales, citation, `replaced`, surgery,
conversion, coarsening) it does on a `cartesian.Geometry`, every field
in `FLAT`, a field built for its layers (a coarsened layer's, say) a
`DepthField`.  Its layers run from the surface down.  Padding below the
base with the lowest properties held is `extended(depths,
fields="hold")`, exact for fields of depth alone.  A flat model type is
a class derived from it alone.  `stretched` places it under a vertical
stretch, topography as a stack of reliefs (`cartesian.layer_linear`).

There is no gravity computed here: gravity from the mass inside a
radius is the sphere's (`behaviours.SelfGravitating` refuses a flat
model).  A flat model's gravity is prescribed, a field like any other
linked with it: `with_uniform_gravity(g)` puts the vocabulary's `g`, a
constant acceleration downward by magnitude, on every layer.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..frames import SphericalFunction
from ..geometry import GeometryBase
from ..layerfunction import constant_layer
from ..model import ModelBase
from .displacement import VerticalDisplacement
from .geometry import Geometry

__all__ = ["Model"]


class Model(ModelBase):
    """A flat-layered model: a `ModelBase` on a `cartesian.Geometry`; see
    the module docstring and `ModelBase` for the arguments."""

    geometry_type: ClassVar[type[GeometryBase]] = Geometry

    def stretched(self, w: VerticalDisplacement | SphericalFunction, *,
                  name: str | None = None, check: bool = True) -> "Model":
        """The model under the vertical stretch driven by `w` (heights,
        positive up); see `cartesian.Geometry.stretched`."""
        return self.replaced(geometry=self.geometry.stretched(w, name=name,
                                                               check=check),
                             check=False)

    def with_uniform_gravity(self, g: float, *, name: str = "g",
                             replace: bool = False) -> "Model":
        """The model with a prescribed gravitational acceleration `g`,
        downward and by magnitude, in the model's units, the same at
        every point: a constant `DepthField` under `name` on every layer,
        computed from nothing.  A negative or infinite `g` is refused, and
        a layer already holding `name` unless `replace`."""
        g = float(g)
        if not (np.isfinite(g) and g >= 0.0):
            raise ValueError(f"g is a finite magnitude, at least zero; got {g:g}")
        make = self.geometry.radial_field_type
        layers = []
        for layer in self.layers:
            if name in layer and not replace:
                raise ValueError(f"layer {layer.index} ({layer.name!r}) already holds "
                                 f"{name!r}; pass replace=True to replace it")
            iv = layer.interval
            held = make(iv, constant_layer(iv, g), name=name)
            layers.append({**layer.fields, name: held})
        return self.replaced(layers=layers)
