"""A flat-layered model: a flat geometry with a bag of fields on every layer.

`Model` is the flat leaf of `model.ModelBase`: everything the spherical
model does (layers, specs, scales, citation, `replaced`, surgery,
conversion, coarsening) it does on a `cartesian.Geometry`, every field
in `FLAT`.  Its layers run from the surface down.  Padding below the
base with the lowest properties held is `extended(depths,
fields="hold")`, exact for fields of depth alone.  A flat model type is
a class derived from it alone.
"""
from __future__ import annotations

from typing import ClassVar

from ..geometry import GeometryBase
from ..model import ModelBase
from .geometry import Geometry

__all__ = ["Model"]


class Model(ModelBase):
    """A flat-layered model: a `ModelBase` on a `cartesian.Geometry`; see
    the module docstring and `ModelBase` for the arguments."""

    geometry_type: ClassVar[type[GeometryBase]] = Geometry
