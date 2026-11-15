"""Flat-layered models: a box or a section, layered in depth below one
free surface.

The counterpart of the spherical library for bodies layered in depth,
reached by name as `planetmodel.cartesian`, the spherical top level
untouched.  What is the same concept in one coordinate is shared: the
skeleton, layer functions, the field kinds and their algebra, mappings
and push-forward, the bases `geometry.GeometryBase` and
`model.ModelBase`, and the executable contracts of `planetmodel.testing`,
which hold the flat objects as they hold the spherical ones.  What is a
different construction has its leaf here.

Points are right-handed Cartesian (x, y, z), z up, the free surface
z = 0 in the reference body.  The layered coordinate is the depth,
positive downward: a skeleton's boundaries are depths from 0 down, and
layers and interfaces are numbered from the surface down, interface 0
the surface with `between = (0, -1)` and interface k the top of layer k
with `between = (k, k - 1)`.  The base of the box, the skeleton's last
boundary, is a face and no interface, as the centre of a ball is none.

Fields are evaluated at (depth, x, y) in `FLAT`, whose local frame
(e_up, e_x, e_y) is vertical first, so the transverse isotropy of
`materials` and the rules written for radial fields carry over.  The
frame is right-handed and its first vector points out of the body, as
e_r does on the sphere; (e_down, e_x, e_y) would be left-handed.  So the
first vector points up while depth increases down: a rank-1 field's
first component is its upward one, and a derivative along the layered
coordinate is the downward one, d/d depth = -d/dz.

A finite box approximates an unbounded domain, laterally and in depth,
and carries no boundary physics.  The lateral domain is a `Box` (x and
y bounded) or a `Rectangle` (x bounded: the section of a body invariant
along y), each with optional periodic padding, none by default, a
model's box being its period.  Below the base a model is padded, where
its base must be moved beyond reach, by `Model.extended(depths,
fields="hold")`, the lowest layer's properties held.  There is no
gravity here.

  FLAT, FlatCoordinates  the system (depth, x, y)          (coordinates)
  Box, Rectangle         lateral domains                    (domains)
  Geometry               a skeleton of depths over a domain (geometry)
  Model                  a flat model                       (model)
  DepthField             a field of depth alone             (fields)
  AnalyticField          a formula of (depth, x, y)         (fields)
  constant_field         a constant field of depth          (fields)
"""
from .coordinates import FLAT, FlatCoordinates
from .domains import Box, Rectangle
from .fields import AnalyticField, DepthField, constant_field
from .geometry import Geometry
from .model import Model

__all__ = ["FLAT", "FlatCoordinates", "Box", "Rectangle", "Geometry", "Model",
           "DepthField", "AnalyticField", "constant_field"]
