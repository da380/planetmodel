"""Flat-layered models: a box or a rectangle, layered in depth below one
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
`materials` and the rules written for radial fields carry over.  A field
of depth alone (`DepthField`) is the flat radial field: in a name shared
with the spherical side (`is_radial`, `radial_profile`), "radial" means
of the layered coordinate alone, depth here.  The frame is right-handed
and its first vector points out of the body, as e_r does on the sphere;
(e_down, e_x, e_y) would be left-handed.  So the first vector points up
while depth increases down: a rank-1 field's first component is its
upward one, and a derivative along the layered coordinate is the
downward one, d/d depth = -d/dz.

A finite box approximates an unbounded domain, laterally and in depth,
and carries no boundary physics.  The lateral domain is a `Box` (x and y
bounded) or a `Rectangle` (x bounded: the cross-section of a body
invariant along y), each with optional periodic padding, none by
default, a model's box being its period.  Below the base a model is
padded, where its base must be moved beyond reach, by
`Model.extended(depths, fields="hold")`, the lowest layer's properties
held.  No gravity is computed here; a model may carry a prescribed one.

A flat body is placed in the physical world by a mapping, as a
spherical one is, and the mapping machinery is shared (`MappingBase`,
the validity report, push-forward and pull-back, a fault's pull-back).
Topography is the vertical stretch (`VerticalStretch`), driven by an
upward displacement (`cartesian.displacement`, where the sign is stated
once: heights, positive up), most often a stack of reliefs
(`layer_linear`): the free surface raised to its topography, each
interface to its own relief, the base held, every layer stretched
linearly in depth between them; `Geometry.stretched` and
`Model.stretched` place a body under one.

A flat geometry is meshed for MFEM by `planetmodel.mesh3d`, as a
spherical one is: `build_layered_mesh(MeshSpec(geometry, sizing), path)`
gives the box over the domain's extent cut by the planes of the
interfaces, or over a `Rectangle` its cross-section in the (x, z) plane,
with the layers and interfaces numbered from the surface down and the
base and the sides labelled faces; `export_mfem` writes it with a
model's fields by layer.  Faults given in a `fault.CartesianFrame`,
whose points are this package's, are built into a box as into a sphere
(`mesh3d.EmbeddedFault`), leaving it through its surface, its base or a
side; under topography the mesh is the reference box, deformed at the
export, and a fault, given where it physically is, is pulled back
through the mapping and its preimage built in.

Lateral structure is held as in the sphere's library, the spherical
harmonics a Fourier series over the box's period: a `Grid` is the
horizontal grid of a `Box` or a `Rectangle`, the nodes of the
`fourier.Transform` of its domain and shape (`grid`); a `FourierField`
is a field whose lateral dependence is a Fourier series with
coefficients piecewise polynomial in depth, and a `FourierFunction` a
function of (x, y) given by one (`expansions`); the flat lateral rules,
bilinear, bicubic and Fourier, periodic along every axis, read values
between a grid's nodes (`lateral`), and a `GriddedField`,
`GriddedFunction` and `GriddedDisplacement` (a vertical displacement on
grids) hold values on a grid at GLL nodes in depth (`gridded`, sharing
the radial half with the sphere's).  Components are in the local frame,
the same at every point of a box, so nothing is rotated between nodes.
`sample` evaluates a flat model on depth nodes times a grid's
(`sampling`); `horizontal_mean` reduces a field to a function of depth,
exactly where the field has a mean of its own (a Fourier field's
constant term, a gridded field's node means), by the grid's quadrature
otherwise, and `horizontally_averaged` a model (`means`).  A Fourier or
gridded function is a relief like any other.  Over a `Rectangle` every
grid, series and rule has the one axis x, and a rectangle's field is the
same at every y.

A flat model is looked at through its slices
(`planetmodel.cartesian.slices`, reached by name as the sphere's
`planetmodel.slices` is): a plane at a depth, a vertical section along a
line, a profile, the horizontal mean with its spread, the spectrum by
horizontal wavenumber and the box seen from a corner, flat or as the
mapping shapes the body, which `planetmodel.plotting` draws depth
downward.

A flat model is written to a model file and read back by
`planetmodel.modelfile` (`write_model`, `read_model`), which take the
flat schema for it, `planetmodel.model/3`, described where it differs
from the spherical file in `cartesian.modelfile`; fields on a horizontal
plane are a flat surface file, `planetmodel.surface/2`
(`cartesian.surfacefile`).  Both modules are reached by name.

A flat model is read from and written to a deck of depths, its first
column the depth from the surface down (`cartesian.deck`): the formats
of the radius-first decks stated for `DEPTH`, one definition each, read
and written by the library's `read_deck`, `write_deck` and
`sample_deck`.  A deck holds no lateral extent, so the model types of
the decks take the box beside the deck.  The catalogue
(`cartesian.catalogue`) holds those, the flat `LayeredIsotropicElastic`
and `CrustOverMantle`, the reference model of the tests and examples.
The behaviours that are pointwise (`Elastic`, `ConstantQ`,
`Viscoelastic`) serve flat models unchanged; gravity from the mass
inside a radius, a mean over the sphere and whatever else is the
sphere's alone refuse a flat object by name, a flat model's gravity
being prescribed (`Model.with_uniform_gravity`).

Matern random fields of flat layers are drawn in coefficients
(`cartesian.randomfield`, the counterpart of `planetmodel.randomfield`'s
samplers): `LayerBasis` holds the eigenbases in depth of every lateral
eigenvalue of a grid's Fourier transform and fixes the order of the
coefficient vector, and `DepthGRF`, `BoxGRF` and `LayeredGRF` sample a
field of depth, of a box or a rectangle, and of a model's layers, with
an exact marginal standard deviation, the mesh in depth padded above
the surface and below the base and the lateral period the grid's.

  FLAT, FlatCoordinates    the system (depth, x, y)          (coordinates)
  Box, Rectangle           lateral domains                    (domains)
  Geometry                 a skeleton of depths over a domain (geometry)
  Model                    a flat model                       (model)
  VerticalStretch          the mapping of topography          (mapping)
  VerticalDisplacement,    the upward displacement driving
  ZeroDisplacement,        it, its adaptors, and a stack of
  CallableDisplacement,    reliefs linear in depth between
  as_displacement,         the boundaries                     (displacement)
  layer_linear
  DepthField               a field of depth alone             (fields)
  AnalyticField            a formula of (depth, x, y)         (fields)
  constant_field           a constant field of depth          (fields)
  Grid                     the horizontal grid of a domain    (grid)
  FourierField,            a Fourier series laterally, in
  FourierFunction          depth and alone                    (expansions)
  GriddedField,            values on a grid at GLL nodes, a
  GriddedFunction,         function and a vertical
  GriddedDisplacement      displacement on grids              (gridded)
  Sample, sample           a model on depth by grid nodes     (sampling)
  horizontal_mean,         the mean over the period at every
  horizontally_averaged    depth, of a field and a model      (means)
  DEPTH, ELASTIC, MAXWELL, the depth of a deck's knots and the
  PRONY, BURGERS           formats of decks of depths         (deck)
  depth_names              a deck's layer and interface names (deck)
  LayeredIsotropicElastic  layers of rho, vp, vs              (catalogue)
  CrustOverMantle          the reference flat model           (catalogue)
  ElasticDeckModel, MaxwellDeckModel, PronyDeckModel, BurgersDeckModel
                           the models of the decks of depths  (catalogue)
  LayerBasis               the coefficient vector of a field
                           over a grid and a mesh in depth    (randomfield)
  DepthGRF, BoxGRF,        Matern random fields of depth, of a
  LayeredGRF               box and of the layers              (randomfield)
"""
from .catalogue import (BurgersDeckModel, CrustOverMantle, ElasticDeckModel,
                        LayeredIsotropicElastic, MaxwellDeckModel, PronyDeckModel)
from .coordinates import FLAT, FlatCoordinates
from .deck import BURGERS, DEPTH, ELASTIC, MAXWELL, PRONY, depth_names
from .displacement import (CallableDisplacement, VerticalDisplacement, ZeroDisplacement,
                           as_displacement, layer_linear)
from .domains import Box, Rectangle
from .expansions import FourierField, FourierFunction
from .fields import AnalyticField, DepthField, constant_field
from .geometry import Geometry
from .grid import Grid
from .gridded import GriddedDisplacement, GriddedField, GriddedFunction
from .mapping import VerticalStretch
from .means import horizontal_mean, horizontally_averaged
from .model import Model
from .randomfield import BoxGRF, DepthGRF, LayerBasis, LayeredGRF
from .sampling import Sample, sample

__all__ = ["FLAT", "FlatCoordinates", "Box", "Rectangle", "Geometry", "Model",
           "DepthField", "AnalyticField", "constant_field", "DEPTH", "ELASTIC",
           "MAXWELL", "PRONY", "BURGERS", "depth_names", "LayeredIsotropicElastic",
           "CrustOverMantle", "ElasticDeckModel", "MaxwellDeckModel", "PronyDeckModel",
           "BurgersDeckModel", "VerticalStretch", "VerticalDisplacement",
           "ZeroDisplacement", "CallableDisplacement", "as_displacement",
           "layer_linear", "Grid", "FourierField", "FourierFunction", "GriddedField",
           "GriddedFunction", "GriddedDisplacement", "Sample", "sample",
           "horizontal_mean", "horizontally_averaged", "LayerBasis", "DepthGRF",
           "BoxGRF", "LayeredGRF"]
