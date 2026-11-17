"""The JSON that travels with every mesh.

A mesh file carries numbered attributes and nothing else.  Which number
is which layer, which boundary is which interface, where each sits,
whether the nodes have already been displaced, and which files beside
the mesh hold which fields: none of that survives into the mesh file,
so it travels alongside in a manifest, and this module is the single
definition of its shape.  A consumer reads the manifest, opens the mesh
the way it says, and knows where to look for every field and what each
one means.

The schema is `planetmodel.mesh.manifest/5` for a mesh of a spherical
geometry without faults, `planetmodel.mesh.manifest/6` for one with
them, and `planetmodel.mesh.manifest/7` for a mesh of a flat-layered
geometry (`planetmodel.cartesian`), with faults or without, with these
blocks:

  mesh          file, format ("msh" or "mfem"), nodes ("reference" or
                "physical"), read_options, displacement
  coordinates   schema 7 alone: the physical axes of the mesh's
                coordinates, in order
  layers[]      attribute, name, r_inner, r_outer (under schema 7
                depth_top, depth_bottom), in_geometry, fluid, and where
                the mesh has faults (schema 6, or 7 with faults)
                attributes
  interfaces[]  attribute, name, radius (under schema 7 depth),
                between_layers, kind, values
  faces[]       schema 7 alone: attribute, name, axis, value, outward
  refinements[] name, size, far_size, decay_width, points, digest
  faults[]      name, attribute, layers, hanging_wall, tip, free_edge,
                junction, meets, ends_on, controls, offset, sizing, and
                under schema 7 face_edge; where the mesh has faults alone
  fields[]      name, file, fe_space, vdim, ordering, rank, weight,
                voigt, unit, layers, radial_degree
  scales        length, mass, time, temperature in SI; null until a model
                is exported (temperature, where absent, is a kelvin)
  constants     the model's constants in its units; empty until then
  model         file, schema: a model file beside the manifest, on the
                mesh's reference skeleton (layer n of the file is
                attribute n + 1); null, or absent, where none was
                written
  meta          whatever the spec's `meta` carried

`fluid`, `kind`, `values` and `radial_degree` are read from the model,
so the model export writes them and everything else writes null: null
means "no model has said", never "no".  `fluid` is the vanishing of
every shear-bearing field of the model's layer (`materials.is_fluid`),
whichever fields are exported: a statement about this model's
description, not about any one timescale; a shell is not fluid.
`kind` classifies an interface from those flags (KINDS below);
`values` holds the one-sided values of each exported real scalar radial
field, `{name: [below, above]}`; `radial_degree` is the degree in r
that reproduces a scalar field within each layer holding it.

The layers and interfaces are the skeleton the mesh was built on, in
the mesh's own lengths: the mesher neither scales nor normalises, so a
radius here is a radius in the mesh.  `refinements` are the extra size
fields the mesh was sized under, recorded for provenance and change
detection — a point cloud by its count and digest, never the points
themselves.  Layers are numbered by
`attribute` 1..N from the centre and `between_layers` gives 0-based
layer indices with -1 for the outside.  Wherever a record names layers
(`fields[].layers`, `faults[].layers`, the model file's layer n being
the layer of attribute n + 1) it names them by that attribute, the
layer's own; under schema 5 it is the one attribute the layer's cells
carry, under schema 6 the first of the layer's `attributes`, the others
being its collars' (below).  A field is a file beside the
mesh, a finite-element space to read it into, a character (`rank`,
`weight`, `voigt`), a unit and the layers on which its values mean
anything.  The displacement m(X) - X is a field like
any other, a vector on every layer; `mesh.displacement` names it where
one was written, and `mesh.nodes` says whether the mesh's coordinates
are reference coordinates or already carry it.  Flat and boring on
purpose: a C++ reader will parse it, nested objects only where the
nesting carries meaning, no polymorphism, every number a number.

A flat mesh (schema 7) is a box, or a rectangle, the cross-section of a
body invariant along y, layered in depth below its free surface: points
are (x, y, z) with z up, the free surface z = 0, and the layered
coordinate is the depth -z (`layered_coordinate`).  `coordinates` names
the physical axes of the mesh's own coordinates: ["x", "y", "z"] for a
box, ["x", "z"] for a rectangle, whose mesh is two-dimensional with z
its second coordinate.  Its layers run from the surface down,
`depth_top` and `depth_bottom` their ends, attribute 1 the top one; its
interfaces are the tops of the layers, numbered the same way, interface
k (from one) at the top of layer k with `between_layers` [k - 1, k - 2],
the deeper layer first, the first the free surface (or a hollow box's
top) with [0, -1].  The base and the sides are no interfaces: they are
the box's faces (`faces[]`), whose boundary attributes follow the
interfaces' in the order of `FACES` (the base, then x_min, x_max, y_min
and y_max; a rectangle has no y faces), each the plane of constant
`axis` at `value` with the body on the side opposite `outward`, the sign
of its outward normal along the axis.  A box approximates an unbounded
body and its faces carry no physics: they are labelled for a solver to
put its own conditions on.  A field's components are those of (x, y, z)
over a rectangle too.  A flat mesh with faults is schema 7 as well: its
manifest has the records of the faults and of their collars as schema 6
has them, and one more curve of a fault, `face_edge`, where it leaves
the body through one of its faces (below); a flat mesh without faults
has neither `faults` nor its layers' `attributes`, as schema 5 has
neither, and a manifest listing one without the other is refused.  A
rectangle takes no faults.

A fault is an interior surface of the mesh, one conforming sheet of
faces with no nodes split, labelled by local sides.  Its faces are the
boundary elements of attribute `attribute`, M + k for the k-th fault
after the M interfaces (in a flat mesh after the interfaces and the
faces of the box), every one oriented with the fault's normal:
its vertices, in the order written, run so that (v1 - v0) x (v2 - v0)
points from the footwall into the hanging wall, so the cell on that
side of a face is the hanging-wall (plus) one and the other the
footwall (minus) one.  `layers` are the layers its faces lie in.  The
curves bounding it are dimension-1 physical groups
of the MSH file (an MFEM mesh keeps no curves; the same curves are the
edges of the fault's faces by the rule the format page gives): `tip`,
its buried tip, where its sides are joined; `free_edge`, where it meets
the model's outer surface (a box's free surface) and its sides are
free; `junction`, where it meets another fault, whose names `meets`
lists; and in a flat mesh `face_edge`, where it leaves the body through
the base or a side, an artificial boundary of a box that stands for an
unbounded body, its sides free as on the surface, which a consumer may
pin or leave free; each null where there is none.  `ends_on` names the
faults among those it ends on: where its
junction with one lies on its own boundary (a splay's edge on its main
fault), its slip is pinned along it as on its tip; where the junction
is interior to it (the main fault under the splay, two faults crossing)
it is not.  Fault k's curve groups are numbered 3k - 2, 3k - 1 and 3k,
in a flat mesh 4k - 3 to 4k (`fault_curve_groups`, in the order of
`FAULT_CURVES` or `FLAT_FAULT_CURVES`), whether or not they exist.
`controls` are the control nodes [n_u,
n_v] of each patch's pull-back through the mapping (of every piece of
a patch built in pieces where it crosses an interface at which the
mapping's derivative jumps), null under the identity, where the fault
is built in as given; `offset` the largest distance found between the
fault and its pull-back mapped forward, zero under the identity;
`sizing` the refinement towards it, {size, far_size, decay_width}, or
null.

A fault's slip lives on its faces' vertices and edges but those of its
pinned edges (its tip, and its junction where it ends on another
fault): its slip-carrying vertices and edges.  The cells holding one
of them, on the hanging-wall side, are the fault's COLLAR, and carry
element attributes of their own: one for each combination of a layer
and a set of faults whose collar a cell is in, numbered N + 1, N + 2,
... after the layers, in the order of the layer and then of the
faults' indices (compared as tuples).  A cell in no collar keeps its
layer's attribute.  `layers[k].attributes` lists the element attributes
the layer's cells carry, its own first, and `faults[k].hanging_wall`
the collar attributes of the fault's hanging wall; a split-node code
takes the indicator of those as its chi_HW, and since the attributes
are the cells' own they survive partitioning, uniform refinement (a
child inherits its parent's) and pre-partitioned files.  A mesh
without faults has no collars, and its manifest is schema 5 exactly
as it was (a flat one's schema 7 without the faults' records).  A
fault's face edge carries slip in the classification as its free edge
does: a consumer that pins it finds the collar cells there touching the
fault only where its slip is zero, which a split-node sum takes as it
takes any such cell, so the one classification serves either choice.

Three rules turn a fault's faces into what a split-node code needs, and
are the format page's: which of a face's two cells is the plus one
(`fault_side`, `plus_cell`); which edges of a fault's faces are its
tip, free edge, face edge and junction, and of its junction where it
ends, read from the faces alone where the mesh format keeps no curves
(`fault_curves`); and which cells are in its hanging wall and which in
its footwall (`fault_walls`, `walls_of_faults`), which the mesher
writes as the collar attributes (`collars`).  All take plain arrays
(the vertices' coordinates, corners as indices into them), so they
apply to a mesh however it was read; the C++ library states them
again.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, asdict, dataclass, field, fields as _fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, get_type_hints

import numpy as np
from numpy.typing import ArrayLike

from ..character import Character
from ..fields import stored_shape
from ..units import Dimensions, Scales, unit_string

if TYPE_CHECKING:
    from ..character import Character
    from ..geometry import InterfaceInfo, LayerInfo
    from ..model import Model
    from .spec import Refinement

__all__ = ["SCHEMA", "SCHEMA_WITH_FAULTS", "SCHEMA_FLAT", "SCHEMAS",
           "FORMATS", "NODES", "KINDS", "AXES", "FACES",
           "FLAT_COORDINATES", "FAULT_CURVES", "FLAT_FAULT_CURVES",
           "MeshManifest", "LayerEntry", "FlatLayerEntry",
           "InterfaceEntry", "FlatInterfaceEntry", "FaceEntry", "RefinementEntry",
           "FaultEntry", "FieldEntry",
           "write", "read", "beside",
           "mesh_block", "scales_block", "constants_block", "fault_curve_groups",
           "validate_structure", "validate_against", "fault_side", "plus_cell",
           "CurveEdges", "fault_curves", "fault_walls", "walls_of_faults",
           "Collars", "collars", "check_collars"]

#: The schema of a manifest whose mesh has no fault, the cells of layer n
#: all carrying attribute n.  Bump only for an incompatible change;
#: consumers check it.
SCHEMA = "planetmodel.mesh.manifest/5"

#: The schema of a manifest whose mesh has faults: the cells of the
#: faults' collars carry attributes beyond the layers', which a reader of
#: `SCHEMA` would take for no layer, so each layer lists the attributes
#: its cells carry (`layers[].attributes`) and each fault its hanging
#: wall's (`faults[].hanging_wall`).
SCHEMA_WITH_FAULTS = "planetmodel.mesh.manifest/6"

#: The schema of a manifest whose mesh is flat-layered, a box or a
#: rectangle, with faults or without: its layers and interfaces by depth
#: from the surface down, its `coordinates` and its `faces[]` (module
#: docstring), which a reader of `SCHEMA` would take for radii and its
#: top for its centre; and where it has faults the faults' and the
#: collars' records as `SCHEMA_WITH_FAULTS` has them, with each fault's
#: `face_edge`, the curve where it leaves the box through its base or a
#: side, which a reader of `SCHEMA_WITH_FAULTS` would take for a tip and
#: pin.
SCHEMA_FLAT = "planetmodel.mesh.manifest/7"

#: The schemas `read` reads.
SCHEMAS = (SCHEMA, SCHEMA_WITH_FAULTS, SCHEMA_FLAT)

#: The curves of a fault, in the order of its dimension-1 groups
#: (`fault_curve_groups`): its buried tip, its free edge on the outer
#: surface and its junction with other faults.
FAULT_CURVES = ("tip", "free_edge", "junction")

#: The curves of a fault in a flat mesh: `FAULT_CURVES` and its face edge,
#: where it leaves the box through its base or a side.
FLAT_FAULT_CURVES = FAULT_CURVES + ("face_edge",)

#: The physical axes, in order; a flat mesh's `coordinates` name some of
#: them.
AXES = ("x", "y", "z")

#: The `coordinates` of a flat mesh: a box's, and a rectangle's, whose mesh
#: is two-dimensional with z its second coordinate.
FLAT_COORDINATES = (("x", "y", "z"), ("x", "z"))

#: The faces of a flat mesh, (name, axis, outward), in the order their
#: boundary attributes take after the interfaces': each is the plane of
#: constant `axis` with the body on the side opposite `outward`, the sign
#: of its outward normal along the axis.  A mesh has those whose axis is
#: among its `coordinates`: a box all five, a rectangle the base and the x
#: faces.  The names are the faces' physical groups in the mesh file.
FACES = (("base", "z", -1), ("x_min", "x", -1), ("x_max", "x", 1),
         ("y_min", "y", -1), ("y_max", "y", 1))

#: The keys only a flat manifest has.
_FLAT_KEYS = ("coordinates", "faces")

#: The MSH format version the mesher writes and MFEM's reader wants.
MSH_VERSION = 2.2

#: The mesh file formats a manifest may sit beside.
FORMATS = ("msh", "mfem")

#: What the mesh's node coordinates are: reference coordinates, or the
#: physical ones of a mesh whose nodes were displaced by a mapping.
NODES = ("reference", "physical")

#: What an interface's `kind` may be: the fluidity of the two model
#: layers it separates; "free" where a model layer meets the outside or
#: a shell (the model's surface, or a hollow model's inner boundary);
#: "outer" for the outer boundary of a mesh whose outermost layer is a
#: shell.  A boundary between two shells has no kind.
KINDS = ("solid-solid", "fluid-solid", "fluid-fluid", "free", "outer")


def beside(path: str | Path, suffix: str) -> Path:
    """`path` with `suffix`, treating the path as a basename.

    Only a mesh or manifest suffix is replaced; anything else is part of
    the name, so `run.v1.5` becomes `run.v1.5.json` rather than
    `run.v1.json`.
    """
    path = Path(path)
    if path.suffix in (".msh", ".mesh", ".json"):
        path = path.with_suffix("")
    return path.with_name(path.name + suffix)


# ------------------------------------------------------------ the entries

@dataclass
class LayerEntry:
    """One `layers[]` record: a layer of the mesh, centre outward."""

    #: The layer's own element attribute, 1..N from the centre, by which
    #: every record names it.
    attribute: int
    #: The layer's name; `layer_<attribute>` where none was given.
    name: str
    #: The inner radius, in the mesh's lengths.
    r_inner: float
    #: The outer radius, in the mesh's lengths.
    r_outer: float
    #: Whether the layer belongs to the geometry (True) or is a shell (False).
    in_geometry: bool
    _: KW_ONLY
    #: Whether shear vanishes throughout the model's layer; False for a
    #: shell, null until a model export reads it (or where it cannot).
    fluid: bool | None = None
    #: The element attributes its cells carry, its own first and then its
    #: collars' (module docstring); a manifest with faults alone has the
    #: key, its cells all carrying `attribute` where there are none.
    attributes: list | None = None

    @classmethod
    def from_layer(cls, layer: LayerInfo, *, attribute: int, r_inner: float,
                   r_outer: float, in_geometry: bool,
                   attributes: Sequence[int] | None = None) -> "LayerEntry":
        """The record of a LayerInfo between the given radii, its cells
        carrying `attributes` where its mesh has faults."""
        return cls(attribute=int(attribute),
                   name=layer.name or f"layer_{attribute}",
                   r_inner=float(r_inner), r_outer=float(r_outer),
                   in_geometry=bool(in_geometry),
                   attributes=None if attributes is None
                   else [int(a) for a in attributes])


@dataclass
class InterfaceEntry:
    """One `interfaces[]` record: a boundary of the mesh, centre outward.

    `between_layers` is `[below, above]` as 0-based layer indices, with
    -1 for the outside of the mesh: the outermost interface has
    `[N - 1, -1]` and the inner boundary of a hollow mesh `[-1, 0]`.
    """

    #: The boundary attribute, 1..M from the centre.
    attribute: int
    #: The interface's name; `interface_<attribute>` where none was given.
    name: str
    #: The radius, in the mesh's lengths.
    radius: float
    #: [layer below, layer above], 0-based, -1 outside.
    between_layers: list
    _: KW_ONLY
    #: One of KINDS; null until a model export can read it.
    kind: str | None = None
    #: {field name: [value below, value above]} for the exported real
    #: scalar radial fields, null on a side holding no such value; null
    #: until a model export writes it.
    values: dict | None = None

    @classmethod
    def from_interface(cls, face: InterfaceInfo, *, attribute: int,
                       radius: float) -> "InterfaceEntry":
        """The record of an InterfaceInfo at the given radius."""
        below, above = face.between
        return cls(attribute=int(attribute),
                   name=face.name or f"interface_{attribute}",
                   radius=float(radius),
                   between_layers=[int(below), int(above)])


@dataclass
class FlatLayerEntry:
    """One `layers[]` record of a flat mesh (schema 7): a layer, from the
    surface down."""

    #: The layer's own element attribute, 1..N from the surface down.
    attribute: int
    #: The layer's name; `layer_<attribute>` where none was given.
    name: str
    #: The depth of its top, in the mesh's lengths.
    depth_top: float
    #: The depth of its bottom, in the mesh's lengths.
    depth_bottom: float
    #: Whether the layer belongs to the geometry: every layer of a flat mesh.
    in_geometry: bool
    _: KW_ONLY
    #: Whether shear vanishes throughout the model's layer; null until a
    #: model export reads it (or where it cannot).
    fluid: bool | None = None
    #: The element attributes its cells carry, its own first and then its
    #: collars', as `LayerEntry.attributes`: where the mesh has faults alone.
    attributes: list | None = None

    @classmethod
    def from_layer(cls, layer: LayerInfo, *, attribute: int, depth_top: float,
                   depth_bottom: float,
                   attributes: Sequence[int] | None = None) -> "FlatLayerEntry":
        """The record of a LayerInfo between the given depths, its cells
        carrying `attributes` where its mesh has faults."""
        return cls(attribute=int(attribute), name=layer.name or f"layer_{attribute}",
                   depth_top=float(depth_top), depth_bottom=float(depth_bottom),
                   in_geometry=True,
                   attributes=None if attributes is None
                   else [int(a) for a in attributes])


@dataclass
class FlatInterfaceEntry:
    """One `interfaces[]` record of a flat mesh (schema 7): the top of a
    layer, from the surface down.

    `between_layers` is `[below, above]` as 0-based layer indices, the
    deeper layer first, with -1 for the outside: the free surface has
    `[0, -1]` and the top of layer k `[k, k - 1]`.
    """

    #: The boundary attribute, 1..N from the surface down.
    attribute: int
    #: The interface's name; `interface_<attribute>` where none was given.
    name: str
    #: The depth, in the mesh's lengths.
    depth: float
    #: [layer below, layer above], 0-based, -1 outside.
    between_layers: list
    _: KW_ONLY
    #: One of KINDS; null until a model export can read it.
    kind: str | None = None
    #: {field name: [value below, value above]}, as `InterfaceEntry.values`.
    values: dict | None = None

    @classmethod
    def from_interface(cls, face: InterfaceInfo, *, attribute: int,
                       depth: float) -> "FlatInterfaceEntry":
        """The record of an InterfaceInfo at the given depth."""
        below, above = face.between
        return cls(attribute=int(attribute), name=face.name or f"interface_{attribute}",
                   depth=float(depth), between_layers=[int(below), int(above)])


@dataclass
class FaceEntry:
    """One `faces[]` record of a flat mesh (schema 7): a face of the body
    that is no interface, its base or a side, labelled for a solver's own
    conditions.

    The face is the plane `axis` = `value` (in the mesh's lengths, z up)
    with the body on the side opposite `outward`, the sign of its outward
    normal along the axis; its name, axis and outward are a row of
    `FACES`, and its boundary attribute follows the interfaces'.
    """

    #: The boundary attribute of its elements, after the interfaces'.
    attribute: int
    #: Its name: "base", "x_min", "x_max", "y_min" or "y_max".
    name: str
    #: The axis normal to it: "x", "y" or "z".
    axis: str
    #: The coordinate along `axis` of its plane.
    value: float
    #: The sign of its outward normal along `axis`, -1 or 1.
    outward: int


@dataclass
class RefinementEntry:
    """One `refinements[]` record: an extra size field the mesh was
    sized under.

    Provenance and change detection only: a consumer needs none of it
    to open the mesh.  `size` and `far_size` are the floor and ceiling
    of the sizes the refinement asked for; `decay_width`, `points` and
    `digest` are what the shipped kinds expose and null for a bare
    callable, with `points` a cloud's count and `digest` the identity
    of its coordinates.
    """

    #: The refinement's class, or a function's name.
    name: str
    #: The smallest size it asked for.
    size: float
    #: The largest size it asked for.
    far_size: float
    _: KW_ONLY
    #: The ramp width; null where the refinement exposes none.
    decay_width: float | None = None
    #: The number of points of a cloud; null where there is no cloud.
    points: int | None = None
    #: The digest identifying a cloud's coordinates; null without one.
    digest: str | None = None

    @classmethod
    def from_refinement(cls, refinement: "Refinement", *,
                        name: str) -> "RefinementEntry":
        """The record of a refinement, reading what it exposes."""
        decay = getattr(refinement, "decay_width", None)
        cloud = getattr(refinement, "points", None)
        digest = getattr(refinement, "digest", None)
        return cls(name=str(name), size=float(refinement.size),
                   far_size=float(refinement.far_size),
                   decay_width=None if decay is None else float(decay),
                   points=None if cloud is None else len(cloud),
                   digest=None if digest is None else str(digest))


def fault_curve_groups(k: int, *, curves: Sequence[str] = FAULT_CURVES
                       ) -> tuple[int, ...]:
    """The dimension-1 physical groups of fault k (from one), one for each
    of its `curves` in their order, n of them numbered n(k - 1) + 1 to
    nk: its tip, free edge and junction curves 3k - 2, 3k - 1 and 3k, and
    in a flat mesh (`FLAT_FAULT_CURVES`) those and its face edge, 4k - 3
    to 4k."""
    n = len(curves)
    return tuple(n * (k - 1) + j for j in range(1, n + 1))


@dataclass
class FaultEntry:
    """One `faults[]` record: a fault built into the mesh as an interior
    surface (module docstring).

    `attribute` is the boundary attribute of its faces, `layers` the
    layers they lie in and `hanging_wall` the element attributes of its
    collar; `tip`, `free_edge`, `junction` and in a flat mesh
    `face_edge` are the dimension-1 physical groups of its curves in the
    MSH file, null where it has none (`face_edge` is a flat mesh's
    alone, and written only there), `meets` the names of the faults it meets
    along its junction
    curves and `ends_on` those of them it ends on, its slip pinned along
    the junction there.  `controls` holds each
    patch's pull-back grid [n_u, n_v], that of each piece of a patch
    built in pieces, null where the mapping is the identity; `offset`
    is the largest distance found between the fault and its pull-back
    mapped forward, in the mesh's lengths; `sizing` the refinement
    towards it, {size, far_size, decay_width}, or null.
    """

    #: The fault's name; `fault_<k>` where none was given.
    name: str
    #: The boundary attribute of its faces: M + k, after the interfaces.
    attribute: int
    #: The layers its faces lie in, by their own attributes.
    layers: list
    _: KW_ONLY
    #: The element attributes of the cells of its hanging wall, its collar.
    hanging_wall: list = field(default_factory=list)
    #: The physical group of its buried tip curves, or null.
    tip: int | None = None
    #: The physical group of its curves on the outer surface, or null.
    free_edge: int | None = None
    #: The physical group of its curves shared with other faults, or null.
    junction: int | None = None
    #: The names of the faults it meets.
    meets: list = field(default_factory=list)
    #: The names of the faults it ends on, of those it meets.
    ends_on: list = field(default_factory=list)
    #: Each patch's pull-back control nodes [n_u, n_v]; null under the identity.
    controls: list | None = None
    #: The largest distance between the fault and its pull-back mapped forward.
    offset: float = 0.0
    #: The refinement towards the fault, {size, far_size, decay_width}, or null.
    sizing: dict | None = None
    #: The physical group of its curves on the faces of a box, its base and
    #: sides, or null: a flat mesh's alone.
    face_edge: int | None = None


@dataclass
class FieldEntry:
    """One `fields[]` record: a file beside the mesh and what it holds.

    `file`, `fe_space`, `vdim` and `ordering` are what a consumer builds
    the space from before it opens the file; `rank`, `weight` and
    `voigt` are the field's character, `unit` its unit string under the
    manifest's scales ("1" where no SI scale is declared, "unknown" for
    a name without dimensions), and `layers` the layers on which its
    values mean anything, by their own attributes (their cells carry
    `MeshManifest.element_attributes` of them): elsewhere the file holds
    zeros.
    """

    name: str
    file: str
    fe_space: str
    vdim: int
    ordering: str
    rank: int
    weight: int
    voigt: bool
    unit: str
    layers: list
    _: KW_ONLY
    #: The polynomial degree in r reproducing a scalar field within
    #: each layer holding it; null where no single degree does, for a
    #: field of higher rank, and until a model export writes it.
    radial_degree: int | None = None

    @classmethod
    def from_field(cls, name: str, file: str | Path, *, fe_space: str, vdim: int,
                   ordering: str, character: Character,
                   dimensions: Dimensions | None, si: bool,
                   layers: Iterable[int],
                   radial_degree: int | None = None) -> "FieldEntry":
        """The record of a field of `character` and `dimensions` written
        to `file` in the given space, on the given layer attributes."""
        return cls(name=str(name), file=Path(file).name, fe_space=str(fe_space),
                   vdim=int(vdim), ordering=str(ordering),
                   rank=int(character.rank), weight=int(character.weight),
                   voigt=character.voigt_shape is not None,
                   unit=unit_string(dimensions, si=si),
                   layers=[int(a) for a in layers],
                   radial_degree=None if radial_degree is None
                   else int(radial_degree))


#: The JSON types each annotated Python type may hold.
_JSON_TYPES = {int: (int,), float: (int, float), bool: (bool,), str: (str,),
               list: (list,), dict: (dict,),
               int | None: (int, type(None)), bool | None: (bool, type(None)),
               str | None: (str, type(None)), dict | None: (dict, type(None)),
               list | None: (list, type(None)),
               float | None: (int, float, type(None))}


def _entry_types(cls: type) -> dict[str, tuple[type, ...]]:
    """Each field of an entry and the JSON types it must hold."""
    hints = get_type_hints(cls)
    return {f.name: _JSON_TYPES[hints[f.name]] for f in _fields(cls)
            if f.name != "_"}


# ------------------------------------------------------------ the blocks

def mesh_block(file: str | Path, *, format: str, nodes: str,
               read_options: Mapping[str, Any] | None = None,
               displacement: str | None = None) -> dict[str, Any]:
    """The `mesh` record: the file, its format, what its coordinates are,
    the options a consumer must open it with (for MFEM, the `mfem::Mesh`
    constructor arguments the dof numbering of the fields was written
    under), and the name of the displacement field where one was written.
    """
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
    if nodes not in NODES:
        raise ValueError(f"nodes must be one of {NODES}, got {nodes!r}")
    return {"file": Path(file).name, "format": str(format), "nodes": str(nodes),
            "read_options": dict(read_options or {}),
            "displacement": None if displacement is None else str(displacement)}


def scales_block(scales: Scales) -> dict[str, float]:
    """The `scales` record: what one stored unit of length, mass, time and
    temperature is in SI, so that every field value and constant is in
    those units."""
    return {"length": float(scales.length), "mass": float(scales.mass),
            "time": float(scales.time), "temperature": float(scales.temperature)}


def constants_block(model: Model) -> dict[str, float]:
    """The `constants` record: the model's constants, in its units."""
    return {str(k): float(model.constant(k)) for k in model.constants}


# ---------------------------------------------------------- the manifest

@dataclass
class MeshManifest:
    """Everything a consumer needs that the mesh file cannot carry."""

    _: KW_ONLY
    #: The mesh file and how to open it; see `mesh_block`.
    mesh: dict[str, Any] = field(default_factory=dict)
    #: The physical axes of the mesh's coordinates, in order, a row of
    #: `FLAT_COORDINATES`: schema 7 alone, None (and not written)
    #: otherwise.
    coordinates: list[str] | None = None
    layers: list[dict[str, Any]] = field(default_factory=list)
    interfaces: list[dict[str, Any]] = field(default_factory=list)
    #: One record per face of a flat mesh that is no interface, its base
    #: and sides; see `FaceEntry`.  Schema 7 alone, and written only
    #: there.
    faces: list[dict[str, Any]] = field(default_factory=list)
    #: One record per refinement the mesh was sized under; see
    #: `RefinementEntry`.
    refinements: list[dict[str, Any]] = field(default_factory=list)
    #: One record per fault built into the mesh; see `FaultEntry`.  The
    #: key is written only where there is one.
    faults: list[dict[str, Any]] = field(default_factory=list)
    #: One record per field written beside the mesh; see `FieldEntry`.
    fields: list[dict[str, Any]] = field(default_factory=list)
    #: The scales the field values are in, or None until a model is exported.
    scales: dict[str, float] | None = None
    #: The model's constants in its units; empty until a model is exported.
    constants: dict[str, float] = field(default_factory=dict)
    #: The model file beside the mesh, `{file, schema}`, or None.
    model: dict[str, str] | None = None
    #: The spec's `meta`, copied through.
    meta: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    @classmethod
    def from_build(cls, *, mesh: Mapping[str, Any],
                   layers: Sequence[LayerEntry | FlatLayerEntry],
                   interfaces: Sequence[InterfaceEntry | FlatInterfaceEntry],
                   refinements: Sequence[RefinementEntry] = (),
                   faults: Sequence[FaultEntry] = (),
                   faces: Sequence[FaceEntry] = (),
                   coordinates: Sequence[str] | None = None,
                   meta: Mapping[str, Any] | None = None) -> "MeshManifest":
        """A manifest from the mesh block and the typed entries.

        The one place a manifest is assembled, so a key cannot drift
        between builders and the schema is described once: where
        `coordinates` are given, a flat mesh's, schema 7 with the
        `coordinates` and `faces`, and with the layers' `attributes` and
        the faults' `face_edge` where there are faults, neither where
        there are none; otherwise schema 6 with the layers' `attributes`
        where there are faults, schema 5 without them where there are
        none, neither with a fault's `face_edge`.
        """
        rows = [asdict(e) for e in layers]
        if not faults:
            for row in rows:
                row.pop("attributes", None)
        flat = coordinates is not None
        faulted = [asdict(e) for e in faults]
        if not flat:
            for row in faulted:
                row.pop("face_edge", None)
        card = cls(mesh=dict(mesh),
                   coordinates=[str(a) for a in coordinates] if flat else None,
                   layers=rows, interfaces=[asdict(e) for e in interfaces],
                   faces=[asdict(e) for e in faces],
                   refinements=[asdict(e) for e in refinements],
                   faults=faulted, meta=dict(meta or {}),
                   schema=_schema(flat=flat, faults=bool(faults)))
        validate_structure(card)
        return card

    # -- the layered coordinate, read by the schema --------------------------

    @property
    def is_flat(self) -> bool:
        """Whether the mesh is flat-layered (schema 7): a box or a rectangle,
        layered in depth."""
        return self.schema == SCHEMA_FLAT

    @property
    def has_faults(self) -> bool:
        """Whether the manifest is one of a mesh with faults, whose layers
        list the attributes of their collars: schema 6, or a flat mesh's
        (schema 7) listing faults."""
        return self.schema == SCHEMA_WITH_FAULTS or (self.is_flat
                                                     and bool(self.faults))

    @property
    def fault_curves(self) -> tuple[str, ...]:
        """The curves of a fault of this mesh, in the order of its groups
        (`fault_curve_groups`): `FLAT_FAULT_CURVES` in a flat mesh,
        `FAULT_CURVES` otherwise."""
        return FLAT_FAULT_CURVES if self.is_flat else FAULT_CURVES

    @property
    def face_attributes(self) -> tuple[int, ...]:
        """The boundary attributes of a flat mesh's faces, its base and
        sides (`faces[]`), where a fault's face edge lies; none otherwise."""
        return tuple(int(e["attribute"]) for e in self.faces)

    def _keys(self) -> tuple[str, str, str]:
        """The keys of a layer's two ends and of an interface's place."""
        if self.is_flat:
            return "depth_top", "depth_bottom", "depth"
        return "r_inner", "r_outer", "radius"

    @property
    def coordinate(self) -> str:
        """The name of the layered coordinate: "radius", or "depth" in a
        flat mesh."""
        return self._keys()[2]

    @property
    def boundaries(self) -> np.ndarray:
        """The layered coordinate at the ends of the layers, (N + 1,), in
        their order: radii from the centre out, depths from the surface
        down."""
        lo, hi, _ = self._keys()
        return np.array([float(self.layers[0][lo])]
                        + [float(e[hi]) for e in self.layers])

    def layer_interval(self, entry: Mapping[str, Any]) -> tuple[float, float]:
        """The ends of a `layers[]` record in the layered coordinate,
        increasing: (r_inner, r_outer), or (depth_top, depth_bottom)."""
        lo, hi, _ = self._keys()
        return float(entry[lo]), float(entry[hi])

    def interface_level(self, entry: Mapping[str, Any]) -> float:
        """The layered coordinate an `interfaces[]` record sits at: its
        radius, or in a flat mesh its depth."""
        return float(entry[self._keys()[2]])

    def mesh_axes(self, dimension: int) -> tuple[int, ...]:
        """The physical axes (0 for x, 1 for y, 2 for z) of the coordinates
        of the mesh, of `dimension` coordinates, in order: a flat mesh's
        `coordinates`, a rectangle's x and z; any other's x, y and z, a
        disc's x and y.  Refused where a flat mesh has another number."""
        if not self.is_flat:
            return tuple(range(int(dimension)))
        axes = tuple(AXES.index(a) for a in self.coordinates)
        if len(axes) != int(dimension):
            raise ValueError(f"a mesh of {dimension} coordinates is not one whose "
                             f"coordinates are {self.coordinates}")
        return axes

    def layered_coordinate(self, points: ArrayLike) -> np.ndarray:
        """The layered coordinate of physical points (..., 3): the radius
        |X|, or in a flat mesh the depth -z."""
        X = np.asarray(points, dtype=float)
        if self.is_flat:
            return -X[..., 2]
        return np.linalg.norm(X, axis=-1)

    def pulled_into(self, points: ArrayLike, interval: tuple[float, float]
                    ) -> np.ndarray:
        """Physical points (..., 3) moved along the layered coordinate
        into `interval`, its [lo, hi], their lateral place kept: radially,
        the direction untouched, or in a flat mesh vertically."""
        lo, hi = (float(x) for x in interval)
        X = np.asarray(points, dtype=float)
        if self.is_flat:
            out = X.copy()
            out[..., 2] = -np.clip(-X[..., 2], lo, hi)
            return out
        r = np.linalg.norm(X, axis=-1)
        safe = np.where(r > 0.0, r, 1.0)
        return X * (np.clip(r, lo, hi) / safe)[..., None]

    @property
    def scale(self) -> float:
        """The length the mesh's tolerances are relative to: its largest
        coordinate in magnitude, the outer radius of a spherical mesh,
        the base's depth or a side's place in a flat one."""
        if not self.is_flat:
            return self.outer_radius
        return max([abs(float(self.boundaries[-1]))]
                   + [abs(float(e["value"])) for e in self.faces])

    def layer_attribute(self, name: str) -> int:
        """The own attribute of a named layer, by which records name it."""
        for entry in self.layers:
            if entry.get("name") == name:
                return int(entry["attribute"])
        raise KeyError(
            f"no layer named {name!r}; layers are "
            f"{[e.get('name') for e in self.layers]}")

    def _attributes(self, entry: Mapping[str, Any]) -> list[int]:
        """The element attributes a layer's cells carry, its own first."""
        if self.has_faults:
            return [int(a) for a in entry["attributes"]]
        return [int(entry["attribute"])]

    def element_attributes(self, *, layers: Iterable[int] | None = None
                           ) -> tuple[int, ...]:
        """The element attributes the cells of the given layers carry,
        each layer's own and then its collars', the layers in the order
        given; every layer's where None.

        The layers are named by their own attributes, as `fields[].layers`
        and `faults[].layers` name them: an MFEM marker for the cells of a
        layer, or of the layers where a field means anything, is made from
        this list.  KeyError for a number that is no layer's own attribute.
        """
        by = {int(e["attribute"]): e for e in self.layers}
        out = []
        for a in (by if layers is None else layers):
            if int(a) not in by:
                raise KeyError(f"no layer has its own attribute {a}; the layers' "
                               f"are {sorted(by)}")
            out += self._attributes(by[int(a)])
        return tuple(out)

    def layer_of(self, attribute: int) -> dict[str, Any]:
        """The record of the layer whose cells carry an element
        attribute: its own, or one of its collars'.  KeyError where no
        layer's cells carry it."""
        for entry in self.layers:
            if int(attribute) in self._attributes(entry):
                return entry
        raise KeyError(f"no layer's cells carry element attribute {attribute}; "
                       f"they carry {list(self.element_attributes())}")

    def interface_attribute(self, name: str) -> int:
        """The attribute number of a named interface."""
        for entry in self.interfaces:
            if entry.get("name") == name:
                return int(entry["attribute"])
        raise KeyError(
            f"no interface named {name!r}; interfaces are "
            f"{[e.get('name') for e in self.interfaces]}")

    def face_attribute(self, name: str) -> int:
        """The boundary attribute of a named face of a flat mesh: "base",
        "x_min", "x_max", "y_min" or "y_max"."""
        for entry in self.faces:
            if entry.get("name") == name:
                return int(entry["attribute"])
        raise KeyError(f"no face named {name!r}; faces are "
                       f"{[e.get('name') for e in self.faces]}")

    def fault_record(self, name: str) -> dict[str, Any]:
        """The record of a named fault."""
        for entry in self.faults:
            if entry.get("name") == name:
                return entry
        raise KeyError(
            f"no fault named {name!r}; faults are "
            f"{[e.get('name') for e in self.faults]}")

    def field_record(self, name: str) -> dict[str, Any]:
        """The record of a named field."""
        for entry in self.fields:
            if entry.get("name") == name:
                return entry
        raise KeyError(
            f"no field named {name!r}; fields are "
            f"{[e.get('name') for e in self.fields]}")

    @property
    def shell_attributes(self) -> tuple[int, ...]:
        """The element attributes of the layers outside the geometry, the
        shells, which hold no fault and so no collar."""
        return self.element_attributes(layers=[int(e["attribute"]) for e in self.layers
                                               if not e.get("in_geometry", True)])

    @property
    def surface_attribute(self) -> int:
        """The attribute of the geometry's outer surface: the interface
        with the geometry's outermost layer below it, a fault's free edge
        lying on it; in a flat mesh its top, `[0, -1]` between."""
        if self.is_flat:
            for e in self.interfaces:
                if list(e["between_layers"]) == [0, -1]:
                    return int(e["attribute"])
            raise ValueError("the manifest lists no top of the flat mesh")
        body = [int(e["attribute"]) for e in self.layers if e.get("in_geometry", True)]
        for e in self.interfaces:
            if body and int(e["between_layers"][0]) + 1 == body[-1]:
                return int(e["attribute"])
        raise ValueError("the manifest lists no surface of the geometry")

    def _radial(self, what: str) -> None:
        """Refuse a radius of a flat mesh, naming what was asked."""
        if self.is_flat:
            raise ValueError(f"a flat mesh has no {what}: its layers run in depth "
                             "(`boundaries`, `scale`)")

    @property
    def inner_radius(self) -> float:
        """The inner radius of the mesh: zero for a full one; a flat mesh
        has none."""
        self._radial("inner radius")
        return float(self.layers[0]["r_inner"])

    @property
    def outer_radius(self) -> float:
        """The outer radius of the mesh, shells included; a flat mesh has
        none."""
        self._radial("outer radius")
        return float(self.layers[-1]["r_outer"])

    def describe(self) -> str:
        """A readable multi-line summary of the manifest.

        The schema, the mesh line, one line per layer, per interface
        and per field, and the scales and constants where a model was
        exported.  Every length is in the mesh's own numbers.
        """
        m = self.mesh
        options = _pairs(m.get("read_options", {}))
        lines = [f"{self.schema}",
                 f"  mesh        {m.get('file', '?')} ({m.get('format', '?')}), "
                 f"{m.get('nodes', '?')} nodes"
                 + (f", read options {options}" if options else "")
                 + (f", displacement {m['displacement']}"
                    if m.get("displacement") else "")]
        if self.coordinates is not None:
            lines.append(f"  coordinates {', '.join(self.coordinates)}")
        lo, hi, at = self._keys()
        lines.append("  layers")
        rows = [(str(e.get("attribute", "?")), str(e.get("name", "")),
                 f"[{_num(e.get(lo))}, {_num(e.get(hi))}]",
                 "in geometry" if e.get("in_geometry", True) else "shell",
                 {True: "fluid", False: "solid"}.get(e.get("fluid"), ""),
                 f"collars {list(e['attributes'][1:])}"
                 if len(e.get("attributes") or ()) > 1 else "")
                for e in self.layers]
        lines.extend(_table(rows))
        lines.append("  interfaces")
        rows = [(str(e.get("attribute", "?")), str(e.get("name", "")),
                 f"{at} {_num(e.get(at))}",
                 f"between layers {list(e.get('between_layers', []))}",
                 str(e.get("kind") or ""))
                for e in self.interfaces]
        lines.extend(_table(rows))
        if self.faces:
            lines.append("  faces")
            rows = [(str(e.get("attribute", "?")), str(e.get("name", "")),
                     f"{e.get('axis', '?')} = {_num(e.get('value'))}",
                     f"outward {_num(e.get('outward'))}")
                    for e in self.faces]
            lines.extend(_table(rows))
        if self.refinements:
            lines.append("  refinements")
            rows = [(str(e.get("name", "?")),
                     f"size {_num(e.get('size'))} to {_num(e.get('far_size'))}"
                     + (f" over {_num(e['decay_width'])}"
                        if e.get("decay_width") is not None else ""),
                     f"{e['points']} points" if e.get("points") is not None
                     else "",
                     str(e.get("digest") or ""))
                    for e in self.refinements]
            lines.extend(_table(rows, numbered=False))
        if self.faults:
            lines.append("  faults")
            rows = [(str(e.get("attribute", "?")), str(e.get("name", "")),
                     f"in layers {list(e.get('layers', []))}",
                     ", ".join(f"{key.replace('_', ' ')} {e[key]}"
                               for key in self.fault_curves
                               if e.get(key) is not None),
                     f"meets {', '.join(e['meets'])}" if e.get("meets") else "",
                     f"ends on {', '.join(e['ends_on'])}" if e.get("ends_on")
                     else "",
                     f"hanging wall {list(e.get('hanging_wall', []))}",
                     f"offset {_num(e.get('offset'))}")
                    for e in self.faults]
            lines.extend(_table(rows))
        if self.fields:
            lines.append("  fields")
            rows = [(str(e.get("name", "?")), str(e.get("file", "?")),
                     f"{e.get('fe_space', '?')} vdim {e.get('vdim', '?')} "
                     f"{e.get('ordering', '?')}",
                     f"rank {e.get('rank', '?')} weight {e.get('weight', '?')}"
                     + (" Voigt" if e.get("voigt") else ""),
                     str(e.get("unit", "?")),
                     f"layers {list(e.get('layers', []))}")
                    for e in self.fields]
            lines.extend(_table(rows, numbered=False))
        if self.scales is not None:
            lines.append("  scales      " + _pairs(self.scales)
                         + (", constants " + _pairs(self.constants)
                            if self.constants else ""))
        if self.model is not None:
            lines.append(f"  model       {self.model.get('file', '?')} "
                         f"({self.model.get('schema', '?')})")
        if self.meta:
            lines.append("  meta        " + _pairs(self.meta))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        faces = f"{len(self.faces)} faces, " if self.faces else ""
        faults = f"{len(self.faults)} faults, " if self.faults else ""
        return (f"MeshManifest({self.schema}, {self.mesh.get('file', '?')}, "
                f"{len(self.layers)} layers, {len(self.interfaces)} interfaces, "
                f"{faces}{faults}{len(self.fields)} fields)")


def _num(value: object) -> str:
    """A number for `describe`, short and exact enough to recognise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _pairs(record: Mapping[str, Any]) -> str:
    """`key value` pairs of a flat record, for `describe`."""
    return ", ".join(f"{k} {_num(v)}" for k, v in record.items())


def _table(rows: Sequence[Sequence[str]], *, numbered: bool = True) -> list[str]:
    """Rows of strings as indented lines with their columns aligned, the
    first column right-aligned as a number when `numbered` and every
    other column left-aligned."""
    if not rows:
        return []
    widths = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    return ["    " + "  ".join(
        cell.rjust(w) if j == 0 and numbered else cell.ljust(w)
        for j, (cell, w) in enumerate(zip(row, widths))).rstrip()
        for row in rows]


def write(path: str | Path, manifest: MeshManifest) -> Path:
    """Write the manifest beside its mesh, and return the path.  The
    `faults` key is written only where the mesh has a fault (schema 6, or
    a flat mesh's 7), so a spherical mesh without one has the manifest
    of schema 5 and a flat one that of schema 7 without the faults'
    records, and the `coordinates` and `faces` keys only for a flat mesh
    (schema 7)."""
    path = beside(path, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    if not data["faults"]:
        del data["faults"]
    if not manifest.is_flat:
        for key in _FLAT_KEYS:
            del data[key]
    with open(path, "w") as fh:
        # allow_nan=False: a strict parser cannot read NaN or Infinity, so
        # a manifest carrying one fails here, where the cause is known.
        json.dump(data, fh, indent=2, sort_keys=False, allow_nan=False)
        fh.write("\n")
    return path


def read(path: str | Path) -> MeshManifest:
    """Read a manifest, checking the schema it declares (one of SCHEMAS)
    and its structure; a key its schema does not have is passed over."""
    with open(path) as fh:
        data = json.load(fh)
    declared = data.get("schema")
    if declared not in SCHEMAS:
        raise ValueError(
            f"{path}: manifest schema is {declared!r}, this planetmodel reads "
            f"{', '.join(map(repr, SCHEMAS[:-1]))} and {SCHEMAS[-1]!r}")
    known = set(MeshManifest.__dataclass_fields__)
    if declared != SCHEMA_FLAT:
        known -= set(_FLAT_KEYS)
    card = MeshManifest(**{k: v for k, v in data.items() if k in known})
    validate_structure(card)
    return card


def _schema(*, flat: bool, faults: bool) -> str:
    """The schema of a manifest of a flat mesh, with faults or without, or
    of a spherical one, which has one with faults and one without (module
    docstring)."""
    if flat:
        return SCHEMA_FLAT
    return SCHEMA_WITH_FAULTS if faults else SCHEMA


# ------------------------------------------------------------ the checks

#: What the `mesh` block must carry, and as what.
_MESH_FIELDS = {"file": (str,), "format": (str,), "nodes": (str,),
                "read_options": (dict,), "displacement": (str, type(None))}

#: What the `scales` block carries where there is one.
_SCALES_FIELDS = {"length": (int, float), "mass": (int, float),
                  "time": (int, float)}


def _typed(value: object, kinds: tuple[type, ...]) -> bool:
    """Whether `value` is one of `kinds`, with bool never passing as int."""
    if isinstance(value, bool) and bool not in kinds:
        return False
    return isinstance(value, kinds)


#: What the structural checks call on the first defect: it raises.
type Fail = Callable[[str], NoReturn]


def _check_record(record: object, types: Mapping[str, tuple[type, ...]],
                  where: str, fail: Fail) -> None:
    """Every key of `types` present in `record` with the type it demands."""
    if not isinstance(record, dict):
        fail(f"{where} is {type(record).__name__}, not an object")
    for key, kinds in types.items():
        if key not in record or not _typed(record[key], kinds):
            names = " or ".join(k.__name__ for k in kinds)
            fail(f"{where}.{key} is {record.get(key)!r}, not {names}")


def validate_structure(manifest: MeshManifest) -> None:
    """Check a manifest has the shape the schema promises.

    Every block is held to the definitions the builders write from, so
    a consumer gets one ValueError naming the field rather than a
    TypeError from deep inside its own reader.
    """
    def fail(msg: str) -> NoReturn:
        raise ValueError(f"malformed manifest: {msg}")

    if manifest.schema not in SCHEMAS:
        fail(f"schema is {manifest.schema!r}, not one of {SCHEMAS}")
    m = manifest.mesh
    _check_record(m, _MESH_FIELDS, "mesh", fail)
    if m["format"] not in FORMATS:
        fail(f"mesh.format {m['format']!r} is not one of {FORMATS}")
    if m["nodes"] not in NODES:
        fail(f"mesh.nodes {m['nodes']!r} is not one of {NODES}")

    flat = manifest.is_flat
    for what, entries, cls in (("layers", manifest.layers,
                                FlatLayerEntry if flat else LayerEntry),
                               ("interfaces", manifest.interfaces,
                                FlatInterfaceEntry if flat else InterfaceEntry),
                               ("faces", manifest.faces, FaceEntry),
                               ("refinements", manifest.refinements,
                                RefinementEntry),
                               ("faults", manifest.faults, FaultEntry),
                               ("fields", manifest.fields, FieldEntry)):
        if not isinstance(entries, list):
            fail(f"{what} must be a list of objects")
        types = _entry_types(cls)
        if cls in (LayerEntry, FlatLayerEntry):
            # a layer's attributes are a mesh with faults', checked with them
            del types["attributes"]
        if cls is FaultEntry and not flat:
            # a fault's face edge is a flat mesh's
            del types["face_edge"]
        for i, e in enumerate(entries):
            _check_record(e, types, f"{what}[{i}]", fail)

    n_layers = len(manifest.layers)
    if not n_layers:
        fail("no layers are listed")
    if flat:
        _check_flat_skeleton(manifest, fail)
    else:
        if manifest.coordinates is not None or manifest.faces:
            fail(f"coordinates and faces are a flat mesh's, under schema "
                 f"{SCHEMA_FLAT!r}, not {manifest.schema!r}")
        _check_skeleton(manifest, fail)
    for i, face in enumerate(manifest.interfaces):
        if face["kind"] is not None and face["kind"] not in KINDS:
            fail(f"interfaces[{i}].kind is {face['kind']!r}, not one of "
                 f"{KINDS} or null")
        for key, pair in (face["values"] or {}).items():
            if (not isinstance(pair, list) or len(pair) != 2
                    or not all(v is None or _typed(v, (int, float))
                               for v in pair)):
                fail(f"interfaces[{i}].values[{key!r}] is {pair!r}, not "
                     "[below, above] as numbers or nulls")

    for i, e in enumerate(manifest.refinements):
        if not e["size"] > 0.0 or e["far_size"] < e["size"]:
            fail(f"refinements[{i}] has size {e['size']} and far_size "
                 f"{e['far_size']}, not 0 < size <= far_size")
        if e["decay_width"] is not None and not e["decay_width"] > 0.0:
            fail(f"refinements[{i}].decay_width is {e['decay_width']}, "
                 "not positive")
        if e["points"] is not None and e["points"] < 1:
            fail(f"refinements[{i}].points is {e['points']}, not a count")

    _check_faults(manifest, fail)
    if manifest.faults:
        _check_collars(manifest, fail)

    names = [e["name"] for e in manifest.fields]
    if len(set(names)) != len(names):
        fail(f"field names repeat: {names}")
    for i, e in enumerate(manifest.fields):
        if not all(_typed(a, (int,)) and 1 <= a <= n_layers for a in e["layers"]):
            fail(f"fields[{i}].layers is {e['layers']}, not attributes in "
                 f"1..{n_layers}")
        d = e["radial_degree"]
        if d is not None and d < 0:
            fail(f"fields[{i}].radial_degree is {d}, not a degree")
        try:
            shape = stored_shape(Character(e["rank"], e["weight"], voigt=e["voigt"]))
        except ValueError as error:
            fail(f"fields[{i}] has no character: {error}")
        stored = int(np.prod(shape, dtype=int))
        # a vector of the mesh's own space, the displacement in 2D, has two
        if e["vdim"] != stored and not (e["rank"] == 1 and e["vdim"] == 2):
            fail(f"fields[{i}].vdim is {e['vdim']}, but a field of rank "
                 f"{e['rank']}{'' if e['voigt'] else ' without Voigt form'} "
                 f"stores {stored} components")
    if m["displacement"] is not None and m["displacement"] not in names:
        fail(f"mesh.displacement names {m['displacement']!r}, which fields "
             "holds no record for")
    if m["displacement"] is not None and m["nodes"] == "physical":
        fail("mesh.displacement is recorded with physical nodes; a mesh whose "
             "nodes are already moved has no displacement to apply")
    for i, face in enumerate(manifest.interfaces):
        for key in face["values"] or {}:
            if key not in names:
                fail(f"interfaces[{i}].values names {key!r}, which fields "
                     "holds no record for")

    if manifest.model is not None:
        _check_record(manifest.model, {"file": (str,), "schema": (str,)}, "model",
                      fail)
    if manifest.scales is not None:
        _check_record(manifest.scales, _SCALES_FIELDS, "scales", fail)
        kelvin = manifest.scales.get("temperature", 1.0)
        if not _typed(kelvin, (int, float)):
            fail(f"scales.temperature is {kelvin!r}, not int or float")
    if not isinstance(manifest.constants, dict):
        fail("constants must be an object")
    for key, value in manifest.constants.items():
        if not _typed(value, (int, float)):
            fail(f"constants.{key} is {value!r}, not a number")
    if not isinstance(manifest.meta, dict):
        fail("meta must be an object")


def _check_skeleton(manifest: MeshManifest, fail: Fail) -> None:
    """The layers and interfaces of a spherical mesh (schemas 5 and 6):
    numbered 1..N from the centre, contiguous in radius, one interface per
    layer and the inner boundary one more where the mesh is hollow, each
    between the layers it bounds.  A full mesh starts at the centre."""
    layers, interfaces = manifest.layers, manifest.interfaces
    n_layers = len(layers)
    for i, lay in enumerate(layers):
        if lay["attribute"] != i + 1:
            fail(f"layers[{i}].attribute is {lay['attribute']}, expected {i + 1}")
        if not 0.0 <= lay["r_inner"] < lay["r_outer"]:
            fail(f"layers[{i}] runs from {lay['r_inner']} to {lay['r_outer']}, "
                 "not 0 <= r_inner < r_outer")
        if i and lay["r_inner"] != layers[i - 1]["r_outer"]:
            fail(f"layers[{i}] starts at {lay['r_inner']} but the "
                 f"layer below ends at {layers[i - 1]['r_outer']}")

    n_faces = len(interfaces)
    if n_faces not in (n_layers, n_layers + 1):
        fail(f"{n_faces} interfaces for {n_layers} layers; a full mesh has one "
             "per layer and a hollow mesh one more")
    first = n_faces - n_layers          # 1 when the inner boundary is a face
    if (layers[0]["r_inner"] > 0.0) != bool(first):
        fail(f"{n_faces} interfaces for {n_layers} layers starting at r = "
             f"{layers[0]['r_inner']}: a full mesh starts at the centre "
             "and a hollow one has a face on its inner boundary")
    for i, face in enumerate(interfaces):
        if face["attribute"] != i + 1:
            fail(f"interfaces[{i}].attribute is {face['attribute']}, expected "
                 f"{i + 1}")
        j = i + 1 - first               # the skeleton boundary the face sits on
        want = [j - 1, j if j < n_layers else -1]
        if list(face["between_layers"]) != want:
            fail(f"interfaces[{i}].between_layers is "
                 f"{face['between_layers']}, expected {want}")


def _check_flat_skeleton(manifest: MeshManifest, fail: Fail) -> None:
    """The coordinates, layers, interfaces and faces of a flat mesh
    (schema 7, module docstring): a box's or a rectangle's
    coordinates; layers numbered 1..N from the surface down, contiguous
    in depth from a top at or below the surface; one interface per
    layer, its top,
    between it and the layer above; and the faces of `FACES` whose axes
    the mesh has, in that order, numbered on from the interfaces, the
    base at the last layer's bottom and each side below its opposite."""
    coordinates = manifest.coordinates
    if not isinstance(coordinates, list) or tuple(coordinates) not in FLAT_COORDINATES:
        fail(f"coordinates is {coordinates!r}, not one of "
             f"{[list(c) for c in FLAT_COORDINATES]}")
    layers, interfaces = manifest.layers, manifest.interfaces
    for i, lay in enumerate(layers):
        if lay["attribute"] != i + 1:
            fail(f"layers[{i}].attribute is {lay['attribute']}, expected {i + 1}")
        if not 0.0 <= lay["depth_top"] < lay["depth_bottom"]:
            fail(f"layers[{i}] runs from depth {lay['depth_top']} to "
                 f"{lay['depth_bottom']}, not 0 <= depth_top < depth_bottom")
        if i and lay["depth_top"] != layers[i - 1]["depth_bottom"]:
            fail(f"layers[{i}] starts at depth {lay['depth_top']} but the layer "
                 f"above ends at {layers[i - 1]['depth_bottom']}")
        if lay["in_geometry"] is not True:
            fail(f"layers[{i}].in_geometry is {lay['in_geometry']}: every layer "
                 "of a flat mesh is the geometry's")
    if len(interfaces) != len(layers):
        fail(f"{len(interfaces)} interfaces for {len(layers)} layers; a flat mesh "
             "has one per layer, its top")
    for i, face in enumerate(interfaces):
        if face["attribute"] != i + 1:
            fail(f"interfaces[{i}].attribute is {face['attribute']}, expected "
                 f"{i + 1}")
        if face["depth"] != layers[i]["depth_top"]:
            fail(f"interfaces[{i}] lies at depth {face['depth']}, not at the top "
                 f"of layers[{i}], {layers[i]['depth_top']}")
        if list(face["between_layers"]) != [i, i - 1]:
            fail(f"interfaces[{i}].between_layers is {face['between_layers']}, "
                 f"expected {[i, i - 1]}")

    rows = [row for row in FACES if row[1] in coordinates]
    got = [e["name"] for e in manifest.faces]
    if got != [name for name, _, _ in rows]:
        fail(f"faces are {got}, not {[name for name, _, _ in rows]} for "
             f"coordinates {coordinates}")
    value = {}
    for k, (e, (name, axis, outward)) in enumerate(zip(manifest.faces, rows)):
        if e["attribute"] != len(interfaces) + k + 1:
            fail(f"faces[{k}].attribute is {e['attribute']}, expected "
                 f"{len(interfaces) + k + 1}, after the {len(interfaces)} "
                 "interfaces")
        if (e["axis"], e["outward"]) != (axis, outward):
            fail(f"faces[{k}] ({name}) is normal to {e['axis']!r} with outward "
                 f"{e['outward']}, not {axis!r} with {outward}")
        value[name] = e["value"]
    if value["base"] != -layers[-1]["depth_bottom"]:
        fail(f"the base lies at z = {value['base']}, not at the last layer's "
             f"bottom, z = {-layers[-1]['depth_bottom']}")
    for axis in coordinates[:-1]:
        if not value[f"{axis}_min"] < value[f"{axis}_max"]:
            fail(f"the faces {axis}_min and {axis}_max lie at {value[f'{axis}_min']} "
                 f"and {value[f'{axis}_max']}, not the first below the second")
    taken = {e["name"] for e in interfaces}.intersection(got)
    if taken:
        fail(f"faces {sorted(taken)} share a name with an interface")


#: What a fault's `sizing` holds where it is not null.
_SIZING_FIELDS = {"size": (int, float), "far_size": (int, float),
                  "decay_width": (int, float)}


def _check_faults(manifest: MeshManifest, fail: Fail) -> None:
    """The `faults[]` records against each other and the skeleton: the
    attributes running on from the interfaces' (and a flat mesh's
    faces'), names unique and not an interface's or a face's, layers of
    the geometry, the curve groups fault k's (`fault_curve_groups` of the
    mesh's `fault_curves`), meetings mutual and with a junction to show
    for them, and the pull-back and sizing records well formed; the
    schema one that takes faults where faults are listed (6, or a flat
    mesh's 7); and a manifest listing none without their records: not
    of schema 6, and a flat one's layers without `attributes`."""
    faults = manifest.faults
    want_schema = _schema(flat=manifest.is_flat, faults=bool(faults))
    if faults and manifest.schema != want_schema:
        fail(f"faults are listed under schema {manifest.schema!r}; a manifest "
             f"with faults is {want_schema!r}")
    if not faults and manifest.has_faults:
        fail(f"faults lists none, and a manifest of schema "
             f"{manifest.schema!r} has faults")
    listed = [i for i, e in enumerate(manifest.layers) if "attributes" in e]
    if not faults and manifest.is_flat and listed:
        fail(f"faults lists none, and layers[{listed[0]}].attributes is listed: a "
             "flat manifest lists its layers' attributes with its faults or neither")
    n_faces = len(manifest.interfaces) + len(manifest.faces)
    after = (f"the {len(manifest.interfaces)} interfaces"
             + (f" and the {len(manifest.faces)} faces" if manifest.faces else ""))
    names = [e["name"] for e in faults]
    if len(set(names)) != len(names):
        fail(f"fault names repeat: {names}")
    taken = {f["name"] for f in manifest.interfaces}
    if taken.intersection(names):
        fail(f"faults {sorted(taken.intersection(names))} share a name with an "
             "interface")
    taken = {f["name"] for f in manifest.faces}
    if taken.intersection(names):
        fail(f"faults {sorted(taken.intersection(names))} share a name with a "
             "face of the box")
    inside = {int(e["attribute"]) for e in manifest.layers if e["in_geometry"]}
    meets = {e["name"]: set(e["meets"]) for e in faults}
    curves = manifest.fault_curves
    for i, e in enumerate(faults):
        k = i + 1
        if e["attribute"] != n_faces + k:
            fail(f"faults[{i}].attribute is {e['attribute']}, expected "
                 f"{n_faces + k}, after {after}")
        if not e["layers"] or not all(_typed(a, (int,)) and a in inside
                                      for a in e["layers"]):
            fail(f"faults[{i}].layers is {e['layers']}, not attributes of "
                 f"layers of the geometry, {sorted(inside)}")
        for key, want in zip(curves, fault_curve_groups(k, curves=curves)):
            if e[key] is not None and e[key] != want:
                fail(f"faults[{i}].{key} is {e[key]}, expected {want} or null")
        if not all(isinstance(m, str) and m in meets and m != e["name"]
                   for m in e["meets"]):
            fail(f"faults[{i}].meets is {e['meets']}, not the names of other "
                 "faults")
        if any(e["name"] not in meets[m] for m in e["meets"]):
            fail(f"faults[{i}] meets {e['meets']}, and not every one of them "
                 f"meets {e['name']!r}")
        if bool(e["meets"]) != (e["junction"] is not None):
            fail(f"faults[{i}] has junction {e['junction']} but meets "
                 f"{e['meets']}: a junction is where it meets another fault")
        ends = e["ends_on"]
        if (not all(isinstance(n, str) and n in e["meets"] for n in ends)
                or len(set(ends)) != len(ends)):
            fail(f"faults[{i}].ends_on is {ends}, not faults it meets, each once")
        controls = e["controls"]
        if controls is not None and not all(
                isinstance(pair, list) and len(pair) == 2
                and all(_typed(n, (int,)) and n >= 4 for n in pair)
                for pair in controls):
            fail(f"faults[{i}].controls is {controls!r}, not [n_u, n_v] per "
                 "patch, each at least 4")
        if not e["offset"] >= 0.0:
            fail(f"faults[{i}].offset is {e['offset']}, not a distance")
        sizing = e["sizing"]
        if sizing is not None:
            _check_record(sizing, _SIZING_FIELDS, f"faults[{i}].sizing", fail)
            if not 0.0 < sizing["size"] <= sizing["far_size"] \
                    or not sizing["decay_width"] > 0.0:
                fail(f"faults[{i}].sizing is {sizing}, not 0 < size <= "
                     "far_size with a positive decay_width")


def _check_collars(manifest: MeshManifest, fail: Fail) -> None:
    """The collar records of a mesh with faults (schema 6, or 7 with
    faults; module docstring): each layer's `attributes` its own and then
    collar attributes in increasing order, none for a shell; the collar
    attributes N + 1..N + C, each one
    layer's and in some fault's hanging wall; each fault's hanging wall
    collar attributes in increasing order; and the collars numbered in
    the order of their layer and then of their faults, so that no two
    share a combination."""
    n = len(manifest.layers)
    owner: dict[int, int] = {}      # collar attribute -> the layer's own
    for i, e in enumerate(manifest.layers):
        where = f"layers[{i}].attributes"
        attrs = e.get("attributes")
        if (not isinstance(attrs, list) or not attrs
                or not all(_typed(a, (int,)) for a in attrs)
                or attrs[0] != e["attribute"] or any(a <= n for a in attrs[1:])
                or any(b <= a for a, b in zip(attrs[1:], attrs[2:]))):
            fail(f"{where} is {attrs!r}, not its own attribute {e['attribute']} "
                 f"and then collar attributes above {n} in increasing order")
        if len(attrs) > 1 and not e["in_geometry"]:
            fail(f"{where} lists collars {attrs[1:]}, and a shell holds no fault")
        for a in attrs[1:]:
            if a in owner:
                fail(f"{where} lists collar {a}, which layers[{owner[a] - 1}] lists "
                     "too")
            owner[a] = e["attribute"]
    if sorted(owner) != list(range(n + 1, n + 1 + len(owner))):
        fail(f"layers list the collar attributes {sorted(owner)}, not "
             f"{n + 1}..{n + len(owner)}")
    walls: dict[int, list[int]] = {a: [] for a in owner}
    for k, e in enumerate(manifest.faults):
        hw = e["hanging_wall"]
        if (not all(_typed(a, (int,)) and a in owner for a in hw)
                or any(b <= a for a, b in zip(hw, hw[1:]))):
            fail(f"faults[{k}].hanging_wall is {hw}, not collar attributes in "
                 "increasing order")
        for a in hw:
            walls[a].append(k)
    previous: tuple[int, tuple[int, ...]] | None = None
    for a in sorted(owner):
        where = f"layers[{owner[a] - 1}].attributes"
        if not walls[a]:
            fail(f"{where} lists collar {a}, which is in no fault's hanging wall")
        key = (owner[a], tuple(walls[a]))
        if previous is not None and key <= previous:
            fail(f"{where} lists collar {a} (layer {owner[a]}, faults "
                 f"{list(walls[a])}) out of the order of layer and faults")
        previous = key


def validate_against(manifest: MeshManifest, *, layer_count: int,
                     interface_count: int, fault_count: int = 0,
                     groups: Mapping[str, Iterable[int]] | None = None) -> None:
    """Check a manifest describes the mesh it was written beside.

    `groups` maps "layers" and "interfaces" to the physical group
    numbers the mesh carries, both of which must run 1..N in the
    skeleton's order, "faces" to those of a flat mesh's faces and
    "faults" to those of the faults' faces, which run on from the
    interfaces' in that order, and "collars" to those of the collars'
    cells, which run on from the layers' and are the manifest's.
    """
    if len(manifest.layers) != layer_count:
        raise ValueError(
            f"manifest lists {len(manifest.layers)} layers, the mesh has "
            f"{layer_count}")
    if len(manifest.interfaces) != interface_count:
        raise ValueError(
            f"manifest lists {len(manifest.interfaces)} interfaces, the mesh "
            f"has {interface_count}")
    if len(manifest.faults) != fault_count:
        raise ValueError(
            f"manifest lists {len(manifest.faults)} faults, the mesh has "
            f"{fault_count}")
    for what, entries in (("layer", manifest.layers),
                          ("interface", manifest.interfaces)):
        attrs = [int(e["attribute"]) for e in entries]
        if attrs != list(range(1, len(entries) + 1)):
            raise ValueError(
                f"manifest {what} attributes are {attrs}, expected "
                f"1..{len(entries)} in the skeleton's order")
    if groups is not None:
        n_faces = len(manifest.faces)
        first = {"layers": 1, "interfaces": 1, "faces": interface_count + 1,
                 "faults": interface_count + n_faces + 1,
                 "collars": layer_count + 1}
        counts = {"layers": layer_count, "interfaces": interface_count,
                  "faces": n_faces, "faults": fault_count,
                  "collars": len(manifest.element_attributes()) - layer_count}
        for what, wanted in groups.items():
            got = sorted(wanted)
            start = first.get(what, 1)
            if got != list(range(start, start + counts.get(what, len(got)))):
                raise ValueError(
                    f"the mesh's {what} physical groups are {got}; the manifest "
                    f"describes {counts.get(what)} numbered from {start}")


# ----------------------------------------------- the rules for a fault's faces

def fault_side(points: ArrayLike, faces: ArrayLike, cells: ArrayLike) -> np.ndarray:
    """The side of a fault's face each cell lies on: +1 the hanging wall
    (plus), -1 the footwall (minus).

    `points` are the mesh's vertices, (n, 3); `faces` the corners of
    faces of a fault, (..., 3), indices into `points` in the order the
    mesh holds them; `cells` the corners of a tetrahedron holding each
    face, (..., 4) in any order, the leading shapes broadcasting.  With
    v0, v1, v2 a face's corners in order and x the cell's corner not on
    it, the cell is on the hanging-wall side where
    (x - v0) . ((v1 - v0) x (v2 - v0)) > 0 and on the footwall side
    where it is negative (the format page's rule).  Refused where a
    cell does not hold its face's three corners, or where x lies in the
    face's plane.  The rule holds for reference and physical
    coordinates alike, the mapping preserving orientation.
    """
    x = np.asarray(points, dtype=float)
    f = np.asarray(faces)
    c = np.asarray(cells)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"points have shape (n, 3), got {x.shape}")
    if f.shape[-1:] != (3,) or c.shape[-1:] != (4,):
        raise ValueError(f"faces have shape (..., 3) and cells (..., 4), got "
                         f"{f.shape} and {c.shape}")
    if not (np.issubdtype(f.dtype, np.integer) and np.issubdtype(c.dtype, np.integer)):
        raise ValueError("faces and cells are corner indices, integers")
    shape = np.broadcast_shapes(f.shape[:-1], c.shape[:-1])
    f = np.broadcast_to(f, shape + (3,))
    c = np.broadcast_to(c, shape + (4,))
    for what, idx in (("faces", f), ("cells", c)):
        if idx.size and (idx.min() < 0 or idx.max() >= len(x)):
            raise ValueError(f"{what} index vertices outside 0..{len(x) - 1}")
    on_face = np.any(c[..., :, None] == f[..., None, :], axis=-1)       # (..., 4)
    held = np.all(np.any(f[..., :, None] == c[..., None, :], axis=-1), axis=-1)
    bad = ~held | (np.count_nonzero(~on_face, axis=-1) != 1)
    if np.any(bad):
        raise ValueError(f"{int(np.count_nonzero(bad))} of {bad.size} cells do not "
                         "hold their face's three corners")
    fourth = np.take_along_axis(c, np.argmax(~on_face, axis=-1)[..., None],
                                axis=-1)[..., 0]
    v0, v1, v2 = (x[f[..., k]] for k in range(3))
    s = np.einsum("...i,...i->...", x[fourth] - v0, np.cross(v1 - v0, v2 - v0))
    if np.any(s == 0.0):
        raise ValueError(f"{int(np.count_nonzero(s == 0.0))} cells have their "
                         "fourth corner in their face's plane")
    return np.where(s > 0.0, 1, -1)


def plus_cell(points: ArrayLike, faces: ArrayLike, cells: ArrayLike) -> np.ndarray:
    """Which of each fault face's two cells is the plus (hanging-wall)
    one: 0 or 1 for each face, (m,); the other is the minus (footwall)
    cell.

    `faces` (m, 3) as `fault_side` takes them, and `cells` (m, 2, 4) the
    corners of the two tetrahedra sharing each face (in MFEM,
    `GetFaceElements` of the face `GetBdrElementFaceIndex` gives).
    Refused where a face's two cells do not lie one on each side of it,
    as the cells of a conforming mesh do.
    """
    f = np.asarray(faces)
    c = np.asarray(cells)
    if f.ndim != 2 or c.shape != (len(f), 2, 4):
        raise ValueError(f"faces have shape (m, 3) and cells (m, 2, 4), got "
                         f"{f.shape} and {c.shape}")
    side = fault_side(points, f[:, None, :], c)
    one_each = side[:, 0] == -side[:, 1]
    if not np.all(one_each):
        raise ValueError(f"{int(np.count_nonzero(~one_each))} of {len(f)} faces "
                         "have both their cells on one side")
    return np.where(side[:, 0] > 0, 0, 1)


def _no_edges() -> np.ndarray:
    """No edges, (0, 2)."""
    return np.empty((0, 2), dtype=np.int64)


@dataclass(frozen=True, eq=False)
class CurveEdges:
    """The edges of one fault's faces that lie on its curves
    (`fault_curves`): each (k, 2), the two corners of an edge as vertex
    indices with the smaller first, the rows in increasing order; and the
    faults it ends on."""

    #: Its buried tip, where its sides are joined.
    tip: np.ndarray
    #: Its edge on the model's outer surface, where its sides are free.
    free_edge: np.ndarray
    #: Where it meets another fault.
    junction: np.ndarray
    #: The part of its junction on its own boundary, where it ends on
    #: another fault: its slip is pinned there as on its tip.
    junction_end: np.ndarray
    #: The faults it ends on, by their indices in the list given.
    ends_on: tuple[int, ...]
    _: KW_ONLY
    #: Its edge on the faces of a box, its base and sides, where it leaves
    #: the body through an artificial boundary: its sides free, as on the
    #: surface, and not pinned.
    face_edge: np.ndarray = field(default_factory=_no_edges)

    @property
    def pinned(self) -> np.ndarray:
        """The edges along which its slip is pinned to zero, (k, 2) in
        increasing order: its tip, and its junction where it ends; never
        its free or face edge."""
        both = np.concatenate([self.tip, self.junction_end])
        return np.unique(both, axis=0).reshape(-1, 2).astype(np.int64)


def _edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """Each edge of a set of triangles, its corners sorted, with how many
    of them use it."""
    out: dict[tuple[int, int], int] = {}
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for lo, hi in np.sort(faces[:, [a, b]], axis=1).tolist():
            out[(lo, hi)] = out.get((lo, hi), 0) + 1
    return out


def _triangles(faces: ArrayLike, what: str) -> np.ndarray:
    """Faces as an (m, 3) array of integer corners."""
    f = np.asarray(faces)
    if f.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError(f"{what} are (m, 3) corner indices, got {f.shape} of "
                         f"{f.dtype}")
    return f


def fault_curves(faults: Sequence[ArrayLike], surface: ArrayLike, *,
                 faces: ArrayLike = ()) -> list[CurveEdges]:
    """The tip, free-edge, face-edge and junction edges of each fault, and
    the part of its junction where it ends, read from the faces alone:
    the rule a consumer of a mesh that keeps no curves applies (the format
    page), and the edges the MSH file's curve groups hold.

    `faults` holds each fault's faces, (m_k, 3) corner indices, `surface`
    the faces of the geometry's outer surface (the boundary elements of
    `MeshManifest.surface_attribute`) and `faces` those of a box's base
    and sides (of `MeshManifest.face_attributes`; none on a sphere), all
    in one numbering of the vertices.  An edge is a pair of corners, and
    a fault's count of it the number of its faces that use it: one on
    the fault's boundary, two inside it.  The edges of a fault's faces
    that are also edges of another fault's faces are its junction; those
    of them it counts once, on its own boundary, are where it ends on
    that other fault (`junction_end`, the faults `ends_on`), and those it
    counts twice where the other ends on it or the two cross.  Of the
    rest, an edge counted once is in its free edge where it is also an
    edge of the surface, in its face edge where it is an edge of the
    box's faces, and in its tip otherwise; an edge counted twice lies
    inside it.  The answer is one `CurveEdges` a fault, in the order
    given.
    """
    counts = [_edge_counts(_triangles(f, "a fault's faces")) for f in faults]
    outer = set(_edge_counts(_triangles(surface, "the surface's faces")))
    box = set(_edge_counts(_triangles(faces, "the faces of the box")))

    def array(edges: set[tuple[int, int]]) -> np.ndarray:
        return np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)

    out = []
    for k, own in enumerate(counts):
        others = [j for j in range(len(counts)) if j != k]
        shared = {e for e in own if any(e in counts[j] for j in others)}
        once = {e for e, n in own.items() if n == 1}
        end = once & shared
        free = (once - shared) & outer
        cut = (once - shared - free) & box
        out.append(CurveEdges(tip=array(once - shared - free - cut),
                              free_edge=array(free), junction=array(shared),
                              junction_end=array(end),
                              ends_on=tuple(j for j in others
                                            if any(e in counts[j] for e in end)),
                              face_edge=array(cut)))
    return out


#: The six edges of a tetrahedron and the three of a triangle, as pairs
#: of corners, and the four faces of a tetrahedron, each opposite one
#: corner.
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_TRI_EDGES = np.array([[0, 1], [1, 2], [0, 2]])
_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def _keys(edges: np.ndarray, n: int) -> np.ndarray:
    """Edges (..., 2) of corners among n vertices as one integer each,
    the same whichever corner comes first."""
    e = np.sort(np.asarray(edges, dtype=np.int64), axis=-1)
    return e[..., 0] * n + e[..., 1]


def _indices(rows: ArrayLike, width: int, n: int, what: str) -> np.ndarray:
    """Rows of `width` vertex indices among n, as an integer array."""
    a = np.asarray(rows)
    if a.size == 0:
        return np.empty((0, width), dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != width or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{what} are (m, {width}) corner indices, got {a.shape} of "
                         f"{a.dtype}")
    if a.min() < 0 or a.max() >= n:
        raise ValueError(f"{what} index vertices outside 0..{n - 1}")
    return a


def fault_walls(points: ArrayLike, cells: ArrayLike, faces: ArrayLike,
                pinned: ArrayLike, *, name: str | None = None) -> np.ndarray:
    """The wall of one fault each cell is in: +1 its hanging wall, -1 its
    footwall, 0 neither, (c,) int8.

    `points` are the mesh's vertices, (n, 3); `cells` the corners of the
    tetrahedra of the body the fault is in, (c, 4): the geometry's
    layers, without the shells a mesh spec adds outside them (through
    which the two sides of a fault meeting the geometry's surface would
    join); `faces` the fault's faces, (m, 3), oriented as the mesh holds
    them (`fault_side`); `pinned` the edges along which its slip is
    pinned, (k, 2), its tip and its junction where it ends
    (`CurveEdges.pinned`); all in one numbering of the vertices.  The
    vertices and edges of its faces that are not on a pinned edge carry
    its slip.  The cells holding one of them are joined across every
    face that is not one of the fault's and holds one of them too; so
    joined, they fall into groups each lying on one side of the fault,
    the side of the faces of the fault its cells hold (the hanging wall
    where a face's vertex order points to the cell, `fault_side`).  An
    edge carries slip in its own right: an edge of the fault joining two
    pinned vertices (across a corner of the fault, say) has slip at its
    interior nodes in a mesh of order two or more, and at its midpoint
    once the mesh is refined, so the cells holding it are in a wall.
    Every other cell touches the fault, if at all, along its pinned
    edges alone, where its slip is zero, and is in neither wall.

    Refused, naming the fault by `name`: a group holding cells on both
    sides of the fault's faces (a face oriented against the others, or a
    tip or junction missing from `pinned`), a group next to none of its
    faces, and a face of the fault that is not a face of two cells.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"points have shape (n, 3), got {x.shape}")
    n = len(x)
    c = _indices(cells, 4, n, "cells")
    f = _indices(faces, 3, n, "a fault's faces")
    p = _indices(pinned, 2, n, "pinned edges")
    what = "the fault" if name is None else f"fault {name!r}"
    walls = np.zeros(len(c), dtype=np.int8)
    if not len(f):
        return walls

    # what carries slip, and the cells holding any of it
    slip_edges = np.setdiff1d(_keys(f[:, _TRI_EDGES], n), _keys(p, n))
    carries = np.zeros(n, dtype=bool)
    carries[f] = True
    carries[p] = False
    touching = (carries[c].any(axis=1)
                | np.isin(_keys(c[:, _TET_EDGES], n), slip_edges).any(axis=1))

    # every face of every cell, the fault's among them, and the cells of each
    sides = np.sort(c[:, _TET_FACES], axis=2).reshape(-1, 3)
    unique, inverse = np.unique(np.concatenate([sides, np.sort(f, axis=1)]), axis=0,
                                return_inverse=True)
    inverse = inverse.ravel()
    of_cells, of_fault = inverse[:len(sides)], inverse[len(sides):]
    held = np.bincount(of_cells, minlength=len(unique))
    if np.any(held > 2):
        raise ValueError(f"{int(np.count_nonzero(held > 2))} faces of the mesh are "
                         "held by more than two cells")
    if np.any(held[of_fault] != 2):
        raise ValueError(f"{what}: {int(np.count_nonzero(held[of_fault] != 2))} of "
                         f"its {len(f)} faces are not a face of two cells")
    order = np.argsort(of_cells, kind="stable")
    owner = order // 4
    start = np.searchsorted(of_cells[order], np.arange(len(unique)))

    # the joins: faces not the fault's, holding a slip-carrying vertex or edge
    two = np.flatnonzero(held == 2)
    pair = np.stack([owner[start[two]], owner[start[two] + 1]], axis=1)
    on = unique[two]
    holds = (carries[on].any(axis=1)
             | np.isin(_keys(on[:, _TRI_EDGES], n), slip_edges).any(axis=1))
    is_fault = np.zeros(len(unique), dtype=bool)
    is_fault[of_fault] = True
    join = pair[holds & ~is_fault[two]]
    idx = np.flatnonzero(touching)
    local = np.full(len(c), -1)
    local[idx] = np.arange(len(idx))
    graph = coo_matrix((np.ones(len(join)), (local[join[:, 0]], local[join[:, 1]])),
                       shape=(len(idx), len(idx)))
    count, group = connected_components(graph, directed=False)

    # each group's side, from the faces of the fault its cells hold
    both = np.stack([owner[start[of_fault]], owner[start[of_fault] + 1]], axis=1)
    side = fault_side(x, f[:, None, :], c[both])
    seen = touching[both]
    plus = np.zeros(count, dtype=bool)
    minus = np.zeros(count, dtype=bool)
    plus[group[local[both[seen & (side > 0)]]]] = True
    minus[group[local[both[seen & (side < 0)]]]] = True
    if np.any(plus & minus):
        raise ValueError(
            f"{what}: {int(np.count_nonzero(plus & minus))} of the {count} groups of "
            "cells touching its slip-carrying vertices and edges hold cells on both "
            "sides of its faces, joining its hanging wall to its footwall: a face "
            "oriented against the others, or a tip or junction missing from its "
            "pinned edges")
    if not np.all(plus | minus):
        raise ValueError(
            f"{what}: {int(np.count_nonzero(~(plus | minus)))} of the {count} groups "
            "of cells touching its slip-carrying vertices and edges are next to none "
            "of its faces")
    walls[idx] = np.where(plus[group], 1, -1)
    return walls


def walls_of_faults(points: ArrayLike, cells: ArrayLike, faults: Sequence[ArrayLike],
                    surface: ArrayLike, *, faces: ArrayLike = (),
                    names: Sequence[str] | None = None) -> np.ndarray:
    """Every fault's walls (`fault_walls`), (F, c) int8, its pinned edges
    read from the faces (`fault_curves`, `CurveEdges.pinned`): `cells`
    are the body's, as `fault_walls` takes them, `faults` each fault's
    faces, (m_k, 3), and `surface` the faces of the geometry's outer
    surface and `faces` those of a box's base and sides, as
    `fault_curves` takes them (a box's faces left out, a fault's face
    edge would be taken for a tip and pinned); `names` names the faults
    in a refusal."""
    curves = fault_curves(faults, surface, faces=faces)
    names = [None] * len(curves) if names is None else list(names)
    if len(names) != len(curves):
        raise ValueError(f"{len(names)} names for {len(curves)} faults")
    n_cells = len(np.asarray(cells))
    return np.stack([fault_walls(points, cells, faces, found.pinned, name=name)
                     for faces, found, name in zip(faults, curves, names)]
                    ) if curves else np.zeros((0, n_cells), dtype=np.int8)


@dataclass(frozen=True, eq=False)
class Collars:
    """The element attributes of the cells of a mesh with faults
    (`collars`)."""

    #: Each cell's element attribute, (c,): its layer's own where it is in
    #: no fault's hanging wall, its collar's otherwise.
    attributes: np.ndarray
    #: The collars in the order of their attributes, from N + 1: each the
    #: layer, by its own attribute, and the faults, by their 0-based
    #: indices in increasing order, whose hanging wall its cells are in.
    combinations: tuple[tuple[int, tuple[int, ...]], ...]
    #: N, the number of layers.
    layer_count: int

    def layer_attributes(self, layer: int) -> list[int]:
        """The element attributes of a layer's cells (the layer by its own
        attribute): its own, then its collars' (`LayerEntry.attributes`)."""
        return [int(layer)] + [self.layer_count + 1 + j for j, (owner, _) in
                               enumerate(self.combinations) if owner == layer]

    def hanging_wall(self, fault: int) -> list[int]:
        """The collar attributes of a fault's hanging wall, the fault by
        its 0-based index (`FaultEntry.hanging_wall`)."""
        return [self.layer_count + 1 + j for j, (_, faults) in
                enumerate(self.combinations) if fault in faults]


def collars(layers: ArrayLike, walls: ArrayLike, *, layer_count: int) -> Collars:
    """The element attributes of the cells of a mesh with faults (module
    docstring): `layers` is each cell's layer by its own attribute, (c,),
    and `walls` the faults' walls (`walls_of_faults`), (F, c).  A cell in
    no fault's hanging wall keeps its layer's attribute; the others carry
    one attribute for each combination of a layer and a set of faults
    whose hanging wall a cell is in, numbered from `layer_count` + 1 in
    the order of the layer and then of the faults' indices, compared as
    tuples."""
    lay = np.asarray(layers)
    if lay.ndim != 1 or (lay.size and not np.issubdtype(lay.dtype, np.integer)):
        raise ValueError(f"layers are one integer a cell, got {lay.shape} of "
                         f"{lay.dtype}")
    if lay.size and (lay.min() < 1 or lay.max() > layer_count):
        raise ValueError(f"layers are attributes in 1..{layer_count}")
    hanging = np.asarray(walls) == 1
    if hanging.ndim != 2 or hanging.shape[1] != len(lay):
        raise ValueError(f"walls have shape (faults, {len(lay)}), got {hanging.shape}")
    out = lay.astype(np.int64)
    inside = hanging.any(axis=0)
    if not np.any(inside):
        return Collars(attributes=out, combinations=(), layer_count=int(layer_count))
    rows = np.concatenate([out[inside, None], hanging[:, inside].T.astype(np.int64)],
                          axis=1)
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    keys = [(int(r[0]), tuple(int(k) for k in np.flatnonzero(r[1:]))) for r in unique]
    order = sorted(range(len(keys)), key=keys.__getitem__)
    rank = np.empty(len(keys), dtype=np.int64)
    rank[order] = np.arange(len(keys))
    out[inside] = layer_count + 1 + rank[inverse.ravel()]
    return Collars(attributes=out, combinations=tuple(keys[j] for j in order),
                   layer_count=int(layer_count))


def check_collars(manifest: MeshManifest, layers: ArrayLike, attributes: ArrayLike,
                  walls: ArrayLike) -> None:
    """Hold a mesh's cells and its manifest's collar records to the
    classification: `layers` is each cell's layer by where it lies (its
    own attribute), `attributes` the element attribute the mesh gives it,
    and `walls` the faults' walls (`walls_of_faults`, the faults in the
    manifest's order).  Every cell carries an attribute of the layer it
    lies in (`MeshManifest.layer_of`), so that a layer's attributes cover
    it, and the one `collars` numbers for it; each layer's `attributes`
    and each fault's `hanging_wall` are the numbering's.  ValueError
    naming the first disagreement."""
    lay = np.asarray(layers)
    got = np.asarray(attributes)
    if lay.shape != got.shape:
        raise ValueError(f"{lay.shape} layers for {got.shape} attributes")
    for a in np.unique(got).tolist():
        try:
            owner = int(manifest.layer_of(a)["attribute"])
        except KeyError as error:
            raise ValueError(f"cells carry element attribute {a}: {error}") from None
        wrong = (got == a) & (lay != owner)
        if np.any(wrong):
            raise ValueError(
                f"{int(np.count_nonzero(wrong))} cells of element attribute {a}, layer "
                f"{owner}'s, lie in layers {sorted(set(lay[wrong].tolist()))}")
    want = collars(lay, walls, layer_count=len(manifest.layers))
    differ = got != want.attributes
    if np.any(differ):
        collar = differ & (want.attributes > len(manifest.layers))
        raise ValueError(
            f"{int(np.count_nonzero(differ))} of {got.size} cells carry another "
            "element attribute than the classification of the faults' walls gives "
            f"them ({int(np.count_nonzero(collar))} of them in its collars)")
    for e in manifest.layers:
        listed = list(manifest.element_attributes(layers=[e["attribute"]]))
        if listed != want.layer_attributes(int(e["attribute"])):
            raise ValueError(
                f"layer {e['name']!r} lists the attributes {listed}, and the "
                f"classification numbers {want.layer_attributes(int(e['attribute']))}")
    for k, e in enumerate(manifest.faults):
        if list(e["hanging_wall"]) != want.hanging_wall(k):
            raise ValueError(
                f"fault {e['name']!r} lists the hanging wall {e['hanging_wall']}, and "
                f"the classification numbers {want.hanging_wall(k)}")
