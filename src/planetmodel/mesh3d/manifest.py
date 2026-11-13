"""The JSON that travels with every mesh.

A mesh file carries numbered attributes and nothing else.  Which number
is which layer, which boundary is which interface, where each sits,
whether the nodes have already been displaced, and which files beside
the mesh hold which fields: none of that survives into the mesh file,
so it travels alongside in a manifest, and this module is the single
definition of its shape.  A consumer reads the manifest, opens the mesh
the way it says, and knows where to look for every field and what each
one means.

The schema is `planetmodel.mesh.manifest/5`, with these blocks:

  mesh          file, format ("msh" or "mfem"), nodes ("reference" or
                "physical"), read_options, displacement
  layers[]      attribute, name, r_inner, r_outer, in_geometry, fluid
  interfaces[]  attribute, name, radius, between_layers, kind, values
  refinements[] name, size, far_size, decay_width, points, digest
  faults[]      name, attribute, layers, tip, free_edge, junction, meets,
                controls, offset, sizing; written only where the mesh
                has faults
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
layer indices with -1 for the outside.  A field is a file beside the
mesh, a finite-element space to read it into, a character (`rank`,
`weight`, `voigt`), a unit and the attributes of the layers on which
its values mean anything.  The displacement m(X) - X is a field like
any other, a vector on every layer; `mesh.displacement` names it where
one was written, and `mesh.nodes` says whether the mesh's coordinates
are reference coordinates or already carry it.  Flat and boring on
purpose: a C++ reader will parse it, nested objects only where the
nesting carries meaning, no polymorphism, every number a number.

A fault is an interior surface of the mesh, one conforming sheet of
faces with no nodes split, labelled by local sides.  Its faces are the
boundary elements of attribute `attribute`, M + k for the k-th fault
after the M interfaces, every one oriented with the fault's normal:
its vertices, in the order written, run so that (v1 - v0) x (v2 - v0)
points from the footwall into the hanging wall, so the cell on that
side of a face is the hanging-wall (plus) one and the other the
footwall (minus) one.  `layers` are the attributes of the layers its
faces lie in.  The curves bounding it are dimension-1 physical groups
of the MSH file (an MFEM mesh keeps no curves; the same curves are the
edges of the fault's faces by the rule the format page gives): `tip`,
its buried tip, where its sides are joined; `free_edge`, where it meets
the model's outer surface and its sides are free; `junction`, where it
meets another fault, whose names `meets` lists; each null where there
is none.  Fault k's curve groups are numbered 3k - 2, 3k - 1 and 3k
whether or not they exist.  `controls` are the control nodes [n_u,
n_v] of each patch's pull-back through the mapping, null under the
identity, where the fault is built in as given; `offset` the largest
distance found between the fault and its pull-back mapped forward,
zero under the identity; `sizing` the refinement towards it, {size,
far_size, decay_width}, or null.

Two rules turn a fault's faces into what a split-node code needs, and
are the format page's: which of a face's two cells is the plus one
(`fault_side`, `plus_cell`), and which edges of a fault's faces are its
tip, free edge and junction, read from the faces alone where the mesh
format keeps no curves (`fault_curves`).  Both take plain arrays (the
vertices' coordinates, corners as indices into them), so they apply to
a mesh however it was read; the C++ library states them again.
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

__all__ = ["SCHEMA", "FORMATS", "NODES", "KINDS", "MeshManifest", "LayerEntry",
           "InterfaceEntry", "RefinementEntry", "FaultEntry", "FieldEntry",
           "write", "read", "beside",
           "mesh_block", "scales_block", "constants_block", "fault_curve_groups",
           "validate_structure", "validate_against", "fault_side", "plus_cell",
           "CurveEdges", "fault_curves"]

#: Bump only for an incompatible change; consumers check it.
SCHEMA = "planetmodel.mesh.manifest/5"

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

    #: The element attribute, 1..N from the centre.
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

    @classmethod
    def from_layer(cls, layer: LayerInfo, *, attribute: int, r_inner: float,
                   r_outer: float, in_geometry: bool) -> "LayerEntry":
        """The record of a LayerInfo between the given radii."""
        return cls(attribute=int(attribute),
                   name=layer.name or f"layer_{attribute}",
                   r_inner=float(r_inner), r_outer=float(r_outer),
                   in_geometry=bool(in_geometry))


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


def fault_curve_groups(k: int) -> tuple[int, int, int]:
    """The dimension-1 physical groups of fault k (from one): its tip,
    free edge and junction curves, numbered 3k - 2, 3k - 1 and 3k."""
    return 3 * k - 2, 3 * k - 1, 3 * k


@dataclass
class FaultEntry:
    """One `faults[]` record: a fault built into the mesh as an interior
    surface (module docstring).

    `attribute` is the boundary attribute of its faces and `layers` the
    attributes of the layers they lie in; `tip`, `free_edge` and
    `junction` are the dimension-1 physical groups of its curves in the
    MSH file, null where it has none, and `meets` the names of the
    faults it meets along its junction curves.  `controls` holds each
    patch's pull-back grid [n_u, n_v], null where the mapping is the
    identity; `offset` is the largest distance found between the fault
    and its pull-back mapped forward, in the mesh's lengths; `sizing`
    the refinement towards it, {size, far_size, decay_width}, or null.
    """

    #: The fault's name; `fault_<k>` where none was given.
    name: str
    #: The boundary attribute of its faces: M + k, after the interfaces.
    attribute: int
    #: The attributes of the layers its faces lie in.
    layers: list
    _: KW_ONLY
    #: The physical group of its buried tip curves, or null.
    tip: int | None = None
    #: The physical group of its curves on the outer surface, or null.
    free_edge: int | None = None
    #: The physical group of its curves shared with other faults, or null.
    junction: int | None = None
    #: The names of the faults it meets.
    meets: list = field(default_factory=list)
    #: Each patch's pull-back control nodes [n_u, n_v]; null under the identity.
    controls: list | None = None
    #: The largest distance between the fault and its pull-back mapped forward.
    offset: float = 0.0
    #: The refinement towards the fault, {size, far_size, decay_width}, or null.
    sizing: dict | None = None


@dataclass
class FieldEntry:
    """One `fields[]` record: a file beside the mesh and what it holds.

    `file`, `fe_space`, `vdim` and `ordering` are what a consumer builds
    the space from before it opens the file; `rank`, `weight` and
    `voigt` are the field's character, `unit` its unit string under the
    manifest's scales ("1" where no SI scale is declared, "unknown" for
    a name without dimensions), and `layers` the attributes of the
    layers on which its values mean anything: elsewhere the file holds
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
    layers: list[dict[str, Any]] = field(default_factory=list)
    interfaces: list[dict[str, Any]] = field(default_factory=list)
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
    def from_build(cls, *, mesh: Mapping[str, Any], layers: Sequence[LayerEntry],
                   interfaces: Sequence[InterfaceEntry],
                   refinements: Sequence[RefinementEntry] = (),
                   faults: Sequence[FaultEntry] = (),
                   meta: Mapping[str, Any] | None = None) -> "MeshManifest":
        """A manifest from the mesh block and the typed entries.

        The one place a manifest is assembled, so a key cannot drift
        between builders and the schema is described once.
        """
        card = cls(mesh=dict(mesh), layers=[asdict(e) for e in layers],
                   interfaces=[asdict(e) for e in interfaces],
                   refinements=[asdict(e) for e in refinements],
                   faults=[asdict(e) for e in faults],
                   meta=dict(meta or {}))
        validate_structure(card)
        return card

    def layer_attribute(self, name: str) -> int:
        """The attribute number of a named layer."""
        for entry in self.layers:
            if entry.get("name") == name:
                return int(entry["attribute"])
        raise KeyError(
            f"no layer named {name!r}; layers are "
            f"{[e.get('name') for e in self.layers]}")

    def interface_attribute(self, name: str) -> int:
        """The attribute number of a named interface."""
        for entry in self.interfaces:
            if entry.get("name") == name:
                return int(entry["attribute"])
        raise KeyError(
            f"no interface named {name!r}; interfaces are "
            f"{[e.get('name') for e in self.interfaces]}")

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
        """Attributes of the layers outside the geometry: the shells."""
        return tuple(int(e["attribute"]) for e in self.layers
                     if not e.get("in_geometry", True))

    @property
    def surface_attribute(self) -> int:
        """The attribute of the geometry's outer surface: the interface
        with the geometry's outermost layer below it, a fault's free edge
        lying on it."""
        body = [int(e["attribute"]) for e in self.layers if e.get("in_geometry", True)]
        for e in self.interfaces:
            if body and int(e["between_layers"][0]) + 1 == body[-1]:
                return int(e["attribute"])
        raise ValueError("the manifest lists no surface of the geometry")

    @property
    def inner_radius(self) -> float:
        """The inner radius of the mesh: zero for a full one."""
        return float(self.layers[0]["r_inner"])

    @property
    def outer_radius(self) -> float:
        """The outer radius of the mesh, shells included."""
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
        lines.append("  layers")
        rows = [(str(e.get("attribute", "?")), str(e.get("name", "")),
                 f"[{_num(e.get('r_inner'))}, {_num(e.get('r_outer'))}]",
                 "in geometry" if e.get("in_geometry", True) else "shell",
                 {True: "fluid", False: "solid"}.get(e.get("fluid"), ""))
                for e in self.layers]
        lines.extend(_table(rows))
        lines.append("  interfaces")
        rows = [(str(e.get("attribute", "?")), str(e.get("name", "")),
                 f"radius {_num(e.get('radius'))}",
                 f"between layers {list(e.get('between_layers', []))}",
                 str(e.get("kind") or ""))
                for e in self.interfaces]
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
                     ", ".join(f"{what} {e[key]}" for key, what in
                               (("tip", "tip"), ("free_edge", "free edge"),
                                ("junction", "junction"))
                               if e.get(key) is not None),
                     f"meets {', '.join(e['meets'])}" if e.get("meets") else "",
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
        faults = f"{len(self.faults)} faults, " if self.faults else ""
        return (f"MeshManifest({self.schema}, {self.mesh.get('file', '?')}, "
                f"{len(self.layers)} layers, {len(self.interfaces)} interfaces, "
                f"{faults}{len(self.fields)} fields)")


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
    `faults` key is written only where the mesh has a fault, so a mesh
    without one has the manifest it always had."""
    path = beside(path, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    if not data["faults"]:
        del data["faults"]
    with open(path, "w") as fh:
        # allow_nan=False: a strict parser cannot read NaN or Infinity, so
        # a manifest carrying one fails here, where the cause is known.
        json.dump(data, fh, indent=2, sort_keys=False, allow_nan=False)
        fh.write("\n")
    return path


def read(path: str | Path) -> MeshManifest:
    """Read a manifest, checking the schema it declares and its structure."""
    with open(path) as fh:
        data = json.load(fh)
    declared = data.get("schema")
    if declared != SCHEMA:
        raise ValueError(
            f"{path}: manifest schema is {declared!r}, this planetmodel reads "
            f"{SCHEMA!r}")
    known = set(MeshManifest.__dataclass_fields__)
    card = MeshManifest(**{k: v for k, v in data.items() if k in known})
    validate_structure(card)
    return card


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

    m = manifest.mesh
    _check_record(m, _MESH_FIELDS, "mesh", fail)
    if m["format"] not in FORMATS:
        fail(f"mesh.format {m['format']!r} is not one of {FORMATS}")
    if m["nodes"] not in NODES:
        fail(f"mesh.nodes {m['nodes']!r} is not one of {NODES}")

    for what, entries, cls in (("layers", manifest.layers, LayerEntry),
                               ("interfaces", manifest.interfaces,
                                InterfaceEntry),
                               ("refinements", manifest.refinements,
                                RefinementEntry),
                               ("faults", manifest.faults, FaultEntry),
                               ("fields", manifest.fields, FieldEntry)):
        if not isinstance(entries, list):
            fail(f"{what} must be a list of objects")
        types = _entry_types(cls)
        for i, e in enumerate(entries):
            _check_record(e, types, f"{what}[{i}]", fail)

    n_layers = len(manifest.layers)
    if not n_layers:
        fail("no layers are listed")
    for i, lay in enumerate(manifest.layers):
        if lay["attribute"] != i + 1:
            fail(f"layers[{i}].attribute is {lay['attribute']}, expected {i + 1}")
        if not 0.0 <= lay["r_inner"] < lay["r_outer"]:
            fail(f"layers[{i}] runs from {lay['r_inner']} to {lay['r_outer']}, "
                 "not 0 <= r_inner < r_outer")
        if i and lay["r_inner"] != manifest.layers[i - 1]["r_outer"]:
            fail(f"layers[{i}] starts at {lay['r_inner']} but the "
                 f"layer below ends at {manifest.layers[i - 1]['r_outer']}")

    n_faces = len(manifest.interfaces)
    if n_faces not in (n_layers, n_layers + 1):
        fail(f"{n_faces} interfaces for {n_layers} layers; a full mesh has one "
             "per layer and a hollow mesh one more")
    first = n_faces - n_layers          # 1 when the inner boundary is a face
    if (manifest.layers[0]["r_inner"] > 0.0) != bool(first):
        fail(f"{n_faces} interfaces for {n_layers} layers starting at r = "
             f"{manifest.layers[0]['r_inner']}: a full mesh starts at the centre "
             "and a hollow one has a face on its inner boundary")
    for i, face in enumerate(manifest.interfaces):
        if face["attribute"] != i + 1:
            fail(f"interfaces[{i}].attribute is {face['attribute']}, expected "
                 f"{i + 1}")
        j = i + 1 - first               # the skeleton boundary the face sits on
        want = [j - 1, j if j < n_layers else -1]
        if list(face["between_layers"]) != want:
            fail(f"interfaces[{i}].between_layers is "
                 f"{face['between_layers']}, expected {want}")
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


#: What a fault's `sizing` holds where it is not null.
_SIZING_FIELDS = {"size": (int, float), "far_size": (int, float),
                  "decay_width": (int, float)}


def _check_faults(manifest: MeshManifest, fail: Fail) -> None:
    """The `faults[]` records against each other and the skeleton: the
    attributes running on from the interfaces', names unique and not an
    interface's, layers of the geometry, the curve groups fault k's,
    meetings mutual and with a junction to show for them, and the
    pull-back and sizing records well formed."""
    faults = manifest.faults
    n_faces = len(manifest.interfaces)
    names = [e["name"] for e in faults]
    if len(set(names)) != len(names):
        fail(f"fault names repeat: {names}")
    taken = {f["name"] for f in manifest.interfaces}
    if taken.intersection(names):
        fail(f"faults {sorted(taken.intersection(names))} share a name with an "
             "interface")
    inside = {int(e["attribute"]) for e in manifest.layers if e["in_geometry"]}
    meets = {e["name"]: set(e["meets"]) for e in faults}
    for i, e in enumerate(faults):
        k = i + 1
        if e["attribute"] != n_faces + k:
            fail(f"faults[{i}].attribute is {e['attribute']}, expected "
                 f"{n_faces + k}, after the {n_faces} interfaces")
        if not e["layers"] or not all(_typed(a, (int,)) and a in inside
                                      for a in e["layers"]):
            fail(f"faults[{i}].layers is {e['layers']}, not attributes of "
                 f"layers of the geometry, {sorted(inside)}")
        for key, want in zip(("tip", "free_edge", "junction"),
                             fault_curve_groups(k)):
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


def validate_against(manifest: MeshManifest, *, layer_count: int,
                     interface_count: int, fault_count: int = 0,
                     groups: Mapping[str, Iterable[int]] | None = None) -> None:
    """Check a manifest describes the mesh it was written beside.

    `groups` maps "layers" and "interfaces" to the physical group
    numbers the mesh carries, both of which must run 1..N from the
    centre, and "faults" to those of the faults' faces, which run on
    from the interfaces'.
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
                f"1..{len(entries)} from the centre outward")
    if groups is not None:
        first = {"layers": 1, "interfaces": 1, "faults": interface_count + 1}
        counts = {"layers": layer_count, "interfaces": interface_count,
                  "faults": fault_count}
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


@dataclass(frozen=True, eq=False)
class CurveEdges:
    """The edges of one fault's faces that lie on its curves
    (`fault_curves`): each (k, 2), the two corners of an edge as vertex
    indices with the smaller first, the rows in increasing order."""

    #: Its buried tip, where its sides are joined.
    tip: np.ndarray
    #: Its edge on the model's outer surface, where its sides are free.
    free_edge: np.ndarray
    #: Where it meets another fault.
    junction: np.ndarray


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


def fault_curves(faults: Sequence[ArrayLike], surface: ArrayLike) -> list[CurveEdges]:
    """The tip, free-edge and junction edges of each fault, read from the
    faces alone: the rule a consumer of a mesh that keeps no curves
    applies (the format page), and the edges the MSH file's curve
    groups hold.

    `faults` holds each fault's faces, (m_k, 3) corner indices, and
    `surface` the faces of the geometry's outer surface (the boundary
    elements of `MeshManifest.surface_attribute`), all in one numbering
    of the vertices.  An edge is a pair of corners.  The edges of a
    fault's faces that are also edges of another fault's faces are its
    junction; of the rest, an edge one of its faces uses alone is in
    its free edge where it is also an edge of the surface, and in its
    tip otherwise; an edge two of its faces use lies inside it.  The
    answer is one `CurveEdges` a fault, in the order given.
    """
    counts = [_edge_counts(_triangles(f, "a fault's faces")) for f in faults]
    outer = set(_edge_counts(_triangles(surface, "the surface's faces")))

    def array(edges: set[tuple[int, int]]) -> np.ndarray:
        return np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)

    out = []
    for k, own in enumerate(counts):
        shared = {e for e in own if any(e in other for j, other in enumerate(counts)
                                        if j != k)}
        once = {e for e, n in own.items() if n == 1} - shared
        free = once & outer
        out.append(CurveEdges(tip=array(once - free), free_edge=array(free),
                              junction=array(shared)))
    return out
