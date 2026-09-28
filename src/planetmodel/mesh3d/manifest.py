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
  fields[]      name, file, fe_space, vdim, ordering, rank, weight,
                voigt, unit, layers, radial_degree
  scales        length, mass, time in SI; null until a model is exported
  constants     the model's constants in its units; empty until then
  meta          whatever the spec's `meta` carried

`fluid`, `kind`, `values` and `radial_degree` are read from the model,
so the model export writes them and everything else writes null: null
means "no model has said", never "no".  `fluid` is the vanishing of
shear in the exported fields (`materials.is_fluid`), a statement about
this model's description, not about any one timescale; a shell is not
fluid.  `kind` classifies an interface from those flags (KINDS below);
`values` holds the one-sided values of each exported real scalar radial
field, `{name: [below, above]}`; `radial_degree` is the degree in r
that reproduces a scalar field within each layer holding it.

The layers and interfaces are the skeleton the mesh was built on, in
the mesh's own lengths: the mesher neither scales nor normalises, so a
radius here is a radius in the mesh.  Layers are numbered by
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
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, asdict, dataclass, field, fields as _fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, get_type_hints

from ..units import Dimensions, Scales, unit_string

if TYPE_CHECKING:
    from ..character import Character
    from ..geometry import InterfaceInfo, LayerInfo
    from ..model import Model

__all__ = ["SCHEMA", "FORMATS", "NODES", "KINDS", "MeshManifest", "LayerEntry",
           "InterfaceEntry", "FieldEntry", "write", "read", "beside",
           "mesh_block", "scales_block", "constants_block",
           "validate_structure", "validate_against"]

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
               str | None: (str, type(None)), dict | None: (dict, type(None))}


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
    """The `scales` record: what one stored unit of length, mass and time
    is in SI, so that every field value and constant is in those units."""
    return {"length": float(scales.length), "mass": float(scales.mass),
            "time": float(scales.time)}


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
    #: One record per field written beside the mesh; see `FieldEntry`.
    fields: list[dict[str, Any]] = field(default_factory=list)
    #: The scales the field values are in, or None until a model is exported.
    scales: dict[str, float] | None = None
    #: The model's constants in its units; empty until a model is exported.
    constants: dict[str, float] = field(default_factory=dict)
    #: The spec's `meta`, copied through.
    meta: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    @classmethod
    def from_build(cls, *, mesh: Mapping[str, Any], layers: Sequence[LayerEntry],
                   interfaces: Sequence[InterfaceEntry],
                   meta: Mapping[str, Any] | None = None) -> "MeshManifest":
        """A manifest from the mesh block and the typed entries.

        The one place a manifest is assembled, so a key cannot drift
        between builders and the schema is described once.
        """
        card = cls(mesh=dict(mesh), layers=[asdict(e) for e in layers],
                   interfaces=[asdict(e) for e in interfaces],
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
        if self.meta:
            lines.append("  meta        " + _pairs(self.meta))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        return (f"MeshManifest({self.schema}, {self.mesh.get('file', '?')}, "
                f"{len(self.layers)} layers, {len(self.interfaces)} interfaces, "
                f"{len(self.fields)} fields)")


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
    """Write the manifest beside its mesh, and return the path."""
    path = beside(path, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        # allow_nan=False: a strict parser cannot read NaN or Infinity, so
        # a manifest carrying one fails here, where the cause is known.
        json.dump(asdict(manifest), fh, indent=2, sort_keys=False,
                  allow_nan=False)
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
        if i and lay["r_inner"] != manifest.layers[i - 1]["r_outer"]:
            fail(f"layers[{i}] starts at {lay['r_inner']} but the "
                 f"layer below ends at {manifest.layers[i - 1]['r_outer']}")

    n_faces = len(manifest.interfaces)
    if n_faces not in (n_layers, n_layers + 1):
        fail(f"{n_faces} interfaces for {n_layers} layers; a full mesh has one "
             "per layer and a hollow mesh one more")
    first = n_faces - n_layers          # 1 when the inner boundary is a face
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
    if m["displacement"] is not None and m["displacement"] not in names:
        fail(f"mesh.displacement names {m['displacement']!r}, which fields "
             "holds no record for")
    for i, face in enumerate(manifest.interfaces):
        for key in face["values"] or {}:
            if key not in names:
                fail(f"interfaces[{i}].values names {key!r}, which fields "
                     "holds no record for")

    if manifest.scales is not None:
        _check_record(manifest.scales, _SCALES_FIELDS, "scales", fail)
    if not isinstance(manifest.constants, dict):
        fail("constants must be an object")
    for key, value in manifest.constants.items():
        if not _typed(value, (int, float)):
            fail(f"constants.{key} is {value!r}, not a number")
    if not isinstance(manifest.meta, dict):
        fail("meta must be an object")


def validate_against(manifest: MeshManifest, *, layer_count: int,
                     interface_count: int,
                     groups: Mapping[str, Iterable[int]] | None = None) -> None:
    """Check a manifest describes the mesh it was written beside.

    `groups` maps "layers" and "interfaces" to the physical group
    numbers the mesh carries; both must run 1..N from the centre.
    """
    if len(manifest.layers) != layer_count:
        raise ValueError(
            f"manifest lists {len(manifest.layers)} layers, the mesh has "
            f"{layer_count}")
    if len(manifest.interfaces) != interface_count:
        raise ValueError(
            f"manifest lists {len(manifest.interfaces)} interfaces, the mesh "
            f"has {interface_count}")
    for what, entries in (("layer", manifest.layers),
                          ("interface", manifest.interfaces)):
        attrs = [int(e["attribute"]) for e in entries]
        if attrs != list(range(1, len(entries) + 1)):
            raise ValueError(
                f"manifest {what} attributes are {attrs}, expected "
                f"1..{len(entries)} from the centre outward")
    if groups is not None:
        counts = {"layers": layer_count, "interfaces": interface_count}
        for what, wanted in groups.items():
            got = sorted(wanted)
            if got != list(range(1, counts.get(what, len(got)) + 1)):
                raise ValueError(
                    f"the mesh's {what} physical groups are {got}; the manifest "
                    f"describes {counts.get(what)} numbered from 1")
