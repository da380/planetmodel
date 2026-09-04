"""The JSON that travels with every mesh.

A mesh file carries numbered attributes and nothing else.  Which number
is which layer, which boundary is which interface, where each sits,
whether the nodes have already been displaced: none of that survives
into the mesh file, so it travels alongside in a manifest, and this
module is the single definition of its shape.  The typed entries say
what a layer and an interface record carry, the block builders say what
every other block holds, `MeshManifest.from_build` assembles a manifest
from them and `validate_structure` checks a file against the same
definitions.

The schema is `planetmodel.mesh.manifest/2`, with these blocks:

  geometry     outer_radius, inner_radius, n_layers
  mesh         dimension, element_order, gmsh_version, msh_version,
               algorithm_2d, algorithm_3d, n_nodes, n_elements,
               high_order_optimised, date
  delivery     "physical" or "referential"
  layers[]     attribute, name, r_inner, r_outer, in_geometry
  interfaces[] attribute, name, mean_radius, between_layers
  mapping      kind, repr, knots, applied_to_nodes
  sizing       policy, per_interface[]
  validation   the counts and warnings of the mesh checks
  provenance   planetmodel_version, mesh_file, perturbation, meta
  files        null until the MFEM export writes it

Every length is in the geometry's own numbers, the ones the mesh file
holds: the mesher neither scales nor normalises, so a radius in the
manifest is a radius in the mesh.  Layers are numbered by `attribute`
1..N from the centre and `between_layers` gives 0-based layer indices
with -1 for the outside.  Flat and boring on purpose: a C++ reader will
parse it, nested objects only where the nesting carries meaning, no
polymorphism, every number a number.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import KW_ONLY, asdict, dataclass, field, fields as _fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, get_type_hints

from .spec import DELIVERIES

__all__ = ["SCHEMA", "MeshManifest", "LayerEntry", "InterfaceEntry",
           "write", "read", "file_digest", "beside", "planetmodel_version",
           "geometry_block", "mesh_block", "mapping_block", "sizing_block",
           "validation_block", "provenance_block", "validate_structure",
           "validate_against"]

#: Bump only for an incompatible change; consumers check it.
SCHEMA = "planetmodel.mesh.manifest/2"

#: The MSH format version the mesher writes and MFEM's reader wants.
MSH_VERSION = 2.2


def beside(path, suffix: str) -> Path:
    """`path` with `suffix`, treating the path as a basename.

    Only a mesh or manifest suffix is replaced; anything else is part of
    the name, so `run.v1.5` becomes `run.v1.5.json` rather than
    `run.v1.json`.
    """
    path = Path(path)
    if path.suffix in (".msh", ".json", ".mesh"):
        path = path.with_suffix("")
    return path.with_name(path.name + suffix)


def planetmodel_version() -> str:
    """The library version a manifest records as its provenance."""
    from .. import __version__
    return __version__


def file_digest(path) -> str | None:
    """The sha256 of a file, or None if it is not there to hash."""
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------ the entries

@dataclass
class LayerEntry:
    """One `layers[]` record: a layer of the mesh, centre outward."""

    #: The element attribute, 1..N from the centre.
    attribute: int
    #: The layer's name; `layer_<attribute>` where none was given.
    name: str
    #: The inner radius, in the geometry's lengths.
    r_inner: float
    #: The outer radius, in the geometry's lengths.
    r_outer: float
    #: Whether the layer belongs to the geometry (True) or is a shell (False).
    in_geometry: bool

    @classmethod
    def from_layer(cls, layer, *, attribute: int, r_inner: float,
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
    #: The mean radius, in the geometry's lengths.
    mean_radius: float
    #: [layer below, layer above], 0-based, -1 outside.
    between_layers: list

    @classmethod
    def from_interface(cls, face, *, attribute: int,
                       mean_radius: float) -> "InterfaceEntry":
        """The record of an InterfaceInfo at the given mean radius."""
        below, above = face.between
        return cls(attribute=int(attribute),
                   name=face.name or f"interface_{attribute}",
                   mean_radius=float(mean_radius),
                   between_layers=[int(below), int(above)])


#: The JSON types each annotated Python type may hold.
_JSON_TYPES = {int: (int,), float: (int, float), bool: (bool,), str: (str,),
               list: (list,), dict: (dict,)}


def _entry_types(cls) -> dict:
    """Each field of an entry and the JSON types it must hold."""
    hints = get_type_hints(cls)
    return {f.name: _JSON_TYPES[hints[f.name]] for f in _fields(cls)
            if f.name != "_"}


# ------------------------------------------------------------ the blocks

def geometry_block(*, outer_radius: float, inner_radius: float, n_layers: int,
                   **extra) -> dict:
    """The `geometry` record: the domain's extent, in the geometry's own
    numbers, and its layer count.

    `extra` adds what a particular builder knows beyond the schema's
    required keys; any length among them is in the same numbers.
    """
    return {"outer_radius": float(outer_radius),
            "inner_radius": float(inner_radius), "n_layers": int(n_layers),
            **extra}


def mesh_block(*, dimension: int, order: int, gmsh_version: str,
               algorithm_2d: int, algorithm_3d: int, counts: dict,
               curving: dict) -> dict:
    """The `mesh` record: what gmsh was asked for and what it produced."""
    return {
        "dimension": int(dimension),
        "element_order": int(order),
        "gmsh_version": str(gmsh_version),
        "msh_version": MSH_VERSION,
        "algorithm_2d": int(algorithm_2d),
        "algorithm_3d": int(algorithm_3d),
        "n_nodes": int(counts.get("nodes", 0)),
        "n_elements": int(counts.get("elements", 0)),
        "high_order_optimised": bool(curving.get("optimized")),
    }


def mapping_block(mapping, *, knots=(), applied_to_nodes: bool) -> dict:
    """The `mapping` record: the mapping's class and repr, its knots, and
    whether the nodes already carry it.

    `mapping` None stands for the identity.  `knots` are the radii where
    the mapping's gradient may jump.
    """
    if mapping is None:
        from ..mapping import IdentityMapping
        mapping = IdentityMapping()
    return {"kind": type(mapping).__name__, "repr": repr(mapping),
            "knots": [float(k) for k in knots],
            "applied_to_nodes": bool(applied_to_nodes)}


def sizing_block(*, policy: str, sizes: dict) -> dict:
    """The `sizing` record: the rule by name and what it gave each interface."""
    return {
        "policy": str(policy),
        "per_interface": [
            {"attribute": i + 1, "size": float(s.size),
             "far_size": float(s.far_size),
             "decay_width": float(s.decay_width)}
            for i, s in sorted(sizes.items())],
    }


def validation_block(report, orientation) -> dict:
    """The `validation` record, from the mesh checks and the orientation repair."""
    return {
        "negative_jacobians": int(report.negative_jacobians),
        "min_sicn": float(report.min_sicn),
        "wrong_orientation": int(report.negative_cells + report.inward_faces),
        "faces_reoriented": int(orientation.faces_flipped),
        "max_interface_radius_error": float(report.max_interface_radius_error),
        "warnings": list(report.warnings),
        "failures": list(getattr(report, "failures", ())),
    }


def provenance_block(*, mesh_file: str, perturbation=None, meta=None) -> dict:
    """The `provenance` record: version, mesh file, what the displacement
    did, and the spec's `meta` copied in."""
    block = {"planetmodel_version": planetmodel_version(),
             "mesh_file": str(mesh_file)}
    block["perturbation"] = (None if perturbation is None else {
        "nodes": int(perturbation.nodes),
        "max_displacement": float(perturbation.max_displacement),
        "validity_margin": _finite(perturbation.validity_margin)})
    block["meta"] = dict(meta or {})
    return block


def _finite(value):
    """A number, or None where there is no number to report."""
    return float(value) if math.isfinite(value) else None


# ---------------------------------------------------------- the manifest

@dataclass
class MeshManifest:
    """Everything a consumer needs that the mesh file cannot carry."""

    _: KW_ONLY
    geometry: dict = field(default_factory=dict)
    mesh: dict = field(default_factory=dict)
    delivery: str = "physical"
    layers: list = field(default_factory=list)
    interfaces: list = field(default_factory=list)
    mapping: dict = field(default_factory=dict)
    sizing: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    #: The MFEM delivery, or None for a bare mesh: which files were
    #: written, how the mesh must be constructed for their dof numbering
    #: to hold, and one record per GridFunction.  Written by the export.
    files: dict | None = None
    schema: str = SCHEMA

    @classmethod
    def from_build(cls, *, geometry: dict, mesh: dict, delivery: str,
                   layers: Sequence[LayerEntry],
                   interfaces: Sequence[InterfaceEntry], mapping: dict,
                   sizing: dict, validation: dict,
                   provenance: dict) -> "MeshManifest":
        """A manifest from typed entries and the blocks the builders make.

        The one place a manifest is assembled, so a key cannot drift
        between builders and the schema is described once.
        """
        card = cls(geometry=dict(geometry), mesh=dict(mesh), delivery=delivery,
                   layers=[asdict(e) for e in layers],
                   interfaces=[asdict(e) for e in interfaces],
                   mapping=dict(mapping), sizing=dict(sizing),
                   validation=dict(validation), provenance=dict(provenance))
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

    @property
    def shell_attributes(self) -> tuple[int, ...]:
        """Attributes of the layers outside the geometry: the shells."""
        return tuple(int(e["attribute"]) for e in self.layers
                     if not e.get("in_geometry", True))

    def describe(self) -> str:
        """A readable multi-line summary of the manifest.

        The schema and the delivery, the geometry block, one line per
        layer and per interface, the mapping, the validation verdict and
        the files block where the MFEM export wrote one.  Every length
        is in the geometry's own numbers.
        """
        lines = [f"{self.schema}", f"  delivery    {self.delivery}",
                 "  geometry    " + _pairs(self.geometry)]
        m = self.mesh
        if m:
            lines.append(
                f"  mesh        {m.get('dimension', '?')}D, order "
                f"{m.get('element_order', '?')}, {m.get('n_elements', '?')} "
                f"elements, {m.get('n_nodes', '?')} nodes, gmsh "
                f"{m.get('gmsh_version', '?')}")
        lines.append("  layers")
        rows = [(str(e.get("attribute", "?")), str(e.get("name", "")),
                 f"[{_num(e.get('r_inner'))}, {_num(e.get('r_outer'))}]",
                 "in geometry" if e.get("in_geometry", True) else "shell")
                for e in self.layers]
        lines.extend(_table(rows))
        lines.append("  interfaces")
        rows = [(str(e.get("attribute", "?")), str(e.get("name", "")),
                 f"mean radius {_num(e.get('mean_radius'))}",
                 f"between layers {list(e.get('between_layers', []))}")
                for e in self.interfaces]
        lines.extend(_table(rows))
        mp = self.mapping
        knots = ", ".join(_num(k) for k in mp.get("knots", ()))
        lines.append(
            f"  mapping     {mp.get('kind', '?')}, applied to nodes "
            f"{mp.get('applied_to_nodes', '?')}, knots [{knots}]")
        v = self.validation
        if v:
            failures = v.get("failures", [])
            ok = (not failures and not v.get("negative_jacobians")
                  and not v.get("wrong_orientation")
                  and v.get("min_sicn", 0.0) > 0.0)
            warns = v.get("warnings", [])
            lines.append(
                f"  validation  {'ok' if ok else 'FAILED'}, minSICN "
                f"{_num(v.get('min_sicn'))}, "
                + (f"{len(warns)} warning(s)" if warns else "no warnings"))
            lines.extend(f"    - FAILED: {w}" for w in failures)
            lines.extend(f"    - {w}" for w in warns)
        f = self.files
        if f is not None:
            lines.append(
                f"  files       mesh {f.get('mesh', '?')}, read options "
                + _pairs(f.get("mesh_read_options", {})))
            for gf in f.get("grid_functions", []):
                lines.append(
                    f"    {gf.get('kind', '?')}  {gf.get('name', '?')}  "
                    f"{gf.get('file', '?')}  {gf.get('fe_space', '?')} "
                    f"vdim {gf.get('vdim', '?')} {gf.get('ordering', '?')}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        return (f"MeshManifest({self.schema}, {self.delivery}, "
                f"{len(self.layers)} layers, {len(self.interfaces)} interfaces)")


def _num(value) -> str:
    """A number for `describe`, short and exact enough to recognise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _pairs(record: dict) -> str:
    """`key value` pairs of a flat record, for `describe`."""
    return ", ".join(f"{k} {_num(v)}" for k, v in record.items())


def _table(rows) -> list:
    """Rows of strings as indented lines with their columns aligned, the
    first column right-aligned as a number and the rest left-aligned."""
    if not rows:
        return []
    widths = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    return ["    " + "  ".join(
        cell.rjust(w) if j == 0 else cell.ljust(w)
        for j, (cell, w) in enumerate(zip(row, widths))).rstrip()
        for row in rows]


def write(path, manifest: MeshManifest) -> Path:
    """Write the manifest beside its mesh, and return the path."""
    path = beside(path, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    data.setdefault("mesh", {}).setdefault(
        "date", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with open(path, "w") as fh:
        # allow_nan=False: a strict parser cannot read NaN or Infinity, so
        # a manifest carrying one fails here, where the cause is known.
        json.dump(data, fh, indent=2, sort_keys=False, allow_nan=False)
        fh.write("\n")
    return path


def read(path) -> MeshManifest:
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

#: What every block must carry, and as what: {block: {key: JSON types}}.
_BLOCKS = {
    "geometry": {"outer_radius": (int, float), "inner_radius": (int, float),
                 "n_layers": (int,)},
    "mesh": {"dimension": (int,), "element_order": (int,),
             "gmsh_version": (str,), "msh_version": (int, float),
             "algorithm_2d": (int,), "algorithm_3d": (int,),
             "n_nodes": (int,), "n_elements": (int,),
             "high_order_optimised": (bool,)},
    "mapping": {"kind": (str,), "repr": (str,), "knots": (list,),
                "applied_to_nodes": (bool,)},
    "sizing": {"policy": (str,), "per_interface": (list,)},
    "validation": {"negative_jacobians": (int,), "min_sicn": (int, float),
                   "wrong_orientation": (int,), "faces_reoriented": (int,),
                   "max_interface_radius_error": (int, float),
                   "warnings": (list,)},
    "provenance": {"planetmodel_version": (str,), "mesh_file": (str,),
                   "perturbation": (dict, type(None)), "meta": (dict,)},
}

#: What every `sizing.per_interface` record carries.
_SIZING_FIELDS = {"attribute": (int,), "size": (int, float),
                  "far_size": (int, float), "decay_width": (int, float)}

#: What every `files.grid_functions` record carries.  A consumer reads
#: these to build the space before it opens the file.
_GF_FIELDS = {"kind": (str,), "name": (str,), "file": (str,),
              "fe_space": (str,), "vdim": (int,), "ordering": (str,)}


def _typed(value, kinds) -> bool:
    """Whether `value` is one of `kinds`, with bool never passing as int."""
    if isinstance(value, bool) and bool not in kinds:
        return False
    return isinstance(value, kinds)


def _check_record(record, types: dict, where: str, fail) -> None:
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
    def fail(msg):
        raise ValueError(f"malformed manifest: {msg}")

    if manifest.delivery not in DELIVERIES:
        fail(f"delivery {manifest.delivery!r} is not one of {DELIVERIES}")
    for name, types in _BLOCKS.items():
        _check_record(getattr(manifest, name), types, name, fail)
    if manifest.mesh["dimension"] not in (2, 3):
        fail(f"mesh.dimension is {manifest.mesh['dimension']}, not 2 or 3")
    if not all(_typed(k, (int, float)) for k in manifest.mapping["knots"]):
        fail("mapping.knots must be a list of numbers")
    for i, entry in enumerate(manifest.sizing["per_interface"]):
        _check_record(entry, _SIZING_FIELDS, f"sizing.per_interface[{i}]", fail)
    if not all(isinstance(w, str) for w in manifest.validation["warnings"]):
        fail("validation.warnings must be a list of strings")
    if not all(isinstance(w, str) for w in manifest.validation.get("failures", [])):
        fail("validation.failures must be a list of strings")

    for what, entries, cls in (("layers", manifest.layers, LayerEntry),
                               ("interfaces", manifest.interfaces,
                                InterfaceEntry)):
        if not isinstance(entries, list):
            fail(f"{what} must be a list of objects")
        types = _entry_types(cls)
        for i, e in enumerate(entries):
            _check_record(e, types, f"{what}[{i}]", fail)

    n_layers = manifest.geometry["n_layers"]
    if len(manifest.layers) != n_layers:
        fail(f"geometry.n_layers is {n_layers} but {len(manifest.layers)} "
             "layers are listed")
    for i, lay in enumerate(manifest.layers):
        if lay["attribute"] != i + 1:
            fail(f"layers[{i}].attribute is {lay['attribute']}, expected {i + 1}")
        if i and lay["r_inner"] != manifest.layers[i - 1]["r_outer"]:
            fail(f"layers[{i}] starts at {lay['r_inner']} but the "
                 f"layer below ends at {manifest.layers[i - 1]['r_outer']}")
    if manifest.layers:
        if manifest.layers[0]["r_inner"] != manifest.geometry["inner_radius"]:
            fail("layers[0].r_inner disagrees with geometry.inner_radius")
        if manifest.layers[-1]["r_outer"] != manifest.geometry["outer_radius"]:
            fail("the last layer's r_outer disagrees with geometry.outer_radius")

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
    _check_files(manifest.files, fail)


def _check_files(files, fail) -> None:
    """The MFEM delivery's `files` block, or nothing where none was written."""
    if files is None:
        return
    if not isinstance(files, dict):
        fail(f"files is {type(files).__name__}, not an object")
    if not isinstance(files.get("mesh"), str):
        fail(f"files.mesh is {files.get('mesh')!r}, not a mesh file name")
    options = files.get("mesh_read_options")
    if not isinstance(options, dict) or "refine" not in options:
        fail("files.mesh_read_options must give the mfem::Mesh constructor "
             "arguments the dof numbering was written under")
    entries = files.get("grid_functions")
    if not isinstance(entries, list):
        fail("files.grid_functions must be a list of objects")
    for i, e in enumerate(entries):
        _check_record(e, _GF_FIELDS, f"files.grid_functions[{i}]", fail)


def validate_against(manifest: MeshManifest, *, layer_count: int,
                     interface_count: int, groups: dict | None = None) -> None:
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
