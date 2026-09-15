"""manifest.py -- the JSON that travels with every mesh.

A mesh file carries numbered attributes and nothing else.  Which number
is the outer core, which boundary is the Moho, what one length unit
means in metres, whether the nodes have already been displaced -- none
of that survives into the .msh, and a consumer that hard-codes it (the
`bdr_attributes.Max() - 2` pattern this replaces) breaks the moment a
layer is added.  So it travels alongside, and this module is the single
definition of its shape: the typed entries below say what a layer and
an interface record carry, `from_build` assembles a manifest from them,
and `validate_structure` checks a file against the same definitions.

Deliberately flat and boring: a C++ reader will parse it.  Nested
objects only where the nesting carries meaning, no polymorphism, and
every number a number rather than a string with units in it.

The `mapping` block is reconstruction-grade, not provenance.  A
referential delivery hands over the reference mesh and expects the
consumer to apply the mapping itself, so what is recorded must be
enough to rebuild it: the rule and its parameters, the surfaces with
their hashes, and the knot radii where dh/dr jumps.

The `files` block is the same idea for the MFEM delivery, and is
written by `mesh3d/export.py` rather than by the mesher: the mesh
file, the `mfem::Mesh` constructor arguments its dof numbering was
written under, and one record per GridFunction saying which space it
lives in and what its numbers mean -- character, dimensions, units,
frame, and the layers it is defined on, since a field zero-filled
outside its domain looks like a field that is zero there.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import KW_ONLY, asdict, dataclass, field, fields as _fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

__all__ = ["provenance_of", "SCHEMA", "MeshManifest", "LayerEntry", "InterfaceEntry",
           "law_name_of", "write", "read", "file_digest", "units_block",
           "mesh_block", "sizing_block", "validation_block",
           "provenance_block", "validate_against", "validate_structure",
           "beside", "planetmodel_version"]

DELIVERIES = ("physical", "referential")
_ROLES = ("material", "control")
#: The name of the frequency-dependent moduli a layer's `law` describes,
#: and of the static moduli a `"static"` law lifts.
VISCOELASTIC_MODULI = "viscoelastic_moduli"
ELASTIC_MODULI = "elastic_moduli"


def beside(path, suffix: str) -> Path:
    """`path` with `suffix`, treating the path as a basename.

    Only a mesh or manifest suffix is replaced; anything else is part of
    the name, so `run.v1.5` becomes `run.v1.5.json` rather than
    `run.v1.json`.
    """
    path = Path(path)
    if path.suffix in (".msh", ".json"):
        path = path.with_suffix("")
    return path.with_name(path.name + suffix)


#: Bump only for an incompatible change; consumers check it.
SCHEMA = "planetmodel.mesh.manifest/1"


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

def provenance_of(topography) -> dict:
    """`Topography.provenance()` with every file entry completed.

    Each entry of `files` carries `file`, `sha256` (computed here where the
    shape did not supply one), `units` and `scale_to_m`; the entries are
    sorted by file name so two writes of one shape agree byte for byte.
    """
    p = dict(topography.provenance())
    files = []
    for entry in p.get("files", ()):
        entry = dict(entry)
        if entry.get("sha256") is None:
            entry["sha256"] = file_digest(entry["file"])
        entry.setdefault("units", None)
        entry.setdefault("scale_to_m", 1.0)
        files.append(entry)
    p["files"] = sorted(files, key=lambda e: e["file"])
    return p


# ------------------------------------------------------------ the entries

def law_name_of(layer) -> str | None:
    """The law behind a layer's frequency-dependent moduli, by name.

    The registered name of the law that built `viscoelastic_moduli` on
    the layer (`"constant_q"`, `"maxwell"`, ...), read from the record
    the field carries; `"static"` where the layer's moduli do not depend
    on frequency -- it holds `elastic_moduli` and either no
    frequency-dependent field or a lift of the static one; and None
    where the layer holds no moduli at all, or a frequency-dependent
    field written by hand that no file can rebuild.  A name only: the
    netCDF file carries the parameters and constants a rebuild needs.
    """
    held = getattr(layer, "fields", {})
    dynamic = held.get(VISCOELASTIC_MODULI)
    if dynamic is None:
        return "static" if ELASTIC_MODULI in held else None
    record = getattr(dynamic, "law", None)
    name = getattr(record, "law", record)
    if isinstance(name, str):
        return name
    # A lifted field carries the static field it stands for.
    if getattr(dynamic, "source", None) is not None:
        return "static"
    return None


@dataclass
class LayerEntry:
    """One `layers[]` record: a layer of the mesh, centre outward."""

    attribute: int
    name: str
    r_inner_nd: float
    r_outer_nd: float
    state: str
    fields: list
    is_vacuum: bool
    _: KW_ONLY
    law: str | None = None

    @classmethod
    def from_layer(cls, layer, *, attribute: int, r_inner_nd: float,
                   r_outer_nd: float) -> "LayerEntry":
        """The record of a `Layer`, its radii already in mesh units."""
        return cls(attribute=int(attribute),
                   name=layer.name or f"layer_{attribute}",
                   r_inner_nd=float(r_inner_nd), r_outer_nd=float(r_outer_nd),
                   state=str(layer.state), fields=list(layer.field_names),
                   is_vacuum=bool(layer.is_vacuum), law=law_name_of(layer))


@dataclass
class InterfaceEntry:
    """One `interfaces[]` record: a boundary of the mesh, centre outward.

    `between_layers` is `[below, above]` with -1 for the space outside
    the outermost boundary.
    """

    attribute: int
    name: str
    mean_radius_nd: float
    between_layers: list
    role: str

    @classmethod
    def from_interface(cls, face, *, attribute: int,
                       mean_radius_nd: float) -> "InterfaceEntry":
        """The record of an `Interface`, its radius already in mesh units."""
        below, above = face.between
        return cls(attribute=int(attribute),
                   name=face.name or f"interface_{attribute}",
                   mean_radius_nd=float(mean_radius_nd),
                   between_layers=[int(below),
                                   -1 if above is None else int(above)],
                   role=str(face.role))


def _entry_types(cls) -> dict:
    """Each field of an entry and the JSON type it must hold."""
    out = {}
    for f in _fields(cls):
        if f.name == "law":
            out[f.name] = (str, type(None))
        elif f.type in ("int",):
            out[f.name] = (int,)
        elif f.type in ("float",):
            out[f.name] = (int, float)
        elif f.type in ("bool",):
            out[f.name] = (bool,)
        elif f.type in ("list",):
            out[f.name] = (list,)
        else:
            out[f.name] = (str,)
    return out


# ------------------------------------------------------------ the blocks

def units_block(scales, divisor: float, rref_m: float) -> dict:
    """The units record: a flat rref_m plus the full scale triple.

    `rref_m` is what a consumer reads to recover metres, and is kept
    flat and first because it is the only part most of them need.  The
    triple beside it is populated where the body carried scales and null
    where it did not, so a future mass or time scale has somewhere to go
    without changing the schema under an existing reader.
    """
    non_si = scales is not None and not scales.is_si
    return {
        "convention": "non-dimensional",
        "rref_m": float(rref_m),
        "geometry_divisor": float(divisor),
        "scales": {
            "length_m": float(scales.length) if non_si else None,
            "mass_kg": float(scales.mass) if non_si else None,
            "time_s": float(scales.time) if non_si else None,
        },
        "gravitational_constant": (float(scales.gravitational_constant)
                                   if non_si else None),
    }


def mesh_block(*, dimension: int, order: int, gmsh_version: str,
               algorithm_2d: int, algorithm_3d: int, counts: dict,
               curving: dict) -> dict:
    """The `mesh` record: what gmsh was asked for and what it produced."""
    return {
        "dimension": int(dimension),
        "element_order": int(order),
        "gmsh_version": gmsh_version,
        "msh_version": 2.2,
        "algorithm_2d": int(algorithm_2d),
        "algorithm_3d": int(algorithm_3d),
        "n_nodes": counts.get("nodes", 0),
        "n_elements": counts.get("elements", 0),
        "high_order_optimised": bool(curving.get("optimized")),
    }


def sizing_block(*, policy: str, sizes: dict) -> dict:
    """The `sizing` record: the rule by name and what it gave each interface."""
    return {
        "policy": policy,
        "per_interface": [
            {"attribute": i + 1, "size_nd": s.size, "far_size_nd": s.far_size,
             "decay_width_nd": s.decay_width}
            for i, s in sorted(sizes.items())],
    }


def validation_block(report, orientation) -> dict:
    """The `validation` record, from the mesh checks and the orientation repair."""
    return {
        "negative_jacobians": report.negative_jacobians,
        "min_sicn": report.min_sicn,
        "wrong_orientation": report.negative_cells + report.inward_faces,
        "faces_reoriented": orientation.faces_flipped,
        "max_interface_radius_error_nd": report.max_interface_radius_error,
        "knots_aligned_with_interfaces": report.knots_aligned,
        "warnings": list(report.warnings),
    }


def provenance_block(*, mesh_file: str, perturbation=None, **extra) -> dict:
    """The `provenance` record: version, mesh file, and what the build knows.

    `extra` is what a recipe run adds -- the recipe file, its hash and
    the command -- and is empty for a mesh built from Python, where
    reproducibility honestly falls back to the user's script.
    """
    block = {"planetmodel_version": planetmodel_version(),
             "mesh_file": mesh_file, **extra}
    block["perturbation"] = (None if perturbation is None else {
        "nodes": perturbation.nodes,
        "max_displacement_nd": perturbation.max_displacement,
        "validity_margin": _finite(perturbation.validity_margin)})
    return block


def _finite(value):
    """A number, or None where there is no number to report."""
    import math
    return float(value) if math.isfinite(value) else None


# ---------------------------------------------------------- the manifest

@dataclass
class MeshManifest:
    """Everything a consumer needs that the .msh cannot carry."""

    _: KW_ONLY
    model: dict = field(default_factory=dict)
    mesh: dict = field(default_factory=dict)
    delivery: str = "physical"
    layers: list = field(default_factory=list)
    interfaces: list = field(default_factory=list)
    coarsening: dict = field(default_factory=dict)
    mapping: dict | None = None
    sizing: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    #: The MFEM delivery, or None for a bare mesh: which files were
    #: written, how the mesh must be constructed for their dof numbering
    #: to hold, and one record per GridFunction.  Written by
    #: `mesh3d/export.py`; the mesher itself writes no fields and leaves
    #: it null.
    files: dict | None = None
    schema: str = SCHEMA

    @classmethod
    def from_build(cls, *, model: dict, mesh: dict, delivery: str,
                   layers: Sequence[LayerEntry],
                   interfaces: Sequence[InterfaceEntry],
                   sizing: dict, validation: dict, provenance: dict,
                   coarsening: dict | None = None,
                   mapping: dict | None = None) -> "MeshManifest":
        """A manifest from typed entries and the blocks the builders make.

        The one place a manifest is assembled: the layered mesher and
        the offset builder both come here, so a key cannot drift
        between them and the schema is described once.
        """
        card = cls(model=dict(model), mesh=dict(mesh), delivery=delivery,
                   layers=[asdict(e) for e in layers],
                   interfaces=[asdict(e) for e in interfaces],
                   coarsening=dict(coarsening or {}), mapping=mapping,
                   sizing=dict(sizing), validation=dict(validation),
                   provenance=dict(provenance))
        validate_structure(card)
        return card

    # -- convenience for consumers -----------------------------------------

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
    def vacuum_attributes(self) -> tuple[int, ...]:
        """Attributes of the vacuum layers: voids and buffers, holding no
        material, which a solver excludes from every material region."""
        return tuple(int(e["attribute"]) for e in self.layers
                     if e.get("is_vacuum"))


    @property
    def rref_m(self) -> float:
        """One mesh length unit, in metres."""
        return float(self.model["rref_m"])


def write(path, manifest: MeshManifest) -> Path:
    """Write the manifest beside its mesh, and return the path."""
    path = beside(path, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    data.setdefault("mesh", {}).setdefault(
        "date", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with open(path, "w") as fh:
        # allow_nan=False on purpose: Python writes NaN and Infinity
        # happily and no strict parser will read them back, so a manifest
        # carrying one is a file the C++ consumer cannot open.  Better to
        # fail here, where the field that produced it is still known.
        json.dump(data, fh, indent=2, sort_keys=False, allow_nan=False)
        fh.write("\n")
    return path


def read(path) -> MeshManifest:
    """Read a manifest, checking the schema it declares."""
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


def validate_structure(manifest: MeshManifest) -> None:
    """Check a manifest has the shape the schema promises.

    The schema string says which shape to expect; this checks the file
    actually has it, against the same entry definitions the builders
    write from, so a consumer gets one ValueError naming the field
    rather than a TypeError from deep inside its own reader.
    """
    def fail(msg):
        raise ValueError(f"malformed manifest: {msg}")

    if manifest.delivery not in DELIVERIES:
        fail(f"delivery {manifest.delivery!r} is not one of {DELIVERIES}")
    for what, entries, cls in (("layers", manifest.layers, LayerEntry),
                               ("interfaces", manifest.interfaces,
                                InterfaceEntry)):
        if not isinstance(entries, list) or not all(
                isinstance(e, dict) for e in entries):
            fail(f"{what} must be a list of objects")
        types = _entry_types(cls)
        for i, e in enumerate(entries):
            for key, kinds in types.items():
                value = e.get(key)
                if key not in e or not isinstance(value, kinds) or (
                        bool in kinds and not isinstance(value, bool)
                        and isinstance(value, int)):
                    fail(f"{what}[{i}].{key} is {value!r}, not "
                         f"{' or '.join(k.__name__ for k in kinds)}")
    for i, lay in enumerate(manifest.layers):
        if not all(isinstance(n, str) for n in lay["fields"]):
            fail(f"layers[{i}].fields is {lay['fields']!r}, not a list of "
                 "field names (an empty list where the layer holds nothing)")
        if i and lay["r_inner_nd"] != manifest.layers[i - 1]["r_outer_nd"]:
            fail(f"layers[{i}] starts at {lay['r_inner_nd']} but the "
                 f"layer below ends at {manifest.layers[i - 1]['r_outer_nd']}")
    for i, face in enumerate(manifest.interfaces):
        if face["role"] not in _ROLES:
            fail(f"interfaces[{i}].role {face['role']!r} is not one of "
                 f"{_ROLES}")
        want = [i, i + 1 if i < len(manifest.interfaces) - 1 else -1]
        if list(face["between_layers"]) != want:
            fail(f"interfaces[{i}].between_layers is "
                 f"{face['between_layers']}, expected {want}")
    if not isinstance(manifest.model.get("rref_m"), (int, float)):
        fail("model.rref_m is missing or not a number")
    _check_files(manifest.files, fail)


#: What every `files.grid_functions` record must carry, and as what.  A
#: consumer reads these to build the space before it opens the file, so a
#: missing one is a mesh it cannot use rather than a cosmetic lapse.
_GF_FIELDS = (("name", str), ("file", str), ("fe_space", str), ("vdim", int),
              ("ordering", str), ("frame", str), ("units", str),
              ("character_rank", int), ("character_weight", int))


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
    if not isinstance(entries, list) or not all(isinstance(e, dict)
                                                for e in entries):
        fail("files.grid_functions must be a list of objects")
    for i, e in enumerate(entries):
        for key, kind in _GF_FIELDS:
            if not isinstance(e.get(key), kind):
                fail(f"files.grid_functions[{i}].{key} is {e.get(key)!r}, "
                     f"not a {kind.__name__}")
        layers = e.get("layers")
        if not isinstance(layers, list) or not all(isinstance(n, int)
                                                   for n in layers):
            fail(f"files.grid_functions[{i}].layers is {layers!r}, not a list "
                 "of the layer indices the quantity is defined on")
        dims = e.get("physical_dimensions")
        if dims is not None and (not isinstance(dims, list) or len(dims) != 3
                                 or not all(isinstance(n, int) for n in dims)):
            fail(f"files.grid_functions[{i}].physical_dimensions is {dims!r}, "
                 "not three integer exponents (mass, length, time) or null")


def validate_against(manifest: MeshManifest, *, layer_count: int,
                     interface_count: int, groups: dict | None = None) -> None:
    """Check a manifest describes the mesh it was written beside.

    The two are written together and could still disagree -- a builder
    that renumbered after filling the manifest, say -- and the failure
    mode is a consumer selecting the wrong material, so it is worth one
    comparison at the end.
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
