"""export.py -- the MFEM delivery: a mesh, GridFunctions and the manifest.

One code path, two deliveries, and everything built on the
**reference** mesh: the mesher writes concentric spheres as MSH 2.2,
PyMFEM reads them, and the model is evaluated at the degrees of freedom
of that mesh, whose coordinates are reference coordinates and therefore
need no inverse mapping anywhere.  `delivery="physical"` then adds the
displacement to the mesh nodes and writes the *pushed-forward* values at
the same dofs; `delivery="referential"` leaves the mesh spherical, writes
the referential values, and puts `m(X) - X` beside them as its own
GridFunction, from which the consumer forms the physical mesh in one
MFEM call.  Neither delivery ever inverts `m`, which is the whole reason
the export walks reference dofs rather than physical ones.

Why L2.  A material discontinuity is the *point* of a layered model, and
an interface of the mesh is where two layers meet with two different
values.  An H1 space has one dof there and has to choose; an L2 space of
the mesh order has per-element dofs -- with the Gauss-Legendre basis,
strictly interior ones -- so each side of an interface carries its own
layer's value and nothing is averaged into existence.  `continuous=` names
the quantities the caller knows to be continuous and wants in H1 instead;
there an interface dof is shared, and the layer that writes last wins,
which is exactly why the flag is the caller's to give and not the
writer's to guess.

Which layer a dof belongs to is read off the mesh: gmsh's physical groups
become MFEM element attributes, attribute `i + 1` is layer `i` centre
outward (the manifest is the map, `_tagging.py` makes it), so the elements
are grouped by attribute and each group is evaluated in **one** vectorised
call with `layer=` fixed.  No per-point Python callback ever crosses into
MFEM.

Domains.  A field belongs to the layers that hold it.  A field written
on a body where some layer has none -- a crust left to
the consumer, a vacuum buffer -- is written as zero there and the
manifest's `files` entry says on which layers it means anything.  A
GridFunction cannot carry NaN usefully (every integrator would spread it
over the whole mesh), so zero plus the truth in the manifest is the honest
combination; a consumer that restricts to the listed attributes, as
mfemElasticity's SubMesh machinery does, never sees the zeros at all.

Units.  The mesher divides *geometry* by `rref` and nothing else
(`_units.py`), so the mesh is non-dimensional in length and the fields
are in the body's own scale system -- SI for an SI body.  This writer
keeps that rule rather than inventing a mass scale the mesh does not fix:
values are the resolved body's own, and every `files` entry carries the
field's dimension exponents and unit string beside the units block's
`rref_m` and scale triple, which is what a consumer needs to reconcile
the two.  For a body that was non-dimensionalised at the model layer the
question does not arise: the divisor is one and everything is already in
one system.

Frames, and the displacement's units.  A dof carries Cartesian
coordinates, and components follow the coordinates, so the values
written are Cartesian components and every `files` entry says
`frame: "cartesian"`.  This is where the MFEM delivery and the netCDF one
part company on purpose: netCDF samples on `(r, theta, phi)` and writes
spherical components, and both are the same rule applied to different
coordinates.  The displacement is the one quantity written in *mesh*
length units rather than the body's, since its whole use is `nodes +=
displacement` in one MFEM call; its entry therefore carries the
dimensions of a length with the non-dimensional unit string, and
`rref_m` in the units block is what turns it into metres.

Reading the result.  `<base>.mesh` is MFEM native and carries the curved
nodes; the `.gf` files are indexed by the dof numbering of *that* mesh,
and MFEM re-marks tetrahedra for refinement on load unless told not to,
which permutes element vertices and with them the nodal and L2 dof
numbering.  So the mesh must be read the way MFEM's own DataCollection
reads a saved mesh, `Mesh(path, 1, 0, false)` -- generate edges, do not
refine, do not fix orientation -- and the manifest's `files` block says so
in the form a C++ reader passes on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import manifest
from ..model.units import unit_string
from ..model.character import VECTOR
from ..model.pushforward import push_forward_field
from ..model.units import Dimensions
from ._units import MeshUnits

__all__ = ["ExportResult", "export_mfem", "MESH_READ_OPTIONS"]

#: The options a consumer must construct `mfem::Mesh` with for the dof
#: numbering of the `.gf` files to be the one they were written in.  The
#: same triple MFEM's own `DataCollection` uses for a saved mesh.
MESH_READ_OPTIONS = {"generate_edges": 1, "refine": 0,
                     "fix_orientation": False}

#: What `evaluate_at` answers in: dof coordinates are Cartesian, so the
#: components are too -- components follow the coordinates.
FRAME = "cartesian"


def _mfem():
    """PyMFEM, imported here and nowhere else in planetmodel."""
    try:
        import mfem.ser as mfem
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "the MFEM exporter needs PyMFEM.  Install it with:\n"
            "    pip install 'planetmodel[mfem]'      "
            "(or: poetry install --extras mfem)"
        ) from exc
    return mfem


@dataclass(frozen=True)
class ExportResult:
    """What an export wrote, and what a consumer reads it back with."""

    mesh_path: Path
    manifest_path: Path
    field_paths: dict
    displacement_path: Path | None
    delivery: str
    files: dict
    counts: dict

    def __repr__(self) -> str:
        n = len(self.field_paths)
        return (f"ExportResult({self.mesh_path.name}, {n} field"
                f"{'' if n == 1 else 's'}, {self.delivery} delivery)")


# ------------------------------------------------------------------ loading

def _load_reference_mesh(result, card):
    """The MSH the mesher wrote, with MFEM's own verdict on it.

    Both counts are read rather than trusted: MFEM's "wrong orientation"
    messages are compiled out of a release build, so the return values
    are the only honest measure, and the boundary check
    judges only the outermost surface because interior faces are skipped.
    """
    mfem = _mfem()
    msh = Path(result.msh_path)
    if not msh.is_file():
        raise FileNotFoundError(f"no mesh at {msh}: the build wrote none")
    if card.mapping is not None and card.mapping.get("applied_to_nodes"):
        raise ValueError(
            f"{msh.name} is a physical mesh: its nodes were displaced by the "
            "mesher, so its coordinates are not reference coordinates and "
            "the fields could only be evaluated by inverting the mapping. "
            "Export from a delivery='referential' build -- the exporter "
            "produces either delivery from the reference mesh.")
    mesh = mfem.Mesh(str(msh))
    wrong_cells = mesh.CheckElementOrientation(True)
    wrong_faces = mesh.CheckBdrElementOrientation(True)
    if wrong_cells or wrong_faces:
        raise ValueError(
            f"MFEM finds {wrong_cells} wrongly oriented elements and "
            f"{wrong_faces} wrongly oriented boundary elements in "
            f"{msh.name}; the mesher's orientation repair did not take")
    if mesh.GetNodes() is None:
        # A straight-sided mesh has no nodal space of its own; giving it
        # the identity one costs nothing and lets everything below speak
        # of dofs and displacements without a special case for order 1.
        mesh.SetCurvature(1)
    return mesh


def _mesh_order(mesh) -> int:
    """The polynomial order of the mesh's own nodal space."""
    return int(mesh.GetNodes().FESpace().GetOrder(0))


def _node_array(gf) -> np.ndarray:
    """A vector GridFunction's values as (ndof, vdim), whatever the ordering."""
    mfem = _mfem()
    fes = gf.FESpace()
    data = gf.GetDataArray()
    vdim = fes.GetVDim()
    if fes.GetOrdering() == mfem.Ordering.byNODES:
        return data.reshape(vdim, -1).T
    return data.reshape(-1, vdim)


def _write_vector(gf, values) -> None:
    """The inverse of `_node_array`: (ndof, vdim) back into a GridFunction."""
    mfem = _mfem()
    fes = gf.FESpace()
    data = gf.GetDataArray()
    if fes.GetOrdering() == mfem.Ordering.byNODES:
        data[:] = np.asarray(values, dtype=float).T.ravel()
    else:
        data[:] = np.asarray(values, dtype=float).ravel()


def _dof_coordinates(mesh, fes):
    """Reference coordinates of every dof of `fes`, in mesh units.

    The plan's route: project the coordinate function -- the mesh's own
    nodal GridFunction, read as a vector coefficient -- into a vector
    space with the same collection as `fes`.  For the nodal bases used
    here that projection *is* interpolation at the dofs, so one C++ call
    returns the point every dof stands for, curved elements included.
    """
    mfem = _mfem()
    vfes = mfem.FiniteElementSpace(mesh, fes.FEColl(), mesh.SpaceDimension(),
                                   mfem.Ordering.byNODES)
    gf = mfem.GridFunction(vfes)
    gf.ProjectCoefficient(mfem.VectorGridFunctionCoefficient(mesh.GetNodes()))
    X = np.array(_node_array(gf), dtype=float, copy=True)
    if X.shape[0] != fes.GetNDofs():
        raise RuntimeError(
            f"the coordinate projection gave {X.shape[0]} points for "
            f"{fes.GetNDofs()} dofs")
    return X


def _elements_by_layer(mesh, fes):
    """`layer index -> the dofs of the elements carrying that attribute`.

    Attribute `i + 1` is layer `i`, centre outward: the numbering the
    manifest records and `_tagging.py` writes into the physical groups.
    """
    attributes = np.fromiter((mesh.GetAttribute(e) for e in range(mesh.GetNE())),
                             dtype=int, count=mesh.GetNE())
    dofs = [np.asarray(fes.GetElementDofs(e), dtype=int)
            for e in range(mesh.GetNE())]
    out = {}
    for attribute in np.unique(attributes):
        chosen = np.flatnonzero(attributes == attribute)
        out[int(attribute) - 1] = np.unique(
            np.concatenate([dofs[e] for e in chosen]))
    return out


# ------------------------------------------------------------ the evaluation

def _clipped_into(X, interval):
    """`X` pulled radially into `[lo, hi]`, direction untouched.

    A curved element approximates its spherical face by a polynomial, so
    a dof of an element in layer `i` can sit a chord's depth outside the
    layer it belongs to.  The element's attribute is the truth -- it is
    what a solver will select on -- so the radius is brought back to the
    interface rather than the value being asked of the wrong layer, and
    the direction, which is what fixes the frame, is left alone.
    """
    lo, hi = float(interval[0]), float(interval[1])
    r = np.linalg.norm(X, axis=-1)
    safe = np.where(r > 0.0, r, 1.0)
    return X * (np.clip(r, lo, hi) / safe)[..., None]


def _component_shape(character):
    """The trailing shape of a value: Voigt where the character reduces."""
    return character.voigt_shape or character.component_shape


def _evaluate_on_layers(field, X, *, skeleton, groups, units, vdim,
                        character, name):
    """One vectorised call per layer, into a flat (ndof, vdim) array.

    `groups` is the field's domain intersected with the layers the mesh
    actually has; everything outside it stays zero, and the manifest's
    `layers` is what says so.
    """
    values = np.zeros((X.shape[0], vdim), dtype=float)
    shape = _component_shape(character)
    for layer, dofs in groups.items():
        points = _clipped_into(units.to_body(X[dofs]), skeleton.interval(layer))
        got = np.asarray(field.evaluate_at(points, layer=layer, frame=FRAME),
                         dtype=float)
        want = (dofs.size,) + tuple(shape)
        if got.shape != want:
            raise ValueError(
                f"field {name!r} answered with shape {got.shape} on layer "
                f"{layer}, expected {want} for a field of {character}")
        values[dofs] = got.reshape(dofs.size, vdim)
    return values


# --------------------------------------------------------------- the manifest

def _file_entry(name, path, fes, *, character, dimensions, layers, si,
                continuous, representation, kind="field"):
    """One `files.grid_functions` record: where it is and what it means."""
    mfem = _mfem()
    ordering = ("byNODES" if fes.GetOrdering() == mfem.Ordering.byNODES
                else "byVDIM")
    return {
        "kind": kind,
        "name": name,
        "file": Path(path).name,
        "fe_space": fes.FEColl().Name(),
        "vdim": int(fes.GetVDim()),
        "ordering": ordering,
        "continuous": bool(continuous),
        "character_rank": int(character.rank),
        "character_weight": int(character.weight),
        "voigt": int(character.voigt_shape is not None),
        "components": [int(n) for n in _component_shape(character)],
        "physical_dimensions": (None if dimensions is None else
                                [int(dimensions.mass), int(dimensions.length),
                                 int(dimensions.time)]),
        "units": unit_string(dimensions, si=si),
        "frame": FRAME,
        "layers": [int(i) for i in layers],
        "attributes": [int(i) + 1 for i in layers],
        "fill_value": 0.0,
        "representation": representation,
    }


# ------------------------------------------------------------------- the API

def export_mfem(result, path_base, *, fields=None, delivery=None,
                order=None, continuous=()) -> ExportResult:
    """Write an MFEM delivery of a built mesh: `.mesh`, `.gf`s, manifest.

    `result` is what `build_layered_mesh` returned -- the reference mesh
    on disk, the resolved body, the units it was meshed in and the
    mapping it was built with.  `path_base` is the basename the files are
    written beside: `<base>.mesh`, one `<base>.<field>.gf` per field,
    `<base>.displacement.gf` in referential mode, and `<base>.json`.
    Giving the mesher's own basename overwrites its manifest with this
    one, which then describes the `.mesh` rather than the `.msh` it was
    built from; a separate basename keeps both.

    `fields` is a sequence of names (default: every field the body has)
    or a mapping `name -> Field` for quantities the caller has built
    itself; `delivery` is `"physical"` or `"referential"` and defaults
    to the one the spec asked for; `order` is the order of the field spaces and
    defaults to the mesh's own; `continuous` names the fields to put in
    H1 rather than L2, which is a promise by the caller that they are
    continuous across every interface.

    A field is written on the elements of the layers that hold it and
    zero elsewhere -- a GridFunction has no room for "not defined here",
    and a NaN would spread through the first integrator that touched it
    -- with the truth in the manifest's `layers`.
    """
    mfem = _mfem()
    path_base = Path(path_base)
    card = manifest.read(result.manifest_path)
    body = result.body
    units = MeshUnits.identity() if result.units is None else result.units
    mapping = result.mapping
    spec_delivery = getattr(getattr(result, "spec", None), "delivery",
                            "physical")
    delivery = spec_delivery if delivery is None else delivery
    if delivery not in ("physical", "referential"):
        raise ValueError(
            "delivery must be 'physical' or 'referential', got "
            f"{delivery!r}")
    if delivery == "referential" and mapping is None:
        raise ValueError(
            "a referential delivery hands the consumer the mapping to apply "
            "and this mesh was built without one; a spherical body wants "
            "delivery='physical'")

    mesh = _load_reference_mesh(result, card)
    order = _mesh_order(mesh) if order is None else int(order)
    si = body.scales.is_si
    continuous = set(continuous)

    chosen = _chosen_fields(body, fields)
    unknown = continuous - set(chosen)
    if unknown:
        raise KeyError(
            f"continuous names {sorted(unknown)}, which are not among the "
            f"fields being written ({sorted(chosen)})")

    # -- the spaces, and the reference coordinates of their dofs ----------
    dim = mesh.Dimension()
    collections = {False: mfem.L2_FECollection(order, dim),
                   True: mfem.H1_FECollection(order, dim)}
    coordinates, groups = {}, {}
    for kind in {name in continuous for name in chosen}:
        fes = mfem.FiniteElementSpace(mesh, collections[kind], 1)
        coordinates[kind] = _dof_coordinates(mesh, fes)
        groups[kind] = _elements_by_layer(mesh, fes)

    # -- the displacement, in the mesh's own nodal space ------------------
    nodes = mesh.GetNodes()
    displacement = mfem.GridFunction(nodes.FESpace())
    X_nodes = _node_array(nodes)
    if mapping is None:
        _write_vector(displacement, np.zeros_like(X_nodes))
    else:
        _write_vector(displacement, units.to_mesh(
            mapping.displacement(units.to_body(X_nodes))))

    # -- the fields, evaluated one vectorised call per layer --------------
    written, entries = {}, []
    for name, field in chosen.items():
        kind = name in continuous
        character = field.character
        vdim = int(np.prod(_component_shape(character), dtype=int))
        source = (field if delivery == "referential"
                  else push_forward_field(field, mapping)
                  if mapping is not None else field)
        # A field per the protocol need not say where it is defined;
        # one that does not is taken to be defined everywhere.
        domain = getattr(field, "domain", range(len(body.layers)))
        layers = tuple(i for i in domain if i in groups[kind])
        fes = mfem.FiniteElementSpace(mesh, collections[kind], vdim,
                                      mfem.Ordering.byNODES)
        gf = mfem.GridFunction(fes)
        values = _evaluate_on_layers(
            source, coordinates[kind], skeleton=body.skeleton,
            groups={i: groups[kind][i] for i in layers}, units=units,
            vdim=vdim, character=character, name=name)
        _write_vector(gf, values)
        path = _beside(path_base, f".{name}.gf")
        gf.Save(str(path), 16)
        written[name] = path
        entries.append(_file_entry(
            name, path, fes, character=character,
            dimensions=getattr(field, "dimensions", None), layers=layers,
            si=si, continuous=kind,
            representation=("referential" if delivery == "referential"
                            else "physical")))

    # -- the mesh: displaced now, after every dof was read ----------------
    if delivery == "physical" and mapping is not None:
        nodes.GetDataArray()[:] += displacement.GetDataArray()
    mesh_path = _beside(path_base, ".mesh")
    mesh.Print(str(mesh_path), 16)

    displacement_path = None
    if delivery == "referential":
        displacement_path = _beside(path_base, ".displacement.gf")
        displacement.Save(str(displacement_path), 16)
        entries.append(_file_entry(
            "displacement", displacement_path, nodes.FESpace(),
            character=VECTOR, dimensions=Dimensions.LENGTH,
            layers=range(len(body.layers)), si=False, continuous=True,
            representation="referential", kind="displacement"))

    files = {
        "mesh": mesh_path.name,
        "mesh_read_options": dict(MESH_READ_OPTIONS),
        "grid_functions": entries,
    }
    card.files = files
    card.delivery = delivery
    if card.mapping is not None:
        card.mapping["applied_to_nodes"] = delivery == "physical"
    manifest.validate_structure(card)
    manifest_path = manifest.write(path_base, card)

    counts = {"elements": mesh.GetNE(), "boundary_elements": mesh.GetNBE(),
              "nodes": nodes.FESpace().GetNDofs(), "order": order}
    return ExportResult(mesh_path=mesh_path, manifest_path=manifest_path,
                        field_paths=written,
                        displacement_path=displacement_path,
                        delivery=delivery, files=files, counts=counts)


def _beside(path_base: Path, suffix: str) -> Path:
    """`<base><suffix>`, with the mesher's own basename convention."""
    return manifest.beside(path_base, suffix)


def _chosen_fields(body, fields) -> dict:
    """The fields to write: named, given outright, or all the body has."""
    if fields is None:
        # Every static field: a frequency- or time-dependent one has no
        # values until frozen (at_frequency, at_time); pass it by name.
        return {name: body[name] for name in body.field_names
                if getattr(body[name], "kind", "static") == "static"}
    if hasattr(fields, "items"):
        return dict(fields)
    return {str(name): body[str(name)] for name in fields}
