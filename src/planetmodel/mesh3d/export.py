"""The MFEM delivery: the mesh, the displacement, and the manifest.

Everything is built on the reference mesh: the mesher writes concentric
spheres as MSH 2.2 in the geometry's own numbers, PyMFEM reads them,
and the geometry's mapping is evaluated at the nodal degrees of freedom
of that mesh, whose coordinates are reference coordinates and need no
inverse mapping.
`delivery="physical"` adds the displacement to the nodes before the
mesh is written; `delivery="referential"` leaves the mesh spherical and
writes `m(X) - X` beside it as a GridFunction in the mesh's own nodal
space, from which the consumer forms the physical mesh in one call.
A 2D mesh has two coordinates per node; they are lifted to the plane
z = 0 for the mapping and dropped again.

`export_mfem` adds the fields of a model, one GridFunction per name in
an L2 space: a material discontinuity is the point of a layered model,
and an L2 space of the mesh gives each element its own dofs, so the two
sides of an interface carry their own layer's value and nothing is
averaged.  Which layer a dof belongs to is the attribute of its
element, and each layer is evaluated in one vectorised call; every
value is referential, the model's own field at the reference point,
with Cartesian components in the model's units, whichever delivery the
mesh is written in.  A dof of a curved element can sit a chord's depth
outside its layer's sphere; the element's attribute is the truth, so
the point is pulled radially to the nearer interface before the layer
is asked.  A shell outside the model, and a layer of the model without
the field, are written as zero: a GridFunction has no room for "not
defined here", and the manifest's `model` block says where each field
means anything.

`<base>.mesh` is MFEM native and carries the curved nodes.  A `.gf`
file is indexed by the dof numbering of that mesh, and MFEM re-marks
tetrahedra for refinement on load unless told not to, which permutes
the numbering; so the mesh must be read as `Mesh(path, 1, 0, false)`
(generate edges, do not refine, do not fix orientation), and the
manifest's `files` block says so in the form a C++ reader passes on.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from collections.abc import Mapping as MappingOf
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..character import Character
from ..fields import stored_shape
from ..mapping import IdentityMapping, Mapping
from . import manifest
from .spec import DELIVERIES, MeshResult

if TYPE_CHECKING:
    from ..model import Model

__all__ = ["ExportResult", "export_mfem_mesh", "export_mfem",
           "MESH_READ_OPTIONS"]

#: The frame the field values are written in: dof coordinates are
#: Cartesian, and components follow the coordinates.
FRAME = "cartesian"

#: The options a consumer must construct `mfem::Mesh` with for the dof
#: numbering of the `.gf` files to be the one they were written in.
MESH_READ_OPTIONS = {"generate_edges": 1, "refine": 0,
                     "fix_orientation": False}


def _mfem() -> ModuleType:
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

    #: The MFEM mesh file.
    mesh_path: Path
    #: The manifest beside it, with its `files` block.
    manifest_path: Path
    #: The displacement GridFunction, in referential delivery; else None.
    displacement_path: Path | None
    #: "physical" or "referential".
    delivery: str
    #: The manifest's `files` block.
    files: dict[str, Any]
    #: Element, boundary element and nodal dof counts, and the order.
    counts: dict[str, int]
    _: KW_ONLY
    #: name -> the field's GridFunction, for the fields `export_mfem` wrote.
    field_paths: dict[str, Path] = field(default_factory=dict)

    def __repr__(self) -> str:
        n = len(self.field_paths)
        fields = f", {n} field{'' if n == 1 else 's'}" if n else ""
        return f"ExportResult({self.mesh_path.name}{fields}, {self.delivery} delivery)"


def _load_reference_mesh(msh_path: str | Path, card: manifest.MeshManifest) -> Any:
    """The MSH the mesher wrote, with MFEM's own verdict on its orientation.

    A mesh whose nodes already carry the mapping is refused: its
    coordinates are not reference coordinates.
    """
    mfem = _mfem()
    msh = Path(msh_path)
    if not msh.is_file():
        raise FileNotFoundError(f"no mesh at {msh}: the build wrote none")
    if card.mapping.get("applied_to_nodes"):
        raise ValueError(
            f"{msh.name} is a physical mesh: its nodes were displaced by the "
            "mesher, so its coordinates are not reference coordinates. "
            "Export from a delivery='referential' build; the exporter "
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
        # A straight-sided mesh has no nodal space of its own; the
        # identity one lets everything below speak of nodal dofs without
        # a special case for order 1.
        mesh.SetCurvature(1)
    return mesh


def _mesh_order(mesh: Any) -> int:
    """The polynomial order of the mesh's own nodal space."""
    return int(mesh.GetNodes().FESpace().GetOrder(0))


def _node_array(gf: Any) -> np.ndarray:
    """A vector GridFunction's values as (ndof, vdim), whatever the ordering."""
    mfem = _mfem()
    fes = gf.FESpace()
    data = gf.GetDataArray()
    vdim = fes.GetVDim()
    if fes.GetOrdering() == mfem.Ordering.byNODES:
        return data.reshape(vdim, -1).T
    return data.reshape(-1, vdim)


def _write_vector(gf: Any, values: ArrayLike) -> None:
    """The inverse of `_node_array`: (ndof, vdim) back into a GridFunction."""
    mfem = _mfem()
    fes = gf.FESpace()
    data = gf.GetDataArray()
    if fes.GetOrdering() == mfem.Ordering.byNODES:
        data[:] = np.asarray(values, dtype=float).T.ravel()
    else:
        data[:] = np.asarray(values, dtype=float).ravel()


def _lifted(X: ArrayLike) -> np.ndarray:
    """Points of shape (n, 2) or (n, 3) as (n, 3), a 2D mesh's in z = 0."""
    X = np.asarray(X, dtype=float)
    n, sdim = X.shape
    X3 = np.zeros((n, 3))
    X3[:, :sdim] = X
    return X3


def _displacement_at(mapping: Mapping, X: ArrayLike, *, scale: float) -> np.ndarray:
    """m(X) - X at nodal coordinates of shape (n, 2) or (n, 3).

    Two-dimensional points are lifted to z = 0 for the mapping, which
    must then keep them in the plane to `1e-12 * scale`, and the third
    component is dropped again.
    """
    X3 = _lifted(X)
    sdim = np.shape(X)[1]
    if hasattr(mapping, "displacement"):
        u = np.asarray(mapping.displacement(X3), dtype=float)
    else:
        u = np.asarray(mapping(X3), dtype=float) - X3
    if u.shape != X3.shape:
        raise ValueError(f"the mapping returned {u.shape} for {X3.shape} points")
    if not np.all(np.isfinite(u)):
        raise ValueError("the mapping is not finite at some nodal dof")
    if sdim == 2:
        out_of_plane = float(np.max(np.abs(u[:, 2])))
        if out_of_plane > 1e-12 * scale:
            raise ValueError(
                f"the mapping moves points of the disc out of its plane by up "
                f"to {out_of_plane:.3g}; a 2D export needs a mapping that "
                "keeps z = 0")
    return u[:, :sdim]


def _file_entry(name: str, path: str | Path, fes: Any, *,
                kind: str) -> dict[str, Any]:
    """One `files.grid_functions` record: where it is and which space it lives in."""
    mfem = _mfem()
    ordering = ("byNODES" if fes.GetOrdering() == mfem.Ordering.byNODES
                else "byVDIM")
    return {"kind": kind, "name": name, "file": Path(path).name,
            "fe_space": fes.FEColl().Name(), "vdim": int(fes.GetVDim()),
            "ordering": ordering}


def export_mfem_mesh(result: MeshResult, path_base: str | Path, *,
                     delivery: str | None = None) -> ExportResult:
    """Write an MFEM delivery of a built mesh: `.mesh`, the displacement
    in referential delivery, and the manifest with its `files` block.

    `result` is what a builder returned: the reference mesh on disk and
    the geometry's mapping (None for a mesh built without one, taken as
    the identity), applied to the node coordinates as they are.
    `path_base` is the basename the files are written beside:
    `<base>.mesh`, `<base>.displacement.gf` in referential delivery, and
    `<base>.json`.  Giving the mesher's own basename overwrites its
    manifest with this one; a separate basename keeps both.  `delivery`
    defaults to the build's.
    """
    mfem = _mfem()
    path_base = Path(path_base)
    card = manifest.read(result.manifest_path)
    delivery = card.delivery if delivery is None else delivery
    if delivery not in DELIVERIES:
        raise ValueError(
            f"delivery must be one of {DELIVERIES}, got {delivery!r}")
    mapping = IdentityMapping() if result.mapping is None else result.mapping
    identity = bool(getattr(mapping, "is_identity", False))

    mesh = _load_reference_mesh(result.msh_path, card)
    nodes = mesh.GetNodes()
    X = np.array(_node_array(nodes), dtype=float, copy=True)
    u = _displacement_at(mapping, X, scale=float(card.geometry["outer_radius"]))
    displacement = mfem.GridFunction(nodes.FESpace())
    _write_vector(displacement, u)

    if delivery == "physical":
        nodes.GetDataArray()[:] += displacement.GetDataArray()
    mesh_path = manifest.beside(path_base, ".mesh")
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.Print(str(mesh_path), 16)

    displacement_path = None
    entries = []
    if delivery == "referential":
        displacement_path = manifest.beside(path_base, ".displacement.gf")
        displacement.Save(str(displacement_path), 16)
        entries.append(_file_entry("displacement", displacement_path,
                                   nodes.FESpace(), kind="displacement"))

    files = {
        "mesh": mesh_path.name,
        "mesh_read_options": dict(MESH_READ_OPTIONS),
        "grid_functions": entries,
    }
    card.files = files
    card.delivery = delivery
    card.mapping["applied_to_nodes"] = delivery == "physical" and not identity
    manifest.validate_structure(card)
    manifest_path = manifest.write(path_base, card)

    counts = {"elements": mesh.GetNE(), "boundary_elements": mesh.GetNBE(),
              "nodes": nodes.FESpace().GetNDofs(), "order": _mesh_order(mesh)}
    return ExportResult(mesh_path=mesh_path, manifest_path=manifest_path,
                        displacement_path=displacement_path,
                        delivery=delivery, files=files, counts=counts)


def _dof_coordinates(mesh: Any, fes: Any) -> np.ndarray:
    """The coordinates of every dof of `fes`, shape (ndof, sdim).

    The mesh's own nodal GridFunction, read as a vector coefficient, is
    projected into a vector space over the same collection as `fes`;
    for a nodal basis that projection is interpolation at the dofs, so
    one call gives the point every dof stands for, curved elements
    included.
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


def _dofs_by_attribute(mesh: Any, fes: Any) -> dict[int, np.ndarray]:
    """attribute -> the dofs of `fes` on the elements carrying it."""
    n = mesh.GetNE()
    attributes = np.fromiter((mesh.GetAttribute(e) for e in range(n)),
                             dtype=int, count=n)
    dofs = [np.asarray(fes.GetElementDofs(e), dtype=int) for e in range(n)]
    return {int(a): np.unique(np.concatenate(
                [dofs[e] for e in np.flatnonzero(attributes == a)]))
            for a in np.unique(attributes)}


def _clipped_into(X: ArrayLike, interval: tuple[float, float]) -> np.ndarray:
    """`X` pulled radially into `[lo, hi]`, direction untouched."""
    lo, hi = (float(x) for x in interval)
    r = np.linalg.norm(X, axis=-1)
    safe = np.where(r > 0.0, r, 1.0)
    return X * (np.clip(r, lo, hi) / safe)[..., None]


def _check_model_sits_on(result: MeshResult, model: Model) -> None:
    """Refuse a mesh not built from a geometry, or a model on another skeleton."""
    if result.geometry is None:
        raise ValueError(
            f"{Path(result.msh_path).name} was not built from a geometry, so "
            "no model sits on it; fields are exported on a layered mesh")
    a = model.skeleton.boundaries
    b = result.geometry.skeleton.boundaries
    if a.size != b.size or not np.allclose(a, b, rtol=model.geometry.rtol,
                                           atol=0.0):
        raise ValueError(
            f"the model's skeleton {a.tolist()} is not the one the mesh was "
            f"built from, {b.tolist()}; export the fields of a model on the "
            "mesh's own geometry")


def _chosen_names(model: Model, fields: Iterable[str] | None) -> tuple[str, ...]:
    """The names to write: every name the model holds, or those given."""
    if fields is None:
        return tuple(model.field_names())
    names = tuple(str(n) for n in fields)
    for name in names:
        if not model.layers_with(name):
            raise KeyError(
                f"no layer of the model holds {name!r}; it holds "
                f"{list(model.field_names())}")
    return names


def _character_of(model: Model, name: str) -> Character:
    """The one character `name` has on every layer holding it."""
    layers = model.layers_with(name)
    characters = {model.layer(i)[name].character for i in layers}
    if len(characters) != 1:
        raise ValueError(
            f"{name!r} has characters {sorted(map(str, characters))} on layers "
            f"{list(layers)}; one GridFunction holds one character")
    return characters.pop()


def _field_values(model: Model, name: str, X: np.ndarray,
                  groups: MappingOf[int, np.ndarray], *, vdim: int) -> np.ndarray:
    """`name` at the dof coordinates `X`, (ndof, vdim), layer by layer.

    `groups` maps attribute -> dofs.  Attribute i + 1 is layer i of the
    model; everything the model does not hold the field on stays zero.
    """
    values = np.zeros((X.shape[0], vdim))
    shape = stored_shape(_character_of(model, name))
    for i in model.layers_with(name):
        dofs = groups.get(i + 1)
        if dofs is None:
            continue
        layer = model.layer(i)
        points = _clipped_into(_lifted(X[dofs]), layer.interval)
        got = np.asarray(layer[name].evaluate_at(points, frame=FRAME), dtype=float)
        want = (dofs.size,) + tuple(shape)
        if got.shape != want:
            raise ValueError(
                f"{name!r} on layer {i} answered with shape {got.shape}, "
                f"expected {want}")
        values[dofs] = got.reshape(dofs.size, vdim)
    return values


def export_mfem(result: MeshResult, path_base: str | Path, *, model: Model,
                fields: Iterable[str] | None = None, delivery: str | None = None,
                order: int | None = None) -> ExportResult:
    """Write an MFEM delivery of a built mesh with the fields of a model
    beside it: `<base>.mesh`, the displacement in referential delivery,
    one `<base>.<name>.gf` per field, and the manifest.

    The mesh and the displacement are `export_mfem_mesh`'s.  Each field
    is a GridFunction in an L2 space of `order` (the mesh's own by
    default) with `vdim` the number of stored components (Voigt for
    ranks 2 and 4), ordered byNODES, holding the referential value at
    every dof: the model's field at the dof's reference coordinates,
    with Cartesian components, in the model's units, whichever delivery
    the mesh is written in.  A dof is evaluated by the layer its
    element's attribute names, pulled radially to that layer's nearer
    interface when a curved element leaves it outside; a shell outside
    the model, and a layer of the model without the field, are written
    as zero.  `fields` is None for every name the model holds, or the
    names to write (KeyError for a name no layer holds).  The model must
    sit on the geometry the mesh was built from: the same skeleton to
    the model's geometry's `rtol`; a mesh not built from a geometry is
    refused.  The manifest's `files.grid_functions` gains a record of
    kind "field" per name and its `model` block says what the values
    mean.
    """
    mfem = _mfem()
    _check_model_sits_on(result, model)
    names = _chosen_names(model, fields)
    card = manifest.read(result.manifest_path)
    mesh = _load_reference_mesh(result.msh_path, card)
    order = _mesh_order(mesh) if order is None else int(order)
    if order < 0:
        raise ValueError(f"the order of an L2 space is not negative, got {order}")
    holders = {name: [i + 1 for i in model.layers_with(name)] for name in names}

    # Every value is computed before anything is written.
    collection = mfem.L2_FECollection(order, mesh.Dimension())
    scalar = mfem.FiniteElementSpace(mesh, collection, 1)
    X = _dof_coordinates(mesh, scalar)
    groups = _dofs_by_attribute(mesh, scalar)
    values = {}
    for name in names:
        vdim = int(np.prod(stored_shape(_character_of(model, name)), dtype=int))
        values[name] = (vdim, _field_values(model, name, X, groups, vdim=vdim))

    export = export_mfem_mesh(result, path_base, delivery=delivery)
    path_base = Path(path_base)
    written, entries = {}, []
    for name, (vdim, array) in values.items():
        fes = mfem.FiniteElementSpace(mesh, collection, vdim, mfem.Ordering.byNODES)
        gf = mfem.GridFunction(fes)
        _write_vector(gf, array)
        path = manifest.beside(path_base, f".{name}.gf")
        gf.Save(str(path), 16)
        written[name] = path
        entries.append(_file_entry(name, path, fes, kind="field"))

    card = manifest.read(export.manifest_path)
    card.files["grid_functions"].extend(entries)
    card.model = manifest.model_block(model, holders=holders)
    manifest.validate_structure(card)
    manifest_path = manifest.write(path_base, card)
    return dataclasses.replace(export, manifest_path=manifest_path,
                               files=card.files, field_paths=written)
