"""The MFEM delivery: the mesh, the displacement, and the manifest.

Everything is built on the reference mesh: the mesher writes concentric
spheres as MSH 2.2 in the geometry's own numbers, PyMFEM reads them,
and the geometry's mapping is evaluated at the nodal degrees of freedom
of that mesh, whose coordinates are reference coordinates and need no
inverse mapping.  This is the one place the mapping meets a mesh: a
displacement is a vector in the mesh's own nodal space, whatever the
mapping is made of.  `delivery="physical"` adds the displacement to the
nodes before the mesh is written, and the moved mesh is checked for
folding through the Jacobian of every element at its quadrature and
nodal points, refused if any is not positive; `delivery="referential"`
leaves the mesh spherical and writes `m(X) - X` beside it as a
GridFunction in the mesh's own nodal space, from which the consumer
forms the physical mesh in one call.  A 2D mesh has two coordinates per
node; they are lifted to the plane z = 0 for the mapping and dropped
again.

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
defined here", and each field's manifest record says on which layers
it means anything.

`<base>.mesh` is MFEM native and carries the curved nodes.  A `.gf`
file is indexed by the dof numbering of that mesh, and MFEM re-marks
tetrahedra for refinement on load unless told not to, which permutes
the numbering; so the mesh must be read as `Mesh(path, 1, 0, false)`
(generate edges, do not refine, do not fix orientation), and the
manifest's `mesh.read_options` says so in the form a C++ reader passes on.
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

from ..character import VECTOR, Character
from ..fields import stored_shape
from ..mapping import IdentityMapping, Mapping
from ..units import LENGTH
from . import manifest
from .spec import DELIVERIES, MeshResult

if TYPE_CHECKING:
    from ..model import Model

__all__ = ["ExportResult", "export_mfem_mesh", "export_mfem",
           "jacobian_report", "MESH_READ_OPTIONS"]

#: The frame the field values are written in: dof coordinates are
#: Cartesian, and components follow the coordinates.
FRAME = "cartesian"

#: The options a consumer must construct `mfem::Mesh` with for the dof
#: numbering of the `.gf` files to be the one they were written in.
MESH_READ_OPTIONS = {"generate_edges": 1, "refine": 0,
                     "fix_orientation": False}

#: The name of the displacement field, m(X) - X, in the manifest.
DISPLACEMENT = "displacement"


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
    #: The manifest beside it.
    manifest_path: Path
    #: The displacement GridFunction, in referential delivery; else None.
    displacement_path: Path | None
    #: "physical" or "referential".
    delivery: str
    #: The `mfem::Mesh` constructor arguments the files were written under.
    read_options: dict[str, Any]
    #: Element, boundary element and nodal dof counts, and the order.
    counts: dict[str, int]
    #: The Jacobian report of the written mesh; see `jacobian_report`.
    quality: dict[str, float]
    _: KW_ONLY
    #: name -> the field's GridFunction, for the fields `export_mfem` wrote.
    field_paths: dict[str, Path] = field(default_factory=dict)

    def __repr__(self) -> str:
        n = len(self.field_paths)
        fields = f", {n} field{'' if n == 1 else 's'}" if n else ""
        return f"ExportResult({self.mesh_path.name}{fields}, {self.delivery} delivery)"


def _load_mesh(msh_path: str | Path) -> Any:
    """The MSH the mesher wrote, with MFEM's own verdict on its orientation."""
    mfem = _mfem()
    msh = Path(msh_path)
    if not msh.is_file():
        raise FileNotFoundError(f"no mesh at {msh}: the build wrote none")
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


def jacobian_report(mesh: Any) -> dict[str, float]:
    """The Jacobian of every element at its quadrature and nodal points.

    `min_ratio` is the smallest, over the elements, of the least
    Jacobian determinant divided by the greatest within the element: one
    for an affine element, small for a strongly curved one, not positive
    for a folded one.  `negative_jacobians` counts the elements with a
    non-positive determinant somewhere.  A curved element can fold
    between corners that are individually fine, which is why the nodal
    points are included.
    """
    mfem = _mfem()
    fes = mesh.GetNodes().FESpace()
    order = _mesh_order(mesh)
    worst = np.inf
    negative = 0
    for e in range(mesh.GetNE()):
        T = mesh.GetElementTransformation(e)
        rules = (mfem.IntRules.Get(mesh.GetElementBaseGeometry(e), 2 * order),
                 fes.GetFE(e).GetNodes())
        dets = []
        for rule in rules:
            for i in range(rule.GetNPoints()):
                T.SetIntPoint(rule.IntPoint(i))
                dets.append(T.Weight())
        low, high = min(dets), max(dets)
        if low <= 0.0:
            negative += 1
        worst = min(worst, low / high if high > 0.0 else -np.inf)
    return {"min_ratio": float(worst), "negative_jacobians": int(negative)}


def _require_unfolded(mesh: Any, what: str) -> dict[str, float]:
    """The Jacobian report of `mesh`, or a refusal naming the folded elements."""
    report = jacobian_report(mesh)
    if report["negative_jacobians"]:
        raise ValueError(
            f"{what} folds {report['negative_jacobians']} element(s): the "
            f"Jacobian is not positive throughout them (worst ratio "
            f"{report['min_ratio']:.3g}).  Nothing was written.")
    return report


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


def _field_entry(name: str, path: str | Path, fes: Any, *, character: Character,
                 dimensions: Any, si: bool, layers: Iterable[int]
                 ) -> manifest.FieldEntry:
    """One `fields[]` record: where the GridFunction is and which space it
    lives in, and what its values are."""
    mfem = _mfem()
    ordering = ("byNODES" if fes.GetOrdering() == mfem.Ordering.byNODES
                else "byVDIM")
    return manifest.FieldEntry.from_field(
        name, path, fe_space=fes.FEColl().Name(), vdim=int(fes.GetVDim()),
        ordering=ordering, character=character, dimensions=dimensions, si=si,
        layers=layers)


def export_mfem_mesh(result: MeshResult, path_base: str | Path, *,
                     delivery: str = "physical") -> ExportResult:
    """Write an MFEM delivery of a built mesh: `.mesh`, the displacement
    in referential delivery, and the manifest.

    `result` is what a builder returned: the reference mesh on disk and
    the geometry's mapping (None for a mesh built without one, taken as
    the identity), applied to the node coordinates as they are.
    Physical delivery moves the nodes and refuses a mesh the mapping
    folds; referential delivery writes the displacement beside them.
    `path_base` is the basename the files are written beside:
    `<base>.mesh`, `<base>.displacement.gf` in referential delivery, and
    `<base>.json`.  Giving the mesher's own basename overwrites its
    manifest with this one; a separate basename keeps both.  The
    manifest's `mesh` block names the MFEM file, says whether its nodes
    are reference or physical coordinates, and names the displacement
    among the `fields` where one was written; the displacement's unit
    is "1" until a model's scales are known.
    """
    mfem = _mfem()
    path_base = Path(path_base)
    card = manifest.read(result.manifest_path)
    if delivery not in DELIVERIES:
        raise ValueError(
            f"delivery must be one of {DELIVERIES}, got {delivery!r}")
    mapping = IdentityMapping() if result.mapping is None else result.mapping
    identity = bool(getattr(mapping, "is_identity", False))

    mesh = _load_mesh(result.msh_path)
    nodes = mesh.GetNodes()
    X = np.array(_node_array(nodes), dtype=float, copy=True)
    u = _displacement_at(mapping, X, scale=card.outer_radius)
    displacement = mfem.GridFunction(nodes.FESpace())
    _write_vector(displacement, u)

    moved = delivery == "physical" and not identity
    if moved:
        nodes.GetDataArray()[:] += displacement.GetDataArray()
        quality = _require_unfolded(mesh, f"the mapping {mapping!r}")
    else:
        quality = jacobian_report(mesh)
    mesh_path = manifest.beside(path_base, ".mesh")
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.Print(str(mesh_path), 16)

    displacement_path = None
    entries = []
    if delivery == "referential":
        displacement_path = manifest.beside(path_base, ".displacement.gf")
        displacement.Save(str(displacement_path), 16)
        entries.append(_field_entry(
            DISPLACEMENT, displacement_path, nodes.FESpace(), character=VECTOR,
            dimensions=LENGTH, si=False,
            layers=[e["attribute"] for e in card.layers]))

    card.mesh = manifest.mesh_block(
        mesh_path, format="mfem", nodes="physical" if moved else "reference",
        read_options=MESH_READ_OPTIONS,
        displacement=DISPLACEMENT if entries else None)
    card.fields = [dataclasses.asdict(e) for e in entries]
    card.scales, card.constants = None, {}
    manifest.validate_structure(card)
    manifest_path = manifest.write(path_base, card)

    counts = {"elements": mesh.GetNE(), "boundary_elements": mesh.GetNBE(),
              "nodes": nodes.FESpace().GetNDofs(), "order": _mesh_order(mesh)}
    return ExportResult(mesh_path=mesh_path, manifest_path=manifest_path,
                        displacement_path=displacement_path,
                        delivery=delivery, read_options=dict(MESH_READ_OPTIONS),
                        counts=counts, quality=quality)


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
                fields: Iterable[str] | None = None, delivery: str = "physical",
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
    mesh = _load_mesh(result.msh_path)
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
    si = model.scales.is_si
    written, entries = {}, []
    for name, (vdim, array) in values.items():
        fes = mfem.FiniteElementSpace(mesh, collection, vdim, mfem.Ordering.byNODES)
        gf = mfem.GridFunction(fes)
        _write_vector(gf, array)
        path = manifest.beside(path_base, f".{name}.gf")
        gf.Save(str(path), 16)
        written[name] = path
        spec = model.spec(name)
        entries.append(_field_entry(
            name, path, fes, character=_character_of(model, name),
            dimensions=None if spec is None else spec.dimensions, si=si,
            layers=holders[name]))

    card = manifest.read(export.manifest_path)
    for entry in card.fields:
        if entry["name"] == DISPLACEMENT:
            entry["unit"] = LENGTH.unit_string(si=si)
    card.fields.extend(dataclasses.asdict(e) for e in entries)
    card.scales = manifest.scales_block(model.scales)
    card.constants = manifest.constants_block(model)
    manifest.validate_structure(card)
    manifest_path = manifest.write(path_base, card)
    return dataclasses.replace(export, manifest_path=manifest_path,
                               field_paths=written)
