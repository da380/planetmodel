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

`<base>.mesh` is MFEM native and carries the curved nodes.  A `.gf`
file is indexed by the dof numbering of that mesh, and MFEM re-marks
tetrahedra for refinement on load unless told not to, which permutes
the numbering; so the mesh must be read as `Mesh(path, 1, 0, false)`
(generate edges, do not refine, do not fix orientation), and the
manifest's `files` block says so in the form a C++ reader passes on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..mapping import IdentityMapping
from . import manifest
from .spec import DELIVERIES

__all__ = ["ExportResult", "export_mfem_mesh", "MESH_READ_OPTIONS"]

#: The options a consumer must construct `mfem::Mesh` with for the dof
#: numbering of the `.gf` files to be the one they were written in.
MESH_READ_OPTIONS = {"generate_edges": 1, "refine": 0,
                     "fix_orientation": False}


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

    #: The MFEM mesh file.
    mesh_path: Path
    #: The manifest beside it, with its `files` block.
    manifest_path: Path
    #: The displacement GridFunction, in referential delivery; else None.
    displacement_path: Path | None
    #: "physical" or "referential".
    delivery: str
    #: The manifest's `files` block.
    files: dict
    #: Element, boundary element and nodal dof counts, and the order.
    counts: dict

    def __repr__(self) -> str:
        return f"ExportResult({self.mesh_path.name}, {self.delivery} delivery)"


def _load_reference_mesh(msh_path, card):
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


def _displacement_at(mapping, X, *, scale: float) -> np.ndarray:
    """m(X) - X at nodal coordinates of shape (n, 2) or (n, 3).

    Two-dimensional points are lifted to z = 0 for the mapping, which
    must then keep them in the plane to `1e-12 * scale`, and the third
    component is dropped again.
    """
    X = np.asarray(X, dtype=float)
    n, sdim = X.shape
    X3 = np.zeros((n, 3))
    X3[:, :sdim] = X
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


def _file_entry(name, path, fes, *, kind: str) -> dict:
    """One `files.grid_functions` record: where it is and which space it lives in."""
    mfem = _mfem()
    ordering = ("byNODES" if fes.GetOrdering() == mfem.Ordering.byNODES
                else "byVDIM")
    return {"kind": kind, "name": name, "file": Path(path).name,
            "fe_space": fes.FEColl().Name(), "vdim": int(fes.GetVDim()),
            "ordering": ordering}


def export_mfem_mesh(result, path_base, *, delivery=None) -> ExportResult:
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
