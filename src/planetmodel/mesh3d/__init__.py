"""mesh3d: 2D and 3D meshes of layered geometries, via gmsh.

A MeshSpec describes what is wanted, `build_layered_mesh` produces it
and a MeshResult says what was written; `build_offset_mesh` makes the
two-body benchmark geometries, `export_mfem_mesh` turns either into an
MFEM delivery and `export_mfem` adds the fields of a model to a layered
one.  The manifest that travels with every mesh is `manifest`.
Geometry construction, tagging, sizing fields, orientation repair and
gmsh session management are private: consumers depend on the mesh and
its manifest, not on how either was built.

gmsh is imported here and nowhere else in planetmodel; PyMFEM is
imported inside the export functions.
"""
from __future__ import annotations

try:
    import gmsh as _gmsh  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised by the extra
    raise ImportError(
        "planetmodel.mesh3d needs gmsh.  Install it with:\n"
        "    pip install 'planetmodel[meshing]'      "
        "(or: poetry install --extras meshing)"
    ) from exc

from . import manifest
from .export import ExportResult, export_mfem, export_mfem_mesh
from .layered import build_layered_mesh
from .offset import build_offset_mesh
from .spec import (AngularResolution, InterfaceSizing, MeshResult, MeshSpec,
                   PerInterface, Shell, SizingRule, UniformInterfaces,
                   ValidationReport)

__all__ = [
    "MeshSpec", "MeshResult", "Shell", "InterfaceSizing", "SizingRule",
    "AngularResolution", "UniformInterfaces", "PerInterface",
    "ValidationReport", "build_layered_mesh", "build_offset_mesh",
    "export_mfem_mesh", "export_mfem", "ExportResult", "manifest",
]
