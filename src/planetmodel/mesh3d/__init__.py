"""mesh3d -- 2D and 3D meshes of layered bodies, via gmsh.

The public surface is deliberately small: a MeshSpec describes what is
wanted, build_layered_mesh produces it, and a MeshResult says what was
written.  Geometry construction, tagging, sizing fields, orientation
repair and gmsh session management are private -- consumers depend on
the mesh and its manifest, not on how either was built.

gmsh is an optional dependency, imported only here.  Nothing else in
planetmodel touches it, so `import planetmodel` never pulls the wheel; CI asserts
that.  PyMFEM is the same one level down: `export_mfem` is named here,
but `export.py` imports `mfem.ser` inside its functions, so importing
this package needs only gmsh.
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

from .offset import build_offset_mesh
from .export import ExportResult, export_mfem
from .layered import build_layered_mesh
from .spec import (AngularResolution, BufferSpec, InterfaceSizing, MeshResult,
                   MeshSpec, PerInterface, UniformInterfaces)

__all__ = [
    "build_layered_mesh", "build_offset_mesh", "export_mfem",
    "MeshSpec", "MeshResult", "ExportResult", "BufferSpec", "InterfaceSizing",
    "AngularResolution", "UniformInterfaces", "PerInterface",
]
