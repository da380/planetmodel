"""mesh3d: 2D and 3D meshes of layered geometries, via gmsh.

A MeshSpec describes what is wanted, `build_layered_mesh` produces the
reference mesh of it through gmsh and a MeshResult says what was
written; `build_offset_mesh` makes the two-body benchmark geometries.
Faults given to a MeshSpec as `EmbeddedFault`s are built into the mesh
from the start, pulled back through the mapping and fragmented in, each
an oriented interior surface with its curves labelled.
`export_mfem_mesh` turns either into an MFEM delivery, where the
geometry's mapping is applied, moving the nodes or written beside them
as a field, and `export_mfem` adds the fields of a model to a layered
one.  The manifest that travels with every mesh is `manifest`, and
`read_mesh` reads a written mesh as plain arrays (`MeshArrays`).
Geometry construction, tagging, sizing fields, orientation repair and
gmsh session management are private: consumers depend on the mesh and
its manifest, not on how either was built.

gmsh is imported by what builds or reads a mesh, where it does:
`build_layered_mesh` and `build_offset_mesh` are loaded from their
modules on first use, and `read_mesh` imports it as it reads.  So the
manifest and its rules, the spec, the arrays and the MFEM export import
without gmsh, and a code holding an MFEM delivery needs no gmsh to read
its manifest; reaching a builder without gmsh raises an ImportError
naming the `meshing` extra.  PyMFEM is imported inside the export
functions.
"""
from __future__ import annotations

import importlib

from . import manifest
from ._extra import GMSH_MISSING, missing_gmsh
from .arrays import MeshArrays, read_mesh
from .export import ExportResult, export_mfem, export_mfem_mesh
from .spec import (AngularResolution, CappedInterfaces, EmbeddedFault,
                   InterfaceSizing, MeshResult, MeshSpec, NearPoints, PerInterface,
                   Refinement, Shell, SizingRule, UniformInterfaces,
                   ValidationReport, near_curve)

__all__ = [
    "MeshSpec", "MeshResult", "Shell", "InterfaceSizing", "SizingRule",
    "AngularResolution", "UniformInterfaces", "CappedInterfaces",
    "PerInterface", "Refinement", "NearPoints", "near_curve", "EmbeddedFault",
    "ValidationReport", "build_layered_mesh",
    "build_offset_mesh", "export_mfem_mesh", "export_mfem", "ExportResult",
    "manifest", "read_mesh", "MeshArrays",
]

#: The names that need gmsh to import, and the modules that hold them.
_MESHING = {"build_layered_mesh": "layered", "build_offset_mesh": "offset"}


def __getattr__(name: str) -> object:
    """A builder, imported from its module on first use; an ImportError
    naming the `meshing` extra where gmsh is missing."""
    if name not in _MESHING:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{_MESHING[name]}")
    except ImportError as error:
        if not missing_gmsh(error):
            raise
        raise ImportError(GMSH_MISSING) from error
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_MESHING))
