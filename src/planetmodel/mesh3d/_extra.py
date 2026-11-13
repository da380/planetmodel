"""gmsh, the `meshing` extra, imported where a mesh is built or read.

The manifest, the spec, the arrays of a mesh and the MFEM export import
without it, so a code holding an MFEM delivery reads its manifest and
applies the manifest's rules with no gmsh installed; what builds or
reads a mesh file asks for it here and is refused, naming the extra,
where it is missing.
"""
from __future__ import annotations

from types import ModuleType

__all__ = ["GMSH_MISSING", "require_gmsh", "missing_gmsh"]

#: What a code is told where gmsh is wanted and missing.
GMSH_MISSING = ("planetmodel.mesh3d needs gmsh to build or read a mesh.  Install it "
                "with:\n    pip install 'planetmodel[meshing]'      "
                "(or: poetry install --extras meshing)")


def missing_gmsh(error: ImportError) -> bool:
    """Whether an ImportError is gmsh's own absence."""
    return error.name == "gmsh" or (error.name or "").startswith("gmsh.")


def require_gmsh() -> ModuleType:
    """gmsh, or an ImportError naming the `meshing` extra."""
    try:
        import gmsh
    except ImportError as error:
        if not missing_gmsh(error):
            raise
        raise ImportError(GMSH_MISSING) from error
    return gmsh
