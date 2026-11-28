"""The extras of `mesh3d`, each imported where it is used.

gmsh, the `meshing` extra, is imported where a mesh is built by it or
an MSH file read; PyMFEM, the `mfem` extra, where a cubed sphere is
built or an MFEM delivery written.  The manifest, the specs and the
arrays of a mesh import without either, so a code holding an MFEM
delivery reads its manifest and applies the manifest's rules with
neither installed; what needs one asks for it here and is refused,
naming the extra, where it is missing.
"""
from __future__ import annotations

from types import ModuleType

__all__ = ["GMSH_MISSING", "MFEM_MISSING", "require_gmsh", "missing_gmsh",
           "require_mfem"]

#: What a code is told where gmsh is wanted and missing.
GMSH_MISSING = ("planetmodel.mesh3d needs gmsh to build or read a mesh.  Install it "
                "with:\n    pip install 'planetmodel[meshing]'      "
                "(or: poetry install --extras meshing)")

#: What a code is told where PyMFEM is wanted and missing.
MFEM_MISSING = ("planetmodel.mesh3d needs PyMFEM to build a cubed sphere or to "
                "write an MFEM delivery.  Install it with:\n"
                "    pip install 'planetmodel[mfem]'      "
                "(or: poetry install --extras mfem)")


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


def require_mfem() -> ModuleType:
    """PyMFEM's serial module, or an ImportError naming the `mfem` extra."""
    try:
        import mfem.ser as mfem
    except ImportError as error:
        if not (error.name or "").split(".")[0] == "mfem":
            raise
        raise ImportError(MFEM_MISSING) from error
    return mfem
