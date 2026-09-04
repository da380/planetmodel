"""planetmodel: spherically layered planetary models, and their meshes.

A model is a skeleton of boundary radii, a geometry that places the
skeleton in the physical world through one continuous mapping, and a
collection of fields on each layer.  This package holds the skeleton and
the geometry, the mappings that place them, and the meshers that hand a
geometry to a solver.  Everything is numbers: units belong to the model
that is built on top.

Subpackages, one level down:

  frames       the local spherical frame and tensors moved between frames
  mesh1d       radial spectral-element meshes (GLL)
  mesh3d       2D and 3D meshes via gmsh, and MFEM export
  testing      the executable contracts, `check_mapping` and its kin
"""
from .displacement import (CallableDisplacement, RadialDisplacement,
                           ZeroDisplacement, as_displacement)
from .geometry import Geometry, InterfaceInfo, LayerInfo
from .mapping import (IdentityMapping, Mapping, MappingBase, MappingPerturbation,
                      RadialStretch, ScaledMapping, ValidityReport,
                      outer_radius_of, validity_lattice)
from .mesh1d import RadialMesh
from .skeleton import CoarseningMap, Location, Skeleton
from . import testing  # noqa: F401

__version__ = "1.0.0.dev1"

__all__ = [
    "Skeleton", "Location", "CoarseningMap",
    "Geometry", "LayerInfo", "InterfaceInfo",
    "Mapping", "MappingBase", "IdentityMapping", "RadialStretch", "ScaledMapping",
    "ValidityReport", "MappingPerturbation", "validity_lattice", "outer_radius_of",
    "RadialDisplacement", "ZeroDisplacement", "CallableDisplacement",
    "as_displacement",
    "RadialMesh",
    "testing",
    "__version__",
]
