"""planetmodel: spherically layered planetary models, and their meshes.

A model is a list of layers.  Each layer is an interval of radius, the
fields it holds by name, and whether it is solid, fluid or vacuum; the
body-wide field `body["rho"]` is a view assembled from the layers'
pieces, defined on the layers that hold one.  Fields carry a tensor
character and physical dimensions, may depend on frequency or on time,
and are evaluated at `(r, theta, phi)` in the local spherical frame.  A
mapping takes the spherical reference body to the physical one, and
fields cross with it by the push-forward their character dictates.  A
model class says what a body guarantees: `ElasticModel`,
`ViscoelasticModel`, `ViscousModel`.

Subpackages, one level down:

  model        the framework: skeleton and layers, fields, laws,
               mapping, model classes, units, frames, vocabulary
  catalogue    named reference models: `prem()`
  io           deck readers, the netCDF model file, the mesh manifest,
               TOML mesh recipes
  mesh1d       radial spectral-element meshes (GLL) and gravity
  mesh3d       3D meshes via gmsh, and MFEM export (extras: meshing, mfem)
  sampling     a body evaluated on radial times angular nodes
  testing      the executable contracts: `check_field` and its kin
  loading      the quasi-static loading problem and Love numbers
               (meant for pyslfp)
  sobolev, randomfield
               radial operator families and Gaussian random fields
               (meant for pygeoinf)
"""
from .model import (
    # geometry and layers
    Skeleton, Layer, Interface, ReferenceBody, fluid_where_vs_zero,
    # fields
    Field, FieldBase, RadialField, AnalyticField, LayerwiseField,
    ComposedField, ElasticField, polynomial_layer, constant_field,
    Character, Symmetry,
    SCALAR, DENSITY, VECTOR, STRESS, ELASTIC, FIRST_ELASTIC,
    Dimensions, Scales, EARTH_MEAN_DENSITY, bond_matrix,
    # frequency and time, laws
    FrequencyDependentField, TimeDependentField,
    lifted_to_frequency, at_frequency, lifted_to_time, at_time,
    LawRecord, constant_q, maxwell, prony,
    # mapping
    Mapping, IdentityMapping, RadialStretch, validity_lattice,
    Topography, AnalyticTopography, GriddedTopography, ZeroTopography,
    Surface, as_topography, as_displacement, layer_linear,
    push_forward, pull_back, push_forward_field,
    # model classes
    ElasticModel, ViscoelasticModel, ViscousModel,
)
from .catalogue import PREM, prem
from .io import read_deck, read_isotropic_deck, read_mineos_deck
from .io.manifest import MeshManifest
from .io.netcdf import read as read_model, write as write_model
from .mesh1d import RadialMesh
from .sampling import AngularGrid, Sample
from .registry import lookup, register, registered
from . import testing  # noqa: F401  (public: the executable contracts)

__version__ = "0.5.0"

# The 3D mesher needs gmsh, so its names resolve on first use rather
# than at import: `import planetmodel` stays light without the extra.
_MESH3D = {"MeshSpec", "MeshResult", "build_layered_mesh", "export_mfem"}


def __getattr__(name: str):
    if name in _MESH3D:
        from . import mesh3d
        return getattr(mesh3d, name)
    raise AttributeError(f"module 'planetmodel' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _MESH3D)


__all__ = [
    # geometry and layers
    "Skeleton", "Layer", "Interface", "ReferenceBody", "fluid_where_vs_zero",
    # fields
    "Field", "FieldBase", "RadialField", "AnalyticField", "LayerwiseField",
    "ComposedField", "ElasticField", "polynomial_layer", "constant_field",
    "Character", "Symmetry",
    "SCALAR", "DENSITY", "VECTOR", "STRESS", "ELASTIC", "FIRST_ELASTIC",
    "Dimensions", "Scales", "EARTH_MEAN_DENSITY", "bond_matrix",
    # frequency and time, laws
    "FrequencyDependentField", "TimeDependentField",
    "lifted_to_frequency", "at_frequency", "lifted_to_time", "at_time",
    "LawRecord", "constant_q", "maxwell", "prony",
    # mapping
    "Mapping", "IdentityMapping", "RadialStretch", "validity_lattice",
    "Topography", "AnalyticTopography", "GriddedTopography", "ZeroTopography",
    "Surface", "as_topography", "as_displacement", "layer_linear",
    "push_forward", "pull_back", "push_forward_field",
    # model classes and catalogue
    "ElasticModel", "ViscoelasticModel", "ViscousModel", "prem", "PREM",
    # io
    "read_deck", "read_mineos_deck", "read_isotropic_deck",
    "read_model", "write_model", "MeshManifest",
    # meshes and sampling
    "RadialMesh", "AngularGrid", "Sample",
    "MeshSpec", "MeshResult", "build_layered_mesh", "export_mfem",
    # components
    "register", "lookup", "registered", "testing",
    "__version__",
]
