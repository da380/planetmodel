"""planetmodel: spherically layered planetary models, and their meshes.

A model is a skeleton of boundary radii, a geometry that places the
skeleton in the physical world through one continuous mapping, and a
bag of fields on each layer.  This package holds the skeleton and the
geometry, the mappings that place them, the fields and the model, the
catalogue of named models, and the meshers that hand a model to a
solver.  Every internal is numbers: the model alone carries units.

Modules reached by name, one level down:

  frames       the local spherical frame, tensors moved between frames, Voigt
  materials    velocity and moduli conversions, the elastic field, what a
               layer's fields imply (fluidity, moduli)
  units        Dimensions, Scales and the dimension constants
  vocabulary   the shipped field names, their characters and dimensions
  catalogue    prem, and simple models for use and testing
  sampling     a model on a radial mesh times an angular grid
  mesh1d       radial spectral-element meshes (GLL), nodal values, gravity
  mesh3d       2D and 3D meshes via gmsh, and MFEM export
  testing      the executable contracts, `check_field` and its kin
"""
from .catalogue import homogeneous, layered, prem
from .character import (DENSITY, ELASTIC, SCALAR, STRESS, VECTOR, Character,
                        Symmetry)
from .displacement import (CallableDisplacement, RadialDisplacement,
                           ZeroDisplacement, as_displacement, flattening,
                           layer_linear)
from .fields import (AnalyticField, ComposedField, Field, FieldBase, RadialField,
                     constant_field)
from .geometry import Geometry, InterfaceInfo, LayerInfo
from .layerfunction import (LayerFunction, NumericLayer, PolynomialLayer,
                            as_layer_function, constant_layer, polynomial_fit,
                            polynomial_layer)
from .mapping import (IdentityMapping, Mapping, MappingBase, MappingPerturbation,
                      RadialStretch, ScaledMapping, ValidityReport,
                      outer_radius_of, validity_lattice)
from .materials import ElasticField, elastic_moduli, is_fluid, kappa_mu, moduli
from .mesh1d import RadialMesh
from .mesh1d.gravity import gravity, mass
from .model import Layer, Model
from .pushforward import (PulledBackField, PushedForwardField, pull_back,
                          push_forward)
from .sampling import AngularGrid, Sample, equiangular, gauss_legendre, sample
from .skeleton import CoarseningMap, Location, Skeleton
from .units import EARTH_MEAN_DENSITY, G_SI, Dimensions, Scales
from .vocabulary import CONSTANTS, VOCABULARY, Constant, FieldSpec
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
    "Character", "SCALAR", "DENSITY", "VECTOR", "STRESS", "ELASTIC", "Symmetry",
    "LayerFunction", "PolynomialLayer", "NumericLayer", "as_layer_function",
    "polynomial_layer", "constant_layer", "polynomial_fit",
    "Field", "FieldBase", "RadialField", "AnalyticField", "ComposedField",
    "constant_field",
    "ElasticField", "is_fluid", "moduli", "elastic_moduli", "kappa_mu",
    "push_forward", "pull_back", "PushedForwardField", "PulledBackField",
    "Dimensions", "Scales", "G_SI", "EARTH_MEAN_DENSITY",
    "FieldSpec", "Constant", "VOCABULARY", "CONSTANTS",
    "Layer", "Model", "prem", "homogeneous", "layered",
    "flattening", "layer_linear",
    "AngularGrid", "Sample", "sample", "gauss_legendre", "equiangular",
    "gravity", "mass",
    "testing",
    "__version__",
]
