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
  rheology     what a layer's rheology fields imply, and a model frozen at
               a frequency with complex moduli
  units        Dimensions, Scales and the dimension constants
  vocabulary   the shipped field names, their characters and dimensions
  behaviours   the free functions as methods: `layer_method` and the mixins
  catalogue    PREM, LayeredIsotropicElastic and MineosModel, model types
  deck         tabulated models: decks, their formats, fields by interpolation
  sampling     a model on a radial mesh times an angular grid
  lateral      values between the nodes of an angular grid: the lateral rules
  modelfile    the model file: a model in one netCDF file (netCDF4, optional)
  surfacefile  the surface file: fields on one spherical surface, static or
               in time (netCDF4, optional)
  harmonics    real spherical harmonics, through ducc0
  slices       a field on a sphere, a section or a profile; spectra and
               correlations by degree
  plotting     profiles, sections, maps and spectra drawn (matplotlib and
               cartopy, optional), and VTK files for ParaView
  mesh1d       radial spectral-element meshes (GLL), nodal values, gravity
               and hydrostatic pressure
  mesh3d       2D and 3D meshes via gmsh, and MFEM export
  randomfield  Matern random fields on balls, annuli and layers
  datasets     published models: their entries, fetched on demand into a
               cache, read into the library's objects
  testing      the executable contracts, `check_field` and its kin
"""

from .behaviours import (
    ConstantQ,
    Elastic,
    SelfGravitating,
    Viscoelastic,
    layer_method,
    with_moduli,
    with_velocities,
)
from .catalogue import PREM, LayeredIsotropicElastic, MineosModel
from .character import DENSITY, ELASTIC, SCALAR, STRESS, VECTOR, Character, Symmetry
from .deck import (
    GIA,
    MINEOS,
    ND,
    TVEL,
    Deck,
    DeckFormat,
    Tabulated,
    deck_layers,
    read_deck,
    write_deck,
)
from .displacement import (
    CallableDisplacement,
    RadialDisplacement,
    ZeroDisplacement,
    as_displacement,
    flattening,
    layer_linear,
)
from .expansions import HarmonicField, HarmonicFunction
from .fields import (
    AnalyticField,
    ComposedField,
    Field,
    FieldBase,
    RadialField,
    constant_field,
)
from .frames import from_geographic, geographic
from .geometry import Geometry, InterfaceInfo, LayerInfo
from .gridded import GriddedDisplacement, GriddedField, GriddedFunction
from .harmonics import real_harmonics, synthesise
from .layerfunction import (
    LayerFunction,
    NumericLayer,
    PolynomialLayer,
    as_layer_function,
    constant_layer,
    polynomial_fit,
    polynomial_layer,
)
from .mapping import (
    IdentityMapping,
    Mapping,
    MappingBase,
    MappingPerturbation,
    RadialStretch,
    ScaledMapping,
    ValidityReport,
    outer_radius_of,
    validity_lattice,
)
from .materials import ElasticField, elastic_moduli, is_fluid, kappa_mu, moduli
from .means import spherical_mean, spherically_averaged
from .mesh1d import RadialMesh
from .mesh1d.gravity import gravity, gravity_fields, mass, pressure, pressure_fields
from .model import CoarseningFit, Layer, Model
from .perturbation import PerturbedField, perturbed, unperturbed
from .pushforward import PulledBackField, PushedForwardField, pull_back, push_forward
from .rheology import (
    dispersive_moduli,
    frozen,
    frozen_moduli,
    is_viscoelastic,
    reference_omega,
)
from .sampling import (
    AngularGrid,
    Sample,
    driscoll_healy,
    equiangular,
    gauss_legendre,
    sample,
)
from .skeleton import CoarseningMap, Location, Skeleton
from .units import EARTH_MEAN_DENSITY, G_SI, Dimensions, Scales
from .vocabulary import CONSTANTS, VOCABULARY, Constant, FieldSpec
from . import testing  # noqa: F401

__version__ = "1.3.2"

__all__ = [
    "Skeleton",
    "Location",
    "CoarseningMap",
    "CoarseningFit",
    "Geometry",
    "LayerInfo",
    "InterfaceInfo",
    "Mapping",
    "MappingBase",
    "IdentityMapping",
    "RadialStretch",
    "ScaledMapping",
    "ValidityReport",
    "MappingPerturbation",
    "validity_lattice",
    "outer_radius_of",
    "RadialDisplacement",
    "ZeroDisplacement",
    "CallableDisplacement",
    "as_displacement",
    "RadialMesh",
    "Character",
    "SCALAR",
    "DENSITY",
    "VECTOR",
    "STRESS",
    "ELASTIC",
    "Symmetry",
    "LayerFunction",
    "PolynomialLayer",
    "NumericLayer",
    "as_layer_function",
    "polynomial_layer",
    "constant_layer",
    "polynomial_fit",
    "Field",
    "FieldBase",
    "RadialField",
    "AnalyticField",
    "ComposedField",
    "constant_field",
    "GriddedField",
    "GriddedFunction",
    "GriddedDisplacement",
    "HarmonicField",
    "HarmonicFunction",
    "PerturbedField",
    "perturbed",
    "unperturbed",
    "spherical_mean",
    "spherically_averaged",
    "ElasticField",
    "is_fluid",
    "moduli",
    "elastic_moduli",
    "kappa_mu",
    "is_viscoelastic",
    "frozen_moduli",
    "dispersive_moduli",
    "frozen",
    "reference_omega",
    "push_forward",
    "pull_back",
    "PushedForwardField",
    "PulledBackField",
    "Dimensions",
    "Scales",
    "G_SI",
    "EARTH_MEAN_DENSITY",
    "FieldSpec",
    "Constant",
    "VOCABULARY",
    "CONSTANTS",
    "Layer",
    "Model",
    "layer_method",
    "with_moduli",
    "with_velocities",
    "Elastic",
    "ConstantQ",
    "SelfGravitating",
    "Viscoelastic",
    "PREM",
    "LayeredIsotropicElastic",
    "MineosModel",
    "Deck",
    "DeckFormat",
    "MINEOS",
    "GIA",
    "ND",
    "TVEL",
    "read_deck",
    "write_deck",
    "deck_layers",
    "Tabulated",
    "flattening",
    "layer_linear",
    "AngularGrid",
    "Sample",
    "sample",
    "gauss_legendre",
    "driscoll_healy",
    "equiangular",
    "geographic",
    "from_geographic",
    "real_harmonics",
    "synthesise",
    "gravity",
    "mass",
    "gravity_fields",
    "pressure",
    "pressure_fields",
    "testing",
    "__version__",
]
