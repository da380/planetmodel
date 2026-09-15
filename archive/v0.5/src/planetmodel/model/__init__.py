"""model -- the reference body: geometry, fields, mappings, model classes.

Conventions fixed here and relied on throughout planetmodel:

  coordinates   (r, theta, phi) with theta the COLATITUDE in radians
                [0, pi] and phi the longitude in radians (-pi, pi].
                Degrees and lat/lon appear only in planetmodel.io adapters,
                which convert on read.

  units         SI at rest: radius in metres, density kg/m^3, moduli
                Pa.  A body carries Scales and can be re-expressed
                exactly (ReferenceBody.nondimensionalised); the 3D
                mesher does that once, internally.

  components    in the frame the coordinates imply.  Field.evaluate
                speaks (r, theta, phi), so it returns components in the
                local (e_r, e_theta, e_phi) frame; frame="cartesian"
                asks for the other.  Rank-2 and rank-4 fields are
                Voigt-reduced in the spherical frame, and the Cartesian
                form of a Voigt matrix is its Bond rotation
                (materials.bond_matrix).

  mapping       m maps the REFERENCE body to the PHYSICAL one.  This is
                MMA26's xi; AAC16's xi is m^-1.

All numerics are float64.
"""
from .body import Interface, Layer, ReferenceBody, fluid_where_vs_zero
from .classes import (ElasticModel, HasDensity, HasModuli, HasViscosity,
                      ModelBase, ViscoelasticModel, ViscousModel)
from .character import (DENSITY, ELASTIC, FIRST_ELASTIC, SCALAR, STRESS,
                        VECTOR, Character, Symmetry)
from .materials import (ElasticField, bond_matrix, kappa_mu_from_moduli,
                        moduli_from_velocities, tensor_to_voigt,
                        velocities_from_moduli, voigt_matrix, voigt_to_tensor)
from .fields.analytic import AnalyticField
from .fields.base import Assemblable, Field
from .fields.frequency import (ComposedFrequencyField, FrequencyDependentField,
                               LiftedFrequencyField, at_frequency,
                               lifted_to_frequency)
from .fields.dependent import FrozenField, LayerwiseDependentField, LiftedField
from .fields.layerwise import LayerwiseField, assemble
from .rheology import LawRecord, constant_q, constant_q_scalar, maxwell, prony
from .fields.time import (ComposedTimeField, TimeDependentField, at_time,
                          lifted_to_time)
from .fields.composite import (ComposedField, FieldBase, RestrictedField,
                               ScaledField, SumField)
from .fields.layer_function import (LayerFunction, as_layer_function,
                                    combine_layer_functions,
                                    multiply_layer_functions,
                                    rescale_layer_function)
from .fields.radial import (RadialField, constant_field, derived_field,
                            make_fitter, polynomial_layer)
from .skeleton import CoarseningMap, Location, Skeleton
from .units import EARTH_MEAN_DENSITY, Dimensions, Scales
from .displacement import (BlendDisplacement, CallableDisplacement,
                           RadialDisplacement, SumDisplacement,
                           ZeroDisplacement, as_displacement, layer_linear)
from .mapping import (IdentityMapping, Mapping, MappingBase,
                      MappingPerturbation, RadialStretch, ValidityReport,
                      validity_lattice)
from .firstelastic import FirstElasticField
from .pullback import (PulledBackElasticField, PulledBackField,
                       pulled_back_elastic)
from .pushforward import (PushedForwardField, pull_back, push_forward,
                          push_forward_field)
from .surface import Surface, ellipsoid_surface, spherical_surface
from .topography import (AnalyticTopography, GriddedTopography,
                         HarmonicTopography, ScaledTopography, SumTopography,
                         Topography, ZeroTopography, as_topography)

__all__ = [
    "Skeleton", "Location", "CoarseningMap", "Layer", "Interface",
    "fluid_where_vs_zero", "Field", "Assemblable", "FieldBase", "RadialField",
    "AnalyticField", "ReferenceBody", "LayerwiseField",
    "LayerwiseDependentField", "assemble", "LiftedField", "FrozenField",
    "FrequencyDependentField", "ComposedFrequencyField",
    "LiftedFrequencyField", "lifted_to_frequency",
    "at_frequency", "TimeDependentField", "ComposedTimeField", "lifted_to_time",
    "at_time", "LawRecord", "constant_q", "constant_q_scalar",
    "maxwell", "prony",
    "ModelBase", "HasDensity", "HasModuli", "HasViscosity",
    "ElasticModel", "ViscoelasticModel", "ViscousModel",
    "make_fitter", "polynomial_layer", "derived_field", "constant_field",
    "Character", "Symmetry", "SCALAR", "DENSITY", "VECTOR", "STRESS",
    "ELASTIC", "FIRST_ELASTIC", "SumField", "ScaledField",
    "ComposedField", "RestrictedField",
    "Topography", "as_topography", "GriddedTopography",
    "AnalyticTopography", "HarmonicTopography", "SumTopography",
    "ScaledTopography", "ZeroTopography",
    "Surface", "ellipsoid_surface", "spherical_surface",
    "RadialDisplacement", "as_displacement", "layer_linear",
    "ZeroDisplacement", "CallableDisplacement", "BlendDisplacement",
    "SumDisplacement", "Mapping", "MappingBase", "IdentityMapping",
    "RadialStretch", "ValidityReport", "validity_lattice",
    "MappingPerturbation",
    "push_forward", "pull_back", "push_forward_field",
    "PushedForwardField", "PulledBackField", "PulledBackElasticField",
    "pulled_back_elastic", "FirstElasticField",
    "Dimensions", "Scales", "EARTH_MEAN_DENSITY",
    "LayerFunction", "as_layer_function", "rescale_layer_function",
    "combine_layer_functions", "multiply_layer_functions",
    "ElasticField", "moduli_from_velocities", "velocities_from_moduli",
    "voigt_matrix", "bond_matrix", "kappa_mu_from_moduli",
    "voigt_to_tensor", "tensor_to_voigt",
]
