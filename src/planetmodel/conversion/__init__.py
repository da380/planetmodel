"""Conversions: seismic fields into thermal and rheological ones.

The laws that turn a model's seismic structure into what a solver of
slow deformation reads (a temperature anomaly from a velocity anomaly, a
density from either, a viscosity from them) are written on numbers in
SI (`laws`), each a short function a reader can hold against its
source; the parameters that vary with radius are profiles (`profiles`);
and a law's values travel with their sources as a parameter set
(`parameters`), so that whatever is built with one cites it.  Applied
to a model (`building`), a law reads an anomaly from the model's fields
and writes what it makes as lazy fields: a viscosity as a log
perturbation of the layer's radial one, a density as a relative
perturbation of its reference.  A viscosity becomes Prony branches
(`prony`): a Maxwell, Burgers or fitted Andrade body put on a model as
the strengths and times `rheology` reads, a Burgers body in named
layers from the model's own viscosity and a Kelvin element's
(`with_burgers`), and a model as the prony and burgers decks hold it
(`prony_deck`, `burgers_deck`).  A table of properties against
temperature and pressure, written by an offline generator, gives a
temperature anomaly by inversion (`tables`).  The base of the
lithosphere is found in a model's fields, at an isotherm, a viscosity
or a Maxwell time (`lithosphere`), and used as geometry
(`datasets.building.moved`) or as a lid of high viscosity.  Nothing
here is particular to the Earth.

Reached through its modules, off the top-level surface, with numpy and
scipy alone::

    from planetmodel.conversion import laws, parameters
    dt = laws.temperature_from_density(
        laws.density_from_velocity(dlnvs, scaling=parameters.SL3D_DENSITY_SCALING
                                   .scaling(r)),
        expansivity=parameters.SL3D_EXPANSIVITY.expansivity(r))
"""
from . import (building, laws, lithosphere, parameters, profiles, prony,
               tables)
from .laws import (GAS_CONSTANT, anelastic_sensitivity, beta_viscosity,
                   density_from_temperature, density_from_velocity,
                   exponential_viscosity, temperature_from_density,
                   temperature_from_velocity)
from .building import (ANOMALIES, Anomaly, attached, sl3d_viscosity, table_anomaly,
                       temperature_anomaly, velocity_anomaly, with_density,
                       with_power_law, with_viscosity)
from .lithosphere import (isotherm, maxwell_threshold, viscosity_threshold,
                          with_lid)
from .tables import PropertyTable
from .parameters import (BetaViscosity, DensityLaw, DensityScaling, Expansivity,
                         ExponentialViscosity, ParameterSet, Sensitivity,
                         ViscosityLaw)
from .profiles import (Profile, ProfileLike, as_profile, constant, polynomial, spline,
                       tabulated)
from .prony import (PronySeries, andrade_series, burgers_branches, burgers_deck,
                    burgers_parameters, burgers_series, burgers_terms, fitted_series,
                    maxwell_series, prony_deck, with_branches, with_burgers)

__all__ = ["building", "laws", "lithosphere", "parameters", "profiles", "prony",
           "tables", "isotherm", "maxwell_threshold", "viscosity_threshold", "with_lid",
           "PropertyTable", "table_anomaly", "PronySeries",
           "andrade_series", "burgers_series", "fitted_series", "maxwell_series",
           "with_branches", "burgers_terms", "burgers_branches", "burgers_parameters",
           "with_burgers", "prony_deck", "burgers_deck", "ANOMALIES", "Anomaly",
           "attached", "sl3d_viscosity", "temperature_anomaly", "velocity_anomaly",
           "with_density", "with_power_law",
           "with_viscosity", "DensityLaw", "ViscosityLaw", "GAS_CONSTANT",
           "anelastic_sensitivity",
           "beta_viscosity", "density_from_temperature", "density_from_velocity",
           "exponential_viscosity", "temperature_from_density",
           "temperature_from_velocity", "BetaViscosity", "DensityScaling",
           "Expansivity", "ExponentialViscosity", "ParameterSet", "Sensitivity",
           "Profile", "ProfileLike", "as_profile", "constant", "polynomial", "spline",
           "tabulated"]
