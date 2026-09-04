"""prem.py -- the Preliminary Reference Earth Model, exactly as published.

Built from the low-order polynomials in radius of Table I of Dziewonski
& Anderson (1981), with no deck file involved, so every field is an
exact piecewise polynomial and evaluation, derivatives and integrals
carry no interpolation or sampling error.
"""
from __future__ import annotations

import numpy as np

from ..io.deck import MINEOS_COLUMNS, attach_moduli, as_model_class
from ..model.body import Layer, ReferenceBody
from ..model.fields.radial import RadialField, polynomial_layer
from ..model.skeleton import Skeleton
from ..model.vocabulary import character_of, dimensions_of

__all__ = ["prem", "PREM"]


# ---------------------------------------------------------------------------
# PREM (Dziewonski & Anderson, 1981)
#
# Polynomial coefficients from Table I of the paper, one region per
# entry (radii in metres; coefficients in the paper's units of g/cm^3,
# km/s or dimensionless, for the normalized radius x = r/a).  They are
# converted to SI on construction.  Regions run centre outwards; where
# the paper lists a single Vp/Vs the region is isotropic and the
# vph/vsh/eta entries are filled in by _prem_coefficients().
# ---------------------------------------------------------------------------

_PREM_A = 6371e3        # normalization radius a, metres

#: The regions of Table I, centre outward, as layer names.
_PREM_LAYER_NAMES = (
    "inner_core", "outer_core", "lowermost_mantle", "lower_mantle",
    "upper_lower_mantle", "transition_zone_lower", "transition_zone_middle",
    "transition_zone_upper", "low_velocity_zone", "lid", "lower_crust",
    "upper_crust", "ocean",
)

_PREM_REGIONS: tuple[tuple[float, float, dict[str, tuple[float, ...]]], ...] = (
    # inner core
    (0.0, 1221500.0, {
        "rho": (13.0885, 0.0, -8.8381),
        "vpv": (11.2622, 0.0, -6.3640),
        "vsv": (3.6678, 0.0, -4.4475),
        "qkappa": (1327.7,), "qmu": (84.6,),
    }),
    # outer core (fluid)
    (1221500.0, 3480000.0, {
        "rho": (12.5815, -1.2638, -3.6426, -5.5281),
        "vpv": (11.0487, -4.0362, 4.8023, -13.5732),
        "vsv": (0.0,),
        "qkappa": (57823.0,), "qmu": (0.0,),
    }),
    # lowermost mantle (D'')
    (3480000.0, 3630000.0, {
        "rho": (7.9565, -6.4761, 5.5283, -3.0807),
        "vpv": (15.3891, -5.3181, 5.5242, -2.5514),
        "vsv": (6.9254, 1.4672, -2.0834, 0.9783),
        "qkappa": (57823.0,), "qmu": (312.0,),
    }),
    # lower mantle
    (3630000.0, 5600000.0, {
        "rho": (7.9565, -6.4761, 5.5283, -3.0807),
        "vpv": (24.9520, -40.4673, 51.4832, -26.6419),
        "vsv": (11.1671, -13.7818, 17.4575, -9.2777),
        "qkappa": (57823.0,), "qmu": (312.0,),
    }),
    # uppermost lower mantle
    (5600000.0, 5701000.0, {
        "rho": (7.9565, -6.4761, 5.5283, -3.0807),
        "vpv": (29.2766, -23.6027, 5.5242, -2.5514),
        "vsv": (22.3459, -17.2473, -2.0834, 0.9783),
        "qkappa": (57823.0,), "qmu": (312.0,),
    }),
    # transition zone: 670-600 km depth
    (5701000.0, 5771000.0, {
        "rho": (5.3197, -1.4836),
        "vpv": (19.0957, -9.8672),
        "vsv": (9.9839, -4.9324),
        "qkappa": (57823.0,), "qmu": (143.0,),
    }),
    # transition zone: 600-400 km depth
    (5771000.0, 5971000.0, {
        "rho": (11.2494, -8.0298),
        "vpv": (39.7027, -32.6166),
        "vsv": (22.3512, -18.5856),
        "qkappa": (57823.0,), "qmu": (143.0,),
    }),
    # transition zone: 400-220 km depth
    (5971000.0, 6151000.0, {
        "rho": (7.1089, -3.8045),
        "vpv": (20.3926, -12.2569),
        "vsv": (8.9496, -4.4597),
        "qkappa": (57823.0,), "qmu": (143.0,),
    }),
    # low velocity zone (transversely isotropic)
    (6151000.0, 6291000.0, {
        "rho": (2.6910, 0.6924),
        "vpv": (0.8317, 7.2180),
        "vsv": (5.8582, -1.4678),
        "vph": (3.5908, 4.6172),
        "vsh": (-1.0839, 5.7176),
        "eta": (3.3687, -2.4778),
        "qkappa": (57823.0,), "qmu": (80.0,),
    }),
    # LID (transversely isotropic)
    (6291000.0, 6346600.0, {
        "rho": (2.6910, 0.6924),
        "vpv": (0.8317, 7.2180),
        "vsv": (5.8582, -1.4678),
        "vph": (3.5908, 4.6172),
        "vsh": (-1.0839, 5.7176),
        "eta": (3.3687, -2.4778),
        "qkappa": (57823.0,), "qmu": (600.0,),
    }),
    # lower crust
    (6346600.0, 6356000.0, {
        "rho": (2.900,), "vpv": (6.800,), "vsv": (3.900,),
        "qkappa": (57823.0,), "qmu": (600.0,),
    }),
    # upper crust
    (6356000.0, 6368000.0, {
        "rho": (2.600,), "vpv": (5.800,), "vsv": (3.200,),
        "qkappa": (57823.0,), "qmu": (600.0,),
    }),
    # ocean (fluid)
    (6368000.0, 6371000.0, {
        "rho": (1.020,), "vpv": (1.450,), "vsv": (0.0,),
        "qkappa": (57823.0,), "qmu": (0.0,),
    }),
)

_PREM_SI_SCALE = {"rho": 1e3, "vpv": 1e3, "vsv": 1e3, "vph": 1e3,
                  "vsh": 1e3, "qkappa": 1.0, "qmu": 1.0, "eta": 1.0}


def _prem_coefficients(name: str,
                       region: dict[str, tuple[float, ...]]) -> tuple[float, ...]:
    """Table I coefficients for one field in one region.

    Fills in the isotropic defaults vph = vpv, vsh = vsv and eta = 1
    for regions where the paper lists a single P and S velocity.
    """
    if name in region:
        return region[name]
    if name == "vph":
        return region["vpv"]
    if name == "vsh":
        return region["vsv"]
    if name == "eta":
        return (1.0,)
    raise KeyError(f"the PREM table defines no field {name!r}")


def prem(*, ocean: bool = True):
    """The Preliminary Reference Earth Model, exactly as published.

    Built from the low-order polynomials in radius of Table I of
    Dziewonski & Anderson (1981), Phys. Earth Planet. Inter. 25,
    297-356, with no deck file involved: every field is an exact
    piecewise polynomial in x = r/a (a = 6371 km), so evaluation,
    derivatives and integrals carry no interpolation or sampling error.

    Conventions match mineos-style decks such as prem.200 so the two
    can be compared field by field: quantities are SI (kg/m^3, m/s,
    radius in metres); the full anisotropic field set is always
    present, with vph = vpv, vsh = vsv and eta = 1 outside the
    transversely isotropic zone (24.4-220 km depth); the infinite Q_mu
    of the fluid outer core and ocean is represented by 0; and elastic
    values hold at the reference period of 1 s.

    Example::

        prem = PREM()
        below, above = prem.skeleton.locate(3480e3).layers
        prem.vsv[above](3480e3)     # vsv on the mantle side of the CMB

    With ocean=False the 3 km water layer is deleted and the model ends
    at the top of the upper crust (6368 km), matching the reference
    Fortran implementation's elastic_PREM(.false.): the solid surface
    becomes the outer boundary and carries the load, and the ocean's
    mass is absent.  The normalization radius for the polynomials
    remains a = 6371 km.

    The value is a `ViscoelasticModel`: every layer holds the moduli
    under constant Q at the reference period of 1 s, and the outer
    core and the ocean are fluid.
    """
    regions = list(_PREM_REGIONS)
    if not ocean:
        regions = regions[:-1]
    boundaries = np.array([lo for lo, _, _ in regions] + [regions[-1][1]])
    skeleton = Skeleton(boundaries)
    fields = {}
    for name in MINEOS_COLUMNS:
        scale = _PREM_SI_SCALE[name]
        funcs = tuple(
            polynomial_layer(
                np.asarray(_prem_coefficients(name, spec)) * scale,
                (lo, hi), scale=_PREM_A)
            for lo, hi, spec in regions)
        fields[name] = RadialField(skeleton, funcs, name=name)
    for name, f in fields.items():
        f.character = character_of(name)
        f.dimensions = dimensions_of(name)
    names = _PREM_LAYER_NAMES[:len(regions)]
    layers = [Layer(index=i, name=n) for i, n in enumerate(names)]
    body = ReferenceBody.from_fields(skeleton, fields, layers=layers, meta={
        "name": "PREM" if ocean else "PREM (oceanless)",
        "ocean": ocean,
        "tref": 1.0,
        "ifanis": 1,
        "reference": ("Dziewonski & Anderson (1981), Phys. Earth "
                      "Planet. Inter. 25, 297-356, Table I"),
    })
    # Moduli are canonical, so they are built here rather than left to
    # the caller.  Each is a product of the exact polynomials above,
    # formed on the coefficients, so PREM stays exact in the moduli --
    # only the velocity views, which need a square root, are merely
    # pointwise exact.
    attach_moduli(body)
    body = body.classify_states()
    return as_model_class(body, reference_period=1.0)


#: The catalogue name: `PREM()` reads naturally in a script.
PREM = prem
