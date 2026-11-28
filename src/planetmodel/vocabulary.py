"""The field names the library itself speaks, and the constants it knows.

A model may hold a field under any name.  What this table fixes is what
a name means when the library reads, writes or guarantees a field by
it: a model looks its names up here, refuses a field attached under a
known name with the wrong character, and converts a field's units by
the dimensions the name carries.  Each entry gives the name's character
(its law under a mapping), its physical dimensions, and what it is; a
model supplies a `FieldSpec` of its own for a name outside the table.

Character and dimensions are independent.  The rule for the character:
a density is rank 0 weight 1; a modulus is a coefficient of the
second elasticity tensor, whose density-like factor of 1/J it shares,
so every modulus (kappa, mu, and the Love moduli A, C, F, L, N) is rank
0 weight 1 too, and the tensor itself is rank 4 weight 1; a velocity, a
quality factor, an anisotropy ratio, a viscosity and a fraction of a
modulus are rank 0 weight 0.  A velocity is the square root of a modulus
over a density, the weights cancelling, which is why it has no law of its
own.

A family is a stem whose every member, the stem with a positive index
(`relaxation_time_3`), means the same thing for its index (`FAMILIES`):
the Prony branches of a generalised Maxwell body in shear, branch k the
fraction `relaxation_strength_k` = mu_k / mu_U of the unrelaxed shear
modulus that relaxes in it and its `relaxation_time_k`, the relaxed part
being (1 - sum w_k) mu_U; a viscosity is the one-branch shorthand, w_1 =
1 and tau_1 = eta / mu.  `member(name)` says whether a name is one and
of which family, `family_spec` gives its spec, and `shipped(name)` the
spec of any name the library speaks, of the table or of a family.  A
Burgers body's Kelvin element is named by `kelvin_viscosity` and
`kelvin_modulus_fraction`; a model holds a Burgers layer as its two
Prony branches.

`CONSTANTS` holds the named constants a model reads in its own units,
each declared with its SI value and dimensions.
"""
from __future__ import annotations

import re
from dataclasses import KW_ONLY, dataclass

from . import units
from .character import DENSITY, ELASTIC, SCALAR, Character
from .units import Dimensions

__all__ = ["FieldSpec", "Constant", "VOCABULARY", "FAMILIES", "CONSTANTS", "spec",
           "member", "family_spec", "shipped"]


@dataclass(frozen=True)
class FieldSpec:
    """What a field name means: its character, dimensions and a gloss.

    `dimensions` is None for a name whose units are not declared; a
    model refuses to convert such a field by name.
    """

    character: Character
    dimensions: Dimensions | None
    _: KW_ONLY
    meaning: str = ""


@dataclass(frozen=True)
class Constant:
    """A named constant: its SI value, its dimensions and a gloss."""

    value_si: float
    dimensions: Dimensions
    _: KW_ONLY
    meaning: str = ""


#: The shipped names, each mapped to its spec.
VOCABULARY: dict[str, FieldSpec] = {
    "rho": FieldSpec(DENSITY, units.DENSITY, meaning="mass density"),
    "vp": FieldSpec(SCALAR, units.VELOCITY,
                    meaning="P-wave speed of an isotropic medium"),
    "vs": FieldSpec(SCALAR, units.VELOCITY,
                    meaning="S-wave speed of an isotropic medium"),
    "vpv": FieldSpec(SCALAR, units.VELOCITY,
                     meaning="P-wave speed along the symmetry axis (vertical)"),
    "vsv": FieldSpec(SCALAR, units.VELOCITY,
                     meaning="S-wave speed polarised along the symmetry axis "
                             "(vertical)"),
    "vph": FieldSpec(SCALAR, units.VELOCITY,
                     meaning="P-wave speed normal to the symmetry axis "
                             "(horizontal)"),
    "vsh": FieldSpec(SCALAR, units.VELOCITY,
                     meaning="S-wave speed polarised normal to the symmetry "
                             "axis (horizontal)"),
    "eta": FieldSpec(SCALAR, units.DIMENSIONLESS,
                     meaning="the anisotropy parameter F / (A - 2L)"),
    "qkappa": FieldSpec(SCALAR, units.DIMENSIONLESS,
                        meaning="bulk quality factor; zero attenuates nothing"),
    "qmu": FieldSpec(SCALAR, units.DIMENSIONLESS,
                     meaning="shear quality factor; zero attenuates nothing "
                             "(a fluid)"),
    "qp": FieldSpec(SCALAR, units.DIMENSIONLESS,
                    meaning="quality factor of P waves; zero attenuates nothing"),
    "kappa": FieldSpec(DENSITY, units.MODULUS,
                       meaning="bulk modulus of an isotropic medium"),
    "mu": FieldSpec(DENSITY, units.MODULUS,
                    meaning="shear modulus of an isotropic medium"),
    "A": FieldSpec(DENSITY, units.MODULUS, meaning="Love modulus A = rho vph^2"),
    "C": FieldSpec(DENSITY, units.MODULUS, meaning="Love modulus C = rho vpv^2"),
    "F": FieldSpec(DENSITY, units.MODULUS,
                   meaning="Love modulus F = eta (A - 2L)"),
    "L": FieldSpec(DENSITY, units.MODULUS, meaning="Love modulus L = rho vsv^2"),
    "N": FieldSpec(DENSITY, units.MODULUS, meaning="Love modulus N = rho vsh^2"),
    "elastic_moduli": FieldSpec(
        ELASTIC, units.MODULUS,
        meaning="the second elasticity tensor, static, Voigt-reduced in the "
                "spherical frame"),
    "viscosity": FieldSpec(SCALAR, units.VISCOSITY, meaning="dynamic viscosity"),
    "kelvin_viscosity": FieldSpec(
        SCALAR, units.VISCOSITY,
        meaning="viscosity of the Kelvin element of a Burgers body, in series with "
                "its Maxwell element"),
    "kelvin_modulus_fraction": FieldSpec(
        SCALAR, units.DIMENSIONLESS,
        meaning="shear modulus of the Kelvin element of a Burgers body, as a fraction "
                "of the unrelaxed shear modulus"),
    "temperature": FieldSpec(SCALAR, units.TEMPERATURE,
                             meaning="absolute temperature"),
    "stress_exponent": FieldSpec(
        SCALAR, units.DIMENSIONLESS,
        meaning="stress exponent n of power-law creep, its strain rate growing as "
                "the stress to the n"),
    "transition_stress": FieldSpec(
        DENSITY, units.MODULUS,
        meaning="deviatoric stress at which power-law creep strains as fast as "
                "linear creep"),
    "pressure": FieldSpec(DENSITY, units.MODULUS,
                          meaning="hydrostatic pressure of the spherically "
                                  "symmetric reference"),
    "g": FieldSpec(SCALAR, units.GRAVITY,
                   meaning="gravitational acceleration of the spherically "
                           "symmetric reference, radially inward, by magnitude"),
    "drho_dphi0": FieldSpec(
        DENSITY, units.DENSITY / (units.GRAVITY * units.LENGTH),
        meaning="stratification of a fluid layer: "
                "d rho / d Phi_0 = rho'(r) / g(r)"),
    "bullen_excess": FieldSpec(
        SCALAR, units.DIMENSIONLESS,
        meaning="excess of the Bullen parameter of a layer over one, "
                "-kappa rho'(r) / (rho^2 g) - 1: zero where the layer is "
                "neutrally stratified, positive where it is stably stratified"),
}

#: The families of names, each stem mapped to the spec every member has.
FAMILIES: dict[str, FieldSpec] = {
    "relaxation_strength": FieldSpec(
        SCALAR, units.DIMENSIONLESS,
        meaning="fraction of the unrelaxed shear modulus that relaxes in Prony "
                "branch k"),
    "relaxation_time": FieldSpec(SCALAR, units.TIME,
                                 meaning="relaxation time of Prony branch k"),
}

_MEMBER = re.compile(r"^(?P<stem>.+)_(?P<index>[1-9][0-9]*)$")


def member(name: str) -> tuple[str, int] | None:
    """(stem, index) of a member of a family, its index positive and
    written without leading zeros; None for any other name."""
    m = _MEMBER.match(name)
    if m is None or m["stem"] not in FAMILIES:
        return None
    return m["stem"], int(m["index"])


def family_spec(name: str) -> FieldSpec | None:
    """The spec of a member of a family, or None."""
    found = member(name)
    return None if found is None else FAMILIES[found[0]]


def shipped(name: str) -> FieldSpec | None:
    """The spec of a name the library speaks, of the table or a family."""
    known = VOCABULARY.get(name)
    return known if known is not None else family_spec(name)


#: The named constants, each with its SI value and dimensions.
CONSTANTS: dict[str, Constant] = {
    "G": Constant(units.G_SI, units.GRAVITATIONAL_CONSTANT,
                  meaning="the gravitational constant"),
}


def spec(name: str) -> FieldSpec:
    """The spec of a shipped name, of the table or a family; KeyError,
    naming it, for any other."""
    known = shipped(name)
    if known is None:
        raise KeyError(
            f"{name!r} is not a name the library speaks; the vocabulary is "
            f"{list(VOCABULARY)}, and the families "
            f"{[f'{stem}_<k>' for stem in FAMILIES]}")
    return known
