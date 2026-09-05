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
quality factor, an anisotropy ratio and a viscosity are rank 0 weight 0.
A velocity is the square root of a modulus over a density, the weights
cancelling, which is why it has no law of its own.

`CONSTANTS` holds the named constants a model reads in its own units,
each declared with its SI value and dimensions.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

from . import units
from .character import DENSITY, ELASTIC, SCALAR, Character
from .units import Dimensions

__all__ = ["FieldSpec", "Constant", "VOCABULARY", "CONSTANTS", "spec"]


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
}

#: The named constants, each with its SI value and dimensions.
CONSTANTS: dict[str, Constant] = {
    "G": Constant(units.G_SI, units.GRAVITATIONAL_CONSTANT,
                  meaning="the gravitational constant"),
}


def spec(name: str) -> FieldSpec:
    """The spec of a shipped name; KeyError, naming it, for any other."""
    try:
        return VOCABULARY[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a name the library speaks; the vocabulary is "
            f"{list(VOCABULARY)}") from None
