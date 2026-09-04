"""vocabulary.py -- the field names the library itself speaks.

A body may hold a field under any name; nothing here restricts that.
What this table fixes is what *planetmodel* means when it reads, writes
or guarantees a field by name: the deck readers file their columns
under these names, `attach_moduli` attaches these, the model classes
require them, and the writers and the mesher look them up.  Each entry
gives the name's character (how it transforms under a mapping), its
physical dimensions, and what it is.

Velocities are SCALAR in character and DIMENSIONLESS-looking quantities
such as the Q factors are genuinely dimensionless; a density is weight
one; the elastic tensor is the ELASTIC character.  Dimensions and
character are independent: a velocity carries the dimensions of a
speed and has no transformation law at all.
"""
from __future__ import annotations

from dataclasses import dataclass

from .character import DENSITY, ELASTIC, SCALAR, Character
from .units import Dimensions

__all__ = ["Entry", "VOCABULARY", "character_of", "dimensions_of",
           "describe", "names"]


@dataclass(frozen=True)
class Entry:
    """One canonical field name: its character, dimensions and meaning."""

    name: str
    character: Character
    dimensions: Dimensions
    meaning: str


_ENTRIES = (
    Entry("rho", DENSITY, Dimensions.DENSITY, "mass density"),
    Entry("vp", SCALAR, Dimensions.VELOCITY, "P-wave speed of an isotropic medium"),
    Entry("vs", SCALAR, Dimensions.VELOCITY, "S-wave speed of an isotropic medium"),
    Entry("vpv", SCALAR, Dimensions.VELOCITY,
          "P-wave speed along the symmetry axis (vertical)"),
    Entry("vsv", SCALAR, Dimensions.VELOCITY,
          "S-wave speed polarised along the symmetry axis (vertical)"),
    Entry("vph", SCALAR, Dimensions.VELOCITY,
          "P-wave speed normal to the symmetry axis (horizontal)"),
    Entry("vsh", SCALAR, Dimensions.VELOCITY,
          "S-wave speed polarised normal to the symmetry axis (horizontal)"),
    Entry("eta", SCALAR, Dimensions.DIMENSIONLESS,
          "the anisotropy parameter F / (A - 2L)"),
    Entry("qkappa", SCALAR, Dimensions.DIMENSIONLESS,
          "bulk quality factor; zero attenuates nothing"),
    Entry("qmu", SCALAR, Dimensions.DIMENSIONLESS,
          "shear quality factor; zero attenuates nothing (a fluid)"),
    Entry("kappa", SCALAR, Dimensions.MODULUS, "bulk modulus of an isotropic medium"),
    Entry("mu", SCALAR, Dimensions.MODULUS, "shear modulus of an isotropic medium"),
    Entry("A", SCALAR, Dimensions.MODULUS, "Love modulus A = rho vph^2"),
    Entry("C", SCALAR, Dimensions.MODULUS, "Love modulus C = rho vpv^2"),
    Entry("F", SCALAR, Dimensions.MODULUS, "Love modulus F = eta (A - 2L)"),
    Entry("L", SCALAR, Dimensions.MODULUS, "Love modulus L = rho vsv^2"),
    Entry("N", SCALAR, Dimensions.MODULUS, "Love modulus N = rho vsh^2"),
    Entry("elastic_moduli", ELASTIC, Dimensions.MODULUS,
          "the second elasticity tensor, static, Voigt-reduced in the "
          "spherical frame"),
    Entry("viscoelastic_moduli", ELASTIC, Dimensions.MODULUS,
          "the frequency-dependent elasticity tensor a law builds from "
          "elastic_moduli and the layer's other fields"),
    Entry("viscosity", SCALAR, Dimensions.VISCOSITY, "dynamic viscosity"),
)

#: The canonical names, each mapped to its entry.
VOCABULARY: dict[str, Entry] = {e.name: e for e in _ENTRIES}


def names() -> tuple[str, ...]:
    """The canonical field names, in the order the table lists them."""
    return tuple(VOCABULARY)


def _entry(name: str) -> Entry:
    try:
        return VOCABULARY[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a field name planetmodel itself speaks; the "
            f"vocabulary is {list(VOCABULARY)}") from None


def character_of(name: str) -> Character:
    """The character of a canonical field name."""
    return _entry(name).character


def dimensions_of(name: str) -> Dimensions:
    """The physical dimensions of a canonical field name."""
    return _entry(name).dimensions


def describe(name: str) -> str:
    """One line saying what a canonical field name means."""
    return _entry(name).meaning
