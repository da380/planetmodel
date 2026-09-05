"""How a field transforms under a mapping, and how elastic symmetry is named.

A field is numbers plus a Character = (rank, weight), and push-forward
is one generic operation driven by that character alone.  For a mapping
m taking the reference body to the physical one, with F[i, j] = d m_i /
d X_j and J = det F, a contravariant field of rank n and weight 1
transforms as

    T_phys[i1..in](x) = (1/J) F[i1, m1] ... F[in, mn] T_ref[m1..mn](X)

with x = m(X); a weight-0 field carries the factors of F but no 1/J,
and a rank-0 weight-0 field simply composes with the mapping.  In
particular rho_phys = rho_ref / J.

    field                                         rank  weight
    rho                                              0       1
    second elasticity tensor                         4       1
    equilibrium stress, stress glut                  2       1
    Q_kappa, Q_mu, eta, relaxation times, viscosity  0       0

The rheological entries are weight 0 on the understanding that the
tensor character of the constitutive law is carried entirely by the
elastic tensor: a scalar relaxation time modulating an elastic tensor is
invariant, whereas a genuine viscosity tensor would be rank 4 weight 1.

Velocities have no transformation law.  v_p and v_s are built from rho
and the moduli, neither of which is invariant, and the combination is a
tensor of no weight; the moduli are the canonical fields and the
velocities are derived from them.

Products of rank-0 fields multiply their characters: weights add, and a
product of two weight-1 fields (a density times a modulus, say) has no
character here and is refused.  Dividing by a weight-1 field is a
product with weight -1 and is likewise refused; a modulus over a density
is a squared velocity, which is not a field.

Symmetry classifies second elasticity tensors only: those with the full
minor and major symmetries, for which the Voigt 6x6 form is faithful.
The first elasticity tensor keeps only the major symmetry and has no
Voigt form, which is what a Character's `voigt` flag records; it is
Character(4, 1, voigt=False), a different character from ELASTIC.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from enum import Enum

__all__ = [
    "Character", "Symmetry",
    "SCALAR", "DENSITY", "VECTOR", "STRESS", "ELASTIC",
]


@dataclass(frozen=True)
class Character:
    """The transformation law of a field: (rank, weight), and Voigt or not.

    `rank` is the number of factors of F, `weight` the power of 1/J, 0
    or 1.  `voigt` says whether a rank-2 or rank-4 value carries the
    symmetries that make the Voigt reduction faithful; it is True for
    the stress and the second elasticity tensor and False for the first
    elasticity tensor.  It takes part in equality, so a field of each
    kind has a different character and the algebra keeps them apart.
    """

    rank: int
    weight: int
    _: KW_ONLY
    voigt: bool = True

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative, got {self.rank}")
        if self.weight not in (0, 1):
            raise ValueError(f"weight must be 0 or 1, got {self.weight}")

    @property
    def component_shape(self) -> tuple[int, ...]:
        """Trailing shape of a value array: (3,) * rank."""
        return (3,) * self.rank

    @property
    def voigt_shape(self) -> tuple[int, ...] | None:
        """Trailing shape under Voigt reduction, or None if not reducible.

        Only the symmetric ranks 2 and 4 reduce, and only where `voigt`
        says the symmetries are there; ranks 0 and 1, and a rank-4 value
        without the minor symmetries, keep their component shape.
        """
        if not self.voigt:
            return None
        return {2: (6,), 4: (6, 6)}.get(self.rank)

    @property
    def is_invariant(self) -> bool:
        """Whether the field merely composes with the mapping."""
        return self.rank == 0 and self.weight == 0

    def __mul__(self, other: Character) -> Character:
        """The character of a pointwise product of two rank-0 fields.

        Weights add and their sum may not exceed 1; a product involving a
        field of positive rank has no character here.  Both refusals are
        ValueError.
        """
        if not isinstance(other, Character):
            return NotImplemented
        if self.rank != 0 or other.rank != 0:
            raise ValueError(
                f"only rank-0 characters multiply, got {self} and {other}")
        weight = self.weight + other.weight
        if weight > 1:
            raise ValueError(
                f"the product of {self} and {other} would have weight "
                f"{weight}, which no field carries")
        return Character(0, weight)

    def __str__(self) -> str:
        tail = "" if self.voigt else ", no Voigt form"
        return f"rank {self.rank}, weight {self.weight}{tail}"


#: Q_kappa, Q_mu, eta, relaxation times, viscosity, and the moduli.
SCALAR = Character(0, 0)
DENSITY = Character(0, 1)
VECTOR = Character(1, 0)
#: Equilibrium stress, stress glut.
STRESS = Character(2, 1)
#: The second elasticity tensor.
ELASTIC = Character(4, 1)


class Symmetry(Enum):
    """Material symmetry of a second elasticity tensor, by moduli count."""

    ISOTROPIC = 2
    VTI = 5

    @property
    def n_independent(self) -> int:
        """The number of independent moduli."""
        return self.value

    def __str__(self) -> str:
        return f"{self.name.lower()} ({self.n_independent} moduli)"
