"""character.py -- how a field transforms, and how elastic symmetry is named.

A field is numbers plus a Character = (rank, weight), and push-forward is
one generic operation driven by that character alone.  For a mapping
m : M_ref -> M_phys with F = (grad m)^T and J = det F, a contravariant
field of rank n and weight 1 transforms as

    T_phys,{i1..in}(x) = (1/J) F_{i1 m1} ... F_{in mn} T_ref,{m1..mn}(X)

with x = m(X), while weight-0 fields are simply composed.  In particular
rho_phys = rho_ref / J, the point at which Al-Attar & Crawford (2016)
eq. (71) and Myhill, Maitra & Al-Attar (2026) agree.

    field                                        rank  weight
    rho                                             0       1
    elastic tensor A                                4       1
    equilibrium stress, stress glut                 2       1
    Q_kappa, Q_mu, eta, relaxation times, viscosity 0       0

The rheological entries are weight 0 **on the assumption that the tensor
character of the constitutive law is carried entirely by the elastic
tensor**: a scalar relaxation time modulating an elastic tensor is
invariant, whereas a genuine viscosity *tensor* would be rank 4 weight 1.
Such parameters are inherently referential and scalar; they compose with
the mapping and never wrap.

Velocities have no transformation law at all.  v_p and v_s are built
from rho and A, neither of which is invariant, and the combination is a
tensor of no weight -- which is why moduli are canonical here and
velocities are derived views.

Symmetry classifies **second** elasticity tensors only: those with the
full minor and major symmetries, for which Voigt 6x6 is faithful and the
2/5/9/21 counting means something.  The *first* elasticity tensors that
referential weak forms consume keep only the major symmetry, so they
have no Voigt form at all -- which is what the Character's `voigt` flag
records, and why FIRST_ELASTIC = Character(4, 1, voigt=False) is a
different character from ELASTIC and will not be added to it.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from enum import Enum

__all__ = [
    "Character", "Symmetry",
    "SCALAR", "DENSITY", "VECTOR", "STRESS", "ELASTIC", "FIRST_ELASTIC",
]


@dataclass(frozen=True)
class Character:
    """The transformation law of a field: (rank, weight), and Voigt or not.

    `voigt` says whether a rank-2 or rank-4 value carries the symmetries
    that make the Voigt reduction faithful.  It is True by default,
    which is the stress and the second elasticity tensor; the one thing
    it is False for is the first elasticity tensor, whose four slots
    keep only the major symmetry.  It participates in equality, so a
    first elasticity field and a second one have different characters
    and SumField refuses to add them -- the type-level statement that
    they are different objects.
    """

    rank: int
    weight: int
    _: KW_ONLY
    voigt: bool = True

    def __post_init__(self) -> None:
        """Reject characters push_forward could not act on."""
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
        says the symmetries are there; everything else -- ranks 0 and 1,
        and the first elasticity tensor -- keeps its full component
        shape, and a caller reading this must fall back to it.
        """
        if not self.voigt:
            return None
        return {2: (6,), 4: (6, 6)}.get(self.rank)

    @property
    def is_invariant(self) -> bool:
        """Whether the field merely composes with the mapping."""
        return self.rank == 0 and self.weight == 0

    def __str__(self) -> str:
        tail = "" if self.voigt else ", no Voigt form"
        return f"rank {self.rank}, weight {self.weight}{tail}"


SCALAR = Character(0, 0)    # Q_kappa, Q_mu, eta, relaxation times, viscosity
DENSITY = Character(0, 1)
VECTOR = Character(1, 0)
STRESS = Character(2, 1)    # equilibrium stress, stress glut
ELASTIC = Character(4, 1)
#: The first elasticity tensor: rank 4, weight 1, major symmetry only.
FIRST_ELASTIC = Character(4, 1, voigt=False)


class Symmetry(Enum):
    """Material symmetry of a second elasticity tensor, by moduli count."""

    ISOTROPIC = 2
    VTI = 5
    ORTHOTROPIC = 9
    GENERAL = 21

    @property
    def n_independent(self) -> int:
        """The number of independent moduli."""
        return self.value

    def promote(self, other: "Symmetry") -> "Symmetry":
        """The least symmetry class containing both.

        The four classes shipped here form a chain, ISOTROPIC < VTI <
        ORTHOTROPIC < GENERAL, so the join is simply the wider of the
        two.  Genuinely incomparable classes (trigonal against
        tetragonal, say) are not represented; adding one means replacing
        this with a real lattice join rather than extending the chain.
        """
        if not isinstance(other, Symmetry):
            raise TypeError(f"expected a Symmetry, got {type(other).__name__}")
        return self if self.n_independent >= other.n_independent else other

    def __str__(self) -> str:
        return f"{self.name.lower()} ({self.n_independent} moduli)"
