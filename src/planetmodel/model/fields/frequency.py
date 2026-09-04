"""frequency.py -- fields that depend on frequency: `evaluate(..., omega=)`.

A frequency-dependent field is a Field with one more keyword, `omega`,
the angular frequency in the body's own units -- rad/s while the body's
scales are SI, and the dimensionless value through the time scale after
`nondimensionalised()`.  It may be complex: the Laplace variable of a
time-domain code enters as `s = i omega`, so a GIA code evaluates off
the real axis and a seismological one on it.  A field says where omega
may lie with `omega_domain`, "real" (constant Q's logarithm) or
"complex" (a Prony series, a lifted static field), and refuses by name
outside it.

Values are complex, and the complex tensor is the object: `part` picks
"complex" (the default, complex128), "real" or "imag" (float64) for a
caller that wants one part.  The character is unchanged, and since the
push-forward is linear a complex tensor pushes forward as its two real
parts do.

**Time convention.**  The harmonic factor is exp(+i omega t): a lossy
modulus has a positive imaginary part on the positive real omega axis,
and the Laplace variable of a relaxation law is s = i omega.  A
consumer working with exp(-i omega t) conjugates.

    lifted_to_frequency(field)      a static field that ignores omega
    ComposedFrequencyField(fn, ...) fn(omega, *values): a law
    at_frequency(field, omega)      the field at one omega, as a static Field

Sums and scalings come from the algebra; a static operand is lifted; a
time-dependent one is refused.  See `dependent.py` for the one
implementation the two kinds share and `time.py` for the other kind.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .dependent import ComposedDependentField, FrozenField, LiftedField

__all__ = ["FrequencyDependentField", "LiftedFrequencyField",
           "ComposedFrequencyField", "lifted_to_frequency", "at_frequency"]


@runtime_checkable
class FrequencyDependentField(Protocol):
    """Values of a Character on a Skeleton, at points and a frequency."""

    skeleton: object
    character: object
    name: str | None
    kind: str                 # "frequency"
    omega_domain: str         # "real" | "complex"

    def evaluate(self, r, theta=None, phi=None, *, omega, layer=None,
                 side: str = "upper", frame: str = "spherical",
                 part: str = "complex"):
        """Values at broadcast (r, theta, phi) and one omega."""
        ...


class LiftedFrequencyField(LiftedField):
    """A static field at every omega."""

    def __init__(self, field, *, kind: str = "frequency",
                 name: str | None = None) -> None:
        if kind != "frequency":
            raise ValueError(f"a LiftedFrequencyField is of kind 'frequency', "
                             f"not {kind!r}")
        super().__init__(field, kind=kind, name=name)


class ComposedFrequencyField(ComposedDependentField):
    """fn(omega, *values): a law applied to its parameter fields."""

    def __init__(self, fn, sources, *, character, kind: str = "frequency",
                 dimensions=None, name: str | None = None,
                 omega_domain: str = "complex", law=None) -> None:
        if kind != "frequency":
            raise ValueError(f"a ComposedFrequencyField is of kind "
                             f"'frequency', not {kind!r}")
        super().__init__(fn, sources, character=character, kind=kind,
                         dimensions=dimensions, name=name,
                         omega_domain=omega_domain, law=law)


def lifted_to_frequency(field, *, name: str | None = None):
    """A static field as a frequency-dependent one that ignores omega.

    An elastic layer beside a viscoelastic one is expressed exactly so:
    not as a label, but as a contribution that does not depend on
    frequency.  A field that is already frequency-dependent is returned
    unchanged.
    """
    if getattr(field, "kind", "static") == "frequency":
        return field
    return LiftedFrequencyField(field, name=name)


def at_frequency(field, omega, *, part: str = "complex") -> FrozenField:
    """A frequency-dependent field at one omega, as a static Field.

    What a push-forward, a sample, a file or `check_field` sees: the
    complex values by default, "real" or "imag" for one part as float64.
    """
    return FrozenField(field, omega, part=part)
