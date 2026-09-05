"""Physical dimensions, and the scales that remove them.

Data arrives with known dimensions, is non-dimensionalised once into
O(1) numbers, is worked with in that form, and is re-dimensionalised on
output.  The two pieces here make each step checkable rather than
remembered.  Nothing inside a field or a mesh consults them: a model
holds a `Scales` and a table of dimensions by name, and conversion is
the model's operation.

**Dimensions** are physical: integer exponents of (mass, length, time).
They are independent of a field's character, which is its law under a
mapping: Q_kappa and a relaxation time are both invariant, yet one is
dimensionless and the other is seconds.  Three base dimensions cover
everything held here; temperature can be added as a fourth exponent
defaulting to zero without disturbing anything.

**Scales** say what one unit of each base dimension is, in SI.  A
quantity of dimensions d stored under scales s relates to its SI value
by

    stored = SI / s.factor(d),   s.factor(d) = mass^d.mass length^d.length
                                              time^d.time

so `Scales.SI` is the identity.  `Scales.geophysical(a, density=rho)`
takes the length a and the mass rho a^3 and fixes the time by
G rho T^2 = 1, so that G is one in those units: factor(G's dimensions)
is G_SI to machine precision, the time scale passing through a square
root.  A solver that takes G as a parameter is handed
`G_SI / scales.factor(GRAVITATIONAL_CONSTANT)`.

The dimension constants are module names.  `DENSITY` here is the
dimensions kg m^-3 and `DENSITY` in `planetmodel.character` is a
transformation law; a module using both refers to one of them through
its module.
"""
from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

__all__ = [
    "Dimensions", "Scales", "unit_string", "G_SI", "EARTH_MEAN_DENSITY",
    "DIMENSIONLESS", "MASS", "LENGTH", "TIME", "DENSITY", "VELOCITY",
    "GRAVITY", "MODULUS", "VISCOSITY", "GRAVITATIONAL_CONSTANT", "FREQUENCY",
]

#: The gravitational constant, m^3 kg^-1 s^-2 (CODATA 2018).  The one
#: definition in the package; everything that needs G reads it here or
#: from a model's constants.
G_SI = 6.6743e-11

#: The conventional mean density of the Earth, kg m^-3.  Prescribed, not
#: computed: a calculated mean would shift with every crustal edit.
EARTH_MEAN_DENSITY = 5515.0


@dataclass(frozen=True)
class Dimensions:
    """Physical dimensions as integer exponents of (mass, length, time).

    Multiplying quantities adds exponents, so
    `MODULUS == DENSITY * VELOCITY**2` holds here as it does on paper.
    """

    _: KW_ONLY
    mass: int = 0
    length: int = 0
    time: int = 0

    def __post_init__(self) -> None:
        for name in ("mass", "length", "time"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name} exponent must be an integer, got {value!r}")

    @property
    def is_dimensionless(self) -> bool:
        return self.mass == 0 and self.length == 0 and self.time == 0

    def __mul__(self, other: Dimensions) -> Dimensions:
        if not isinstance(other, Dimensions):
            return NotImplemented
        return Dimensions(mass=self.mass + other.mass,
                          length=self.length + other.length,
                          time=self.time + other.time)

    def __truediv__(self, other: Dimensions) -> Dimensions:
        if not isinstance(other, Dimensions):
            return NotImplemented
        return Dimensions(mass=self.mass - other.mass,
                          length=self.length - other.length,
                          time=self.time - other.time)

    def __pow__(self, n: int) -> Dimensions:
        if not isinstance(n, int) or isinstance(n, bool):
            return NotImplemented
        return Dimensions(mass=self.mass * n, length=self.length * n,
                          time=self.time * n)

    def unit_string(self, *, si: bool = True) -> str:
        """A udunits-style unit string: "kg m-3" in SI, "1" otherwise.

        A non-dimensional model's numbers are pure whatever their
        physical dimensions, so `si=False` gives "1" for every quantity.
        """
        if not si:
            return "1"
        parts = []
        for symbol, n in (("kg", self.mass), ("m", self.length), ("s", self.time)):
            if n == 1:
                parts.append(symbol)
            elif n != 0:
                parts.append(f"{symbol}{n}")
        return " ".join(parts) if parts else "1"

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "dimensionless"
        return " ".join(f"{sym}^{e}" for sym, e in
                        (("M", self.mass), ("L", self.length), ("T", self.time))
                        if e != 0)


def unit_string(dims: Dimensions | None, *, si: bool = True) -> str:
    """`dims.unit_string(si=si)`, or "unknown" where no dimensions are declared."""
    return "unknown" if dims is None else dims.unit_string(si=si)


DIMENSIONLESS = Dimensions()
MASS = Dimensions(mass=1)
LENGTH = Dimensions(length=1)
TIME = Dimensions(time=1)
FREQUENCY = Dimensions(time=-1)
DENSITY = Dimensions(mass=1, length=-3)
VELOCITY = Dimensions(length=1, time=-1)
GRAVITY = Dimensions(length=1, time=-2)
MODULUS = Dimensions(mass=1, length=-1, time=-2)
VISCOSITY = Dimensions(mass=1, length=-1, time=-1)
GRAVITATIONAL_CONSTANT = Dimensions(mass=-1, length=3, time=-2)


@dataclass(frozen=True)
class Scales:
    """What one unit of each base dimension is, in SI.

    A quantity of dimensions d stored under these scales relates to its
    SI value by stored = SI / factor(d).  `Scales.SI` is the identity and
    the default everywhere: a model is SI until it says otherwise.
    """

    _: KW_ONLY
    length: float = 1.0
    mass: float = 1.0
    time: float = 1.0

    SI: ClassVar[Scales]

    def __post_init__(self) -> None:
        for name in ("length", "mass", "time"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value)
                    and value > 0.0):
                raise ValueError(
                    f"{name} scale must be a positive finite number, "
                    f"got {value!r}")
            object.__setattr__(self, name, float(value))

    @classmethod
    def geophysical(cls, length: float,
                    *, density: float = EARTH_MEAN_DENSITY) -> Scales:
        """Scales for self-gravitating problems: G becomes one.

        `length` is the body's radius in metres and `density` a
        prescribed reference density in kg m^-3, conventionally the
        Earth's mean.  The mass scale is density length^3 and the time
        scale is fixed by G density T^2 = 1.
        """
        length = float(length)
        density = float(density)
        if length <= 0.0 or density <= 0.0:
            raise ValueError(
                f"length and density must be positive, got {length}, {density}")
        return cls(length=length, mass=density * length ** 3,
                   time=1.0 / math.sqrt(G_SI * density))

    @property
    def is_si(self) -> bool:
        return self.length == 1.0 and self.mass == 1.0 and self.time == 1.0

    def factor(self, dims: Dimensions) -> float:
        """The SI size of one stored unit of a quantity with `dims`."""
        return (self.mass ** dims.mass * self.length ** dims.length
                * self.time ** dims.time)

    def __repr__(self) -> str:
        if self.is_si:
            return "Scales.SI"
        return (f"Scales(length={self.length:.6g}, mass={self.mass:.6g}, "
                f"time={self.time:.6g})")


Scales.SI = Scales()
