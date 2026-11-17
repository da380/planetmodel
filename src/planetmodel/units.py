"""Physical dimensions, and the scales that remove them.

Data arrives with known dimensions, is non-dimensionalised once into
O(1) numbers, is worked with in that form, and is re-dimensionalised on
output.  The two pieces here make each step checkable rather than
remembered.  Nothing inside a field or a mesh consults them: a model
holds a `Scales` and a table of dimensions by name, and conversion is
the model's operation.

**Dimensions** are physical: integer exponents of (mass, length, time,
temperature).  They are independent of a field's character, which is
its law under a mapping: Q_kappa and a relaxation time are both
invariant, yet one is dimensionless and the other is seconds.

**Scales** say what one unit of each base dimension is, in SI.  A
quantity of dimensions d stored under scales s relates to its SI value
by ::

    stored = SI / s.factor(d),   s.factor(d) = mass^d.mass length^d.length
                                              time^d.time
                                              temperature^d.temperature

so `Scales.SI` is the identity.  `Scales.geophysical(a, density=rho)`
takes the length a and the mass rho a^3 and fixes the time by
G rho T^2 = 1, so that G is one in those units: factor(G's dimensions)
is G_SI to machine precision, the time scale passing through a square
root; the temperature scale is a kelvin unless it is given.  A solver
that takes G as a parameter is handed
`G_SI / scales.factor(GRAVITATIONAL_CONSTANT)`.

The dimension constants are module names.  `DENSITY` here is the
dimensions kg m^-3 and `DENSITY` in `planetmodel.character` is a
transformation law; a module using both refers to one of them through
its module.

**Unit strings of other sources.**  `parse_unit(text)` reads the unit a
published file states, "km.s-1", "kilometers/second", "g/cm^3", "%",
into its dimensions and the SI size of one of it, so that a value v in
that unit is v times the factor in SI.  It is a small grammar, not a
table of whole strings: symbols (m, g, s, K, Pa, N, J, W, Hz, min, h,
day, yr, bar, cc, %, deg, rad) with an SI prefix where one applies,
products by ".", "*" or a space, a quotient of the next factor by "/",
powers by "^n", `"**n"` or a trailing integer ("s-1", "m3"), numbers as
factors (so that `scaled_unit`'s own strings read back), the spelled-out
words sources use (kilometer, meters, second, sec, gram, per, percent,
degrees), and the words they use for a pure number (1, count, ratio,
dimensionless).  What it cannot express is refused by name, never
guessed: an empty string, a unit with an offset from zero (Celsius,
Fahrenheit), a logarithm, a symbol it does not know (electric units
among them, there being no base dimension of current), a non-positive
factor.  A source that states its unit wrongly is corrected where the
source is described, not here.
"""
from __future__ import annotations

import math
import numbers
import re
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

__all__ = [
    "Dimensions", "Scales", "unit_string", "scaled_unit", "parse_unit", "G_SI",
    "EARTH_MEAN_DENSITY",
    "DIMENSIONLESS", "MASS", "LENGTH", "TIME", "DENSITY", "VELOCITY",
    "GRAVITY", "MODULUS", "VISCOSITY", "GRAVITATIONAL_CONSTANT", "FREQUENCY",
    "TEMPERATURE", "BASE",
]

#: The base dimensions, in the order of every exponent tuple and scale.
BASE = ("mass", "length", "time", "temperature")

#: The gravitational constant, m^3 kg^-1 s^-2 (CODATA 2018).  The one
#: definition in the package; everything that needs G reads it here or
#: from a model's constants.
G_SI = 6.6743e-11

#: The conventional mean density of the Earth, kg m^-3.  Prescribed, not
#: computed: a calculated mean would shift with every crustal edit.
EARTH_MEAN_DENSITY = 5515.0


@dataclass(frozen=True)
class Dimensions:
    """Physical dimensions as integer exponents of (mass, length, time,
    temperature).

    Multiplying quantities adds exponents, so
    `MODULUS == DENSITY * VELOCITY**2` holds here as it does on paper.
    """

    _: KW_ONLY
    mass: int = 0
    length: int = 0
    time: int = 0
    temperature: int = 0

    def __post_init__(self) -> None:
        for name in BASE:
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError(
                    f"{name} exponent must be an integer, got {value!r}")
            object.__setattr__(self, name, int(value))

    @property
    def is_dimensionless(self) -> bool:
        """Whether every exponent is zero."""
        return all(getattr(self, name) == 0 for name in BASE)

    def __mul__(self, other: Dimensions) -> Dimensions:
        if not isinstance(other, Dimensions):
            return NotImplemented
        return Dimensions(**{n: getattr(self, n) + getattr(other, n) for n in BASE})

    def __truediv__(self, other: Dimensions) -> Dimensions:
        if not isinstance(other, Dimensions):
            return NotImplemented
        return Dimensions(**{n: getattr(self, n) - getattr(other, n) for n in BASE})

    def __pow__(self, n: int) -> Dimensions:
        if not isinstance(n, numbers.Integral) or isinstance(n, bool):
            return NotImplemented
        return Dimensions(**{b: getattr(self, b) * int(n) for b in BASE})

    def unit_string(self, *, si: bool = True) -> str:
        """A udunits-style unit string: "kg m-3" in SI, "1" otherwise.

        A non-dimensional model's numbers are pure whatever their
        physical dimensions, so `si=False` gives "1" for every quantity.
        """
        if not si:
            return "1"
        parts = []
        for symbol, n in (("kg", self.mass), ("m", self.length), ("s", self.time),
                          ("K", self.temperature)):
            if n == 1:
                parts.append(symbol)
            elif n != 0:
                parts.append(f"{symbol}{n}")
        return " ".join(parts) if parts else "1"

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "dimensionless"
        return " ".join(f"{sym}^{e}" for sym, e in
                        (("M", self.mass), ("L", self.length), ("T", self.time),
                         ("Theta", self.temperature))
                        if e != 0)


def unit_string(dims: Dimensions | None, *, si: bool = True) -> str:
    """`dims.unit_string(si=si)`, or "unknown" where no dimensions are declared."""
    return "unknown" if dims is None else dims.unit_string(si=si)


def scaled_unit(dims: Dimensions | None, scales: Scales) -> str:
    """The stored unit as a UDUNITS scaled unit: the SI unit, preceded by
    the size of one stored unit when the scales make it other than one
    ("m s-1" in SI, "3865.300165 m s-1" under the geophysical scales of
    the Earth); "1" for a
    pure number and "unknown" where no dimensions are declared."""
    if dims is None:
        return "unknown"
    si = dims.unit_string()
    factor = scales.factor(dims)
    if factor == 1.0:
        return si
    return f"{factor:.10g}" if si == "1" else f"{factor:.10g} {si}"


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
TEMPERATURE = Dimensions(temperature=1)


# -- unit strings ---------------------------------------------------------------

#: Each symbol `parse_unit` reads: its dimensions and the SI size of one.
_SYMBOLS: dict[str, tuple[Dimensions, float]] = {
    "m": (LENGTH, 1.0),
    "g": (MASS, 1e-3),
    "s": (TIME, 1.0),
    "K": (TEMPERATURE, 1.0),
    "Pa": (MODULUS, 1.0),
    "N": (MASS * LENGTH / TIME ** 2, 1.0),
    "J": (MASS * LENGTH ** 2 / TIME ** 2, 1.0),
    "W": (MASS * LENGTH ** 2 / TIME ** 3, 1.0),
    "Hz": (FREQUENCY, 1.0),
    "min": (TIME, 60.0),
    "h": (TIME, 3600.0),
    "day": (TIME, 86400.0),
    "yr": (TIME, 31557600.0),                      # the Julian year
    "bar": (MODULUS, 1e5),
    "cc": (LENGTH ** 3, 1e-6),
    "%": (DIMENSIONLESS, 0.01),
    "deg": (DIMENSIONLESS, math.pi / 180.0),
    "rad": (DIMENSIONLESS, 1.0),
}

#: The symbols that take an SI prefix.
_PREFIXABLE = frozenset({"m", "g", "s", "K", "Pa", "N", "J", "W", "Hz", "yr", "bar"})

#: The SI prefixes `parse_unit` reads, with both forms of the micro sign.
_PREFIXES = {"G": 1e9, "M": 1e6, "k": 1e3, "h": 1e2, "c": 1e-2, "m": 1e-3,
             "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "n": 1e-9}

#: Spelled-out words, in any case, and the symbols they stand for.
_WORDS = {
    **dict.fromkeys(("kilometer", "kilometers", "kilometre", "kilometres"), "km"),
    **dict.fromkeys(("meter", "meters", "metre", "metres"), "m"),
    **dict.fromkeys(("centimeter", "centimeters", "centimetre", "centimetres"), "cm"),
    **dict.fromkeys(("millimeter", "millimeters", "millimetre", "millimetres"), "mm"),
    **dict.fromkeys(("second", "seconds", "sec", "secs"), "s"),
    **dict.fromkeys(("minute", "minutes"), "min"),
    **dict.fromkeys(("hour", "hours"), "h"),
    **dict.fromkeys(("day", "days"), "day"),
    **dict.fromkeys(("year", "years"), "yr"),
    **dict.fromkeys(("gram", "grams"), "g"),
    **dict.fromkeys(("kilogram", "kilograms"), "kg"),
    "kelvin": "K",
    **dict.fromkeys(("pascal", "pascals"), "Pa"),
    **dict.fromkeys(("gigapascal", "gigapascals"), "GPa"),
    **dict.fromkeys(("megapascal", "megapascals"), "MPa"),
    "percent": "%",
    **dict.fromkeys(("degree", "degrees"), "deg"),
    **dict.fromkeys(("radian", "radians"), "rad"),
    "per": "/",
}

#: Whole strings, in any case, that are a unit in one word.
_WHOLE = {
    **dict.fromkeys(("1", "count", "ratio", "dimensionless", "unitless"),
                    (DIMENSIONLESS, 1.0)),
    "kmps": (VELOCITY, 1e3),
}

#: Whole strings, in any case, of units with an offset from zero.
_OFFSETS = frozenset({
    "c", "degc", "deg c", "degree c", "degrees c", "celsius", "degree celsius",
    "degrees celsius", "°c", "℃", "f", "degf", "deg f", "degree f",
    "degrees f", "fahrenheit", "degrees fahrenheit", "°f",
})

#: One token of a unit string: a number, a symbol with its power, or an operator.
_TOKEN = re.compile(r"""\s*(?:
      (?P<number>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)(?![A-Za-z%µμ])
    | (?P<symbol>[A-Za-z%µμ]+)
      (?:\s*(?:\^|\*\*)\s*(?P<caret>[+-]?\d+)|(?P<power>[+-]?\d+))?
    | (?P<op>[/.*])
    )""", re.VERBOSE)


def _symbol(name: str, text: str) -> tuple[Dimensions, float]:
    """The dimensions and SI size of one symbol, prefixed or not."""
    if name in _SYMBOLS:
        return _SYMBOLS[name]
    prefix, rest = name[:1], name[1:]
    if prefix in _PREFIXES and rest in _PREFIXABLE:
        dims, size = _SYMBOLS[rest]
        return dims, _PREFIXES[prefix] * size
    raise ValueError(
        f"{name!r} in {text!r} is not a unit planetmodel reads (electric units "
        "among them: its dimensions have no base of current); the data set's "
        "entry must state the unit")


def parse_unit(text: str) -> tuple[Dimensions, float]:
    """The dimensions of a unit string and the SI size of one of its unit.

    A value v stated in `text` is v times the factor in SI: "km.s-1"
    gives (VELOCITY, 1000.0), "g/cm^3" (DENSITY, 1000.0), "%"
    (DIMENSIONLESS, 0.01).  The grammar is the module docstring's; an
    empty string, an offset, a logarithm, an unknown symbol and a
    non-positive factor are refused by name.
    """
    s = " ".join(str(text).split())
    low = s.lower()
    if not s:
        raise ValueError("an empty unit string says nothing of its unit; the data "
                         "set's entry must state it")
    if low in _OFFSETS:
        raise ValueError(
            f"{text!r} has an offset from zero (a temperature in Celsius or "
            "Fahrenheit); planetmodel's scales are factors alone, so the values "
            "must be stated in kelvin")
    if re.match(r"(log|ln)\w*\s*[\[(]", low):
        raise ValueError(f"{text!r} is the logarithm of a quantity, not a unit")
    if low in _WHOLE:
        return _WHOLE[low]
    s = re.sub(r"[A-Za-z]+", lambda w: _WORDS.get(w.group(0).lower(), w.group(0)), s)
    dims, factor = DIMENSIONLESS, 1.0
    sign, expect_term, pos = 1, True, 0
    while pos < len(s):
        match = _TOKEN.match(s, pos)
        if match is None or match.end() == pos:
            raise ValueError(f"cannot read {text!r} as a unit at {s[pos:]!r}")
        pos = match.end()
        if match["op"] is not None:
            if expect_term:
                raise ValueError(f"cannot read {text!r} as a unit: {match['op']!r} "
                                 "follows no factor")
            sign, expect_term = (-1 if match["op"] == "/" else 1), True
            continue
        if match["number"] is not None:
            factor *= float(match["number"]) ** sign
        else:
            d, size = _symbol(match["symbol"], text)
            n = int(match["caret"] or match["power"] or 1) * sign
            dims, factor = dims * d ** n, factor * size ** n
        sign, expect_term = 1, False
    if expect_term:
        raise ValueError(f"cannot read {text!r} as a unit: it ends in an operator")
    if not factor > 0.0:
        raise ValueError(f"{text!r} has the factor {factor:g}; a unit's factor is "
                         "positive (a misplaced separator?)")
    return dims, factor


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
    temperature: float = 1.0

    SI: ClassVar[Scales]

    def __post_init__(self) -> None:
        for name in ("length", "mass", "time", "temperature"):
            value = getattr(self, name)
            if not (isinstance(value, numbers.Real) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0.0):
                raise ValueError(
                    f"{name} scale must be a positive finite number, "
                    f"got {value!r}")
            object.__setattr__(self, name, float(value))

    @classmethod
    def geophysical(cls, length: float, *, density: float = EARTH_MEAN_DENSITY,
                    temperature: float = 1.0) -> Scales:
        """Scales for self-gravitating problems: G becomes one.

        `length` is the body's radius in metres and `density` a
        prescribed reference density in kg m^-3, conventionally the
        Earth's mean.  The mass scale is density length^3 and the time
        scale is fixed by G density T^2 = 1; `temperature` is the
        temperature scale in kelvin, one by default.
        """
        length = float(length)
        density = float(density)
        if length <= 0.0 or density <= 0.0:
            raise ValueError(
                f"length and density must be positive, got {length}, {density}")
        return cls(length=length, mass=density * length ** 3,
                   time=1.0 / math.sqrt(G_SI * density), temperature=temperature)

    @property
    def is_si(self) -> bool:
        """Whether one stored unit of each base dimension is one SI unit."""
        return (self.length == 1.0 and self.mass == 1.0 and self.time == 1.0
                and self.temperature == 1.0)

    def factor(self, dims: Dimensions) -> float:
        """The SI size of one stored unit of a quantity with `dims`."""
        return (self.mass ** dims.mass * self.length ** dims.length
                * self.time ** dims.time * self.temperature ** dims.temperature)

    def __repr__(self) -> str:
        if self.is_si:
            return "Scales.SI"
        kelvin = ("" if self.temperature == 1.0
                  else f", temperature={self.temperature:.6g}")
        return (f"Scales(length={self.length:.6g}, mass={self.mass:.6g}, "
                f"time={self.time:.6g}{kelvin})")


Scales.SI = Scales()
