"""dependent.py -- fields that take one more argument: a frequency, or a time.

A static field is evaluated at points.  A frequency-dependent field is
evaluated at points and an angular frequency `omega`; a time-dependent
one at points and a time `t`.  Everything else about a field -- its
skeleton, character, dimensions, frames, the layers it is defined on --
is unchanged, and everything the library does to a static field is done
to each kind here, once, with the kind a value on the instance rather
than a class of its own:

- a **lift** wraps a static field as one that ignores the argument,
  which is how an elastic layer is expressed beside a viscoelastic one;
- the **algebra** (+, -, unary -, scalar * and /) builds sums and
  scalings of the same kind, lifting a static operand, and refuses to
  mix the two kinds -- passing between frequency and time is a
  transform, never silent;
- **composition**, `fn(arg, *values)`, is the vehicle for a law: the
  function sees the argument as well as the operands' values, so
  constant Q's logarithm in omega is a composed field and nothing more;
- **restriction**, **re-statement on an interval**, **assembly** from a
  body's pieces and **rescaling** go through the operands, exactly as
  the static composites do, because these classes are the same
  `Assemblable` machinery with one argument threaded through;
- a **view** of dependent pieces held by a body's layers is a
  `LayerwiseDependentField`, dispatching each radius to its piece;
- **freezing** at one value of the argument gives back a static Field,
  which is what a push-forward, a sampler, a file or `check_field` sees.

`omega` may be complex -- the Laplace variable of a time-domain code
enters as `s = i omega` -- and a field says where it may lie with
`omega_domain`, "real" or "complex"; `t` is real.  Both are scalars; a
caller loops.  Values of a frequency field are complex; `part` picks
"complex" (the default), "real" or "imag".  A time field's values are
real.

The public names for each kind, with the two protocols and the
convenience functions, are in `frequency.py` and `time.py`.
"""
from __future__ import annotations

import numpy as np

from ..character import Character
from .composite import (_CompositeBase, all_radial,
                        checked_sum_metadata, require_same_skeleton)
from .layerwise import LayerwiseField

__all__ = ["DependentFieldBase", "LiftedField", "SumDependentField",
           "ScaledDependentField", "ComposedDependentField",
           "LayerwiseDependentField", "FrozenField", "lift"]

KINDS = ("frequency", "time")
ARGUMENTS = {"frequency": "omega", "time": "t"}


def kind_of(field) -> str:
    """"static", "frequency" or "time"; a bare protocol object is static."""
    return getattr(field, "kind", "static")


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return kind


def take_part(v, part: str, *, kind: str = "frequency"):
    """Real, imaginary or complex values, float64 for the first two.

    A time field's values are real whatever `part` says, and "imag" is
    refused for it: there is no imaginary part to take.
    """
    v = np.asarray(v)
    if kind == "time":
        if part == "imag":
            raise ValueError("a time-dependent field is real; part='imag' is "
                             "for frequency fields")
        return np.array(np.real(v), dtype=float, order="C")
    if part == "complex":
        return np.array(v, dtype=complex, order="C")
    if part == "imag":
        return np.array(np.imag(v), dtype=float, order="C")
    return np.array(np.real(v), dtype=float, order="C")




class DependentFieldBase(_CompositeBase):
    """A field of one kind, "frequency" or "time", apart from its argument.

    `kind` is set on the instance by each construction; `argument_name`
    is "omega" or "t" accordingly, and `evaluate` takes that keyword.
    The work is `evaluate_with(r, theta, phi, arg, *, layer, side,
    frame)`, the values at one value of the argument, complex for a
    frequency field and real for a time field.  Operands are what
    restriction, re-statement, assembly and rescaling act on, as for
    the static composites this class extends.
    """

    kind: str = "dependent"

    @property
    def argument_name(self) -> str:
        """The keyword `evaluate` takes: "omega" or "t"."""
        return ARGUMENTS[self.kind]

    @property
    def omega_domain(self) -> str:
        """Where omega may lie: "real" or "complex".  Frequency fields only."""
        return "complex"

    def check_argument(self, arg):
        """A scalar, real where the field says so; returns it as a number.

        A complex `omega` with zero imaginary part is real, and is
        returned as a float where the field's form lives on the real
        axis, so a caller working in complex arithmetic is not refused
        for a value that is in the domain.
        """
        a = np.asarray(arg)
        name = self.argument_name
        if a.ndim != 0:
            raise ValueError(
                f"{name} must be a scalar in this version; got shape "
                f"{a.shape} (loop over values)")
        is_complex = np.iscomplexobj(a)
        if self.kind == "time":
            if is_complex and np.imag(a) != 0:
                raise ValueError(f"t must be real, got {arg!r}")
            return float(np.real(a))
        if self.omega_domain == "real":
            if is_complex and np.imag(a) != 0:
                raise ValueError(
                    f"omega must be real for {self!r} (omega_domain='real'): "
                    f"got {arg!r}. This field's form is defined on the real "
                    "frequency axis only")
            return float(np.real(a))
        return complex(a) if is_complex else float(a)


    @property
    def is_radial(self) -> bool:
        """Direction-independent when every operand is."""
        return all_radial(self.operands())

    # -- what subclasses supply -------------------------------------------

    def evaluate_with(self, r, theta, phi, arg, *, layer, side, frame):
        """The values at points and one value of the argument, uncast."""
        raise NotImplementedError

    def _evaluate(self, r, theta, phi, arg, *, layer, side, frame):
        return self.evaluate_with(r, theta, phi, arg, layer=layer, side=side,
                                  frame=frame)

    # -- evaluation ------------------------------------------------------

    def _argument(self, omega, t):
        """The one argument this kind takes, refusing the other by name."""
        name = self.argument_name
        given = {"omega": omega, "t": t}
        other = "t" if name == "omega" else "omega"
        if given[other] is not None:
            raise TypeError(
                f"a {self.kind}-dependent field takes {name}=, not {other}=")
        if given[name] is None:
            raise TypeError(
                f"a {self.kind}-dependent field needs {name}=")
        return self.check_argument(given[name])

    def evaluate(self, r, theta=None, phi=None, *, omega=None, t=None,
                 layer=None, side: str = "upper", frame: str = "spherical",
                 part: str = "complex"):
        """Values at (r, theta, phi) and one `omega` or `t`, in the frame asked.

        A frequency field takes `omega`, a scalar, real or complex as
        `omega_domain` allows, and `part` picks "complex", "real" or
        "imag" of its values; a time field takes `t`, real, and its
        values are real.  `layer`, `side` and `frame` mean what they
        mean for a static field.
        """
        if part not in ("real", "imag", "complex"):
            raise ValueError(f"part must be 'real', 'imag' or 'complex', "
                             f"got {part!r}")
        arg = self._argument(omega, t)
        v = self.evaluate_with(r, theta, phi, arg, layer=layer, side=side,
                               frame=frame)
        return take_part(v, part, kind=self.kind)

    def evaluate_at(self, X, *, omega=None, t=None, layer=None,
                    side: str = "upper", frame: str = "cartesian",
                    part: str = "complex"):
        """Values at Cartesian points X of shape (..., 3), at one argument."""
        from ..frames import spherical_coordinates
        r, theta, phi, _ = spherical_coordinates(X)
        return self.evaluate(r, theta, phi, omega=omega, t=t, layer=layer,
                             side=side, frame=frame, part=part)

    def __call__(self, r, theta=None, phi=None, *, omega=None, t=None):
        """`evaluate` with the defaults, at one `omega` or `t`."""
        return self.evaluate(r, theta, phi, omega=omega, t=t)

    def at(self, arg, *, part: str = "complex") -> "FrozenField":
        """This field at one omega or t, as a static Field."""
        return FrozenField(self, arg, part=part)

    # -- the algebra, closed over the kind --------------------------------

    def _coerce(self, other):
        """An operand of this kind: lifted if static, refused if the other."""
        k = kind_of(other)
        if k == self.kind:
            return other
        if k == "static":
            return LiftedField(other, kind=self.kind)
        raise TypeError(
            f"a {self.kind}-dependent field and a {k}-dependent field do not "
            "combine in one expression: passing between frequency and time "
            "is a transform, and is never applied silently")

    def __add__(self, other):
        if not hasattr(other, "evaluate"):
            return NotImplemented
        return SumDependentField((self, self._coerce(other)))

    __radd__ = __add__

    def __sub__(self, other):
        if not hasattr(other, "evaluate"):
            return NotImplemented
        return SumDependentField((self, ScaledDependentField(
            self._coerce(other), -1.0)))

    def __rsub__(self, other):
        if not hasattr(other, "evaluate"):
            return NotImplemented
        return SumDependentField((self._coerce(other),
                                  ScaledDependentField(self, -1.0)))

    def __neg__(self):
        return ScaledDependentField(self, -1.0)

    def __mul__(self, other):
        if hasattr(other, "evaluate"):
            return NotImplemented
        return ScaledDependentField(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if hasattr(other, "evaluate"):
            return NotImplemented
        return ScaledDependentField(self, 1.0 / other)


# ---------------------------------------------------------------------------
# the four constructions
# ---------------------------------------------------------------------------

class LiftedField(DependentFieldBase):
    """A static field that ignores the argument: an elastic layer's moduli.

    Values are the static field's at every omega or t; for a frequency
    field they are returned complex with zero imaginary part, so the
    `part` machinery sees one kind of number throughout.
    """

    def __init__(self, field, *, kind: str, name: str | None = None) -> None:
        if kind_of(field) != "static":
            raise TypeError(f"only a static field lifts; {field!r} is "
                            f"{kind_of(field)}-dependent")
        self.kind = _check_kind(kind)
        self._field = field
        self.skeleton = field.skeleton
        self.character = field.character
        self.dimensions = getattr(field, "dimensions", None)
        self.name = name if name is not None else field.name

    @property
    def source(self):
        """The static field lifted."""
        return self._field

    def operands(self) -> tuple:
        return (self._field,)

    def rebuilt_from(self, operands, *, name=None):
        return type(self)(operands[0], kind=self.kind, name=name)

    def matches(self, other) -> bool:
        return type(other) is type(self) and other.kind == self.kind

    def evaluate_with(self, r, theta, phi, arg, *, layer, side, frame):
        v = np.asarray(self._field.evaluate(r, theta, phi, layer=layer,
                                            side=side, frame=frame))
        return v.astype(complex) if self.kind == "frequency" else v

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._field!r})"


class SumDependentField(DependentFieldBase):
    """The sum of fields of one kind, skeleton- and character-checked."""

    def __init__(self, terms, *, name: str | None = None) -> None:
        terms = tuple(terms)
        if not terms:
            raise ValueError("a sum needs at least one term")
        kinds = {kind_of(f) for f in terms}
        if len(kinds) != 1 or "static" in kinds:
            raise TypeError(
                f"a dependent sum takes terms of one kind; got {sorted(kinds)}")
        self.kind = kinds.pop()
        self._terms = terms
        self.skeleton = require_same_skeleton(terms, "add")
        self.character, self.dimensions = checked_sum_metadata(terms)
        self.name = name

    @property
    def terms(self):
        """The summed fields, in order."""
        return self._terms

    @property
    def omega_domain(self) -> str:
        return ("real" if any(getattr(f, "omega_domain", "complex") == "real"
                              for f in self._terms) else "complex")

    @property
    def is_radial(self) -> bool:
        return all_radial(self._terms)

    def operands(self) -> tuple:
        return self._terms

    def rebuilt_from(self, operands, *, name=None):
        return SumDependentField(operands, name=name)

    def evaluate_with(self, r, theta, phi, arg, *, layer, side, frame):
        out = None
        for f in self._terms:
            v = f.evaluate_with(r, theta, phi, arg, layer=layer, side=side,
                                frame=frame)
            out = v if out is None else out + v
        return out

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"{type(self).__name__}({len(self._terms)} terms{nm})"


class ScaledDependentField(DependentFieldBase):
    """A field of one kind times a constant (complex allowed for frequency)."""

    def __init__(self, field, factor, *, name: str | None = None) -> None:
        kind = kind_of(field)
        if kind == "static":
            raise TypeError(f"{field!r} is static; scale it with ScaledField")
        self.kind = kind
        self._field = field
        self.factor = (complex(factor) if np.iscomplexobj(factor)
                       else float(factor))
        if kind == "time" and isinstance(self.factor, complex):
            raise ValueError("a time-dependent field is real; the factor "
                             f"{factor!r} is not")
        self.skeleton = field.skeleton
        self.character = field.character
        self.dimensions = getattr(field, "dimensions", None)
        self.name = name

    @property
    def source(self):
        """The field scaled."""
        return self._field

    @property
    def omega_domain(self) -> str:
        return getattr(self._field, "omega_domain", "complex")

    @property
    def is_radial(self) -> bool:
        return bool(getattr(self._field, "is_radial", False))

    def operands(self) -> tuple:
        return (self._field,)

    def rebuilt_from(self, operands, *, name=None):
        return ScaledDependentField(operands[0], self.factor, name=name)

    def matches(self, other) -> bool:
        return type(other) is type(self) and other.factor == self.factor

    def evaluate_with(self, r, theta, phi, arg, *, layer, side, frame):
        return self.factor * self._field.evaluate_with(
            r, theta, phi, arg, layer=layer, side=side, frame=frame)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"{type(self).__name__}({self.factor} x {self._field!r}{nm})"


class ComposedDependentField(DependentFieldBase):
    """fn(arg, *values): the vehicle for a law.

    `fn` receives the argument (omega or t) first and then the operands'
    values at the same points, and returns the components of a field of
    `character`.  Static operands are lifted; a law's parameter fields
    are static almost always.  `kind` is given, or taken from a
    dependent source.  `omega_domain` is declared, since it is a
    property of the formula: constant Q's logarithm lives on the real
    axis, a Prony series anywhere.

    A composed field does not rescale: `fn` closes over the argument,
    whose units are the body's, and over any constant it was written
    with.  A law's field, which knows its constants, overrides that.
    """

    def __init__(self, fn, sources, *, character: Character, kind=None,
                 dimensions=None, name: str | None = None,
                 omega_domain: str = "complex", law=None) -> None:
        sources = tuple(sources)
        if not sources:
            raise ValueError("a composed field needs at least one source")
        if omega_domain not in ("real", "complex"):
            raise ValueError(
                f"omega_domain must be 'real' or 'complex', got {omega_domain!r}")
        found = {kind_of(f) for f in sources} - {"static"}
        if kind is None:
            if len(found) != 1:
                raise ValueError(
                    "give kind='frequency' or kind='time': none of the sources "
                    "is dependent" if not found else
                    f"the sources mix kinds {sorted(found)}")
            kind = found.pop()
        elif found - {kind}:
            raise TypeError(
                f"a {kind}-dependent composition cannot take "
                f"{sorted(found - {kind})}-dependent sources")
        self.kind = _check_kind(kind)
        self._fn = fn
        self._sources = tuple(self._coerce(f) for f in sources)
        self.skeleton = require_same_skeleton(self._sources, "compose")
        self.character = character
        self.dimensions = dimensions
        self.name = name
        self._omega_domain = omega_domain
        #: Provenance, a rheology.LawRecord, where a law built this field;
        #: None for a hand-written composition.  Carried through
        #: restriction, re-statement and assembly.
        self.law = law

    @property
    def sources(self):
        return self._sources

    @property
    def fn(self):
        return self._fn

    @property
    def omega_domain(self) -> str:
        if self._omega_domain == "real":
            return "real"
        return ("real" if any(getattr(f, "omega_domain", "complex") == "real"
                              for f in self._sources) else "complex")

    @property
    def is_radial(self) -> bool:
        return all_radial(self._sources)

    def operands(self) -> tuple:
        return self._sources

    def rebuilt_from(self, operands, *, name=None):
        return type(self)(self._fn, operands, character=self.character,
                          kind=self.kind, dimensions=self.dimensions, name=name,
                          omega_domain=self._omega_domain, law=self.law)

    def matches(self, other) -> bool:
        return (type(other) is type(self) and other._fn is self._fn
                and len(other._sources) == len(self._sources)
                and other.law == self.law)

    def rescaled(self, convert, old, new):
        raise TypeError(
            f"cannot rescale {type(self).__name__} {self.name!r}: its formula "
            "closes over the argument and any constants in the old units (a "
            "reference period, a relaxation time) that this body cannot "
            "re-express. Rebuild it after the rescale.")

    def evaluate_with(self, r, theta, phi, arg, *, layer, side, frame):
        vals = [f.evaluate_with(r, theta, phi, arg, layer=layer, side=side,
                                frame=frame) for f in self._sources]
        out = np.asarray(self._fn(arg, *vals))
        return out.astype(complex) if self.kind == "frequency" else out

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        srcs = ", ".join(f.name or "?" for f in self._sources)
        return f"{type(self).__name__}({srcs}{nm})"


class LayerwiseDependentField(DependentFieldBase, LayerwiseField):
    """The generic view of dependent pieces: each radius to its piece.

    What `assemble` builds when a body's layers hold frequency- or
    time-dependent fields of no one reassemblable type -- an elastic
    lithosphere lifted beside a Maxwell mantle.  The pieces' kind is
    the view's, and their `omega_domain` the narrower of theirs.
    """

    def __init__(self, skeleton, pieces, *, name: str | None = None) -> None:
        LayerwiseField.__init__(self, skeleton, pieces, name=name)
        kinds = {kind_of(p) for p in self.pieces}
        if len(kinds) != 1 or "static" in kinds:
            raise ValueError(
                f"{self!r} mixes kinds {sorted(kinds)}: a view is one kind "
                "of field, so lift the static pieces first")
        self.kind = kinds.pop()

    @property
    def omega_domain(self) -> str:
        return ("real" if any(getattr(p, "omega_domain", "complex") == "real"
                              for p in self.pieces) else "complex")

    domain = LayerwiseField.domain
    is_radial = LayerwiseField.is_radial
    restricted = LayerwiseField.restricted
    on_interval = LayerwiseField.on_interval
    rescaled = LayerwiseField.rescaled
    integrate = LayerwiseField.integrate

    def operands(self) -> tuple:
        return self.pieces

    def rebuilt_from(self, operands, *, name=None):
        from .layerwise import assemble
        return assemble(self.skeleton, operands, name=name)

    @classmethod
    def assembled(cls, skeleton, pieces, *, name=None):
        return NotImplemented

    def evaluate_with(self, r, theta, phi, arg, *, layer, side, frame):
        return self._dispatch(r, theta, phi, layer=layer, side=side,
                              frame=frame, arg=arg)


# ---------------------------------------------------------------------------
# freezing: back to a static field at one value of the argument
# ---------------------------------------------------------------------------

class FrozenField(_CompositeBase):
    """A dependent field at one value of its argument: a static Field.

    What a push-forward, a sampler, a file and `check_field` see.  For
    a frequency field `part` picks "complex" (the default: the complex
    tensor is the object), "real" or "imag"; the last two give float64
    values.  A time field's values are real and its `part` is "real".
    The layers and `is_radial` are the source's.
    """

    kind = "static"

    def __init__(self, field, arg, *, part: str = "complex",
                 name: str | None = None) -> None:
        if part not in ("real", "imag", "complex"):
            raise ValueError(f"part must be 'real', 'imag' or 'complex', "
                             f"got {part!r}")
        if field.kind == "time":
            if part == "imag":
                raise ValueError("a time-dependent field is real; part='imag' "
                                 "is for frequency fields")
            part = "real"
        self._field = field
        self._arg = field.check_argument(arg)
        self.part = part
        self.skeleton = field.skeleton
        self.character = field.character
        self.dimensions = getattr(field, "dimensions", None)
        src = field.name
        self.name = name if name is not None else (
            f"{src}@{field.argument_name}={self._arg:.6g}" if src else None)

    @property
    def source(self):
        """The dependent field frozen."""
        return self._field

    @property
    def argument(self):
        """The omega or t this field is frozen at."""
        return self._arg

    @property
    def is_radial(self) -> bool:
        return bool(getattr(self._field, "is_radial", False))

    @property
    def domain(self) -> tuple[int, ...]:
        return self._field.domain

    def operands(self) -> tuple:
        return (self._field,)

    def rebuilt_from(self, operands, *, name=None):
        return FrozenField(operands[0], self._arg, part=self.part,
                           name=self.name if name is None else name)

    def matches(self, other) -> bool:
        return (isinstance(other, FrozenField) and other.part == self.part
                and other._arg == self._arg)

    def rescaled(self, convert, old, new):
        raise TypeError(
            f"cannot rescale FrozenField {self.name!r}: the "
            f"{self._field.argument_name} "
            "it is frozen at is in the old units. Freeze the rescaled field "
            "instead.")

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical"):
        v = self._field.evaluate_with(r, theta, phi, self._arg, layer=layer,
                                      side=side, frame=frame)
        return take_part(v, self.part, kind=self._field.kind)

    def __repr__(self) -> str:
        return (f"FrozenField({self._field!r} at "
                f"{self._field.argument_name}={self._arg})")


def lift(field, *, kind: str):
    """A static field as a frequency- or time-dependent one; a field of
    that kind already is returned unchanged."""
    if kind_of(field) == kind:
        return field
    return LiftedField(field, kind=kind)
