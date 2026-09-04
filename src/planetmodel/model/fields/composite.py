"""composite.py -- fields built lazily from other fields.

Three ways to combine fields, all evaluated on demand rather than
sampled and refitted:

  SumField        f + g, character- and skeleton-checked
  ScaledField     c * f
  ComposedField   fn(f1, f2, ...), an arbitrary pointwise function
  RestrictedField f on one layer, on a skeleton holding only that layer

ComposedField is the vehicle for derived views.  Velocities, for
instance, are sqrt(A / rho): exact at every point it is asked about, but
not a polynomial, so it cannot be integrated exactly.  Sampling and
refitting would hide that by producing a plausible spline; composing
lazily keeps the model honest, with `integrate` documented as
approximate and implemented by quadrature.

Character arithmetic is deliberately strict.  Adding a density to a
shear modulus is a mistake worth an exception, not a broadcast, so
SumField requires equal characters.  Scaling leaves the character alone.
ComposedField cannot infer one, so it is given explicitly and defaults
to SCALAR.

Values keep their dtype.  A composite of real fields is real and one
with a complex operand -- a frequency-dependent field frozen at one
omega -- is complex; nothing here casts to float, so an imaginary part
is never discarded on the way through an expression.

FieldBase lives here rather than in base.py because the operators build
these three classes, and the three inherit it: that circle is what makes
the algebra closed.  (a + b) + c is a SumField of a SumField, and
2 * (a + b) a ScaledField of one, each still an operand for the next
operation.  base.py stays a pure protocol, importing nothing.

The composites are `Assemblable` (base.py): each says what its operands
are and how to rebuild itself from new ones, and restriction,
re-statement on an interval, assembly from pieces and a change of
scales are then the one generic operation on the operands.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from ..character import SCALAR, Character
from .base import Field

__all__ = ["SumField", "ScaledField", "ComposedField", "RestrictedField",
           "FieldBase"]


def dependent(field) -> bool:
    """Whether a field is of the frequency or time kind."""
    return getattr(field, "kind", "static") != "static"


def require_same_skeleton(fields, what: str):
    """All operands must live on one geometry; returns it."""
    first = fields[0].skeleton
    for f in fields[1:]:
        if f.skeleton != first:
            raise ValueError(
                f"cannot {what} fields on different skeletons: "
                f"{first!r} and {f.skeleton!r}")
    return first




def all_radial(fields) -> bool:
    """Whether every operand is known to be independent of direction.

    A field that does not declare is_radial is assumed to depend on
    angle: silence is not a promise.
    """
    return all(bool(getattr(f, "is_radial", False)) for f in fields)




class FieldBase:
    """The optional [extra] tier: algebra and conveniences for any Field.

    The protocol (base.py) asks only for `evaluate`; this supplies what
    every shipped field wants anyway.  Mixing it in gives +, -, unary -,
    scalar * on either side, and / for free, each of them
    skeleton- and character-checked by the composite it builds; the
    plain call `field(r)`; `evaluate_at` for Cartesian points; and
    `domain`, `restricted`, `on_interval` and `rescaled` with the
    generic behaviour a field type may then improve on.

    It is mixed into the composites themselves, so the algebra is
    *closed*: (a + b) + c and 2 * (a + b) are ordinary fields, not
    TypeErrors, and an expression can be built up in whatever order it
    reads best.
    """

    #: "static" here; the frequency- and time-dependent families say
    #: theirs.  The algebra, the lifts and a body's views dispatch on it.
    kind = "static"

    @property
    def is_radial(self) -> bool:
        """Whether the field's values are independent of direction.

        False by default, which is the safe answer for a field that has
        not said: a caller may then only omit theta and phi where the
        field itself promises they are ignorable.  RadialField overrides
        it to True and every composite to the conjunction over its parts.

        It is a statement about the field's own components, in its own
        frame.  A VTI elastic tensor is radial in that sense and its
        *Cartesian* components still depend on direction, because the
        frame does.
        """
        return False

    def evaluate_at(self, X, *, layer=None, side: str = "upper",
                    frame: str = "cartesian"):
        """Values at Cartesian points X of shape (..., 3).

        A field's components are given in the frame its coordinates
        imply: `evaluate` speaks (r, theta, phi) and answers in the
        local (e_r, e_theta, e_phi) frame, so `evaluate_at` speaks
        Cartesian points and answers in Cartesian components.  Either
        accepts `frame=` to ask for the other; nothing else differs,
        because this converts the coordinates once and delegates, so a
        field is never handed a colatitude computed a second way.
        """
        from ..frames import spherical_coordinates
        r, theta, phi, _ = spherical_coordinates(X)
        return self.evaluate(r, theta, phi, layer=layer, side=side,
                             frame=frame)

    def __call__(self, r, theta=None, phi=None):
        """`evaluate` with the defaults: the plain callable a layer offers.

        A field on one layer has no sides, so calling it is unambiguous.
        On a many-layer field the call is `evaluate` with `side="upper"`,
        exactly as `evaluate(r)` would be.
        """
        return self.evaluate(r, theta, phi)

    @property
    def domain(self) -> tuple[int, ...]:
        """The layers of the skeleton on which the field is defined.

        Every layer unless a field says otherwise; RadialField and the
        composites report the layers their functions or sources cover.
        """
        return tuple(range(self.skeleton.nlayers))

    def restricted(self, layer):
        """This field on one layer alone, refusing radii outside it.

        A thin wrapper, not a copy: the values are this field's, taken
        with `layer=` fixed so the one-sided value at either end is the
        one that layer owns.  The skeleton of the result is the single
        span, so a radius outside it is refused by the same check that
        refuses a radius outside a body -- which is what makes a
        restriction usable wherever a field is, `check_field` included.

        Field types that can do better override this to return their
        own kind on one layer (a RadialField piece, an ElasticField of
        restricted moduli), which is what lets a body reassemble them.
        """
        return RestrictedField(self, layer)

    def on_interval(self, lo: float, hi: float):
        """A single-layer field re-stated on [lo, hi].

        Clipping only, by default: the interval must lie inside this
        field's one layer, and the result is a RestrictedField on it.
        Extrapolation beyond the layer is refused here, since a generic
        field's rule of evaluation is not known to extend; types whose
        rule does (a layer function, a formula) override this.
        """
        sk = self.skeleton
        if sk.nlayers != 1:
            raise ValueError(
                f"{self!r} spans {sk.nlayers} layers; restrict to one first")
        if not sk.contains(lo, hi):
            b = sk.boundaries
            raise TypeError(
                f"{self!r} cannot be extrapolated to [{lo:.6g}, {hi:.6g}]: "
                f"it is defined on [{b[0]:.6g}, {b[-1]:.6g}] and its rule of "
                "evaluation is not known to extend beyond that; attach a "
                "field for the new layer explicitly")
        return RestrictedField(self, 0, interval=(lo, hi))

    def rescaled(self, convert, old, new):
        """This field re-expressed in the scales `new`, from `old`.

        `convert(field)` is the body's converter for any field this one
        is built from (it memoises, so shared operands stay shared);
        `old` and `new` are the two `Scales`, for a type that carries a
        constant with units of its own.  A field that cannot be
        re-expressed refuses by name rather than keeping numbers in the
        wrong units; the composites convert their operands and rebuild,
        a RadialField its layer functions, a law its constants.
        """
        raise TypeError(
            f"cannot rescale {type(self).__name__} {self.name!r}: it does not "
            "say how to re-express itself in other scales (no `rescaled`). "
            "Rebuild it after the rescale.")

    def __add__(self, other):
        if not isinstance(other, Field) or dependent(other):
            # A frequency- or time-dependent operand takes over (its
            # __radd__ lifts this static field to its kind).
            return NotImplemented
        return SumField((self, other))

    def __sub__(self, other):
        if not isinstance(other, Field) or dependent(other):
            return NotImplemented
        return SumField((self, ScaledField(other, -1.0)))

    def __neg__(self):
        return ScaledField(self, -1.0)

    def __mul__(self, other):
        if isinstance(other, Field):
            raise TypeError(
                "field * field is not defined: the product of two fields has "
                "no character in general.  Use ComposedField to say what you "
                "mean.")
        return ScaledField(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Field):
            raise TypeError(
                "field / field is not defined; use ComposedField to say what "
                "you mean.")
        return ScaledField(self, 1.0 / other)


class _CompositeBase(FieldBase):
    """Shared plumbing: the Assemblable contract, and an approximate integrate.

    Inheriting FieldBase is what closes the algebra: a composite is
    itself an operand, so (a + b) + c, 2 * (a + b) and (a + b) / 2 build
    further composites instead of raising.

    Restriction, re-statement on an interval, reassembly and rescaling
    all go through the operands: a composite on one layer is the same
    composite of its operands on that layer.  Subclasses say what their
    operands are (`operands`) and how to rebuild from new ones
    (`rebuilt_from`); the four operations here are then generic.
    """

    def operands(self) -> tuple:
        """The fields this one is built from, in a fixed order."""
        raise NotImplementedError

    def rebuilt_from(self, operands, *, name=None):
        """The same construction on other operands."""
        raise NotImplementedError

    def matches(self, other) -> bool:
        """Whether `other` is the same composition of as many operands."""
        return type(other) is type(self) and (
            len(other.operands()) == len(self.operands()))

    def _operands(self) -> tuple:
        return self.operands()

    def _rebuilt_from(self, operands, *, name=None):
        return self.rebuilt_from(operands, name=name)

    def _matches(self, other) -> bool:
        return self.matches(other)

    @property
    def domain(self) -> tuple[int, ...]:
        """The layers every operand is defined on."""
        doms = [set(getattr(f, "domain", range(self.skeleton.nlayers)))
                for f in self.operands()]
        return tuple(sorted(set.intersection(*doms))) if doms else ()

    def restricted(self, layer):
        """The same composition of the operands restricted to `layer`."""
        i = self.skeleton.layer_index(layer)
        return self.rebuilt_from([f.restricted(i) for f in self.operands()],
                                 name=self.name)

    def on_interval(self, lo: float, hi: float):
        """The same composition of the operands on [lo, hi]."""
        return self.rebuilt_from([f.on_interval(lo, hi)
                                  for f in self.operands()], name=self.name)

    def rescaled(self, convert, old, new):
        """The same composition of the operands, each converted."""
        return self.rebuilt_from([convert(f) for f in self.operands()],
                                 name=self.name)

    @classmethod
    def assembled(cls, skeleton, pieces, *, name=None):
        """The same composition of the operands, each assembled."""
        from .layerwise import assemble
        first = pieces[0]
        if not all(first.matches(p) for p in pieces):
            return NotImplemented
        n = len(first.operands())
        ops = [assemble(skeleton, [p.operands()[j] for p in pieces])
               for j in range(n)]
        return first.rebuilt_from(ops, name=name if name is not None
                                  else first.name)

    @classmethod
    def _assembled(cls, skeleton, pieces, *, name=None):
        return cls.assembled(skeleton, pieces, name=name)

    def integrate(self, a: float, b: float, *, limit: int = 200) -> float:
        """Integral over [a, b] by quadrature.

        Approximate by construction: a composite has no closed form to
        integrate, so this is adaptive quadrature over the interval,
        subdivided at any skeleton boundary it crosses so the
        integrator never straddles a discontinuity.
        """
        a, b = float(a), float(b)
        if a == b:
            return 0.0
        sign = 1.0
        if a > b:
            a, b, sign = b, a, -1.0
        bnd = np.asarray(self.skeleton.boundaries, dtype=float)
        cuts = [a, *[x for x in bnd if a < x < b], b]
        total = 0.0
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            mid = 0.5 * (lo + hi)
            layer = self.skeleton.locate(mid).layers[0]
            val, _ = quad(
                lambda x, layer=layer: float(
                    np.asarray(self.evaluate(np.array([x]), layer=layer))[0]),
                lo, hi, limit=limit)
            total += val
        return sign * total


def checked_sum_metadata(terms):
    """The character and dimensions a sum inherits, or why it cannot."""
    char = terms[0].character
    for f in terms[1:]:
        if f.character != char:
            raise ValueError(
                f"cannot add fields of different character: {char} and "
                f"{f.character}")
    dims = {f.dimensions for f in terms
            if getattr(f, "dimensions", None) is not None}
    if len(dims) > 1:
        raise ValueError(
            f"cannot add fields of different dimensions: "
            f"{sorted(map(str, dims))}")
    return char, (dims.pop() if dims else None)


class SumField(_CompositeBase):
    """The sum of fields sharing a skeleton and a character."""

    def __init__(self, terms, *, name: str | None = None) -> None:
        terms = tuple(terms)
        if not terms:
            raise ValueError("SumField needs at least one term")
        self._terms = terms
        self.skeleton = require_same_skeleton(terms, "add")
        self.character, self.dimensions = checked_sum_metadata(terms)
        self.name = name

    @property
    def terms(self):
        """The summed fields, in order."""
        return self._terms

    def operands(self) -> tuple:
        return self._terms

    def rebuilt_from(self, operands, *, name=None):
        return SumField(operands, name=name)

    @property
    def is_radial(self) -> bool:
        """A sum is direction-independent when every term is."""
        return all_radial(self._terms)

    def evaluate(self, r, theta=None, phi=None, **kw):
        """The sum of the terms, evaluated pointwise."""
        out = None
        for f in self._terms:
            v = np.asarray(f.evaluate(r, theta, phi, **kw))
            out = v if out is None else out + v
        return out

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"SumField({len(self._terms)} terms{nm})"


class ScaledField(_CompositeBase):
    """A field multiplied by a constant; the character is unchanged."""

    def __init__(self, field, factor, *, name: str | None = None) -> None:
        self._field = field
        self.factor = (complex(factor) if np.iscomplexobj(factor)
                       else float(factor))
        self.skeleton = field.skeleton
        self.character = field.character
        self.dimensions = getattr(field, "dimensions", None)
        self.name = name

    @property
    def source(self):
        """The field scaled."""
        return self._field

    @property
    def is_radial(self) -> bool:
        """Scaling cannot introduce a direction."""
        return bool(getattr(self._field, "is_radial", False))

    def operands(self) -> tuple:
        return (self._field,)

    def rebuilt_from(self, operands, *, name=None):
        return ScaledField(operands[0], self.factor, name=name)

    def matches(self, other) -> bool:
        return isinstance(other, ScaledField) and other.factor == self.factor

    def evaluate(self, r, theta=None, phi=None, **kw):
        """The underlying field, scaled."""
        return self.factor * np.asarray(self._field.evaluate(r, theta, phi, **kw))

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"ScaledField({self.factor:g} x {self._field!r}{nm})"


class RestrictedField(_CompositeBase):
    """One layer of a field, on a skeleton holding only that layer.

    What `FieldBase.restricted` builds.  Two things are restricted, not
    one: the *values* are taken with `layer=` fixed, so at either end of
    the span the one-sided value belonging to this layer is returned
    rather than whichever side a tie-break chose; and the *domain* is
    the span itself, so a radius outside it is refused with a message
    naming the layer, instead of quietly returning a neighbour's
    material.

    `is_radial` is inherited: restricting the domain cannot introduce a
    direction, nor remove one.
    """

    def __init__(self, field, layer, *, interval=None,
                 name: str | None = None) -> None:
        """Bind a field and one of its layers, by index.

        `interval`, when given, narrows the span further: it must lie
        inside the layer, and is how a generic field is clipped when a
        body is truncated or refined through it.
        """
        from ..skeleton import Skeleton
        i = field.skeleton.layer_index(layer)
        lo, hi = field.skeleton.interval(i)
        if interval is not None:
            a, b = (float(x) for x in interval)
            tol = field.skeleton.tolerance
            if a < lo - tol or b > hi + tol or not b > a:
                raise ValueError(
                    f"interval [{a:.6g}, {b:.6g}] is not inside layer {i} "
                    f"[{lo:.6g}, {hi:.6g}]")
            lo, hi = a, b
        self._field = field
        self._layer = i
        self.skeleton = Skeleton([lo, hi])
        self.character = field.character
        self.dimensions = getattr(field, "dimensions", None)
        src = getattr(field, "name", None)
        self.name = name if name is not None else (
            f"{src}[{i}]" if src else None)

    @property
    def source(self):
        """The field this restricts."""
        return self._field

    @property
    def layer(self) -> int:
        """The index, in the source's skeleton, of the layer kept."""
        return self._layer

    @property
    def is_radial(self) -> bool:
        """Whatever the source promises: a restriction changes no values."""
        return bool(getattr(self._field, "is_radial", False))

    @property
    def domain(self) -> tuple[int, ...]:
        return (0,)

    def operands(self) -> tuple:
        return (self._field,)

    def rebuilt_from(self, operands, *, name=None):
        lo, hi = (float(x) for x in self.skeleton.boundaries)
        return RestrictedField(operands[0], self._layer, interval=(lo, hi),
                               name=name)

    def restricted(self, layer):
        """A restriction has one layer, and that is itself."""
        self.skeleton.layer_index(layer)
        return self

    @classmethod
    def assembled(cls, skeleton, pieces, *, name=None):
        """Restrictions of different sources have no common rebuild."""
        return NotImplemented

    def on_interval(self, lo: float, hi: float):
        """Narrow the span; a restriction never extrapolates."""
        if not self.skeleton.contains(lo, hi):
            raise TypeError(
                f"{self!r} cannot be extrapolated to [{lo:.6g}, {hi:.6g}]")
        return RestrictedField(self._field, self._layer, interval=(lo, hi),
                               name=self.name)

    def rescaled(self, convert, old, new):
        """The converted source, restricted to the same layer and span."""
        k = old.length / new.length
        lo, hi = (float(x) for x in self.skeleton.boundaries)
        return RestrictedField(convert(self._field), self._layer,
                               interval=(lo * k, hi * k), name=self.name)

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical"):
        """The source's values on this layer; other radii are refused.

        `layer` may be omitted or may name this field's single layer;
        it is not the source's numbering, since the restriction is a
        field in its own right and its skeleton has one layer.
        """
        if layer is not None and self.skeleton.layer_index(layer) != 0:
            raise IndexError("layer index out of range for 1 layer")
        lo, hi = (float(x) for x in self.skeleton.boundaries)
        rr = np.asarray(r, dtype=float)
        tol = self.skeleton.tolerance
        if rr.size and (np.min(rr) < lo - tol or np.max(rr) > hi + tol):
            bad = float(rr.reshape(-1)[
                np.argmax((rr < lo - tol) | (rr > hi + tol))])
            raise ValueError(
                f"radius {bad:.6g} outside layer {self._layer} of "
                f"{self._field!r}, which is [{lo:.6g}, {hi:.6g}]: this field "
                "is the restriction to that layer, so a radius beyond it has "
                "no value here even though the field it restricts has one")
        return self._field.evaluate(rr, theta, phi, layer=self._layer,
                                    side=side, frame=frame)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"RestrictedField({self._field!r}, layer={self._layer}{nm})"


class ComposedField(_CompositeBase):
    """fn(*sources) evaluated on demand, never sampled and refitted.

    The derived views of a model: velocities from moduli and density,
    Q from relaxation times, anything pointwise.  Exact wherever it is
    asked, approximate under integration, and honest about the
    difference -- which sampling and refitting would not be.

    `character` cannot be inferred from an arbitrary function, so it is
    given; it defaults to SCALAR, right for the invariant quantities
    most derived views produce.

    Under a change of scales the sources are converted and `fn` is
    kept, which is right when `fn` is dimensionally homogeneous --
    sqrt(A / rho) maps scaled inputs to the scaled output -- and is the
    requirement on a user-supplied one.
    """

    def __init__(self, fn, sources, *, name: str | None = None,
                 character: Character = SCALAR, dimensions=None) -> None:
        sources = tuple(sources)
        if not sources:
            raise ValueError("ComposedField needs at least one source")
        self._fn = fn
        self._sources = sources
        self.skeleton = require_same_skeleton(sources, "compose")
        self.character = character
        self.dimensions = dimensions
        self.name = name

    @property
    def sources(self):
        """The fields this view is computed from."""
        return self._sources

    @property
    def fn(self):
        """The pointwise function of the sources' values."""
        return self._fn

    def operands(self) -> tuple:
        return self._sources

    def rebuilt_from(self, operands, *, name=None):
        return ComposedField(self._fn, operands, name=name,
                             character=self.character,
                             dimensions=self.dimensions)

    def matches(self, other) -> bool:
        return (isinstance(other, ComposedField) and other._fn is self._fn
                and len(other._sources) == len(self._sources))

    @property
    def is_radial(self) -> bool:
        """A pointwise function of radial fields is radial.

        fn sees only values, never the coordinates, so it cannot
        manufacture an angular dependence the sources do not have.
        """
        return all_radial(self._sources)

    def evaluate(self, r, theta=None, phi=None, **kw):
        """fn applied to the sources' values at the same points."""
        vals = [np.asarray(f.evaluate(r, theta, phi, **kw))
                for f in self._sources]
        return np.asarray(self._fn(*vals))

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        srcs = ", ".join(f.name or "?" for f in self._sources)
        return f"ComposedField({srcs}{nm})"
