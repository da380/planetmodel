"""Static fields on one interval, and the small algebra between them.

A field is data on one layer: an `interval`, a `character` (the rank and
weight that say how it transforms), a `name`, and
`evaluate(r, theta, phi, *, frame)`.  It knows no skeleton, no sides and
no units: a discontinuity is two layers asked separately, and what the
numbers mean is the model's business.

`evaluate(r, theta, phi)` broadcasts its coordinates and returns float64
of the broadcast shape followed by the character's Voigt shape for
ranks 2 and 4, or its component shape otherwise.  Components are given
in the local spherical frame (e_r, e_theta, e_phi) at the point unless
`frame="cartesian"` asks for Cartesian ones; a rank-1 field rotates as
R v, a Voigt rank-2 as M v and a Voigt rank-4 as M C M^T with M the Bond
matrix of R, anything else slot by slot.  Calling a field, `f(r, theta,
phi)`, is `evaluate` in the spherical frame, and a radial field of rank
0 may be called with the radius alone, `f(r)`.  A radius outside the
interval by more than `rtol` of its width is refused: stepping beyond a
layer on purpose is `on_interval`.

Three kinds are shipped.  `RadialField` holds one layer function per
stored component and is exact when they are polynomial.
`AnalyticField` is a formula of (r, theta, phi).  `ComposedField` is a
pointwise function of other fields, the one escape from the algebra
below, which is: `+` and `-` between fields of one character on one
interval, `*` and `/` between fields of rank 0 (weights add and
subtract), `**` of a rank-0 weight-0 field, and scaling by a number.
When every operand is radial the result is a `RadialField` on the
combined layer functions, exact when they are polynomial; otherwise it
is a `ComposedField`.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .character import SCALAR, Character
from .frames import (bond_matrix, rotate_slots, spherical_coordinates,
                     spherical_frame, tensor_to_voigt)
from .layerfunction import as_layer_function, constant_layer, same_interval

__all__ = ["Field", "FieldBase", "RadialField", "AnalyticField",
           "ComposedField", "constant_field", "FRAMES", "check_frame",
           "stored_shape"]

#: The frames a field can present its components in.
FRAMES = ("spherical", "cartesian")


def check_frame(frame: str) -> None:
    """Refuse a frame outside FRAMES, naming it."""
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}: expected one of {FRAMES}")


def stored_shape(character: Character) -> tuple[int, ...]:
    """The trailing shape a field of this character presents: Voigt for
    ranks 2 and 4 when the character says so, full components otherwise."""
    v = character.voigt_shape
    return v if v is not None else character.component_shape


@runtime_checkable
class Field(Protocol):
    """What every field has: three attributes and `evaluate`."""

    interval: tuple[float, float]
    character: Character
    name: str | None

    def evaluate(self, r, theta, phi, *, frame: str = "spherical"): ...


def _interval(interval) -> tuple[float, float]:
    lo, hi = (float(x) for x in interval)
    if not hi > lo:
        raise ValueError(f"an interval must increase, got ({lo:g}, {hi:g})")
    return lo, hi


def _rotate(values, R, character: Character):
    """Components rotated by R: Bond on a Voigt form, a factor per slot else."""
    rank = character.rank
    if rank == 0:
        return values
    if character.voigt_shape is None:
        return rotate_slots(values, R, rank)
    M = bond_matrix(R)
    if rank == 2:
        return np.einsum("...ab,...b->...a", M, values)
    return np.einsum("...ab,...bc,...dc->...ad", M, values, M)


class FieldBase:
    """The shared behaviour of the shipped fields.

    A subclass sets `_interval`, `_character`, `_name` and `_rtol`, and
    implements `_values(r, theta, phi)` returning the stored components
    in the spherical frame, plus `on_interval`, `rescaled` and
    `renamed`.  This class checks the coordinates, presents the values
    in the frame asked for, and supplies the algebra.
    """

    is_radial = False
    __array_ufunc__ = None

    @property
    def interval(self) -> tuple[float, float]:
        return self._interval

    @property
    def character(self) -> Character:
        return self._character

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def rtol(self) -> float:
        """The fraction of the interval's width a radius may overshoot."""
        return self._rtol

    @property
    def stored_shape(self) -> tuple[int, ...]:
        return stored_shape(self._character)

    # -- the question -------------------------------------------------------

    def _needs_angles(self) -> bool:
        return not (self.is_radial and self._character.rank == 0)

    def _points(self, r, theta, phi):
        """(r, theta, phi) broadcast and checked; angles None when allowed."""
        r = np.asarray(r, dtype=float)
        if theta is None or phi is None:
            if self._needs_angles():
                raise ValueError(
                    f"{self!r} needs theta and phi as well as r: only a radial "
                    "field of rank 0 is called with the radius alone")
            theta = phi = None
        else:
            r, theta, phi = np.broadcast_arrays(
                r, np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
        lo, hi = self._interval
        tol = self._rtol * (hi - lo)
        if r.size and (np.any(r < lo - tol) or np.any(r > hi + tol)):
            bad = float(r[(r < lo - tol) | (r > hi + tol)].flat[0])
            raise ValueError(
                f"radius {bad:.6g} is outside the interval [{lo:.6g}, {hi:.6g}] "
                f"of {self!r}; on_interval re-states a field on purpose")
        return r, theta, phi

    def evaluate(self, r, theta, phi, *, frame: str = "spherical"):
        """The components at (r, theta, phi) in `frame`, float64."""
        check_frame(frame)
        r, theta, phi = self._points(r, theta, phi)
        values = np.asarray(self._values(r, theta, phi), dtype=float)
        if frame == "cartesian" and self._character.rank:
            values = _rotate(values, spherical_frame(theta, phi), self._character)
        return values

    def __call__(self, *coordinates):
        """`evaluate` in the spherical frame: f(r, theta, phi), or f(r) for a
        radial field of rank 0."""
        if len(coordinates) == 1:
            return self.evaluate(coordinates[0], None, None)
        if len(coordinates) != 3:
            raise TypeError(f"a field is called with (r, theta, phi) or with r "
                            f"alone, not {len(coordinates)} arguments")
        return self.evaluate(*coordinates)

    def evaluate_at(self, X, *, frame: str = "cartesian"):
        """The components at Cartesian points X of shape (..., 3), in `frame`."""
        r, theta, phi, _ = spherical_coordinates(X)
        return self.evaluate(r, theta, phi, frame=frame)

    # -- the algebra --------------------------------------------------------

    def _same_interval(self, other) -> None:
        if not same_interval(self._interval, other.interval, rtol=self._rtol):
            raise ValueError(
                f"fields on different intervals cannot be combined: "
                f"{self._interval} and {other.interval}")

    def __add__(self, other):
        if not isinstance(other, Field):
            return NotImplemented
        self._same_interval(other)
        if other.character != self._character:
            raise ValueError(
                f"cannot add a {other.character} field to a {self._character} one")
        return _combine((self, other), np.add, self._character)

    def __sub__(self, other):
        if not isinstance(other, Field):
            return NotImplemented
        self._same_interval(other)
        if other.character != self._character:
            raise ValueError(
                f"cannot subtract a {other.character} field from a "
                f"{self._character} one")
        return _combine((self, other), np.subtract, self._character)

    def __neg__(self):
        return _scale(self, -1.0)

    def __mul__(self, other):
        if isinstance(other, Field):
            self._same_interval(other)
            return _combine((self, other), np.multiply,
                            self._character * other.character)
        if _is_number(other):
            return _scale(self, float(other))
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Field):
            self._same_interval(other)
            if self._character.rank or other.character.rank:
                raise ValueError("only fields of rank 0 divide")
            char = Character(0, self._character.weight - other.character.weight)
            return _combine((self, other), np.divide, char)
        if _is_number(other):
            return _scale(self, 1.0 / float(other))
        return NotImplemented

    def __pow__(self, n):
        if self._character.rank or self._character.weight:
            raise ValueError("only a field of rank 0 and weight 0 is raised to a power")
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("a field is raised to a non-negative integer")
        if isinstance(self, RadialField):
            return RadialField(self._interval, self.function ** int(n),
                               character=self._character, rtol=self._rtol)
        return ComposedField(lambda x: x ** int(n), (self,), character=self._character)


def _is_number(x) -> bool:
    return np.isscalar(x) and not isinstance(x, (str, bytes))


def _elementwise(fn, *arrays) -> np.ndarray:
    """fn applied element by element to object arrays of one shape."""
    out = np.empty(arrays[0].shape, dtype=object)
    for idx in np.ndindex(arrays[0].shape):
        out[idx] = fn(*(a[idx] for a in arrays))
    return out


def _combine(fields, op, character):
    if all(isinstance(f, RadialField) for f in fields):
        a, b = fields
        return RadialField(a.interval, _elementwise(op, a._fs, b._fs),
                           character=character, rtol=a.rtol)
    return ComposedField(op, fields, character=character)


def _scale(field, c: float):
    if isinstance(field, RadialField):
        return RadialField(field.interval, _elementwise(lambda f: f * c, field._fs),
                           character=field.character, rtol=field.rtol)
    return ComposedField(lambda x: c * x, (field,), character=field.character)


class RadialField(FieldBase):
    """A field of the radius alone: one layer function per stored component.

    `function` is a layer function, a `PPoly`, a number or a callable of
    r for a rank-0 field; for a higher rank it is a nested sequence of
    those in the stored shape (Voigt for ranks 2 and 4), giving the
    components in the spherical frame.  `functions` is that array of
    layer functions; `function` the single one of a rank-0 field.
    """

    is_radial = True

    def __init__(self, interval, function, *, character: Character = SCALAR,
                 name: str | None = None, rtol: float = 1e-9) -> None:
        self._interval = _interval(interval)
        self._character = character
        self._name = name
        self._rtol = float(rtol)
        shape = stored_shape(character)
        fs = np.empty(shape, dtype=object)
        if isinstance(function, np.ndarray) and function.dtype == object:
            if function.shape != shape:
                raise ValueError(
                    f"a {character} radial field has components of shape "
                    f"{shape}, got {function.shape}")
            for idx in np.ndindex(shape):
                fs[idx] = as_layer_function(function[idx], self._interval)
        elif shape == ():
            fs[()] = as_layer_function(function, self._interval)
        else:
            try:
                for idx in np.ndindex(shape):
                    item = function
                    for i in idx:
                        item = item[i]
                    fs[idx] = as_layer_function(item, self._interval)
            except (TypeError, IndexError):
                raise ValueError(
                    f"a {character} radial field takes its components as a "
                    f"nested sequence of shape {shape}") from None
        self._fs = fs

    @property
    def functions(self) -> np.ndarray:
        """The layer functions, an object array of the stored shape."""
        return self._fs

    @property
    def function(self):
        """The one layer function of a rank-0 field."""
        if self._fs.shape:
            raise ValueError(f"{self!r} has {self._fs.size} components; use functions")
        return self._fs[()]

    def _values(self, r, theta, phi):
        if not self._fs.shape:
            return self._fs[()](r)
        out = np.empty(r.shape + self._fs.shape)
        for idx in np.ndindex(self._fs.shape):
            out[(...,) + idx] = self._fs[idx](r)
        return out

    def _map(self, fn, *, character=None, interval=None, name=None):
        fs = np.empty(self._fs.shape, dtype=object)
        for idx in np.ndindex(self._fs.shape):
            fs[idx] = fn(self._fs[idx])
        return RadialField(self._interval if interval is None else interval, fs,
                           character=self._character if character is None
                           else character, name=name, rtol=self._rtol)

    def derivative(self, *, nu: int = 1) -> "RadialField":
        """The nu-th radial derivative of every component; the character is kept."""
        return self._map(lambda f: f.derivative(nu=nu))

    def integrate(self, a: float, b: float):
        """The signed integral from a to b: a float for rank 0, else an array."""
        if not self._fs.shape:
            return self._fs[()].integrate(a, b)
        out = np.empty(self._fs.shape)
        for idx in np.ndindex(self._fs.shape):
            out[idx] = self._fs[idx].integrate(a, b)
        return out

    def on_interval(self, lo: float, hi: float) -> "RadialField":
        """The same layer functions re-stated on [lo, hi], by their own rule."""
        lo, hi = _interval((lo, hi))
        return self._map(lambda f: f.on_interval(lo, hi), interval=(lo, hi),
                         name=self._name)

    def rescaled(self, *, k: float, v: float) -> "RadialField":
        """v f(r / k) on the interval scaled by k."""
        lo, hi = self._interval
        return self._map(lambda f: f.rescaled(k=k, v=v), interval=(k * lo, k * hi),
                         name=self._name)

    def renamed(self, name: str | None) -> "RadialField":
        return self._map(lambda f: f, name=name)

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return f"RadialField({nm}{self._character} on [{lo:g}, {hi:g}])"


class AnalyticField(FieldBase):
    """A formula `fn(r, theta, phi)` returning components in `frame`.

    The formula receives broadcast coordinate arrays and returns the
    components at each point, full or Voigt for ranks 2 and 4 (a
    constant tensor is broadcast); the result is presented Voigt.  A
    formula returning complex values is refused.
    """

    def __init__(self, interval, fn, *, character: Character = SCALAR,
                 name: str | None = None, frame: str = "spherical",
                 rtol: float = 1e-9) -> None:
        if not callable(fn):
            raise TypeError(f"expected a callable, got {type(fn).__name__}")
        check_frame(frame)
        self._interval = _interval(interval)
        self._fn = fn
        self._character = character
        self._name = name
        self._frame = frame
        self._rtol = float(rtol)

    @property
    def fn(self):
        """The formula of (r, theta, phi)."""
        return self._fn

    @property
    def frame(self) -> str:
        """The frame the formula returns its components in."""
        return self._frame

    def _values(self, r, theta, phi):
        raw = np.asarray(self._fn(r, theta, phi))
        if np.iscomplexobj(raw):
            raise TypeError(f"{self!r} returned complex values; a field is real")
        vals = _to_stored(raw, r.shape, self._character, self)
        if self._frame == "cartesian" and self._character.rank:
            R = spherical_frame(theta, phi)
            vals = _rotate(vals, np.swapaxes(R, -1, -2), self._character)
        return vals

    def on_interval(self, lo: float, hi: float) -> "AnalyticField":
        """The formula on [lo, hi]: it extends wherever it is asked."""
        return AnalyticField((lo, hi), self._fn, character=self._character,
                             name=self._name, frame=self._frame, rtol=self._rtol)

    def rescaled(self, *, k: float, v: float) -> "AnalyticField":
        """v fn(r / k, theta, phi) on the interval scaled by k."""
        k, v = float(k), float(v)
        fn = self._fn
        lo, hi = self._interval
        return AnalyticField(
            (k * lo, k * hi), lambda r, t, p: v * np.asarray(fn(r / k, t, p)),
            character=self._character, name=self._name, frame=self._frame,
            rtol=self._rtol)

    def renamed(self, name: str | None) -> "AnalyticField":
        return AnalyticField(self._interval, self._fn, character=self._character,
                             name=name, frame=self._frame, rtol=self._rtol)

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return f"AnalyticField({nm}{self._character} on [{lo:g}, {hi:g}])"


def _to_stored(raw, point_shape, character, owner):
    """A formula's return value as stored components at every point."""
    rank = character.rank
    stored = stored_shape(character)
    full = character.component_shape
    if rank == 0:
        vals = raw
    elif raw.shape[-len(stored):] == stored:
        vals = raw
    elif character.voigt_shape is not None and raw.shape[-rank:] == full:
        vals = tensor_to_voigt(raw, rank=rank)
    else:
        want = f"{stored} or {full}" if stored != full else f"{full}"
        raise ValueError(
            f"{owner!r} returns components of trailing shape {want}, "
            f"got {raw.shape}")
    target = point_shape + stored
    if vals.shape != target:
        try:
            vals = np.broadcast_to(vals, target)
        except ValueError:
            raise ValueError(
                f"{owner!r} should return {target} at {point_shape} points, "
                f"got {raw.shape}") from None
    return np.array(vals, dtype=float)


class ComposedField(FieldBase):
    """`fn(*values)` pointwise on fields of one interval.

    The sources' stored components in the spherical frame are passed to
    `fn`, which returns the stored components of `character`; it is
    never sampled or refitted.  Radial when every source is.
    """

    def __init__(self, fn, sources, *, character: Character,
                 name: str | None = None) -> None:
        if not callable(fn):
            raise TypeError(f"expected a callable, got {type(fn).__name__}")
        sources = tuple(sources)
        if not sources:
            raise ValueError("a composed field needs at least one source")
        for s in sources:
            if not isinstance(s, Field):
                raise TypeError(f"{s!r} is not a Field")
        first = sources[0]
        rtol = getattr(first, "rtol", 1e-9)
        for s in sources[1:]:
            if not same_interval(first.interval, s.interval, rtol=rtol):
                raise ValueError(
                    f"sources on different intervals: {first.interval} and "
                    f"{s.interval}")
        self._fn = fn
        self._sources = sources
        self._interval = _interval(first.interval)
        self._character = character
        self._name = name
        self._rtol = float(rtol)
        self.is_radial = all(getattr(s, "is_radial", False) for s in sources)

    @property
    def fn(self):
        return self._fn

    @property
    def sources(self) -> tuple:
        return self._sources

    def _values(self, r, theta, phi):
        args = [s.evaluate(r, theta, phi) for s in self._sources]
        raw = np.asarray(self._fn(*args))
        if np.iscomplexobj(raw):
            raise TypeError(f"{self!r} returned complex values; a field is real")
        return _to_stored(raw, r.shape, self._character, self)

    def on_interval(self, lo: float, hi: float) -> "ComposedField":
        return ComposedField(self._fn, [s.on_interval(lo, hi) for s in self._sources],
                             character=self._character, name=self._name)

    def rescaled(self, *, k: float, v: float) -> "ComposedField":
        """The sources on the scaled coordinate, the result times v."""
        v = float(v)
        fn = self._fn
        return ComposedField(lambda *a: v * np.asarray(fn(*a)),
                             [s.rescaled(k=k, v=1.0) for s in self._sources],
                             character=self._character, name=self._name)

    def renamed(self, name: str | None) -> "ComposedField":
        return ComposedField(self._fn, self._sources, character=self._character,
                             name=name)

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return (f"ComposedField({nm}{self._character} of {len(self._sources)} "
                f"on [{lo:g}, {hi:g}])")


def constant_field(value, interval, *, character: Character = SCALAR,
                   name: str | None = None) -> RadialField:
    """The constant `value` (a number, or an array of the stored shape) as an
    exact radial field."""
    lo, hi = _interval(interval)
    v = np.asarray(value, dtype=float)
    shape = stored_shape(character)
    if v.shape != shape:
        raise ValueError(f"a {character} constant has shape {shape}, got {v.shape}")
    if shape == ():
        return RadialField((lo, hi), constant_layer(float(v), (lo, hi)),
                           character=character, name=name)
    fs = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        fs[idx] = constant_layer(float(v[idx]), (lo, hi))
    return RadialField((lo, hi), fs, character=character, name=name)
