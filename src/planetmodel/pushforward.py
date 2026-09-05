"""Fields carried across a mapping, forwards and back.

For a mapping m taking the reference body to the physical one, with
F[i, j] = d m_i / d X_j and J = det F, a field of rank n and weight w
transforms as

    T_phys[i1..in](x) = J^-w F[i1, A1] ... F[in, An] T_ref[A1..An](X)

with x = m(X): one factor of F on every slot, and a factor of 1/J when
the weight is 1.  That is the whole rule, driven by the Character and
nothing else.  Rank 0 weight 1 is rho / J; rank 0 weight 0 composes
with the mapping; rank 2 weight 1 carries the second Piola-Kirchhoff
stress to the Cauchy stress F S F^T / J; rank 4 weight 1 carries the
second elasticity tensor.  The pull-back is the same rule with F^-1 in
place of F and J^+w in place of J^-w, so the two are inverses at one
(F, J).  Identical wrapping on every slot preserves the minor and major
symmetries, so a Voigt reduction after the rule is faithful.

Two levels.  `push_forward` and `pull_back` act on arrays: values, F
and J already evaluated at the same points, in Cartesian components, at
full tensor rank.  A Voigt array is a bookkeeping device with no slots
to wrap and is refused; `full_components` is the one place a field's
Voigt presentation is expanded.  Complex values pass through the array
level, since the rule is linear.

`PushedForwardField` and `PulledBackField` act on fields and are lazy:
they hold a source and a mapping and do the work in `evaluate`.  Both
are asked at reference coordinates and neither inverts the mapping.  A
pushed-forward field returns the physical tensor at m(X), in Cartesian
components or in the spherical frame at m(X), the frame a physical
quantity is named in.  A pulled-back field takes a callable of the
physical point's spherical coordinates returning components in the
frame there, evaluates it at m(X), and gives the referential tensor at
X in the frame at X, so it is a field like any other.
"""
from __future__ import annotations

import numpy as np

from .character import Character
from .fields import FieldBase, _to_stored, check_frame
from .frames import (cartesian_points, rotate_slots, spherical_coordinates,
                     spherical_frame, tensor_to_voigt, voigt_to_tensor)
from .mapping import ScaledMapping

__all__ = ["push_forward", "pull_back", "full_components",
           "PushedForwardField", "PulledBackField"]


def _as_values(values, rank: int, what: str) -> np.ndarray:
    """`values` as an array with the trailing shape the rank demands.

    Real input becomes float64 and complex input stays complex.  A
    Voigt array has the right number of entries and the wrong shape, so
    it is refused by name rather than left to fail in the contraction.
    """
    values = np.asarray(values)
    if not np.iscomplexobj(values):
        values = np.asarray(values, dtype=float)
    want = (3,) * rank
    if rank and values.shape[-rank:] != want:
        voigt = {2: (6,), 4: (6, 6)}.get(rank)
        if voigt is not None and values.shape[-len(voigt):] == voigt:
            raise ValueError(
                f"{what} takes full components, not Voigt: a rank-{rank} value "
                f"has trailing shape {want}, and {values.shape} is a Voigt "
                "array; expand it with voigt_to_tensor first")
        raise ValueError(
            f"a rank-{rank} value has trailing shape {want}, got {values.shape}")
    return values


def _weight_factor(J, rank: int) -> np.ndarray:
    """J reshaped to divide or multiply a rank-n value slot-wise."""
    J = np.asarray(J, dtype=float)
    return J.reshape(J.shape + (1,) * rank)


def push_forward(values, F, J, character: Character) -> np.ndarray:
    """The physical values of a field of `character` from its referential ones.

    `values` has shape broadcast + (3,) * rank in full Cartesian
    components at reference points X; `F` (..., 3, 3) and `J` (...) are
    the mapping's deformation gradient and Jacobian there.  Every slot
    receives one factor of F and a weight-1 field is divided by J:

        T_ij.. = J^-w F_iA F_jB ... T_AB..

    A rank-0 weight-0 field is returned unchanged.  Complex values stay
    complex; a Voigt array is refused.
    """
    values = _as_values(values, character.rank, "push_forward")
    out = rotate_slots(values, F, character.rank)
    if character.weight:
        out = out / _weight_factor(J, character.rank)
    return out


def pull_back(values, F, J, character: Character) -> np.ndarray:
    """The referential values from the physical ones at the same (F, J).

    The inverse of `push_forward`: F^-1 on every slot and J^+w,

        T_AB.. = J^w (F^-1)_Ai (F^-1)_Bj ... T_ij..

    `values` are the physical components at x = m(X); the result is
    referential, at X.  Complex values stay complex; a Voigt array is
    refused.
    """
    values = _as_values(values, character.rank, "pull_back")
    Finv = np.linalg.inv(np.asarray(F, dtype=float)) if character.rank else F
    out = rotate_slots(values, Finv, character.rank)
    if character.weight:
        out = out * _weight_factor(J, character.rank)
    return out


def full_components(field, r, theta, phi, *, frame: str = "cartesian") -> np.ndarray:
    """A field's values at full tensor rank, trailing shape (3,) * rank.

    The one place a Voigt-presenting field is expanded: a character with
    a Voigt shape has its (6,) or (6, 6) values expanded to (3, 3) or
    (3, 3, 3, 3), anything else is asked plainly.
    """
    char = field.character
    values = np.asarray(field.evaluate(r, theta, phi, frame=frame))
    if char.voigt_shape is not None:
        values = voigt_to_tensor(values, rank=char.rank)
    return _as_values(values, char.rank, "full_components")


def _reduce(values, character: Character) -> np.ndarray:
    """Full components presented the way the character stores them."""
    if character.voigt_shape is not None:
        return tensor_to_voigt(values, rank=character.rank)
    return values


class PushedForwardField(FieldBase):
    """The image of a referential field under a mapping, asked at reference points.

    `evaluate(r, theta, phi)` takes reference coordinates and returns
    the tensor the field has at the physical point m(X): the source's
    Cartesian components are pushed forward with F(X) and J(X), then
    given in Cartesian components or in the spherical frame at m(X),
    Voigt-reduced when the character has a Voigt shape.  Interval and
    character are the source's; the name defaults to `<source>_phys`
    where the source is named.  All three coordinates are required, F
    depending on direction even where the source does not, so the field
    is never radial.  `rescaled` rescales the source and conjugates the
    mapping by the same length scale.
    """

    is_radial = False

    def __init__(self, field, mapping, *, name: str | None = None) -> None:
        self._source = field
        self._mapping = mapping
        self._interval = tuple(float(x) for x in field.interval)
        self._character = field.character
        self._rtol = float(getattr(field, "rtol", 1e-9))
        src = getattr(field, "name", None)
        self._name = name if name is not None else (f"{src}_phys" if src else None)

    @property
    def source(self):
        """The referential field carried across."""
        return self._source

    @property
    def mapping(self):
        """The mapping it is carried across."""
        return self._mapping

    def _physical(self, r, theta, phi):
        """(full Cartesian components at m(X), the points m(X)) for reference
        coordinates already broadcast and checked."""
        X = cartesian_points(r, theta, phi)
        vals = full_components(self._source, r, theta, phi, frame="cartesian")
        F = np.asarray(self._mapping.deformation_gradient(X), dtype=float)
        J = np.asarray(self._mapping.jacobian(X), dtype=float)
        return push_forward(vals, F, J, self._character), self._mapping(X)

    def _values(self, r, theta, phi):
        out, x = self._physical(r, theta, phi)
        if self._character.rank:
            _, _, _, R = spherical_coordinates(x)
            out = rotate_slots(out, np.swapaxes(R, -1, -2), self._character.rank)
        return _reduce(out, self._character)

    def evaluate(self, r, theta, phi, *, frame: str = "spherical"):
        """The physical tensor at m(X) for the reference point X = (r, theta, phi).

        In Cartesian components for `frame="cartesian"`, in the local
        spherical frame at m(X) for `frame="spherical"`.
        """
        check_frame(frame)
        r, theta, phi = self._points(r, theta, phi)
        if frame == "spherical":
            return self._values(r, theta, phi)
        out, _ = self._physical(r, theta, phi)
        return _reduce(out, self._character)

    def on_interval(self, lo: float, hi: float) -> "PushedForwardField":
        """The source re-stated on [lo, hi], under the same mapping."""
        return PushedForwardField(self._source.on_interval(lo, hi), self._mapping,
                                  name=self._name)

    def rescaled(self, *, k: float, v: float) -> "PushedForwardField":
        """The source rescaled by (k, v) under the mapping conjugated by k."""
        return PushedForwardField(self._source.rescaled(k=k, v=v),
                                  ScaledMapping(self._mapping, k), name=self._name)

    def renamed(self, name: str | None) -> "PushedForwardField":
        out = PushedForwardField(self._source, self._mapping, name=name)
        out._name = name
        return out

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        return f"PushedForwardField({nm}{self._source!r} under {self._mapping!r})"


class PulledBackField(FieldBase):
    """A field known on the physical body, stated referentially.

    `physical(r, theta, phi)` is a callable of the physical point's
    spherical coordinates returning components in the spherical frame
    there, full or Voigt for ranks 2 and 4 (a constant tensor is
    broadcast).  At a reference point X the field evaluates `physical`
    at m(X), rotates to Cartesian components with the frame at m(X),
    pulls back with F(X) and J(X), and presents the referential tensor
    in the frame at X.  No inverse mapping is involved.  All three
    coordinates are required and the field is never radial.  A formula
    returning complex values is refused.  `rescaled` conjugates the
    mapping by k and reads `physical` at r / k, times v.
    """

    is_radial = False

    def __init__(self, interval, physical, mapping, *, character: Character,
                 name: str | None = None, rtol: float = 1e-9) -> None:
        if not callable(physical):
            raise TypeError(f"expected a callable, got {type(physical).__name__}")
        lo, hi = (float(x) for x in interval)
        if not hi > lo:
            raise ValueError(f"an interval must increase, got ({lo:g}, {hi:g})")
        self._interval = (lo, hi)
        self._physical = physical
        self._mapping = mapping
        self._character = character
        self._name = name
        self._rtol = float(rtol)

    @property
    def physical(self):
        """The formula of the physical point's spherical coordinates."""
        return self._physical

    @property
    def mapping(self):
        """The mapping the physical values are pulled back through."""
        return self._mapping

    def _values(self, r, theta, phi):
        rank = self._character.rank
        X = cartesian_points(r, theta, phi)
        rp, tp, pp, Rp = spherical_coordinates(self._mapping(X))
        raw = np.asarray(self._physical(rp, tp, pp))
        if np.iscomplexobj(raw):
            raise TypeError(f"{self!r} returned complex values; a field is real")
        vals = _to_stored(raw, r.shape, self._character, self)
        if self._character.voigt_shape is not None:
            vals = voigt_to_tensor(vals, rank=rank)
        vals = rotate_slots(vals, Rp, rank)
        F = np.asarray(self._mapping.deformation_gradient(X), dtype=float)
        J = np.asarray(self._mapping.jacobian(X), dtype=float)
        ref = pull_back(vals, F, J, self._character)
        if rank:
            R = spherical_frame(theta, phi)
            ref = rotate_slots(ref, np.swapaxes(R, -1, -2), rank)
        return _reduce(ref, self._character)

    def on_interval(self, lo: float, hi: float) -> "PulledBackField":
        """The same physical formula and mapping on [lo, hi]."""
        return PulledBackField((lo, hi), self._physical, self._mapping,
                               character=self._character, name=self._name,
                               rtol=self._rtol)

    def rescaled(self, *, k: float, v: float) -> "PulledBackField":
        """v physical(r / k, theta, phi) under the mapping conjugated by k."""
        k, v = float(k), float(v)
        fn = self._physical
        lo, hi = self._interval
        return PulledBackField(
            (k * lo, k * hi), lambda r, t, p: v * np.asarray(fn(r / k, t, p)),
            ScaledMapping(self._mapping, k), character=self._character,
            name=self._name, rtol=self._rtol)

    def renamed(self, name: str | None) -> "PulledBackField":
        return PulledBackField(self._interval, self._physical, self._mapping,
                               character=self._character, name=name,
                               rtol=self._rtol)

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return (f"PulledBackField({nm}{self._character} on [{lo:g}, {hi:g}] "
                f"under {self._mapping!r})")
