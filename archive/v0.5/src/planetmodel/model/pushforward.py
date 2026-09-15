"""pushforward.py -- carrying fields across a mapping.

For a mapping m with F = (grad m)^T and J = det F, a contravariant field
of rank n and weight w transforms as

    T_phys,{i1..in}(x) = J^-w F_{i1 A1} ... F_{in An} T_ref,{A1..An}(X)

with x = m(X): one factor of F per slot, and the weight deciding whether
a factor of J divides it.  That is the whole rule (Appendix B.8.1), and
it is driven by the Character and nothing else -- rank 0 weight 1 is
rho/J, the point at which Al-Attar & Crawford (2016) eq. (71) and
Myhill, Maitra & Al-Attar (2026) agree; rank 0 weight 0 is composition;
rank 2 weight 1 carries the second Piola-Kirchhoff stress to the Cauchy
stress, F S F^T / J; rank 4 weight 1 carries the second elasticity
tensor.  The contraction is built from the rank, so nothing here is
written out per rank and nothing is special-cased.

The **pull-back** is the same rule with F^-1 in place of F and J^+w in
place of J^-w, so push_forward(pull_back(T)) is T.  A pull-back is what
a quantity known on the *physical* body needs in order to be stated
referentially; the physically isotropic and VTI closed forms of
Appendix B.8.2 and B.8.3 live in pullback.py, with the generic einsum
here as their oracle.

Two argument conventions
------------------------

`push_forward(values, F, J, character)` and `pull_back(...)` are the
**array** level: values, F and J already evaluated at the same points,
in Cartesian components, at full tensor rank.  Voigt input is not
accepted -- a Voigt matrix is a bookkeeping device, not a tensor with
four slots to wrap -- so expand first (materials.voigt_to_tensor).

`push_forward_field(field, mapping)` is the **field** level: a lazy
Field on the same skeleton whose evaluate takes REFERENCE coordinates
and returns the physical value there, forming F and J itself.  That is
the direction the exporter wants -- it walks reference dofs -- and it is
why nothing here inverts the mapping.

Complex values.  The rule is linear, so a complex tensor (a frozen
viscoelastic modulus, say) pushes forward as its real and imaginary
parts do, and the dtype follows the input.
"""
from __future__ import annotations

import numpy as np

from .character import Character
from .fields.composite import FieldBase
from .frames import (MAX_RANK, cartesian_points, rotate_slots,
                     rotation_subscripts, spherical_frame)
from .materials import check_frame, tensor_to_voigt, voigt_to_tensor

__all__ = ["push_forward", "pull_back", "push_forward_field",
           "PushedForwardField", "check_tensor_symmetries", "full_components",
           "MAX_RANK"]


def _subscripts(rank: int) -> str:
    """The einsum string of Appendix B.8.1 read off index by index."""
    return rotation_subscripts(rank)


def _as_values(values, rank: int, what: str) -> np.ndarray:
    """Values as an array, with the trailing shape the rank demands.

    Real input becomes float64 and complex input stays complex.  The
    error a caller is most likely to earn is handing over a Voigt
    matrix, which has the right number of numbers and the wrong shape,
    so it is named rather than left to a cryptic einsum failure.
    """
    values = np.asarray(values)
    if not np.iscomplexobj(values):
        values = np.asarray(values, dtype=float)
    want = (3,) * rank
    if rank and values.shape[-rank:] != want:
        voigt = {2: (6,), 4: (6, 6)}.get(rank)
        if voigt is not None and values.shape[-len(voigt):] == voigt:
            raise ValueError(
                f"{what} takes full components, not Voigt: a rank-{rank} "
                f"value has trailing shape {want}, and {values.shape} looks "
                "like a Voigt array.  Expand it with "
                "materials.voigt_to_tensor first")
        raise ValueError(
            f"a rank-{rank} value has trailing shape {want}, got {values.shape}")
    return values


def full_components(field, character: Character, r, theta, phi, *,
                    layer=None, side: str = "upper",
                    frame: str = "cartesian") -> np.ndarray:
    """A field's values at full tensor rank, in the asked-for frame.

    The one place that knows how to get 3**rank components out of a
    field that would rather hand over a Voigt array: a field offering
    `evaluate_full` is asked through it, and what `evaluate` returns is
    expanded otherwise.  A character whose `voigt_shape` is None -- rank
    0 or 1, or a first elasticity tensor -- has nothing to expand and is
    asked plainly.  Rank 0 is asked without a frame at all, since a
    scalar has no components to rotate and a field supporting only its
    own frame should not be refused over numbers that would be
    identical.  Real values become float64; complex ones stay complex.
    """
    rank = character.rank
    if rank == 0:
        return _as_values(field.evaluate(r, theta, phi, layer=layer, side=side),
                          0, "full_components")
    kw = dict(layer=layer, side=side, frame=frame)
    if character.voigt_shape is not None:
        full = getattr(field, "evaluate_full", None)
        if full is not None:
            return _as_values(full(r, theta, phi, **kw), rank, "full_components")
        return voigt_to_tensor(np.asarray(field.evaluate(r, theta, phi, **kw)),
                               rank=rank)
    return _as_values(field.evaluate(r, theta, phi, **kw), rank,
                      "full_components")


#: The name `full_components` carried while it was private.


def _weight_factor(J, rank: int) -> np.ndarray:
    """J reshaped so it divides or multiplies a rank-n value."""
    J = np.asarray(J, dtype=float)
    return J.reshape(J.shape + (1,) * rank)


def push_forward(values, F, J, character: Character):
    """The pushed-forward values of a field of the given character.

    `values` are the referential values at points X, in full component
    form of shape broadcast + (3,) * rank; `F` (..., 3, 3) and `J` (...)
    are the mapping's deformation gradient and Jacobian there.  Every
    slot gets one factor of F, and a weight-1 field is divided by J:

        T_ij..  =  J^-w F_iA F_jB ... T_AB..

    Weight-0 rank-0 fields are therefore returned unchanged -- for them
    composition with the mapping is the whole transformation -- and
    weight-1 scalars are divided by J.

    Components are whatever frame F is given in; the mapping layer is
    Cartesian, so in practice that is Cartesian.
    """
    values = _as_values(values, character.rank, "push_forward")
    out = rotate_slots(values, F, character.rank)
    if character.weight:
        out = out / _weight_factor(J, character.rank)
    return out


def pull_back(values, F, J, character: Character):
    """The pulled-back values: F^-1 on every slot, and J^+w.

    The inverse of push_forward at the same (F, J), so pushing forward
    what this returns recovers the input:

        T_AB..  =  J^+w (F^-1)_Ai (F^-1)_Bj ... T_ij..

    `values` are the physical values at x = m(X); the result is
    referential, at X.  Written with the inverse rather than by solving,
    because F is 3x3 and the same inverse serves every slot.
    """
    values = _as_values(values, character.rank, "pull_back")
    Finv = np.linalg.inv(np.asarray(F, dtype=float)) if character.rank else F
    out = rotate_slots(values, Finv, character.rank)
    if character.weight:
        out = out * _weight_factor(J, character.rank)
    return out


def check_tensor_symmetries(T, *, rtol: float = 1e-10) -> None:
    """Assert the minor and major symmetries of a rank-4 tensor.

    A debug helper, not part of any hot path: push-forward wraps all
    four slots identically, so it *cannot* break these symmetries in
    exact arithmetic (Appendix B.8.1), and this is what checks that the
    implementation does what the algebra says before a result is
    Voigt-reduced -- a reduction that keeps one slot per symmetry class
    and would silently discard the evidence of a bug.
    """
    T = np.asarray(T)
    if T.shape[-4:] != (3, 3, 3, 3):
        raise ValueError(f"expected (..., 3, 3, 3, 3), got {T.shape}")
    scale = float(np.max(np.abs(T))) or 1.0
    for name, swapped in (("minor (ij)", np.einsum("...ijkl->...jikl", T)),
                          ("minor (kl)", np.einsum("...ijkl->...ijlk", T)),
                          ("major", np.einsum("...ijkl->...klij", T))):
        err = float(np.max(np.abs(T - swapped))) / scale
        assert err <= rtol, (
            f"the {name} symmetry is broken by {err:.3e} relative, so this "
            "is not a second elasticity tensor and Voigt would not be "
            "faithful to it")


class PushedForwardField(FieldBase):
    """A field carried across a mapping, evaluated at reference points.

    Lazy: it holds the source field and the mapping and does the work in
    `evaluate`, so nothing is sampled and refitted.  `evaluate` takes
    REFERENCE coordinates (r, theta, phi) and returns the value the
    field has at the physical point m(X) -- which is what an exporter
    walking reference degrees of freedom asks for, and why the mapping
    is never inverted.

    All three coordinates are required, even for a source that is
    radial: F depends on direction, so the pushed-forward value does
    too.  That is also why `is_radial` is False.

    Frames.  The work is done in Cartesian components, because that is
    the frame the mapping speaks.  `frame="spherical"` rotates the
    result back with the local frame at the **reference** point X,
    R^T c R slot by slot.  For a radial mapping the frames at X and at
    m(X) coincide -- direction is preserved -- so that is also the local
    frame at the physical point, and the components are the ones a
    seismologist would name.  For a general mapping it is not: the frame
    reported is the reference one, said here so the choice is visible
    rather than assumed.

    Ranks 2 and 4 are Voigt-reduced on the way out, as everywhere else
    in planetmodel, unless `voigt=False` (or through `evaluate_full`).
    The reduction is faithful because identical wrapping on every slot
    preserves the minor and major symmetries (Appendix B.8.1);
    check_tensor_symmetries is the check, used by the tests rather than
    on every call.
    """

    def __init__(self, field, mapping, *, name: str | None = None) -> None:
        """Bind a source field and the mapping to carry it across."""
        self.source = field
        self.mapping = mapping
        self.skeleton = field.skeleton
        self.character = field.character
        self.dimensions = getattr(field, "dimensions", None)
        src_name = getattr(field, "name", None)
        self.name = name if name is not None else (
            f"{src_name}_phys" if src_name else None)

    @property
    def is_radial(self) -> bool:
        """False: the value depends on direction through the mapping.

        Even for a radial source under a radial stretch, J varies with
        direction wherever the relief does, so this is not a promise
        that can be made from the source alone.  An identity mapping is
        the case where it could be -- and push_forward_field returns the
        source itself there, so no PushedForwardField ever needs it.
        """
        return False

    def _reference_points(self, r, theta, phi):
        """(r, theta, phi) broadcast, and the Cartesian reference points X."""
        if theta is None or phi is None:
            raise ValueError(
                f"{type(self).__name__} needs theta and phi as well as r: the "
                "push-forward is F applied to every slot, and F depends on "
                "direction even where the source field does not")
        r, theta, phi = np.broadcast_arrays(
            np.asarray(r, dtype=float), np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float))
        return r, theta, phi, cartesian_points(r, theta, phi)

    def _source_cartesian(self, r, theta, phi, *, layer, side):
        """The source's values in Cartesian components, at full rank."""
        return full_components(self.source, self.character, r, theta, phi,
                               layer=layer, side=side, frame="cartesian")

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical",
                 voigt: bool = True):
        """The physical value at m(X), for X the given reference point.

        The source is evaluated first, so a radius outside the skeleton
        is refused by the field that owns the skeleton rather than
        silently mapped.
        """
        check_frame(frame)
        rank = self.character.rank
        r, theta, phi, X = self._reference_points(r, theta, phi)
        vals = self._source_cartesian(r, theta, phi, layer=layer, side=side)

        F = np.asarray(self.mapping.deformation_gradient(X), dtype=float)
        J = np.asarray(self.mapping.jacobian(X), dtype=float)
        out = push_forward(vals, F, J, self.character)

        if frame == "spherical" and rank:
            R = spherical_frame(theta, phi)
            out = rotate_slots(out, np.swapaxes(R, -1, -2), rank)
        if rank in (2, 4) and voigt:
            out = tensor_to_voigt(out, rank=rank)
        return out

    def evaluate_full(self, r, theta=None, phi=None, *, layer=None,
                      side: str = "upper", frame: str = "spherical"):
        """The value at full tensor rank: `evaluate` with `voigt=False`."""
        return self.evaluate(r, theta, phi, layer=layer, side=side,
                             frame=frame, voigt=False)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"PushedForwardField({self.source!r}, {self.mapping!r}{nm})"


def push_forward_field(field, mapping):
    """The image of `field` under `mapping`, as a lazy Field.

    The returned field lives on the same skeleton and has the same
    character; its `evaluate` takes reference coordinates and returns
    the physical value there (PushedForwardField).  It is named
    `<source>_phys` where the source is named.

    A mapping that moves nothing has nothing to do: F = I and J = 1
    exactly, so the source field *is* its own image and is returned
    unchanged, keeping its name, its `is_radial` and its cheap
    evaluation.  That is the case a spherically symmetric body is,
    which is why it needs no special-casing anywhere else.
    """
    if getattr(mapping, "is_identity", False):
        return field
    return PushedForwardField(field, mapping)
