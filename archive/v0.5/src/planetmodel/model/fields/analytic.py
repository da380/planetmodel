"""analytic.py -- a field given by a formula rather than by a table.

`RadialField` is a model read from disk; `AnalyticField` is a model
written down.  It wraps any `fn(r, theta, phi)` returning the components
of a field of the given Character, and is the landing point for the
three things planetmodel otherwise has no home for: a random-field sample
evaluated on demand, a manufactured solution whose exact answer is
known, and the closed-form test cases the push-forward and pull-back
oracles are built from.

It is deliberately the *general* field of the library, so it makes none
of RadialField's promises.  All three coordinates are required and
`is_radial` is False: a formula that happens to ignore theta and phi is
indistinguishable from one that does not, and silence is not a promise.
Where a formula really is radial, RadialField says so and
means it.

**Frames.**  `frame=` in the constructor names the frame `fn` speaks;
`frame=` in `evaluate` names the frame the caller wants.  Where they
differ the components are rotated with the local frame matrix
R = [e_r, e_theta, e_phi] of `frames.spherical_frame`, one factor per
slot -- R going spherical to Cartesian and R^T coming back -- which is
the same `rotate_slots` the push-forward uses, so there is one rotation in the
library and not two.  The rotation is done on full components and the
Voigt reduction happens last, which is why a Voigt matrix rotates
correctly here without the Bond matrix ever being named: Bond is the
Voigt shadow of exactly this operation.

**Voigt in, Voigt or full out.**  For ranks 2 and 4 with the symmetries,
`fn` may return either the full `(..., 3, 3)` / `(..., 3, 3, 3, 3)`
components or the Voigt `(..., 6)` / `(..., 6, 6)` form -- the trailing
shapes distinguish themselves -- and
`evaluate` presents the Voigt form, as everything else in planetmodel does.
"""
from __future__ import annotations

import numpy as np

from ..character import SCALAR, Character
from .composite import FieldBase

__all__ = ["AnalyticField"]


class AnalyticField(FieldBase):
    """A Field defined by a callable of (r, theta, phi).

    `fn(r, theta, phi)` receives the three coordinates already
    broadcast against one another and returns the components of a field
    of `character`, in the frame `frame` names ("spherical", the
    default, or "cartesian").  Ranks 2 and 4 may be returned either at
    full rank or in Voigt form.

    `skeleton` is the geometry the field is stated on; radii outside it
    are refused rather than extrapolated, because a formula will
    cheerfully continue and the skeleton belongs to the field rather
    than to the formula.

    `layer` and `side` are accepted and ignored.  They select one side
    of a discontinuity, and an analytic function of position has no
    sides: it takes one value at a radius, whichever layer the caller
    has in mind.  A model with interfaces is a RadialField, or a
    formula that resolves them itself.
    """

    def __init__(self, fn, skeleton, *, character: Character = SCALAR,
                 dimensions=None, name: str | None = None,
                 frame: str = "spherical") -> None:
        """Bind a formula, the geometry it is stated on, and what it is."""
        if not callable(fn):
            raise TypeError(
                "fn must be a callable of (r, theta, phi), got "
                f"{type(fn).__name__}")
        self._check_frame(frame)
        self._fn = fn
        self.skeleton = skeleton
        self.character = character
        self.dimensions = dimensions
        self.name = name
        #: The frame `fn` returns its components in.
        self.frame = frame

    @staticmethod
    def _check_frame(frame: str) -> None:
        """Refuse a frame name the library does not define."""
        if frame not in ("spherical", "cartesian"):
            raise ValueError(
                f"unknown frame {frame!r}: components are given in the "
                "'spherical' frame or the 'cartesian' one")

    @property
    def is_radial(self) -> bool:
        """False: a formula is not asked whether it uses its angles."""
        return False

    def restated(self, skeleton, *, name=None) -> "AnalyticField":
        """The same formula on another skeleton."""
        return AnalyticField(self._fn, skeleton, character=self.character,
                             dimensions=self.dimensions,
                             name=self.name if name is None else name,
                             frame=self.frame)

    def restricted(self, layer) -> "AnalyticField":
        """The formula on one layer: it has no sides, so nothing changes."""
        from ..skeleton import Skeleton
        return self.restated(Skeleton(self.skeleton.interval(layer)))

    def on_interval(self, lo: float, hi: float) -> "AnalyticField":
        """The formula on [lo, hi]; a formula extends wherever it is asked."""
        from ..skeleton import Skeleton
        return self.restated(Skeleton([lo, hi]))

    @property
    def fn(self):
        """The formula of (r, theta, phi)."""
        return self._fn

    def rescaled(self, convert, old, new):
        """A formula in one set of scales is not known in another."""
        raise TypeError(
            f"cannot rescale AnalyticField {self.name!r}: a formula in SI "
            "coordinates does not know how to be restated in others. "
            "Rebuild it after the rescale.")

    @classmethod
    def assembled(cls, skeleton, pieces, *, name=None):
        """One formula on the whole skeleton, if every piece is that formula."""
        first = pieces[0]
        same = all(isinstance(p, AnalyticField) and p._fn is first._fn
                   and p.frame == first.frame and p.character == first.character
                   for p in pieces)
        if not same:
            return NotImplemented
        covered = {i for i in range(skeleton.nlayers)
                   if any(p.skeleton.nlayers == 1
                          and skeleton.spans(*p.skeleton.boundaries, layer=i)
                          for p in pieces)}
        if covered != set(range(skeleton.nlayers)):
            return NotImplemented
        return first.restated(skeleton, name=name)

    _assembled = assembled

    # -- where the question is asked ---------------------------------------

    def _points(self, r, theta, phi):
        """(r, theta, phi) broadcast, with the radii checked.

        The angles are required.  A field that depends on angle must
        raise when they are omitted, and an arbitrary
        formula does depend on them until it says otherwise -- which
        this class, unlike RadialField, has no way to say.
        """
        if theta is None or phi is None:
            raise ValueError(
                f"{type(self).__name__} needs theta and phi as well as r: an "
                "analytic field is a formula in all three coordinates, and "
                "planetmodel will not guess which of them it happens to ignore")
        r, theta, phi = np.broadcast_arrays(
            np.asarray(r, dtype=float), np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float))

        b = np.asarray(self.skeleton.boundaries, dtype=float)
        tol = 1e-9 * (b[-1] - b[0])
        if r.size and (np.any(r < b[0] - tol) or np.any(r > b[-1] + tol)):
            raise ValueError(
                f"radius outside the skeleton [{b[0]:.6g}, {b[-1]:.6g}]: the "
                "formula would extrapolate rather than refuse")
        return r, theta, phi

    def _values(self, r, theta, phi) -> np.ndarray:
        """The formula's components at the points, at full tensor rank.

        The component axes must be there already -- a rank-2 field that
        returns one number per point has not said what it means -- but
        the point axes need not be: a constant field is a formula
        returning one tensor, and it is broadcast here.
        """
        rank = self.character.rank
        voigt = self.character.voigt_shape
        full = r.shape + self.character.component_shape
        vals = np.asarray(self._fn(r, theta, phi), dtype=float)

        if voigt is not None and vals.shape[-len(voigt):] == voigt:
            from ..materials import voigt_to_tensor
            vals = voigt_to_tensor(
                np.broadcast_to(vals, r.shape + voigt), rank=rank)
        elif rank and vals.shape[-rank:] != self.character.component_shape:
            want = (f"{self.character.component_shape} or {voigt}"
                    if voigt is not None
                    else str(self.character.component_shape))
            raise ValueError(
                f"a rank-{rank} analytic field returns components of "
                f"trailing shape {want}, got {vals.shape}")

        if vals.shape != full:
            try:
                # Copied rather than left a read-only broadcast view, so
                # a caller may write to what it is given.
                vals = np.broadcast_to(vals, full).copy()
            except ValueError:
                raise ValueError(
                    f"a rank-{rank} analytic field should return {full} at "
                    f"{r.shape} points, got {vals.shape}") from None
        return vals

    # -- how the answer is presented ---------------------------------------

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical"):
        """The components at (r, theta, phi), in the requested frame.

        Voigt-reduced for ranks 2 and 4 that have the symmetries, as
        everywhere else in planetmodel.  `layer` and `side` are accepted and
        ignored: an analytic function has no sides.
        """
        self._check_frame(frame)
        rank = self.character.rank
        r, theta, phi = self._points(r, theta, phi)
        vals = self._values(r, theta, phi)

        if rank and frame != self.frame:
            from ..frames import rotate_slots, spherical_frame
            R = spherical_frame(theta, phi)
            # R takes spherical components to Cartesian ones; R^T comes
            # back.  One factor per slot, which is what rotate_slots does.
            vals = rotate_slots(vals, R if frame == "cartesian"
                                else np.swapaxes(R, -1, -2), rank)

        if self.character.voigt_shape is not None:
            from ..materials import tensor_to_voigt
            vals = tensor_to_voigt(vals, rank=rank)
        return vals

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return (f"AnalyticField({self._fn!r}{nm}, {self.character}, "
                f"{self.frame})")

