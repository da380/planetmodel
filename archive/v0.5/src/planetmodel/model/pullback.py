"""pullback.py -- a quantity known on the physical body, stated referentially.

Physical inputs are pulled back at construction, so that everything
downstream sees referential fields and one mapping, and no model
carries a second representation of itself.  A CRUST-1.0 crustal
modulus, a mantle tomography, a homogeneous body's density: all of them
are functions of where a point *is*, and the canonical state wants
functions of where a point *came from*.

The generic route is one line of algebra (Appendix B.8.1, run backwards):
evaluate the physical quantity at x = m(X), rotate its components into
Cartesian with the frame at x, and apply F^-1 to every slot with a
factor of J^+w.  `PulledBackField` is that, lazily, for any character.

The fast paths avoid materialising anything.  A physically isotropic
tensor is two scalars and the identity, and its pull-back is the same
two scalars and `Cinv = (F^T F)^-1` (Appendix B.8.2); a physically VTI
tensor is five scalars, the identity and an axis n, and its pull-back is
those five scalars, Cinv and the pulled axis `Ntil = F^-1 n` (B.8.3).
Both are the physical tensor's invariant form with every `d` replaced by
Cinv and every `n` by Ntil, times J: the pull-back is a change of
variables and the invariant form is written in the metric.
`PulledBackElasticField` is the pair of them; the generic route is its
oracle, not its fallback.

**The referential tensor is not isotropic.**  Cinv carries the
anisotropy the mapping induces, so a physically isotropic medium under a
non-uniform stretch pulls back to a fully general (21-modulus)
referential tensor.  That is why these classes report the *physical*
symmetry and store the physical moduli, and why the result is a Voigt
matrix rather than a `Symmetry` and two numbers.  It keeps the full
minor and major symmetries -- identical wrapping on every slot -- so the
Voigt reduction is faithful.

**Radial mappings.**  A radial map preserves direction, so the frames at
X and at m(X) coincide and the physical coordinates of m(X) are simply
`(r + h, theta, phi)`: the composition costs one displacement
evaluation, with no Cartesian round trip and no arccos.  `F e_r =
(1 + dh/dr) e_r` for the same reason, so `Ntil = e_r / (1 + dh/dr)` with
no inverse.  Both shortcuts are taken for a `RadialStretch` and the
general route runs otherwise; the tests check that they agree.
"""
from __future__ import annotations

import numpy as np

from .character import ELASTIC, Character, Symmetry
from .fields.composite import FieldBase
from .frames import (cartesian_points, rotate_slots, spherical_coordinates,
                     spherical_frame)
from .mapping import RadialStretch
from .materials import (MODULI_NAMES, check_frame, tensor_to_voigt,
                        voigt_to_tensor)
from .pushforward import pull_back
from .units import Dimensions

__all__ = ["PullBackBase", "PulledBackField", "PulledBackElasticField",
           "pulled_back_elastic"]

#: The moduli each supported physical symmetry class is stated by.
_MODULI_OF = {Symmetry.ISOTROPIC: ("kappa", "mu"), Symmetry.VTI: MODULI_NAMES}


def _four(c) -> np.ndarray:
    """A per-point coefficient reshaped to multiply a rank-4 value."""
    return np.asarray(c, dtype=float)[..., None, None, None, None]


class PullBackBase(FieldBase):
    """Geometry shared by the pull-back constructors.

    Subclasses set `mapping`, `skeleton`, `character` and `name`, and
    supply the referential Cartesian value; everything about *where* the
    physical quantity is asked for, and how the answer is presented,
    lives here so that the fast paths and the generic route cannot drift
    apart in their geometry.  A user's own pull-back (a physical
    quantity of some other symmetry, say) subclasses this and overrides
    `evaluate`, using `_points`, `_physical_point` and `_present`.
    """

    mapping = None
    skeleton = None

    @property
    def is_radial(self) -> bool:
        """False: the value depends on direction through the mapping.

        Even where the physical quantity is spherically symmetric and
        the mapping radial, F varies with direction wherever the relief
        does, so this is not a promise that can be made from the
        physical field alone.
        """
        return False

    # -- where the question is asked ---------------------------------------

    def _points(self, r, theta, phi):
        """(r, theta, phi) broadcast, and the Cartesian reference points X.

        The radius is checked against the skeleton here, because the
        physical quantity is an arbitrary callable and will happily
        extrapolate: the skeleton belongs to this field, not to it.
        """
        if theta is None or phi is None:
            raise ValueError(
                f"{type(self).__name__} needs theta and phi as well as r: the "
                "pull-back applies F^-1 to every slot and evaluates the "
                "physical quantity at m(X), both of which depend on "
                "direction even where the physical quantity does not")
        r, theta, phi = np.broadcast_arrays(
            np.asarray(r, dtype=float), np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float))

        b = np.asarray(self.skeleton.boundaries, dtype=float)
        tol = 1e-9 * (b[-1] - b[0])
        if np.any(r < b[0] - tol) or np.any(r > b[-1] + tol):
            raise ValueError(
                f"radius outside the skeleton [{b[0]:.6g}, {b[-1]:.6g}]: a "
                "pull-back is stated on the reference body, and the physical "
                "quantity would extrapolate rather than refuse")
        return r, theta, phi, cartesian_points(r, theta, phi)

    def _physical_point(self, r, theta, phi, X):
        """(r, theta, phi) of x = m(X), and the local frame R there.

        For a `RadialStretch` this is the shortcut of the module
        docstring: the direction is untouched, so the physical point is
        (r + h, theta, phi) and its frame is the frame at X.  One
        displacement evaluation, against a mapping call, a norm and an
        arccos for the general route.
        """
        if isinstance(self.mapping, RadialStretch):
            h = np.asarray(self.mapping.h(r, theta, phi), dtype=float)
            return r + h, theta, phi, spherical_frame(theta, phi)
        return spherical_coordinates(self.mapping(X))

    # -- how the answer is presented ---------------------------------------

    def _present(self, out, theta, phi, *, frame: str, voigt: bool):
        """Rotate to the asked-for frame at X, and Voigt-reduce.

        The referential value is Cartesian throughout the algebra,
        because that is the frame the mapping speaks.  `spherical`
        rotates with the local frame at the **reference** point X, which
        for a radial mapping is also the frame at m(X) and so the one a
        seismologist would name.
        """
        rank = self.character.rank
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


#: The name the base carried while it was private.


class PulledBackField(PullBackBase):
    """A physical quantity expressed on the reference body, lazily.

    `physical(r, theta, phi)` is a callable of **physical** spherical
    coordinates returning the quantity's components in the spherical
    frame *at the physical point*, at full tensor rank: a scalar for
    rank 0, `(..., 3)` for rank 1, and the Voigt `(..., 6)` or
    `(..., 6, 6)` for ranks 2 and 4, which are expanded here.  Full
    components are accepted for those ranks too, since their trailing
    shapes are unambiguous.

    `evaluate` takes REFERENCE coordinates and returns the referential
    value there:

        X <- (r, theta, phi);  x = m(X);  T = physical(x) in the frame
        at x;  T_ref = J^w (F^-1 ...) T,  presented in the frame at X.

    All three coordinates are required (a field that depends on angle
    raises when they are omitted), and `is_radial` is False, because F
    depends on direction whatever the physical quantity does.

    `layer` and `side` are accepted and ignored: they select a side of a
    discontinuity of a *referential* field, and a plain function of
    physical position has no such notion.  A physical model with
    interfaces should be given as a callable that resolves them itself.

    This is the generic route -- one evaluation, one rotation and one
    einsum per slot -- and it is the oracle for the elastic fast paths
    below rather than their fallback.
    """

    def __init__(self, physical, mapping, *, skeleton, character: Character,
                 dimensions=None, name: str | None = None) -> None:
        """Bind a physical callable, the mapping, and what it is."""
        if not callable(physical):
            raise TypeError(
                "physical must be a callable of physical spherical "
                f"coordinates (r, theta, phi), got {type(physical).__name__}")
        self.physical = physical
        self.mapping = mapping
        self.skeleton = skeleton
        self.character = character
        self.dimensions = dimensions
        self.name = name

    def _physical_values(self, rp, tp, pp) -> np.ndarray:
        """The physical components at (rp, tp, pp), at full rank."""
        rank = self.character.rank
        vals = np.asarray(self.physical(rp, tp, pp), dtype=float)
        if rank not in (2, 4):
            want = (3,) * rank
            if rank and vals.shape[-rank:] != want:
                raise ValueError(
                    f"a rank-{rank} physical value has trailing shape {want}, "
                    f"got {vals.shape}")
            return vals
        voigt = {2: (6,), 4: (6, 6)}[rank]
        if vals.shape[-len(voigt):] == voigt:
            return voigt_to_tensor(vals, rank=rank)
        if vals.shape[-rank:] == (3,) * rank:
            return vals
        raise ValueError(
            f"a rank-{rank} physical value has trailing shape {voigt} in "
            f"Voigt form or {(3,) * rank} in full, got {vals.shape}")

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical",
                 voigt: bool = True):
        """The referential value at the reference point (r, theta, phi)."""
        check_frame(frame)
        rank = self.character.rank
        r, theta, phi, X = self._points(r, theta, phi)
        rp, tp, pp, Rp = self._physical_point(r, theta, phi, X)

        vals = self._physical_values(rp, tp, pp)
        if rank:
            vals = rotate_slots(vals, Rp, rank)     # spherical at x -> Cartesian

        F = np.asarray(self.mapping.deformation_gradient(X), dtype=float)
        J = np.asarray(self.mapping.jacobian(X), dtype=float)
        out = pull_back(vals, F, J, self.character)
        return self._present(out, theta, phi, frame=frame, voigt=voigt)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return (f"PulledBackField({self.character}, {self.mapping!r}{nm})")


class PulledBackElasticField(PullBackBase):
    """The referential second elasticity tensor of a physical medium.

    `moduli` are physical scalar callables `(r, theta, phi)` in physical
    spherical coordinates: `kappa` and `mu` for `Symmetry.ISOTROPIC`,
    and `A, C, F, L, N` for `Symmetry.VTI` with the symmetry axis radial
    at the *physical* point, `n = e_r(x)`.  `symmetry` reports that
    physical class; the referential tensor it builds is in general fully
    anisotropic, as the module docstring explains.

    Appendix B.8.2, physically isotropic, with lam = kappa - 2 mu / 3:

        A_ref = J [ lam Cinv (x) Cinv + mu (Cinv (x) Cinv)_sym ]

    Appendix B.8.3, physically VTI, is the five-term invariant form with
    every `d` pair replaced by Cinv and `n` by Ntil = F^-1 n, times J,
    coefficients

        c1 = A - 2N   c2 = N   c3 = F - A + 2N
        c4 = L - N    c5 = A + C - 2F - 4L.

    Cinv = (F^T F)^-1 is formed numerically from F, as the appendix
    directs: it is one 3x3 inverse per point and the closed form would
    be a second place for the same algebra to be wrong.

    **The VTI axis.**  `n = e_r(x)`, the radial direction at the
    *physical* point, which is what "VTI" means and what B.8.3 says
    ("radial: the direction of X for a radial map").  In the spherical
    frame at x that vector has components (1, 0, 0), so the physical
    tensor is the invariant form with the axis at the **first** index
    and its Voigt matrix carries C at entry (1, 1) -- the layout
    `materials.voigt_matrix` builds, pinned there against the invariant
    form with n = e_r.

    The result is Cartesian in the reference frame before presentation,
    and is Voigt-reduced on the way out unless `voigt=False`; the
    reduction is faithful because the construction is symmetric in
    (A,B), in (C,D) and under the pair swap by inspection.
    """

    character = ELASTIC

    def __init__(self, symmetry: Symmetry, moduli: dict, mapping, *,
                 skeleton, name: str | None = None) -> None:
        """Bind the physical moduli of a symmetry class and the mapping."""
        expected = _MODULI_OF.get(symmetry)
        if expected is None:
            raise NotImplementedError(
                f"PulledBackElasticField has no fast path for {symmetry}; "
                "ISOTROPIC (B.8.2) and VTI (B.8.3) are the closed forms the "
                "appendix gives, and anything else goes through "
                "PulledBackField with the generic rule")
        missing = [k for k in expected if k not in moduli]
        if missing:
            raise ValueError(
                f"{symmetry} needs {list(expected)}; missing {missing}")
        extra = [k for k in moduli if k not in expected]
        if extra:
            raise ValueError(f"{symmetry} takes only {list(expected)}; got {extra}")

        self.symmetry = symmetry
        self.moduli = dict(moduli)
        self.mapping = mapping
        self.skeleton = skeleton
        self.name = name

    @property
    def dimensions(self):
        """An elastic tensor's components are moduli: pressure."""
        return Dimensions.MODULUS

    @property
    def moduli_names(self) -> tuple[str, ...]:
        """The physical moduli held, in the order the class names them."""
        return tuple(self.moduli)

    # -- the two closed forms ----------------------------------------------

    @staticmethod
    def _isotropic(vals, Cinv, J):
        """B.8.2, verbatim."""
        kappa, mu = vals["kappa"], vals["mu"]
        lam = kappa - 2.0 * mu / 3.0
        return _four(J) * (
            _four(lam) * np.einsum("...AB,...CD->...ABCD", Cinv, Cinv)
            + _four(mu) * (np.einsum("...AC,...BD->...ABCD", Cinv, Cinv)
                           + np.einsum("...AD,...BC->...ABCD", Cinv, Cinv)))

    @staticmethod
    def _vti(vals, Cinv, J, Nt):
        """B.8.3, verbatim, with NN_AB = Ntil_A Ntil_B."""
        A, C, F_, L, N = (vals[k] for k in MODULI_NAMES)
        c1, c2, c3 = A - 2.0 * N, N, F_ - A + 2.0 * N
        c4, c5 = L - N, A + C - 2.0 * F_ - 4.0 * L
        NN = Nt[..., :, None] * Nt[..., None, :]
        return _four(J) * (
            _four(c1) * np.einsum("...AB,...CD->...ABCD", Cinv, Cinv)
            + _four(c2) * (np.einsum("...AC,...BD->...ABCD", Cinv, Cinv)
                           + np.einsum("...AD,...BC->...ABCD", Cinv, Cinv))
            + _four(c3) * (np.einsum("...AB,...CD->...ABCD", Cinv, NN)
                           + np.einsum("...AB,...CD->...ABCD", NN, Cinv))
            + _four(c4) * (np.einsum("...AC,...BD->...ABCD", NN, Cinv)
                           + np.einsum("...AD,...BC->...ABCD", NN, Cinv)
                           + np.einsum("...BC,...AD->...ABCD", NN, Cinv)
                           + np.einsum("...BD,...AC->...ABCD", NN, Cinv))
            + _four(c5) * np.einsum("...A,...B,...C,...D->...ABCD",
                                    Nt, Nt, Nt, Nt))

    def _pulled_axis(self, F, Rp, r, theta, phi):
        """Ntil = F^-1 n, for n = e_r at the physical point.

        A radial stretch takes e_r to (1 + dh/dr) e_r, so the inverse
        acts on it by a division and nothing is solved.  For any other
        mapping it is one linear solve per point, which is where the
        general route pays for its generality.
        """
        n = Rp[..., :, 0]                       # e_r(x), Cartesian
        if isinstance(self.mapping, RadialStretch):
            dr = np.asarray(self.mapping.h.radial_derivative(r, theta, phi),
                            dtype=float)
            return n / (1.0 + dr)[..., None]
        return np.linalg.solve(F, n[..., None])[..., 0]

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical",
                 voigt: bool = True):
        """The referential elastic tensor at the reference point.

        `layer` and `side` are accepted and ignored, as in
        PulledBackField: the moduli are functions of physical position.
        """
        check_frame(frame)
        r, theta, phi, X = self._points(r, theta, phi)
        rp, tp, pp, Rp = self._physical_point(r, theta, phi, X)
        vals = {k: np.asarray(f(rp, tp, pp), dtype=float)
                for k, f in self.moduli.items()}

        F = np.asarray(self.mapping.deformation_gradient(X), dtype=float)
        J = np.asarray(self.mapping.jacobian(X), dtype=float)
        Cinv = np.linalg.inv(np.einsum("...kA,...kB->...AB", F, F))

        if self.symmetry is Symmetry.ISOTROPIC:
            out = self._isotropic(vals, Cinv, J)
        else:
            out = self._vti(vals, Cinv, J,
                            self._pulled_axis(F, Rp, r, theta, phi))
        return self._present(out, theta, phi, frame=frame, voigt=voigt)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return (f"PulledBackElasticField({self.symmetry.name.lower()}{nm}, "
                f"moduli={list(self.moduli)}, {self.mapping!r})")


def pulled_back_elastic(symmetry: Symmetry, moduli: dict, mapping, *,
                        skeleton, name: str | None = None):
    """The referential elastic field of a physical medium, by symmetry.

    Dispatch on the physical symmetry: ISOTROPIC and VTI have the closed
    forms of B.8.2 and B.8.3 and get `PulledBackElasticField`; anything
    else is refused by name, pointing at the generic route, because a
    wrong fast path is worse than none.
    """
    if symmetry in _MODULI_OF:
        return PulledBackElasticField(symmetry, moduli, mapping,
                                      skeleton=skeleton, name=name)
    raise NotImplementedError(
        f"no pull-back fast path for {symmetry}: Appendix B.8 gives closed "
        "forms for the physically isotropic and VTI cases only.  Express the "
        "medium as a callable returning its Voigt matrix in the spherical "
        "frame at the physical point and use PulledBackField, whose generic "
        "rule (B.8.1 with F^-1) covers every symmetry class")
