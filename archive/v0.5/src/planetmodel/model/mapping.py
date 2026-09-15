"""mapping.py -- m : M_ref -> M_phys, and its derivatives.

**Direction.**  Throughout planetmodel, m maps the REFERENCE body to the
PHYSICAL one.  This is the direction of Myhill, Maitra & Al-Attar
(2026), whose xi is our m.  Al-Attar & Crawford (2016) use xi in the
opposite sense, so AAC16's xi is our m^-1: transcribing their equations
without applying that inverse introduces a spurious one.  With

    F = (grad m)^T,    J = det F,

a contravariant field of rank n and weight 1 pushes forward as
(1/J) F ... F applied to its indices; in particular rho_phys =
rho_ref / J, which is where the two papers agree.

**What is committed.**  The Mapping protocol is m, F and J.  Consumers
that need a = J C^-1, or C itself, form them from F and J in their own
weak form; the versions here exist for plotting and diagnostics and
carry no stability promise.  That division is deliberate: the geometry
is planetmodel's, the constitutive algebra is the consumer's.

**Coordinates.**  Mappings are Cartesian-first -- points arrive as
(..., 3) arrays -- because meshes and solvers are.  The field layer is
(r, theta, phi)-first because models are.  The asymmetry is accepted
rather than papered over, and each side converts once at its boundary,
through frames.py.

**Generality.**  Only radial mappings are shipped: both papers use them
alone, and MMA26 says plainly that for most applications there is little
benefit in more general ones.  A general mapping -- AAC16's time-one
flow of a vector field, say, for a configuration a radial map cannot
reach -- is any object exposing the three methods of the protocol, and
nothing here needs to know about it.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .character import Character
from .displacement import ZeroDisplacement, as_displacement
from .frames import cartesian_points, spherical_coordinates
from .pushforward import push_forward

__all__ = [
    "Mapping", "MappingBase", "IdentityMapping", "RadialStretch",
    "ValidityReport", "MappingPerturbation", "validity_lattice",
]

#: A point closer to the origin than this fraction of the body's size is
#: treated as the origin, where the radial frame is undefined.  A radial
#: map fixes the origin, so F is the identity there provided h(0) = 0,
#: which every sane displacement gives.  Relative, so the mapping means
#: the same thing in metres and in non-dimensional units.
_R_FLOOR_FRACTION = 1e-9

#: sin(theta) is clipped here so the 1/(r sin theta) entry stays finite
#: on the polar axis.  For a continuous topography dh/dphi vanishes
#: there, so the ratio has a finite limit and the clip only decides how
#: it is approached numerically.
_SIN_FLOOR = 1e-12


@runtime_checkable
class Mapping(Protocol):
    """m : M_ref -> M_phys, with its deformation gradient and Jacobian.

    Three methods, all taking Cartesian points of shape (..., 3):

        __call__(X)              -> x = m(X),          (..., 3)
        deformation_gradient(X)  -> F = (grad m)^T,    (..., 3, 3)
        jacobian(X)              -> J = det F,         (...)

    Anything exposing these is a Mapping, whether or not it inherits
    from anything here -- so a consumer's own object can be handed to
    the mesher without importing planetmodel at all.
    """

    def __call__(self, X): ...

    def deformation_gradient(self, X, *, frame: str = "cartesian"): ...

    def jacobian(self, X): ...


@dataclass(frozen=True)
class ValidityReport:
    """Whether a mapping preserves orientation, and where it is worst.

    A bare bool would say a mapping is about to tangle without saying
    where, which is the one thing the caller needs in order to act:
    lower an exaggeration, move a floor, refine a region.
    """

    valid: bool
    margin: float                       # min(1 + dh/dr), or min J - style
    _: KW_ONLY
    worst_point: tuple[float, float, float] | None = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        if self.valid:
            return f"ValidityReport(valid, margin={self.margin:.4g})"
        where = ("" if self.worst_point is None
                 else f" at (r, theta, phi) = ("
                      f"{self.worst_point[0]:.6g}, {self.worst_point[1]:.4f}, "
                      f"{self.worst_point[2]:.4f})")
        return f"ValidityReport(INVALID: {self.reason}{where})"


@dataclass(frozen=True)
class MappingPerturbation:
    """dF and dJ induced by a perturbation of the mapping.

    What a consumer contracts against to linearise its own weak form --
    MMA26 eq. (37) forms delta a from exactly these.
    """

    dF: np.ndarray
    dJ: np.ndarray


def validity_lattice(skeleton, *, n_r: int = 8, n_theta: int = 25,
                     n_phi: int = 16):
    """A (r, theta, phi) sample covering a body, for is_valid(sample=...).

    The verdict of is_valid is only as good as its sample, and the two
    ways to build a bad one are both easy: cover only part of the
    sphere, and a fold near one pole hides behind the other; or lay
    radii uniformly over the whole body, and a thin span -- PREM's
    outermost is 12 km in 6371 -- receives no points at all, though thin
    spans are exactly where dh/dr is largest.  So the radii here are
    laid per span, n_r strictly inside each one, and the angles cover
    the full sphere including both poles.

    Returns broadcastable arrays: r with shape (nspans*n_r, 1, 1), theta
    (1, n_theta, 1), phi (1, 1, n_phi).
    """
    b = np.asarray(skeleton.boundaries, dtype=float)
    rs = []
    for lo, hi in zip(b[:-1], b[1:]):
        inset = 1e-6 * (hi - lo)
        rs.append(np.linspace(lo + inset, hi - inset, n_r))
    r = np.concatenate(rs)
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    return (r[:, None, None], theta[None, :, None], phi[None, None, :])


#: The names the frame helpers carried while they lived here.


class MappingBase:
    """The [extra] tier: conveniences built on m, F and J.

    Optional. Implementations may mix it in for free derived quantities,
    or provide their own; nothing requires it.
    """

    # -- derived tensors, for diagnostics only ------------------------------

    def right_cauchy_green(self, X):
        """C = F^T F.  A convenience; no stability promise."""
        F = self.deformation_gradient(X)
        return np.einsum("...ki,...kj->...ij", F, F)

    def displacement(self, X):
        """u(X) = m(X) - X, the mapping as a vector field on the reference.

        The form every export writes.  A mesh already holds the
        reference coordinates, so what a consumer needs from the
        mapping is what to add to them: MFEM takes this as a
        GridFunction and gplspec writes it into the netCDF file.  For a
        RadialStretch it is h e_r and for a general mapping it is
        whatever it is, which is exactly why it is defined here in terms
        of m alone rather than read off a displacement.

        Cartesian in and Cartesian out, like every other method here.
        """
        X = np.asarray(X, dtype=float)
        return np.asarray(self(X), dtype=float) - X

    def gravity_tensor(self, X):
        """a = J C^-1, the combination MMA26's weak form contracts.

        Provided for plotting and for checking a consumer's own algebra,
        not as the committed interface: forming it here would put a
        gravity-specific quantity in a model library, and consumers can
        build it from F and J in one line.
        """
        J = self.jacobian(X)
        C = self.right_cauchy_green(X)
        return J[..., None, None] * np.linalg.inv(C)

    # -- validity -----------------------------------------------------------

    def is_valid(self, *, X=None, sample=None) -> ValidityReport:
        """Whether the mapping preserves orientation on the given points.

        The generic test is J > 0 at sampled points; RadialStretch
        sharpens it to the analytic conditions, which hold everywhere
        rather than only where they were checked.
        """
        if X is None and sample is not None:
            X = cartesian_points(*sample)
        if X is None:
            raise ValueError("give points X to check, or a sample")
        J = np.asarray(self.jacobian(X), dtype=float)
        worst = float(np.min(J))
        if worst > 0.0:
            return ValidityReport(True, worst)
        i = int(np.argmin(J.reshape(-1)))
        P = np.asarray(X, dtype=float).reshape(-1, 3)[i]
        r, theta, phi, _ = spherical_coordinates(P)
        return ValidityReport(
            False, worst, worst_point=(float(r), float(theta), float(phi)),
            reason=f"J = {worst:.4g} is not positive")

    # -- not provided by default -------------------------------------------

    def inverse(self, x, **kw):
        """X such that m(X) = x, where the mapping can invert itself."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an inverse; radial "
            "stretches do, by a scalar root-find")

    def linearise(self, delta, *, X=None) -> MappingPerturbation:
        """dF and dJ at points X, for a perturbation of the mapping."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a linearisation")

    # -- convenience --------------------------------------------------------

    def push_forward(self, field_values, X, character: Character):
        """Carry referential values at X across to the physical body."""
        return push_forward(field_values, self.deformation_gradient(X),
                            self.jacobian(X), character)

    @property
    def is_identity(self) -> bool:
        """Whether this mapping is known to move nothing."""
        return False


class IdentityMapping(MappingBase):
    """The mapping that moves nothing: F = I, J = 1.

    Why a spherically symmetric body needs no special case anywhere: it
    is a body whose mapping happens to be this one.
    """

    def __call__(self, X):
        return np.array(np.asarray(X, dtype=float))

    def deformation_gradient(self, X, *, frame: str = "cartesian"):
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(np.eye(3), X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        X = np.asarray(X, dtype=float)
        return np.ones(X.shape[:-1])

    def inverse(self, x, **kw):
        return np.array(np.asarray(x, dtype=float))

    def is_valid(self, *, X=None, sample=None) -> ValidityReport:
        return ValidityReport(True, 1.0)

    def linearise(self, delta, *, X=None) -> MappingPerturbation:
        """Perturbing the identity: dF and dJ of the perturbation alone."""
        return RadialStretch(ZeroDisplacement()).linearise(delta, X=X)

    @property
    def is_identity(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "IdentityMapping()"


class RadialStretch(MappingBase):
    """m(X) = (r + h) e_r: points move along their own radius.

    The workhorse.  Both papers use radial mappings only, and MMA26 says
    plainly that for most applications there is little benefit in more
    general ones.

    Everything is closed form.  Writing F in the local orthonormal frame
    (e_r, e_theta, e_phi) at X is legitimate for *both* index slots
    because a radial map preserves direction, so the frames at X and at
    m(X) coincide -- which is what makes the expressions this simple:

        F_rr    = 1 + dh/dr
        F_r,th  = (1/r) dh/dtheta
        F_r,ph  = (1/(r sin theta)) dh/dphi
        F_th,th = F_ph,ph = 1 + h/r
        (all other entries zero)

        J = det F = (1 + dh/dr) (1 + h/r)^2

    and orientation is preserved exactly when 1 + dh/dr > 0 and h > -r,
    which is the analytic half of the mesh-validity condition.

    Cartesian components follow from one batched conjugation, F_cart =
    R F_sph R^T, with R the frame matrix.
    """

    def __init__(self, h, *, rmax: float | None = None,
                 name: str | None = None) -> None:
        """Bind a displacement; anything callable is adapted.

        `h` is required: the map with no displacement is IdentityMapping,
        and a RadialStretch of a ZeroDisplacement is the same thing said
        the long way, which `is_identity` still recognises.  `rmax` is
        the body's outer radius, taken from the displacement's outermost
        knot when not given; it brackets the inverse and sets the scale
        of "close to the origin".
        """
        self.h = as_displacement(h)
        knots = tuple(getattr(self.h, "knots", ()))
        self.rmax = float(rmax) if rmax is not None else (
            float(knots[-1]) if knots else None)
        self.name = name

    # -- the pieces ---------------------------------------------------------

    def _floor(self, r) -> float:
        """The radius below which a point is the origin, for these points.

        A fraction of the body's size -- `rmax` where it is known, the
        largest radius in the batch otherwise -- so that the floor is
        the same geometric statement whatever the units.
        """
        scale = self.rmax if self.rmax is not None else float(np.max(r))
        return _R_FLOOR_FRACTION * (scale if scale > 0.0 else 1.0)

    def _parts(self, X):
        """(r, theta, phi, R, h, dh/dr, dh/dtheta, dh/dphi) at X."""
        r, theta, phi, R = spherical_coordinates(X)
        h = np.asarray(self.h(r, theta, phi), dtype=float)
        dr = np.asarray(self.h.radial_derivative(r, theta, phi), dtype=float)
        dth, dph = self.h.angular_gradient(r, theta, phi)
        return (r, theta, phi, R, h, dr,
                np.asarray(dth, dtype=float), np.asarray(dph, dtype=float))

    def __call__(self, X):
        """The displaced points, x = (r + h) e_r."""
        X = np.asarray(X, dtype=float)
        r, theta, phi, _ = spherical_coordinates(X)
        h = np.asarray(self.h(r, theta, phi), dtype=float)
        floor = self._floor(r)
        scale = np.where(r > floor, (r + h) / np.maximum(r, floor), 1.0)
        return X * scale[..., None]

    def deformation_gradient(self, X, *, frame: str = "cartesian"):
        """F, in Cartesian components by default.

        frame="spherical" returns the components in the local
        (e_r, e_theta, e_phi) frame, which is where the expressions are
        sparse and where they are easiest to check by eye.
        """
        if frame not in ("cartesian", "spherical"):
            raise ValueError("frame must be 'cartesian' or 'spherical'")
        r, theta, phi, R, h, dr, dth, dph = self._parts(X)
        rs = np.maximum(r, self._floor(r))
        # theta comes from arccos, so sin(theta) >= 0 and the floor is
        # the only guard needed.
        sin_t = np.maximum(np.sin(theta), _SIN_FLOOR)

        F = np.zeros(np.shape(r) + (3, 3))
        stretch = 1.0 + h / rs                       # tangential stretch
        F[..., 0, 0] = 1.0 + dr
        F[..., 0, 1] = dth / rs
        F[..., 0, 2] = dph / (rs * sin_t)
        F[..., 1, 1] = stretch
        F[..., 2, 2] = stretch
        if frame == "spherical":
            return F
        return np.einsum("...ik,...kl,...jl->...ij", R, F, R)

    def jacobian(self, X):
        """J = (1 + dh/dr)(1 + h/r)^2, in closed form."""
        r, theta, phi, _, h, dr, _, _ = self._parts(X)
        rs = np.maximum(r, self._floor(r))
        return (1.0 + dr) * (1.0 + h / rs) ** 2

    # -- validity, sharpened ------------------------------------------------

    def is_valid(self, *, X=None, sample=None) -> ValidityReport:
        """The analytic conditions 1 + dh/dr > 0 and h > -r.

        Unlike the generic J > 0 test these are conditions on the
        mapping itself, so checking them on a sample says something
        about the map everywhere it is smooth -- but the *discrete*
        condition, that each perturbed element keeps a positive
        Jacobian, is a different check and the mesher runs both.

        **The verdict is only as good as the sample.**  A mapping folds
        where the relief is steepest, which is usually at particular
        directions, so a sample covering only part of the sphere can
        report a folding mapping valid: relief that drives dh/dr
        strongly negative somewhere may be harmlessly positive
        everywhere the sample looked.  Pass points that cover the
        angular range -- the mesher passes its actual nodes, which do --
        or use validity_lattice(skeleton), which covers every span and
        the whole sphere by construction.
        """
        if X is not None:
            r, theta, phi, _ = spherical_coordinates(X)
        elif sample is not None:
            # Broadcast to one shape up front: the report indexes these
            # arrays with a flat index into the *broadcast* result, so
            # ragged inputs (a lattice of (n,1,1), (1,m,1), (1,1,k)
            # axes) must be expanded before anything is computed.
            r, theta, phi = np.broadcast_arrays(
                np.asarray(sample[0], dtype=float),
                np.asarray(sample[1], dtype=float),
                np.asarray(sample[2], dtype=float))
        else:
            raise ValueError("give either points X or a (r, theta, phi) sample")

        h = np.asarray(self.h(r, theta, phi), dtype=float)
        dr = np.asarray(self.h.radial_derivative(r, theta, phi), dtype=float)
        floor = self._floor(r)

        def at(k):
            return (float(np.ravel(r)[k]), float(np.ravel(theta)[k]),
                    float(np.ravel(phi)[k]))

        # The two factors of J = (1 + dh/dr)(1 + h/r)^2.  Both are
        # dimensionless, which is what makes a margin built from them
        # mean the same thing in SI and in scaled units, and both must
        # be positive for orientation to be preserved.
        #
        # Written instead as r + h > 0 -- the form the condition usually
        # takes -- the second test fails at r = 0 for *every* radial
        # map, since the centre is the fixed point and stays put.  It
        # also mixes a length with a pure number, so the reported margin
        # changes with the choice of units.
        radial = 1.0 + dr
        ratio = np.where(r > floor, h / np.maximum(r, floor), dr)
        tangential = 1.0 + ratio

        # A non-zero displacement AT the centre has no radial direction
        # to act along, and h/r diverges there: a broken displacement
        # rather than a fold, so it is reported as its own thing.
        centre = (r <= floor) & (np.abs(h) > floor)
        if np.any(centre):
            k = int(np.flatnonzero(centre.reshape(-1))[0])
            return ValidityReport(
                False, float(h.reshape(-1)[k]), worst_point=at(k),
                reason=f"h = {float(h.reshape(-1)[k]):.4g} at the centre, where a "
                "radial displacement has no direction to act along")

        i = int(np.argmin(radial.reshape(-1)))
        j = int(np.argmin(tangential.reshape(-1)))
        worst_radial = float(radial.reshape(-1)[i])
        worst_tangential = float(tangential.reshape(-1)[j])

        if worst_radial <= 0.0:
            return ValidityReport(
                False, worst_radial, worst_point=at(i),
                reason=f"1 + dh/dr = {worst_radial:.4g} is not positive, so the "
                "mapping folds radially")
        if worst_tangential <= 0.0:
            return ValidityReport(
                False, worst_tangential, worst_point=at(j),
                reason=f"1 + h/r = {worst_tangential:.4g} is not positive, so shells "
                "cross through the origin")
        return ValidityReport(True, min(worst_radial, worst_tangential))

    # -- inverse ------------------------------------------------------------

    def inverse(self, x, *, rmax: float | None = None, tol: float = 1e-12):
        """X with m(X) = x, by a scalar root-find along each ray.

        A radial map leaves the direction alone, so inverting it is one
        scalar problem per point: find s with s + h(s, theta, phi) equal
        to the physical radius.  Where the mapping is valid that map is
        strictly increasing, so the root is unique and bracketed by
        [0, rmax].  `tol` is relative to the body's size.
        """
        from scipy.optimize import brentq

        x = np.asarray(x, dtype=float)
        rho, theta, phi, _ = spherical_coordinates(x)
        top = (float(rmax) if rmax is not None
               else (self.rmax if self.rmax is not None
                     else float(np.max(rho)) * 1.5))
        scale = top if top > 0.0 else 1.0

        flat = np.broadcast_arrays(rho, theta, phi)
        out = np.empty(np.shape(rho))
        it = np.nditer(flat[0], flags=["multi_index"])
        for _ in it:
            k = it.multi_index
            target = float(flat[0][k])
            th, ph = float(flat[1][k]), float(flat[2][k])

            def residual(s, *, th=th, ph=ph, target=target):
                return float(s + self.h(s, th, ph)) - target

            lo, hi = 0.0, max(top, target * 1.5)
            if hi <= 0.0 or residual(lo) > 0.0:
                out[k] = 0.0
                continue
            grow = 0
            while residual(hi) < 0.0 and grow < 60:
                hi *= 1.5
                grow += 1
            out[k] = brentq(residual, lo, hi, xtol=tol * scale)

        return cartesian_points(out, theta, phi)

    # -- linearisation ------------------------------------------------------

    def linearise(self, delta, *, X=None) -> MappingPerturbation:
        """dF and dJ for a perturbation delta of the displacement.

        Both are linear in delta and its derivatives:

            dF_rr    = d(dh)/dr
            dF_r,th  = (1/r) d(dh)/dtheta
            dF_r,ph  = (1/(r sin theta)) d(dh)/dphi
            dF_th,th = dF_ph,ph = dh / r

            dJ = d(dh)/dr (1 + h/r)^2
                 + 2 (1 + dh/dr)(1 + h/r) dh / r

        which is what a consumer contracts to linearise its own weak
        form.  Call with the points at which the perturbation is wanted.
        """
        if X is None:
            raise ValueError("give the points X at which to linearise")
        dh_map = RadialStretch(delta, rmax=self.rmax)
        r, theta, phi, R, h, dr, _, _ = self._parts(X)
        rs = np.maximum(r, self._floor(r))

        d = np.asarray(dh_map.h(r, theta, phi), dtype=float)
        ddr = np.asarray(dh_map.h.radial_derivative(r, theta, phi), dtype=float)
        dth, dph = dh_map.h.angular_gradient(r, theta, phi)
        # theta comes from arccos, so sin(theta) >= 0 and the floor is
        # the only guard needed.
        sin_t = np.maximum(np.sin(theta), _SIN_FLOOR)

        dF = np.zeros(np.shape(r) + (3, 3))
        dF[..., 0, 0] = ddr
        dF[..., 0, 1] = np.asarray(dth, dtype=float) / rs
        dF[..., 0, 2] = np.asarray(dph, dtype=float) / (rs * sin_t)
        dF[..., 1, 1] = d / rs
        dF[..., 2, 2] = d / rs
        dF_cart = np.einsum("...ik,...kl,...jl->...ij", R, dF, R)

        dJ = ddr * (1.0 + h / rs) ** 2 + 2.0 * (1.0 + dr) * (1.0 + h / rs) * d / rs
        return MappingPerturbation(dF_cart, dJ)

    @property
    def is_identity(self) -> bool:
        return isinstance(self.h, ZeroDisplacement)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"RadialStretch({self.h!r}{nm})"
