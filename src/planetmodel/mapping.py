"""Mappings from the reference body to the physical one.

A mapping m takes points of the spherical reference body to points of
the physical body.  With

    F = (grad m)^T,  F[i, j] = d m_i / d X_j,      J = det F,

the committed interface is three methods on Cartesian points of shape
(..., 3):

    class Mapping(Protocol):
        def __call__(self, X)              -> m(X),  (..., 3)
        def deformation_gradient(self, X)  -> F,     (..., 3, 3)
        def jacobian(self, X)              -> J,     (...)

Anything exposing these is a Mapping.  An optional tier, discovered by
attribute, adds `displacement(X) = m(X) - X`, `is_valid(...)`,
`inverse(x)`, `linearise(delta, X=...)`, `knots` (radii where F may
jump), `is_identity`, and `scaled(k)`, the mapping k m(X / k) of a body
k times the size in the mapping's own form; MappingBase provides
generic versions of the first two, and `scaled_mapping(m, k)` asks for
`scaled` and falls back on a `ScaledMapping`.  A radial mapping also
carries `h`, its radial displacement, and
`deformation_gradient_spherical(X)`.

Mappings are Cartesian-first because meshes and solvers are; the
conversion to and from (r, theta, phi) happens once, through
`planetmodel.frames`.  Nothing here knows about units: every tolerance
is relative to a length the mapping is given.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from .coordinates import SPHERICAL
from .displacement import (CallableDisplacement, RadialDisplacement, SphericalFunction,
                           ZeroDisplacement, as_displacement)
from .frames import cartesian_points, spherical_coordinates

if TYPE_CHECKING:
    from .geometry import Geometry
    from .skeleton import Skeleton

__all__ = [
    "Mapping", "MappingBase", "IdentityMapping", "RadialStretch",
    "ScaledMapping", "ValidityReport", "MappingPerturbation",
    "validity_lattice", "outer_radius_of", "SphericalSample",
    "R_FLOOR_FRACTION", "SIN_FLOOR", "validity", "scaled_mapping",
]

#: A set of points given as (r, theta, phi) arrays that broadcast against
#: one another, as `validity_lattice` builds and `is_valid(sample=...)`
#: takes.
type SphericalSample = tuple[ArrayLike, ArrayLike, ArrayLike]

#: A point closer to the origin than this fraction of `rmax` is treated as
#: the origin, where the radial frame is undefined.
R_FLOOR_FRACTION = 1e-9

#: The inverse of a radial stretch: at most this many widenings of each
#: bracket by half, and this many safeguarded Newton steps, every one
#: of which halves the bracket when it does not take the Newton step.
_WIDENINGS = 60
_ITERATIONS = 100

#: sin(theta) below this is the polar axis, where the 1/(r sin theta)
#: entry of F is taken as its limit along the meridian.
SIN_FLOOR = 1e-12


def _per_sin(h: RadialDisplacement, r: np.ndarray, theta: np.ndarray,
             phi: np.ndarray, dph: ArrayLike) -> np.ndarray:
    """(1 / sin theta) dh/dphi, the component of h's angular gradient along
    e_phi.  On the polar axis both factors vanish and the component is its
    limit along the meridian: dh/dtheta a quarter turn on, at phi + pi / 2,
    negated at the south pole, where e_phi is minus that meridian's
    e_theta."""
    dph = np.asarray(dph, dtype=float)
    sin_t = np.sin(theta)
    out = dph / np.maximum(sin_t, SIN_FLOOR)
    axis = sin_t < SIN_FLOOR
    if np.any(axis):
        rr, tt, pp = (np.broadcast_to(a, axis.shape)[axis] for a in (r, theta, phi))
        quarter = np.asarray(h.angular_gradient(rr, tt, pp + 0.5 * np.pi)[0],
                             dtype=float)
        out = np.array(np.broadcast_to(out, axis.shape))
        out[axis] = np.where(np.cos(tt) > 0.0, 1.0, -1.0) * quarter
    return out


@runtime_checkable
class Mapping(Protocol):
    """m: reference body -> physical body, with its gradient and Jacobian.

    Three methods on Cartesian points of shape (..., 3): `__call__(X)`,
    `deformation_gradient(X)` returning (..., 3, 3) with
    F[i, j] = d m_i / d X_j, and `jacobian(X)` returning det F.
    """

    def __call__(self, X: ArrayLike) -> np.ndarray:
        """m(X): the physical points of reference points X, (..., 3)."""
        ...

    def deformation_gradient(self, X: ArrayLike) -> np.ndarray:
        """F at X, (..., 3, 3), with F[i, j] = d m_i / d X_j."""
        ...

    def jacobian(self, X: ArrayLike) -> np.ndarray:
        """J = det F at X, of the points' shape."""
        ...


@dataclass(frozen=True)
class ValidityReport:
    """Whether a mapping preserves orientation, and where it is worst.

    `margin` is dimensionless: the smallest factor of J found.  For an
    invalid mapping `worst_point` is (r, theta, phi) of the failure and
    `reason` names the failing factor.
    """

    valid: bool
    margin: float
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
    """dF (..., 3, 3) and dJ (...) induced by a perturbation of the mapping."""

    dF: np.ndarray
    dJ: np.ndarray


def validity_lattice(skeleton: Skeleton, *, n_r: int = 8, n_theta: int = 25,
                     n_phi: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A (r, theta, phi) sample covering a skeleton, for `is_valid(sample=...)`.

    `n_r` radii are laid strictly inside every layer of the skeleton, so
    a thin layer is sampled as densely as a thick one, and the centre is
    sampled where the skeleton holds it; theta covers [0, pi] including
    both poles and phi covers [-pi, pi).  Returns broadcastable arrays of
    shapes (n, 1, 1), (1, n_theta, 1) and (1, 1, n_phi), n being
    nlayers*n_r, one more for the centre.  It is `SPHERICAL.lattice`
    under the names of the spherical coordinates.
    """
    return SPHERICAL.lattice(skeleton, n_c=n_r, n_a=n_theta, n_b=n_phi)


def outer_radius_of(domain: float | Skeleton | Geometry) -> float:
    """The outer radius a mapping is meant for: a number, or the outer
    boundary of a skeleton or of a geometry."""
    sk = getattr(domain, "skeleton", domain)
    b = getattr(sk, "boundaries", None)
    r = float(b[-1]) if b is not None else float(domain)
    if not r > 0.0:
        raise ValueError(f"the outer radius must be positive, got {r}")
    return r


def _points_or_sample(X: ArrayLike | None, sample: SphericalSample | None
                      ) -> np.ndarray:
    """Cartesian points from either argument of `is_valid`, or ValueError."""
    if X is None and sample is not None:
        X = cartesian_points(*sample)
    if X is None:
        raise ValueError("give either points X or a (r, theta, phi) sample")
    return np.asarray(X, dtype=float)


def validity(m: Mapping, *, X: ArrayLike | None = None,
             sample: SphericalSample | None = None) -> ValidityReport:
    """A mapping's verdict on the points or sample: its own `is_valid`
    where it has one, else J > 0 there (`MappingBase.is_valid`)."""
    if hasattr(m, "is_valid"):
        return m.is_valid(X=X, sample=sample)
    return MappingBase.is_valid(m, X=X, sample=sample)


def scaled_mapping(m: Mapping, k: float) -> Mapping:
    """m conjugated by a change of length scale, k m(X / k), for k > 0:
    the mapping's own `scaled(k)` where it has one, which keeps its form
    (a stack of boundary reliefs stays a stack), else a `ScaledMapping`."""
    k = float(k)
    if not k > 0.0:
        raise ValueError(f"the scale factor must be positive, got {k}")
    own = getattr(m, "scaled", None)
    return own(k) if callable(own) else ScaledMapping(m, k)


class MappingBase:
    """Generic implementations of the optional tier, built on m, F and J.

    Mix it in for `displacement`, `right_cauchy_green`, `is_valid`, and
    refusals of `inverse` and `linearise`; nothing requires it.
    """

    def __call__(self, X: ArrayLike) -> np.ndarray:
        """m(X); the concrete mapping supplies it."""
        raise NotImplementedError

    def deformation_gradient(self, X: ArrayLike) -> np.ndarray:
        """F at X; the concrete mapping supplies it."""
        raise NotImplementedError

    def jacobian(self, X: ArrayLike) -> np.ndarray:
        """J = det F at X; the concrete mapping supplies it."""
        raise NotImplementedError

    def right_cauchy_green(self, X: ArrayLike) -> np.ndarray:
        """C = F^T F."""
        F = self.deformation_gradient(X)
        return np.einsum("...ki,...kj->...ij", F, F)

    def displacement(self, X: ArrayLike) -> np.ndarray:
        """u(X) = m(X) - X, Cartesian in and out."""
        X = np.asarray(X, dtype=float)
        return np.asarray(self(X), dtype=float) - X

    def is_valid(self, *, X: ArrayLike | None = None,
                 sample: SphericalSample | None = None) -> ValidityReport:
        """Whether J > 0 on the given points or on a (r, theta, phi) sample.

        The verdict is only as good as the sample; `validity_lattice`
        builds one that covers every layer and the whole sphere.
        """
        X = _points_or_sample(X, sample)
        J = np.asarray(self.jacobian(X), dtype=float)
        worst = float(np.min(J))
        if worst > 0.0:
            return ValidityReport(True, worst)
        i = int(np.argmin(J.reshape(-1)))
        P = X.reshape(-1, 3)[i]
        r, theta, phi, _ = spherical_coordinates(P)
        return ValidityReport(
            False, worst, worst_point=(float(r), float(theta), float(phi)),
            reason=f"J = {worst:.4g} is not positive")

    def inverse(self, x: ArrayLike) -> np.ndarray:
        """X such that m(X) = x, where the mapping can invert itself."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an inverse")

    def linearise(self, delta: RadialDisplacement | SphericalFunction, *,
                  X: ArrayLike) -> MappingPerturbation:
        """dF and dJ at points X for a perturbation of the mapping."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a linearisation")

    @property
    def is_identity(self) -> bool:
        """Whether this mapping is known to move nothing."""
        return False


class IdentityMapping(MappingBase):
    """The mapping that moves nothing: F = I, J = 1."""

    knots: tuple[float, ...] = ()

    def __call__(self, X: ArrayLike) -> np.ndarray:
        """A float copy of X."""
        return np.array(np.asarray(X, dtype=float))

    def deformation_gradient(self, X: ArrayLike) -> np.ndarray:
        """The identity on every point."""
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(np.eye(3), X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X: ArrayLike) -> np.ndarray:
        """Ones of the points' shape."""
        X = np.asarray(X, dtype=float)
        return np.ones(X.shape[:-1])

    def inverse(self, x: ArrayLike) -> np.ndarray:
        """A float copy of x."""
        return np.array(np.asarray(x, dtype=float))

    def scaled(self, k: float) -> "IdentityMapping":
        """The identity, on a body of any size."""
        return self

    def is_valid(self, *, X: ArrayLike | None = None,
                 sample: SphericalSample | None = None) -> ValidityReport:
        """Valid everywhere, whatever the sample."""
        return ValidityReport(True, 1.0)

    @property
    def is_identity(self) -> bool:
        """True: this is the identity."""
        return True

    def __repr__(self) -> str:
        return "IdentityMapping()"


class RadialStretch(MappingBase):
    """m(X) = (r + h) e_r: every point moves along its own radius.

    In the local frame (e_r, e_theta, e_phi), which is the same at X and
    at m(X) because direction is preserved,

        F_rr    = 1 + dh/dr
        F_r,th  = (1/r) dh/dtheta
        F_r,ph  = (1/(r sin theta)) dh/dphi
        F_th,th = F_ph,ph = 1 + h/r
        J = (1 + dh/dr) (1 + h/r)^2

    and orientation is preserved exactly when both factors are positive.
    Cartesian components are R F R^T with R the frame matrix.  `rmax` is
    the outer radius of the domain the mapping is meant for, given as a
    number or as a skeleton or geometry whose outer boundary it is: it
    sets the scale of "close to the origin" and brackets the inverse.
    """

    def __init__(self, h: RadialDisplacement | SphericalFunction, *,
                 rmax: float | Skeleton | Geometry, name: str | None = None) -> None:
        self.h = as_displacement(h)
        self.rmax = outer_radius_of(rmax)
        self.name = name

    @property
    def knots(self) -> tuple[float, ...]:
        """The radii where dh/dr, and so F, may jump."""
        return tuple(getattr(self.h, "knots", ()))

    def _floor(self) -> float:
        return R_FLOOR_FRACTION * self.rmax

    def _ratio(self, r: np.ndarray, h: np.ndarray, dr: np.ndarray) -> np.ndarray:
        """h / r, and its limit dh/dr at the centre, below the floor."""
        floor = self._floor()
        return np.where(r > floor, h / np.maximum(r, floor), dr)

    def _radial(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """(h, dh/dr) at the points."""
        return (np.asarray(self.h(r, theta, phi), dtype=float),
                np.asarray(self.h.radial_derivative(r, theta, phi), dtype=float))

    def _spherical(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """(F in the local frame, the frame matrix R) at X."""
        r, theta, phi, R = spherical_coordinates(X)
        h, dr = self._radial(r, theta, phi)
        dth, dph = self.h.angular_gradient(r, theta, phi)
        rs = np.maximum(r, self._floor())
        F = np.zeros(np.shape(r) + (3, 3))
        stretch = 1.0 + self._ratio(r, h, dr)
        F[..., 0, 0] = 1.0 + dr
        F[..., 0, 1] = np.asarray(dth, dtype=float) / rs
        F[..., 0, 2] = _per_sin(self.h, r, theta, phi,
                                np.asarray(dph, dtype=float)) / rs
        F[..., 1, 1] = stretch
        F[..., 2, 2] = stretch
        return F, R

    def __call__(self, X: ArrayLike) -> np.ndarray:
        """The displaced points x = (r + h) e_r; the origin stays put."""
        X = np.asarray(X, dtype=float)
        r, theta, phi, _ = spherical_coordinates(X)
        if np.any(r <= self._floor()):
            h, dr = self._radial(r, theta, phi)
            return X * (1.0 + self._ratio(r, h, dr))[..., None]
        h = np.asarray(self.h(r, theta, phi), dtype=float)
        return X * (1.0 + h / r)[..., None]

    def deformation_gradient_spherical(self, X: ArrayLike) -> np.ndarray:
        """F in the local (e_r, e_theta, e_phi) frame, shape (..., 3, 3)."""
        return self._spherical(X)[0]

    def deformation_gradient(self, X: ArrayLike) -> np.ndarray:
        """F in Cartesian components, R F_sph R^T."""
        F, R = self._spherical(X)
        return R @ F @ np.swapaxes(R, -1, -2)

    def jacobian(self, X: ArrayLike) -> np.ndarray:
        """J = (1 + dh/dr)(1 + h/r)^2."""
        r, theta, phi, _ = spherical_coordinates(X)
        h, dr = self._radial(r, theta, phi)
        return (1.0 + dr) * (1.0 + self._ratio(r, h, dr)) ** 2

    def is_valid(self, *, X: ArrayLike | None = None,
                 sample: SphericalSample | None = None) -> ValidityReport:
        """The analytic conditions 1 + dh/dr > 0 and 1 + h/r > 0.

        Conditions on the mapping itself rather than on a discrete
        Jacobian, checked at the given points or sample.  A displacement
        that is not zero at the origin is reported as its own failure,
        with margin h / rmax.
        """
        if X is not None:
            r, theta, phi, _ = spherical_coordinates(X)
        elif sample is not None:
            r, theta, phi = np.broadcast_arrays(
                np.asarray(sample[0], dtype=float),
                np.asarray(sample[1], dtype=float),
                np.asarray(sample[2], dtype=float))
        else:
            raise ValueError("give either points X or a (r, theta, phi) sample")

        h, dr = self._radial(r, theta, phi)
        floor = self._floor()

        def at(k: int) -> tuple[float, float, float]:
            return (float(np.ravel(r)[k]), float(np.ravel(theta)[k]),
                    float(np.ravel(phi)[k]))

        radial = 1.0 + dr
        tangential = 1.0 + self._ratio(r, h, dr)

        centre = (r <= floor) & (np.abs(h) > floor)
        if np.any(centre):
            k = int(np.flatnonzero(centre.reshape(-1))[0])
            hk = float(h.reshape(-1)[k])
            return ValidityReport(
                False, hk / self.rmax, worst_point=at(k),
                reason=f"h = {hk:.4g} at the centre, where a radial "
                "displacement has no direction to act along")

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
                reason=f"1 + h/r = {worst_tangential:.4g} is not positive, so "
                "shells cross through the origin")
        return ValidityReport(True, min(worst_radial, worst_tangential))

    def inverse(self, x: ArrayLike, *, tol: float = 1e-12) -> np.ndarray:
        """X with m(X) = x, by a safeguarded Newton iteration on every ray.

        Direction is preserved, so each point is one scalar problem:
        find s with s + h(s, theta, phi) equal to the physical radius.
        Where the mapping is valid the residual increases with s and the
        root is unique.  Each point keeps a bracket, [0, max(rmax, 1.5
        times its radius)] widened until it holds the root, and takes the
        Newton step when it lands strictly inside the bracket and the
        bisection when it does not, so that Newton cycling between two
        layers, where the residual is linear in s on each and every step
        lands on an end of the bracket, gives way to the bisection; a
        point whose residual is already positive at the centre goes to
        the centre.  `tol` is relative to `rmax`.
        """
        x = np.asarray(x, dtype=float)
        rho, theta, phi, _ = spherical_coordinates(x)
        shape = np.shape(rho)
        rho, theta, phi = (np.ravel(a) for a in np.broadcast_arrays(rho, theta, phi))
        xtol = tol * self.rmax

        def residual(s: np.ndarray, k: np.ndarray) -> np.ndarray:
            return s + np.asarray(self.h(s, theta[k], phi[k]), dtype=float) - rho[k]

        every = np.arange(rho.size)
        lo, hi = np.zeros(rho.size), np.maximum(self.rmax, 1.5 * rho)
        live = residual(lo, every) <= 0.0
        for _ in range(_WIDENINGS):
            k = every[live]
            k = k[residual(hi[k], k) < 0.0]
            if not k.size:
                break
            hi[k] *= 1.5
        s = np.where(live, np.clip(rho, lo, hi), 0.0)
        for _ in range(_ITERATIONS):
            k = every[live]
            if not k.size:
                break
            sk = s[k]
            f = residual(sk, k)
            df = 1.0 + np.asarray(
                self.h.radial_derivative(sk, theta[k], phi[k]), dtype=float)
            lo[k] = np.where(f < 0.0, sk, lo[k])
            hi[k] = np.where(f < 0.0, hi[k], sk)
            with np.errstate(divide="ignore", invalid="ignore"):
                step = sk - f / df
            inside = (df > 0.0) & (step > lo[k]) & (step < hi[k])
            new = np.where(inside, step, 0.5 * (lo[k] + hi[k]))
            new = np.where(f == 0.0, sk, new)
            s[k] = new
            live[k] = np.abs(new - sk) > xtol
        if np.any(live):
            raise RuntimeError(
                f"the inverse did not converge at {int(live.sum())} points in "
                f"{_ITERATIONS} iterations")
        return cartesian_points(*(a.reshape(shape) for a in (s, theta, phi)))

    def linearise(self, delta: RadialDisplacement | SphericalFunction, *,
                  X: ArrayLike) -> MappingPerturbation:
        """dF and dJ at X for a perturbation `delta` of the displacement.

            dF_rr    = d(dh)/dr
            dF_r,th  = (1/r) d(dh)/dtheta
            dF_r,ph  = (1/(r sin theta)) d(dh)/dphi
            dF_th,th = dF_ph,ph = dh / r
            dJ = d(dh)/dr (1 + h/r)^2 + 2 (1 + dh/dr)(1 + h/r) dh / r

        dF is returned in Cartesian components.
        """
        dh = as_displacement(delta)
        r, theta, phi, R = spherical_coordinates(X)
        h, dr = self._radial(r, theta, phi)
        rs = np.maximum(r, self._floor())

        d = np.asarray(dh(r, theta, phi), dtype=float)
        ddr = np.asarray(dh.radial_derivative(r, theta, phi), dtype=float)
        dth, dph = dh.angular_gradient(r, theta, phi)
        stretch, dstretch = 1.0 + self._ratio(r, h, dr), self._ratio(r, d, ddr)

        dF = np.zeros(np.shape(r) + (3, 3))
        dF[..., 0, 0] = ddr
        dF[..., 0, 1] = np.asarray(dth, dtype=float) / rs
        dF[..., 0, 2] = _per_sin(dh, r, theta, phi, dph) / rs
        dF[..., 1, 1] = dstretch
        dF[..., 2, 2] = dstretch
        dF_cart = R @ dF @ np.swapaxes(R, -1, -2)

        dJ = ddr * stretch ** 2 + 2.0 * (1.0 + dr) * stretch * dstretch
        return MappingPerturbation(dF_cart, dJ)

    def scaled(self, k: float) -> "RadialStretch | ScaledMapping":
        """k m(X / k): the stretch of the displacement's own `scaled(k)`,
        `rmax` multiplied by k, or a `ScaledMapping` of this stretch where
        the displacement has none."""
        own = getattr(self.h, "scaled", None)
        if not callable(own):
            return ScaledMapping(self, k)
        return RadialStretch(own(float(k)), rmax=float(k) * self.rmax, name=self.name)

    @property
    def is_identity(self) -> bool:
        """Whether the driving displacement is the zero one."""
        return isinstance(self.h, ZeroDisplacement)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"RadialStretch({self.h!r}{nm}, rmax={self.rmax:g})"


class ScaledMapping(MappingBase):
    """A mapping conjugated by a change of length scale: k m(X / k).

    Points of the scaled body are divided by k, mapped, and multiplied
    back.  F and J pass through unchanged, since both are dimensionless;
    `knots` are multiplied by k.  `is_valid` is the underlying mapping's
    `validity` in its own coordinates; `inverse` and `linearise` are its
    own where it has them (a perturbation delta of the scaled body is
    delta(k r) / k to it), refused otherwise.
    """

    def __init__(self, mapping: Mapping, k: float) -> None:
        self.mapping = mapping
        self.k = float(k)
        if not self.k > 0.0:
            raise ValueError(f"the scale factor must be positive, got {k}")

    def __call__(self, X: ArrayLike) -> np.ndarray:
        """k m(X / k)."""
        return self.k * np.asarray(self.mapping(np.asarray(X, dtype=float) / self.k),
                                   dtype=float)

    def deformation_gradient(self, X: ArrayLike) -> np.ndarray:
        """F of the underlying mapping at X / k, dimensionless as it is."""
        return self.mapping.deformation_gradient(np.asarray(X, dtype=float) / self.k)

    def jacobian(self, X: ArrayLike) -> np.ndarray:
        """J of the underlying mapping at X / k."""
        return self.mapping.jacobian(np.asarray(X, dtype=float) / self.k)

    def is_valid(self, *, X: ArrayLike | None = None,
                 sample: SphericalSample | None = None) -> ValidityReport:
        """The underlying mapping's verdict, in its own coordinates."""
        if X is not None:
            return validity(self.mapping, X=np.asarray(X, dtype=float) / self.k)
        if sample is not None:
            r, theta, phi = sample
            return validity(self.mapping,
                            sample=(np.asarray(r, dtype=float) / self.k, theta, phi))
        raise ValueError("give either points X or a (r, theta, phi) sample")

    def inverse(self, x: ArrayLike) -> np.ndarray:
        """k m^-1(x / k), where the underlying mapping inverts itself."""
        if not hasattr(self.mapping, "inverse"):
            return MappingBase.inverse(self, x)
        return self.k * np.asarray(
            self.mapping.inverse(np.asarray(x, dtype=float) / self.k), dtype=float)

    def linearise(self, delta: RadialDisplacement | SphericalFunction, *,
                  X: ArrayLike) -> MappingPerturbation:
        """The underlying mapping's dF and dJ at X / k, for delta(k r) / k."""
        inner = getattr(self.mapping, "linearise", None)
        if inner is None:
            return MappingBase.linearise(self, delta, X=X)
        d, k = as_displacement(delta), self.k

        def gradient(r: np.ndarray, theta: np.ndarray, phi: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
            gt, gp = d.angular_gradient(k * r, theta, phi)
            return np.asarray(gt, dtype=float) / k, np.asarray(gp, dtype=float) / k

        scaled = CallableDisplacement(
            lambda r, theta, phi: np.asarray(d(k * r, theta, phi), dtype=float) / k,
            radial_derivative=lambda r, theta, phi: d.radial_derivative(
                k * r, theta, phi),
            angular_gradient=gradient,
            knots=[x / k for x in getattr(d, "knots", ())])
        return inner(scaled, X=np.asarray(X, dtype=float) / k)

    def scaled(self, k: float) -> Mapping:
        """The underlying mapping scaled by this factor times k, through
        `scaled_mapping`."""
        return scaled_mapping(self.mapping, self.k * float(k))

    @property
    def knots(self) -> tuple[float, ...]:
        """The underlying mapping's knots, multiplied by k."""
        return tuple(self.k * float(k) for k in getattr(self.mapping, "knots", ()))

    @property
    def is_identity(self) -> bool:
        """Whether the underlying mapping is the identity."""
        return bool(getattr(self.mapping, "is_identity", False))

    def __repr__(self) -> str:
        return f"ScaledMapping({self.mapping!r}, k={self.k:g})"
