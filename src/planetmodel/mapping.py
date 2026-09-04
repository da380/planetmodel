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
jump) and `is_identity`; MappingBase provides generic versions of the
first two.  A radial mapping also carries `h`, its radial displacement,
and `deformation_gradient_spherical(X)`.

Mappings are Cartesian-first because meshes and solvers are; the
conversion to and from (r, theta, phi) happens once, through
`planetmodel.frames`.  Nothing here knows about units: every tolerance
is relative to a length the mapping is given.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .displacement import ZeroDisplacement, as_displacement
from .frames import cartesian_points, spherical_coordinates

__all__ = [
    "Mapping", "MappingBase", "IdentityMapping", "RadialStretch",
    "ScaledMapping", "ValidityReport", "MappingPerturbation",
    "validity_lattice",
]

#: A point closer to the origin than this fraction of `rmax` is treated as
#: the origin, where the radial frame is undefined.
_R_FLOOR_FRACTION = 1e-9

#: sin(theta) is clipped here so the 1/(r sin theta) entry of F stays
#: finite on the polar axis, where dh/dphi vanishes for any continuous h.
_SIN_FLOOR = 1e-12


@runtime_checkable
class Mapping(Protocol):
    """m: reference body -> physical body, with its gradient and Jacobian.

    Three methods on Cartesian points of shape (..., 3): `__call__(X)`,
    `deformation_gradient(X)` returning (..., 3, 3) with
    F[i, j] = d m_i / d X_j, and `jacobian(X)` returning det F.
    """

    def __call__(self, X): ...

    def deformation_gradient(self, X): ...

    def jacobian(self, X): ...


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


def validity_lattice(skeleton, *, n_r: int = 8, n_theta: int = 25,
                     n_phi: int = 16):
    """A (r, theta, phi) sample covering a skeleton, for `is_valid(sample=...)`.

    `n_r` radii are laid strictly inside every layer of the skeleton, so
    a thin layer is sampled as densely as a thick one; theta covers
    [0, pi] including both poles and phi covers [-pi, pi).  Returns
    broadcastable arrays of shapes (nlayers*n_r, 1, 1), (1, n_theta, 1)
    and (1, 1, n_phi).
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


def _points_or_sample(X, sample):
    """Cartesian points from either argument of `is_valid`, or ValueError."""
    if X is None and sample is not None:
        X = cartesian_points(*sample)
    if X is None:
        raise ValueError("give either points X or a (r, theta, phi) sample")
    return np.asarray(X, dtype=float)


class MappingBase:
    """Generic implementations of the optional tier, built on m, F and J.

    Mix it in for `displacement`, `right_cauchy_green`, `is_valid`, and
    refusals of `inverse` and `linearise`; nothing requires it.
    """

    def right_cauchy_green(self, X):
        """C = F^T F."""
        F = self.deformation_gradient(X)
        return np.einsum("...ki,...kj->...ij", F, F)

    def displacement(self, X):
        """u(X) = m(X) - X, Cartesian in and out."""
        X = np.asarray(X, dtype=float)
        return np.asarray(self(X), dtype=float) - X

    def is_valid(self, *, X=None, sample=None) -> ValidityReport:
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

    def inverse(self, x):
        """X such that m(X) = x, where the mapping can invert itself."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an inverse")

    def linearise(self, delta, *, X) -> MappingPerturbation:
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

    def __call__(self, X):
        return np.array(np.asarray(X, dtype=float))

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(np.eye(3), X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        X = np.asarray(X, dtype=float)
        return np.ones(X.shape[:-1])

    def inverse(self, x):
        return np.array(np.asarray(x, dtype=float))

    def is_valid(self, *, X=None, sample=None) -> ValidityReport:
        return ValidityReport(True, 1.0)

    @property
    def is_identity(self) -> bool:
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
    the outer radius of the domain the mapping is meant for: it sets the
    scale of "close to the origin" and brackets the inverse.
    """

    def __init__(self, h, *, rmax: float, name: str | None = None) -> None:
        self.h = as_displacement(h)
        self.rmax = float(rmax)
        if not self.rmax > 0.0:
            raise ValueError(f"rmax must be positive, got {rmax}")
        self.name = name

    @property
    def knots(self) -> tuple[float, ...]:
        """The radii where dh/dr, and so F, may jump."""
        return tuple(getattr(self.h, "knots", ()))

    def _floor(self) -> float:
        return _R_FLOOR_FRACTION * self.rmax

    def _parts(self, X):
        """(r, theta, phi, R, h, dh/dr, dh/dtheta, dh/dphi) at X."""
        r, theta, phi, R = spherical_coordinates(X)
        h = np.asarray(self.h(r, theta, phi), dtype=float)
        dr = np.asarray(self.h.radial_derivative(r, theta, phi), dtype=float)
        dth, dph = self.h.angular_gradient(r, theta, phi)
        return (r, theta, phi, R, h, dr,
                np.asarray(dth, dtype=float), np.asarray(dph, dtype=float))

    def __call__(self, X):
        """The displaced points x = (r + h) e_r; the origin stays put."""
        X = np.asarray(X, dtype=float)
        r, theta, phi, _ = spherical_coordinates(X)
        h = np.asarray(self.h(r, theta, phi), dtype=float)
        floor = self._floor()
        scale = np.where(r > floor, (r + h) / np.maximum(r, floor), 1.0)
        return X * scale[..., None]

    def deformation_gradient_spherical(self, X):
        """F in the local (e_r, e_theta, e_phi) frame, shape (..., 3, 3)."""
        r, theta, phi, R, h, dr, dth, dph = self._parts(X)
        rs = np.maximum(r, self._floor())
        sin_t = np.maximum(np.sin(theta), _SIN_FLOOR)
        F = np.zeros(np.shape(r) + (3, 3))
        stretch = 1.0 + h / rs
        F[..., 0, 0] = 1.0 + dr
        F[..., 0, 1] = dth / rs
        F[..., 0, 2] = dph / (rs * sin_t)
        F[..., 1, 1] = stretch
        F[..., 2, 2] = stretch
        return F

    def deformation_gradient(self, X):
        """F in Cartesian components, R F_sph R^T."""
        _, _, _, R = spherical_coordinates(X)
        F = self.deformation_gradient_spherical(X)
        return np.einsum("...ik,...kl,...jl->...ij", R, F, R)

    def jacobian(self, X):
        """J = (1 + dh/dr)(1 + h/r)^2."""
        r, theta, phi, _, h, dr, _, _ = self._parts(X)
        rs = np.maximum(r, self._floor())
        return (1.0 + dr) * (1.0 + h / rs) ** 2

    def is_valid(self, *, X=None, sample=None) -> ValidityReport:
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

        h = np.asarray(self.h(r, theta, phi), dtype=float)
        dr = np.asarray(self.h.radial_derivative(r, theta, phi), dtype=float)
        floor = self._floor()

        def at(k):
            return (float(np.ravel(r)[k]), float(np.ravel(theta)[k]),
                    float(np.ravel(phi)[k]))

        radial = 1.0 + dr
        ratio = np.where(r > floor, h / np.maximum(r, floor), dr)
        tangential = 1.0 + ratio

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

    def inverse(self, x, *, tol: float = 1e-12):
        """X with m(X) = x, by a scalar root-find along each ray.

        Direction is preserved, so each point is one scalar problem:
        find s with s + h(s, theta, phi) equal to the physical radius.
        Where the mapping is valid the root is unique.  `tol` is relative
        to `rmax`.
        """
        from scipy.optimize import brentq

        x = np.asarray(x, dtype=float)
        rho, theta, phi, _ = spherical_coordinates(x)
        top = self.rmax

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
            if residual(lo) > 0.0:
                out[k] = 0.0
                continue
            grow = 0
            while residual(hi) < 0.0 and grow < 60:
                hi *= 1.5
                grow += 1
            out[k] = brentq(residual, lo, hi, xtol=tol * top)

        return cartesian_points(out, theta, phi)

    def linearise(self, delta, *, X) -> MappingPerturbation:
        """dF and dJ at X for a perturbation `delta` of the displacement.

            dF_rr    = d(dh)/dr
            dF_r,th  = (1/r) d(dh)/dtheta
            dF_r,ph  = (1/(r sin theta)) d(dh)/dphi
            dF_th,th = dF_ph,ph = dh / r
            dJ = d(dh)/dr (1 + h/r)^2 + 2 (1 + dh/dr)(1 + h/r) dh / r

        dF is returned in Cartesian components.
        """
        dh = as_displacement(delta)
        r, theta, phi, R, h, dr, _, _ = self._parts(X)
        rs = np.maximum(r, self._floor())

        d = np.asarray(dh(r, theta, phi), dtype=float)
        ddr = np.asarray(dh.radial_derivative(r, theta, phi), dtype=float)
        dth, dph = dh.angular_gradient(r, theta, phi)
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
        return f"RadialStretch({self.h!r}{nm}, rmax={self.rmax:g})"


class ScaledMapping(MappingBase):
    """A mapping conjugated by a change of length scale: k m(X / k).

    Points of the scaled body are divided by k, mapped, and multiplied
    back.  F and J pass through unchanged, since both are dimensionless;
    `knots` are multiplied by k.  `is_valid` and `inverse` delegate to
    the underlying mapping in its own coordinates.
    """

    def __init__(self, mapping, k: float) -> None:
        self.mapping = mapping
        self.k = float(k)
        if not self.k > 0.0:
            raise ValueError(f"the scale factor must be positive, got {k}")

    def __call__(self, X):
        return self.k * np.asarray(self.mapping(np.asarray(X, dtype=float) / self.k),
                                   dtype=float)

    def deformation_gradient(self, X):
        return self.mapping.deformation_gradient(np.asarray(X, dtype=float) / self.k)

    def jacobian(self, X):
        return self.mapping.jacobian(np.asarray(X, dtype=float) / self.k)

    def is_valid(self, *, X=None, sample=None) -> ValidityReport:
        if X is not None:
            return self.mapping.is_valid(X=np.asarray(X, dtype=float) / self.k)
        if sample is not None:
            r, theta, phi = sample
            return self.mapping.is_valid(
                sample=(np.asarray(r, dtype=float) / self.k, theta, phi))
        raise ValueError("give either points X or a (r, theta, phi) sample")

    def inverse(self, x):
        return self.k * np.asarray(
            self.mapping.inverse(np.asarray(x, dtype=float) / self.k), dtype=float)

    @property
    def knots(self) -> tuple[float, ...]:
        return tuple(self.k * float(k) for k in getattr(self.mapping, "knots", ()))

    @property
    def is_identity(self) -> bool:
        return bool(getattr(self.mapping, "is_identity", False))

    def __repr__(self) -> str:
        return f"ScaledMapping({self.mapping!r}, k={self.k:g})"
