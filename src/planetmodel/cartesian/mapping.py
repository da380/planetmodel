"""The vertical stretch: every point of a flat body moves along the vertical.

The flat counterpart of the radial stretch (`mapping.RadialStretch`):
driven by one scalar function w(depth, x, y), the upward displacement
(`cartesian.displacement`, which states the sign once: heights, positive
up)::

    m(X) = X + w(-Z, X_x, X_y) e_z,

so that x and y are kept and the point at depth d goes to the physical
depth d - w.  With d/d depth = -d/dz::

    F = I + e_z (grad w)^T:  F_zx = dw/dx,  F_zy = dw/dy,  F_zz = 1 - dw/d depth,
    J = 1 - dw/d depth,

in the local frame (e_up, e_x, e_y) the same matrix with the vertical
row first, as F_rr, F_r,theta and F_r,phi head the radial stretch's.
Orientation is preserved exactly where 1 - dw/d depth > 0: where the
surfaces of the reference body keep their order, every layer a
positive thickness.  There is no centre, axis or tangential factor: a
column is its own one-dimensional problem, which is what the inverse
solves, by a safeguarded Newton iteration in each column.  The vertical
stretch is the mapping of topography: `cartesian.Geometry.stretched`
places a flat geometry under the stretch of a `layer_linear` stack of
reliefs, the free surface raised to its topography and the base held.

Every other part of the mapping machinery is shared: the `Mapping`
protocol and `MappingBase`, `IdentityMapping`, `scaled_mapping`, the
validity report, the push-forward and pull-back of fields, and the
pull-back of a fault (`fault.pull_back`), which asks the mapping for
its `inverse` alone.  Any object under the protocol serves a flat
geometry; the stretch is the one with a closed form here.  Nothing here
knows about units: every tolerance is relative to the depth of the
base the stretch is meant for.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..coordinates import SPHERICAL, Coordinates
from ..frames import SphericalFunction
from ..mapping import MappingBase, MappingPerturbation, SphericalSample, ValidityReport
from .coordinates import FLAT, LOCAL_FRAME
from .displacement import (CallableDisplacement, VerticalDisplacement, ZeroDisplacement,
                           as_displacement)

if TYPE_CHECKING:
    from ..geometry import GeometryBase
    from ..skeleton import Skeleton

__all__ = ["VerticalStretch"]

#: The inverse of a vertical stretch: at most this many widenings of each
#: column's bracket, and this many safeguarded Newton steps, every one of
#: which halves the bracket when it does not take the Newton step.
_WIDENINGS = 60
_ITERATIONS = 100


def _base_of(domain: float | Skeleton | GeometryBase) -> float:
    """The depth of the base a stretch is meant for: a number, or the last
    boundary of a skeleton or of a geometry's."""
    sk = getattr(domain, "skeleton", domain)
    b = getattr(sk, "boundaries", None)
    d = float(b[-1]) if b is not None else float(domain)
    if not d > 0.0:
        raise ValueError(f"the depth of the base must be positive, got {d}")
    return d


class VerticalStretch(MappingBase):
    """m(X) = X + w e_z: every point rises by w(depth, x, y), positive up;
    see the module docstring.

    `w` is any vertical displacement (a callable of (depth, x, y) is
    adapted by `cartesian.as_displacement`).  `base` is the depth of the
    base of the body the stretch is meant for, given as a number or as a
    skeleton or geometry whose last boundary it is: it sets the scale of
    the inverse's tolerance and of its first bracket.
    """

    def __init__(self, w: VerticalDisplacement | SphericalFunction, *,
                 base: float | Skeleton | GeometryBase, name: str | None = None
                 ) -> None:
        self.w = as_displacement(w)
        self.base = _base_of(base)
        self.name = name

    @property
    def knots(self) -> tuple[float, ...]:
        """The depths where dw/d depth, and so F, may jump."""
        return tuple(getattr(self.w, "knots", ()))

    def _columns(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(depth, x, y) of points X."""
        depth, x, y, _ = FLAT.split(X)
        return depth, x, y

    def __call__(self, X: ArrayLike) -> np.ndarray:
        """The displaced points: z raised by w, x and y kept."""
        X = np.asarray(X, dtype=float)
        out = np.array(X)
        out[..., 2] += np.asarray(self.w(*self._columns(X)), dtype=float)
        return out

    def displacement(self, X: ArrayLike) -> np.ndarray:
        """u(X) = m(X) - X = w e_z, Cartesian, exactly."""
        X = np.asarray(X, dtype=float)
        u = np.zeros(X.shape)
        u[..., 2] = np.asarray(self.w(*self._columns(X)), dtype=float)
        return u

    def _local(self, X: ArrayLike) -> np.ndarray:
        """F in the local frame (e_up, e_x, e_y) at X."""
        depth, x, y = self._columns(X)
        dd = np.asarray(self.w.depth_derivative(depth, x, y), dtype=float)
        gx, gy = self.w.lateral_gradient(depth, x, y)
        F = np.broadcast_to(np.eye(3), np.shape(depth) + (3, 3)).copy()
        F[..., 0, 0] = 1.0 - dd
        F[..., 0, 1] = np.asarray(gx, dtype=float)
        F[..., 0, 2] = np.asarray(gy, dtype=float)
        return F

    def deformation_gradient_local(self, X: ArrayLike) -> np.ndarray:
        """F in the local frame (e_up, e_x, e_y), (..., 3, 3): the vertical
        row (1 - dw/d depth, dw/dx, dw/dy) over the identity's."""
        return self._local(X)

    def deformation_gradient(self, X: ArrayLike) -> np.ndarray:
        """F in Cartesian components, R F_local R^T with R `LOCAL_FRAME`."""
        return LOCAL_FRAME @ self._local(X) @ LOCAL_FRAME.T

    def jacobian(self, X: ArrayLike) -> np.ndarray:
        """J = 1 - dw/d depth."""
        return 1.0 - np.asarray(self.w.depth_derivative(*self._columns(X)),
                                dtype=float)

    def validity(self, *, X: ArrayLike | None = None,
                 sample: SphericalSample | None = None,
                 coordinates: Coordinates | None = None) -> ValidityReport:
        """The analytic condition 1 - dw/d depth > 0, checked at the given
        points or sample.

        A condition on the mapping itself rather than on a discrete
        Jacobian: where it fails a surface of the reference body rises
        above the one over it, a layer turned inside out.  The sample is
        in `coordinates`, spherical when None as the protocol has it,
        and a geometry passes its own (`FLAT`); the worst point is
        reported in (depth, x, y), the coordinates of a vertical stretch.
        """
        if X is None and sample is not None:
            X = (SPHERICAL if coordinates is None else coordinates).points(*sample)
        if X is None:
            raise ValueError("give either points X or a (depth, x, y) sample")
        depth, x, y = self._columns(X)
        J = 1.0 - np.asarray(self.w.depth_derivative(depth, x, y), dtype=float)
        J = np.broadcast_to(J, np.broadcast(depth, x, y).shape)
        i = int(np.argmin(J.reshape(-1)))
        worst = float(J.reshape(-1)[i])
        if worst > 0.0:
            return ValidityReport(True, worst, labels=FLAT.labels)
        at = tuple(float(np.broadcast_to(v, J.shape).reshape(-1)[i])
                   for v in (depth, x, y))
        return ValidityReport(
            False, worst, worst_point=at, labels=FLAT.labels,
            reason=f"1 - dw/d depth = {worst:.4g} is not positive, so the mapping "
            "folds vertically: a surface of the reference body rises above the one "
            "over it")

    def inverse(self, x: ArrayLike, *, tol: float = 1e-12) -> np.ndarray:
        """X with m(X) = x, by a safeguarded Newton iteration in every column.

        x and y are kept, so each point is one scalar problem: the depth
        s with s - w(s, x, y) equal to the physical depth.  Where the
        mapping is valid the residual increases with s and the root is
        unique.  Each point keeps a bracket, its physical depth plus or
        minus `base`, widened until it holds the root, and takes the
        Newton step when it lands strictly inside the bracket and the
        bisection when it does not, so that Newton cycling between two
        layers of a piecewise-linear stretch gives way to the
        bisection.  `tol` is relative to `base`.
        """
        x = np.asarray(x, dtype=float)
        shape = x.shape[:-1]
        target, a, b = (np.ravel(v) for v in self._columns(x))
        xtol = tol * self.base

        def residual(s: np.ndarray, k: np.ndarray) -> np.ndarray:
            return s - np.asarray(self.w(s, a[k], b[k]), dtype=float) - target[k]

        every = np.arange(target.size)
        lo, hi = target - self.base, target + self.base
        for _ in range(_WIDENINGS):
            low = every[residual(lo, every) > 0.0]
            high = every[residual(hi, every) < 0.0]
            if not (low.size or high.size):
                break
            width = hi - lo
            lo[low] -= width[low]
            hi[high] += width[high]
        else:
            raise RuntimeError("the inverse found no bracket of the root at some "
                               "points: the mapping does not preserve orientation "
                               "in their columns")
        s = np.clip(target, lo, hi)
        live = np.ones(target.size, dtype=bool)
        for _ in range(_ITERATIONS):
            k = every[live]
            if not k.size:
                break
            sk = s[k]
            f = residual(sk, k)
            df = 1.0 - np.asarray(self.w.depth_derivative(sk, a[k], b[k]),
                                  dtype=float)
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
        return FLAT.points(*(v.reshape(shape) for v in (s, a, b)))

    def linearise(self, delta: VerticalDisplacement | SphericalFunction, *,
                  X: ArrayLike) -> MappingPerturbation:
        """dF and dJ at X for a perturbation `delta` of the displacement::

            dF_zx = d(dw)/dx,  dF_zy = d(dw)/dy,  dF_zz = -d(dw)/d depth,
            dJ = -d(dw)/d depth,

        exact, since F and J are linear in w; dF in Cartesian components.
        """
        dw = as_displacement(delta)
        depth, x, y = self._columns(X)
        dd = np.asarray(dw.depth_derivative(depth, x, y), dtype=float)
        gx, gy = dw.lateral_gradient(depth, x, y)
        dF = np.zeros(np.shape(depth) + (3, 3))
        dF[..., 2, 0] = np.asarray(gx, dtype=float)
        dF[..., 2, 1] = np.asarray(gy, dtype=float)
        dF[..., 2, 2] = -dd
        return MappingPerturbation(dF, -dd)

    def scaled(self, k: float) -> "VerticalStretch":
        """k m(X / k): the stretch of k w(X / k), the displacement's own
        `scaled(k)`, or a `CallableDisplacement`'s of it where it has none,
        `base` multiplied by k: a vertical stretch at any scale."""
        k = float(k)
        if not k > 0.0:
            raise ValueError(f"the scale factor must be positive, got {k}")
        own = getattr(self.w, "scaled", None)
        w = own(k) if callable(own) else CallableDisplacement(self.w).scaled(k)
        return VerticalStretch(w, base=k * self.base, name=self.name)

    @property
    def is_identity(self) -> bool:
        """Whether the driving displacement is the zero one."""
        return isinstance(self.w, ZeroDisplacement)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"VerticalStretch({self.w!r}{nm}, base={self.base:g})"
