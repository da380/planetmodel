"""The radial displacement h(r, theta, phi) that drives a radial stretch.

A radial mapping is driven by one scalar function: how far each point
moves along its own radius.  The callable is the interface::

    class RadialDisplacement(Protocol):
        def __call__(self, r, theta, phi) -> array

and everything else is an optional enrichment discovered by attribute::

    radial_derivative(r, theta, phi)   -> dh/dr
    angular_gradient(r, theta, phi)    -> (dh/dtheta, dh/dphi)
    knots                              -> radii where dh/dr may jump
    bounds()                           -> (min h, max h)
    scaled(k)                          -> k h(r / k, theta, phi)

the last the same displacement on a body k times the size, in its own
form, so that a change of length scale keeps a stack of reliefs a stack.

A bare function is accepted wherever a displacement is wanted, with the
derivatives taken by central differences; supplying them exactly gives
exact deformation gradients.  `as_displacement(h)` adapts any callable
to an object carrying every method, exact where `h` supplied it.  Two
displacements are shipped: `flattening`, the degree-2 shape of a
hydrostatic ellipsoid, and `layer_linear`, which gives every boundary of
a skeleton an analytic relief and interpolates linearly in r between
them within each layer.

Knots make smoothness checkable.  A displacement that is C0 but whose
radial derivative jumps declares the radii where it does, which a
geometry takes anywhere within its skeleton and a radial mesh takes as
element edges.  A displacement declaring no knots asserts that it is
smooth.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from .frames import SphericalFunction
from .skeleton import Skeleton

__all__ = ["RadialDisplacement", "ZeroDisplacement", "CallableDisplacement",
           "as_displacement", "flattening", "layer_linear", "LayerLinear",
           "AngularGradientFunction", "Relief", "angular_difference",
           "radial_difference"]

#: A function of (r, theta, phi) returning (dh/dtheta, dh/dphi).
type AngularGradientFunction = Callable[[np.ndarray, np.ndarray, np.ndarray],
                                        tuple[ArrayLike, ArrayLike]]

#: The shape of one boundary: a callable of (theta, phi), which may carry
#: an `angular_gradient(theta, phi)` attribute returning its two derivatives
#: and a `scaled(k)` one returning k times itself in its own form.
type Relief = Callable[[np.ndarray, np.ndarray], ArrayLike]


@runtime_checkable
class RadialDisplacement(Protocol):
    """h(r, theta, phi): how far a point moves along its radius.

    Required: `__call__(r, theta, phi) -> array`, broadcasting.
    Optional, used when present: `radial_derivative`, `angular_gradient`,
    `knots`, `bounds`, `scaled`.
    """

    def __call__(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                 ) -> np.ndarray:
        """The displacement at the broadcast coordinates."""
        ...


def _broadcast(r: ArrayLike, theta: ArrayLike, phi: ArrayLike
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three arguments broadcast to a common float array shape."""
    return np.broadcast_arrays(np.asarray(r, dtype=float),
                               np.asarray(theta, dtype=float),
                               np.asarray(phi, dtype=float))


def _across_pole(theta: np.ndarray, phi: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A colatitude stepped past a pole as the point it names: theta < 0
    is (-theta, phi + pi), theta > pi is (2 pi - theta, phi + pi)."""
    past = (theta < 0.0) | (theta > np.pi)
    t = np.where(theta < 0.0, -theta, np.where(theta > np.pi, 2.0 * np.pi - theta,
                                                theta))
    return t, np.where(past, phi + np.pi, phi)


def angular_difference(f: Callable[[np.ndarray, np.ndarray], ArrayLike],
                       theta: np.ndarray, phi: np.ndarray, *, step: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """(df/dtheta, df/dphi) of a function on the sphere by central
    differences of `step` radians; a colatitude stepped past a pole is
    read across it, at the opposite longitude, so that every evaluation
    lies on the sphere and the difference through a pole is that of the
    function's smooth continuation."""
    up, down = _across_pole(theta + step, phi), _across_pole(theta - step, phi)
    dt = (np.asarray(f(*up), dtype=float) - np.asarray(f(*down), dtype=float)) \
        / (2.0 * step)
    dp = (np.asarray(f(theta, phi + step), dtype=float)
          - np.asarray(f(theta, phi - step), dtype=float)) / (2.0 * step)
    return dt, dp


def radial_difference(f: Callable[[np.ndarray], ArrayLike], r: np.ndarray, *,
                      step: float) -> np.ndarray:
    """df/dr by central differences of `max(step, step |r|)`, one-sided
    (second order) where the lower point would fall below r = 0."""
    h = np.maximum(step, step * np.abs(r))
    central = r - h >= 0.0
    lo = np.where(central, r - h, r)
    mid = np.where(central, r, r + h)
    hi = r + np.where(central, h, 2.0 * h)
    f_lo, f_mid, f_hi = (np.asarray(f(x), dtype=float) for x in (lo, mid, hi))
    return np.where(central, (f_hi - f_lo) / (2.0 * h),
                    (-3.0 * f_lo + 4.0 * f_mid - f_hi) / (2.0 * h))


class ZeroDisplacement:
    """No displacement: the identity mapping's h."""

    knots: tuple[float, ...] = ()

    def __call__(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """Zeros of the broadcast shape."""
        r, _, _ = _broadcast(r, theta, phi)
        return np.zeros(r.shape)

    def radial_derivative(self, r: ArrayLike, theta: ArrayLike,
                          phi: ArrayLike) -> np.ndarray:
        """dh/dr: zeros of the broadcast shape."""
        return self(r, theta, phi)

    def angular_gradient(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(dh/dtheta, dh/dphi): zeros of the broadcast shape."""
        z = self(r, theta, phi)
        return z, z.copy()

    def bounds(self) -> tuple[float, float]:
        """(min h, max h): both zero."""
        return 0.0, 0.0

    def scaled(self, k: float) -> "ZeroDisplacement":
        """No displacement on a body of any size."""
        return self

    def __repr__(self) -> str:
        return "ZeroDisplacement()"


class CallableDisplacement:
    """Any callable h(r, theta, phi), with its derivatives exact or differenced.

    Derivatives passed in, or carried by `fn` as attributes named
    `radial_derivative` and `angular_gradient`, are used as given;
    otherwise differences are taken.  In the angles they are central
    with step `dstep` (`angular_difference`), a colatitude stepped past
    a pole being read across it at the opposite longitude, so that they
    stay central there; in the radius they are central with step
    `max(dstep, dstep * |r|)` (`radial_difference`), and one-sided where
    the lower point would fall below the centre.  `knots` declares the
    radii where dh/dr jumps, by default those `fn` carries; leaving them
    empty asserts smoothness.  `bounds` is `fn`'s where it has one.
    """

    def __init__(self, fn: SphericalFunction, *, knots: Iterable[float] = (),
                 radial_derivative: SphericalFunction | None = None,
                 angular_gradient: AngularGradientFunction | None = None,
                 name: str | None = None, dstep: float = 1e-6) -> None:
        if not callable(fn):
            raise TypeError(f"expected a callable, got {type(fn).__name__}")
        self._fn = fn
        self._dr = (radial_derivative if radial_derivative is not None
                    else getattr(fn, "radial_derivative", None))
        self._da = (angular_gradient if angular_gradient is not None
                    else getattr(fn, "angular_gradient", None))
        self._h = float(dstep)
        knots = tuple(knots) or tuple(getattr(fn, "knots", ()))
        self.knots = tuple(sorted(float(k) for k in knots))
        if callable(getattr(fn, "bounds", None)):
            self.bounds = fn.bounds
        self.name = name

    def __call__(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The wrapped callable at the broadcast coordinates."""
        r, theta, phi = _broadcast(r, theta, phi)
        return np.asarray(self._fn(r, theta, phi), dtype=float)

    def radial_derivative(self, r: ArrayLike, theta: ArrayLike,
                          phi: ArrayLike) -> np.ndarray:
        """dh/dr, exact if supplied, else a central difference."""
        if self._dr is not None:
            return np.asarray(self._dr(*_broadcast(r, theta, phi)), dtype=float)
        r, theta, phi = _broadcast(r, theta, phi)
        return radial_difference(lambda x: self(x, theta, phi), r, step=self._h)

    def angular_gradient(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(dh/dtheta, dh/dphi), exact if supplied, else differenced."""
        if self._da is not None:
            gt, gp = self._da(*_broadcast(r, theta, phi))
            return np.asarray(gt, dtype=float), np.asarray(gp, dtype=float)
        r, theta, phi = _broadcast(r, theta, phi)
        return angular_difference(lambda t, p: self(r, t, p), theta, phi,
                                  step=self._h)

    def scaled(self, k: float) -> "CallableDisplacement":
        """k h(r / k, theta, phi): dh/dr read at r / k and the angular
        gradient multiplied by k, each exact where this displacement's is
        and differenced as it is otherwise; the knots and the bounds
        multiplied by k."""
        k = float(k)

        def h(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
            return k * self(r / k, theta, phi)

        def dh_dr(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
            return self.radial_derivative(r / k, theta, phi)

        def dh_dangles(r: np.ndarray, theta: np.ndarray, phi: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
            gt, gp = self.angular_gradient(r / k, theta, phi)
            return k * gt, k * gp

        if callable(getattr(self, "bounds", None)):
            def bounds() -> tuple[float, float]:
                lo, hi = self.bounds()
                return k * float(lo), k * float(hi)
            h.bounds = bounds
        return CallableDisplacement(h, knots=[k * x for x in self.knots],
                                    radial_derivative=dh_dr,
                                    angular_gradient=dh_dangles, name=self.name,
                                    dstep=self._h)

    def __repr__(self) -> str:
        fn = getattr(self._fn, "__name__", None) or repr(self._fn)
        nm = f" {self.name!r}" if self.name else ""
        knots = f", knots={list(self.knots)}" if self.knots else ""
        return f"CallableDisplacement({fn}{nm}{knots})"


def as_displacement(fn: RadialDisplacement | SphericalFunction, *,
                    knots: Iterable[float] = (), **kw: object) -> RadialDisplacement:
    """Adapt any callable of (r, theta, phi) to the protocol.

    An object that already carries `radial_derivative`, `angular_gradient`
    and `knots` is returned unchanged; extra arguments are then refused
    rather than ignored.  Anything else is wrapped in a
    CallableDisplacement, which keeps what it carries of the three and
    its `bounds`.
    """
    if all(hasattr(fn, a) for a in
           ("radial_derivative", "angular_gradient", "knots")):
        if knots or kw:
            raise ValueError(
                f"{fn!r} already declares its knots and derivatives; the "
                "extra arguments would be ignored")
        return fn
    return CallableDisplacement(fn, knots=knots, **kw)


def flattening(f: float) -> CallableDisplacement:
    """h = -(2f/3) r P2(cos theta): the degree-2 shape of flattening `f`.

    Every sphere of the reference body becomes a spheroid of the same
    mean radius with polar radius r (1 - 2f/3) and equatorial radius
    r (1 + f/3), so that its flattening, the difference of the two over
    the equatorial radius, is f / (1 + f/3), f to first order.  h is
    proportional to r, so the shape is the same at every radius and on
    a body of any size.
    """
    f = float(f)
    a = 2.0 * f / 3.0

    def h(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return -a * r * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)

    def dh_dr(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return -a * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0) + 0.0 * r

    def dh_dangles(r: np.ndarray, theta: np.ndarray, phi: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
        return 3.0 * a * r * np.cos(theta) * np.sin(theta), 0.0 * r

    return CallableDisplacement(h, radial_derivative=dh_dr,
                                angular_gradient=dh_dangles,
                                name=f"flattening({f:g})")


class LayerLinear:
    """Reliefs on a skeleton's boundaries, interpolated linearly in r.

    `reliefs` gives one callable `relief(theta, phi)` per boundary of
    `skeleton`, or None for a boundary that stays spherical; within
    layer i the displacement is the linear interpolation in r between
    the reliefs of its two boundaries, so h is continuous, its radial
    derivative jumps only at the boundaries (the knots), and every
    boundary takes exactly its relief.  A relief carrying an
    `angular_gradient(theta, phi)` attribute is used for the angular
    gradient; any other is differenced by `angular_difference` with step
    `dstep`.
    """

    def __init__(self, skeleton: Skeleton, reliefs: Iterable[Relief | None], *,
                 dstep: float = 1e-6) -> None:
        b = np.asarray(skeleton.boundaries, dtype=float)
        reliefs = list(reliefs)
        if len(reliefs) != b.size:
            raise ValueError(
                f"{b.size} boundaries need {b.size} reliefs (None where "
                f"spherical), got {len(reliefs)}")
        for rel in reliefs:
            if rel is not None and not callable(rel):
                raise TypeError(f"a relief is a callable of (theta, phi) or None, "
                                f"got {type(rel).__name__}")
        self._b = b
        self._reliefs = reliefs
        self._h = float(dstep)
        self.knots = tuple(float(x) for x in b[1:-1])
        self.name = "layer_linear"

    @property
    def boundaries(self) -> np.ndarray:
        """The skeleton boundaries the reliefs sit on, innermost first."""
        return self._b

    @property
    def reliefs(self) -> tuple[Relief | None, ...]:
        """One relief per boundary, None where a boundary carries none."""
        return tuple(self._reliefs)

    def _relief(self, j: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        rel = self._reliefs[j]
        if rel is None:
            return np.zeros(np.broadcast(theta, phi).shape)
        return np.asarray(rel(theta, phi), dtype=float) + 0.0 * theta

    def _pieces(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                ) -> tuple[np.ndarray, ...]:
        """The layer of every point and the reliefs at its two boundaries:
        (r, theta, phi, lo, hi, below, above)."""
        r, theta, phi = _broadcast(r, theta, phi)
        i = np.clip(np.searchsorted(self._b, r, side="right") - 1,
                    0, self._b.size - 2)
        lo, hi = self._b[i], self._b[i + 1]
        below = np.empty(r.shape)
        above = np.empty(r.shape)
        for j in np.unique(i):
            m = i == j
            below[m] = self._relief(j, theta[m], phi[m])
            above[m] = self._relief(j + 1, theta[m], phi[m])
        return r, theta, phi, lo, hi, below, above

    def __call__(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The reliefs of the point's layer, interpolated linearly in r."""
        r, theta, phi, lo, hi, below, above = self._pieces(r, theta, phi)
        t = (r - lo) / (hi - lo)
        return (1.0 - t) * below + t * above

    def radial_derivative(self, r: ArrayLike, theta: ArrayLike,
                          phi: ArrayLike) -> np.ndarray:
        """dh/dr: the slope of the layer's interpolation, exact."""
        r, theta, phi, lo, hi, below, above = self._pieces(r, theta, phi)
        return (above - below) / (hi - lo)

    def angular_gradient(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(dh/dtheta, dh/dphi), exact where the reliefs supply theirs."""
        r, theta, phi = _broadcast(r, theta, phi)
        i = np.clip(np.searchsorted(self._b, r, side="right") - 1,
                    0, self._b.size - 2)
        lo, hi = self._b[i], self._b[i + 1]
        t = (r - lo) / (hi - lo)
        dt = np.zeros(r.shape)
        dp = np.zeros(r.shape)
        for j in np.unique(i):
            m = i == j
            for k, w in ((j, 1.0 - t[m]), (j + 1, t[m])):
                rel = self._reliefs[k]
                if rel is None:
                    continue
                if hasattr(rel, "angular_gradient"):
                    gt, gp = rel.angular_gradient(theta[m], phi[m])
                else:
                    gt, gp = angular_difference(
                        lambda a, b, k=k: self._relief(k, a, b), theta[m], phi[m],
                        step=self._h)
                dt[m] += w * np.asarray(gt, dtype=float)
                dp[m] += w * np.asarray(gp, dtype=float)
        return dt, dp

    def scaled(self, k: float) -> "LayerLinear":
        """The same reliefs on a body k times the size: the boundaries and
        every relief multiplied by k, a relief by its own `scaled` where
        it has one (a gridded relief stays gridded, a harmonic one
        harmonic)."""
        k = float(k)
        return LayerLinear(Skeleton(k * self._b),
                           [None if rel is None else _scaled_relief(rel, k, self._h)
                            for rel in self._reliefs], dstep=self._h)

    def __repr__(self) -> str:
        n = sum(rel is not None for rel in self._reliefs)
        return f"LayerLinear({n} reliefs on {self._b.size} boundaries)"


def _scaled_relief(relief: Relief, k: float, step: float) -> Relief:
    """k times a relief: its own `scaled(k)`, else a `_ScaledRelief`."""
    own = getattr(relief, "scaled", None)
    return own(k) if callable(own) else _ScaledRelief(relief, k, step=step)


class _ScaledRelief:
    """k relief(theta, phi), with k times the relief's angular gradient,
    its own or differenced by `angular_difference` with `step`, as
    `LayerLinear` would difference the relief itself."""

    def __init__(self, relief: Relief, k: float, *, step: float) -> None:
        self._relief, self._k, self._step = relief, float(k), float(step)

    def __call__(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """k times the relief at (theta, phi)."""
        return self._k * np.asarray(self._relief(theta, phi), dtype=float)

    def angular_gradient(self, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(d/dtheta, d/dphi) of k times the relief."""
        own = getattr(self._relief, "angular_gradient", None)
        if own is not None:
            gt, gp = own(theta, phi)
        else:
            gt, gp = angular_difference(self._relief, np.asarray(theta, dtype=float),
                                        np.asarray(phi, dtype=float), step=self._step)
        return (self._k * np.asarray(gt, dtype=float),
                self._k * np.asarray(gp, dtype=float))

    def scaled(self, k: float) -> "_ScaledRelief":
        """The relief scaled by this factor times k."""
        return _ScaledRelief(self._relief, self._k * float(k), step=self._step)


def layer_linear(skeleton: Skeleton, reliefs: Iterable[Relief | None], *,
                 dstep: float = 1e-6) -> LayerLinear:
    """The displacement interpolating boundary reliefs linearly in r."""
    return LayerLinear(skeleton, reliefs, dstep=dstep)
