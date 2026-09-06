"""The radial displacement h(r, theta, phi) that drives a radial stretch.

A radial mapping is driven by one scalar function: how far each point
moves along its own radius.  The callable is the interface,

    class RadialDisplacement(Protocol):
        def __call__(self, r, theta, phi) -> array

and everything else is an optional enrichment discovered by attribute:

    radial_derivative(r, theta, phi)   -> dh/dr
    angular_gradient(r, theta, phi)    -> (dh/dtheta, dh/dphi)
    knots                              -> radii where dh/dr may jump
    bounds()                           -> (min h, max h)

A bare function is accepted wherever a displacement is wanted, with the
derivatives taken by central differences; supplying them exactly gives
exact deformation gradients.  `as_displacement(h)` adapts any callable
to an object carrying every method, exact where `h` supplied it.  Two
displacements are shipped: `flattening`, the degree-2 shape of a
hydrostatic ellipsoid, and `layer_linear`, which gives every boundary of
a skeleton an analytic relief and interpolates linearly in r between
them within each layer.

Knots make smoothness checkable.  A displacement that is C0 but whose
radial derivative jumps declares the radii where it does, and a
geometry requires every declared knot to lie on one of its boundaries.
A displacement declaring no knots asserts that it is smooth.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from .frames import SphericalFunction

if TYPE_CHECKING:
    from .skeleton import Skeleton

__all__ = ["RadialDisplacement", "ZeroDisplacement", "CallableDisplacement",
           "as_displacement", "flattening", "layer_linear", "LayerLinear",
           "AngularGradientFunction", "Relief"]

#: A function of (r, theta, phi) returning (dh/dtheta, dh/dphi).
type AngularGradientFunction = Callable[[np.ndarray, np.ndarray, np.ndarray],
                                        tuple[ArrayLike, ArrayLike]]

#: The shape of one boundary: a callable of (theta, phi), which may carry
#: an `angular_gradient(theta, phi)` attribute returning its two derivatives.
type Relief = Callable[[np.ndarray, np.ndarray], ArrayLike]


@runtime_checkable
class RadialDisplacement(Protocol):
    """h(r, theta, phi): how far a point moves along its radius.

    Required: `__call__(r, theta, phi) -> array`, broadcasting.
    Optional, used when present: `radial_derivative`, `angular_gradient`,
    `knots`, `bounds`.
    """

    def __call__(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                 ) -> np.ndarray: ...


def _broadcast(r: ArrayLike, theta: ArrayLike, phi: ArrayLike
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three arguments broadcast to a common float array shape."""
    return np.broadcast_arrays(np.asarray(r, dtype=float),
                               np.asarray(theta, dtype=float),
                               np.asarray(phi, dtype=float))


class ZeroDisplacement:
    """No displacement: the identity mapping's h."""

    knots: tuple[float, ...] = ()

    def __call__(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        r, _, _ = _broadcast(r, theta, phi)
        return np.zeros(r.shape)

    def radial_derivative(self, r: ArrayLike, theta: ArrayLike,
                          phi: ArrayLike) -> np.ndarray:
        return self(r, theta, phi)

    def angular_gradient(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        z = self(r, theta, phi)
        return z, z.copy()

    def bounds(self) -> tuple[float, float]:
        return 0.0, 0.0

    def __repr__(self) -> str:
        return "ZeroDisplacement()"


class CallableDisplacement:
    """Any callable h(r, theta, phi), with its derivatives exact or differenced.

    Derivatives passed in, or carried by `fn` as attributes named
    `radial_derivative` and `angular_gradient`, are used as given;
    otherwise central differences are taken, with step `dstep` in the
    angles and `max(dstep, dstep * |r|)` in the radius.  `knots` declares
    the radii where dh/dr jumps; leaving them empty asserts smoothness.
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
        self.knots = tuple(sorted(float(k) for k in knots))
        self.name = name

    def __call__(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        r, theta, phi = _broadcast(r, theta, phi)
        return np.asarray(self._fn(r, theta, phi), dtype=float)

    def radial_derivative(self, r: ArrayLike, theta: ArrayLike,
                          phi: ArrayLike) -> np.ndarray:
        """dh/dr, exact if supplied, else a central difference."""
        if self._dr is not None:
            return np.asarray(self._dr(*_broadcast(r, theta, phi)), dtype=float)
        r, theta, phi = _broadcast(r, theta, phi)
        h = np.maximum(self._h, self._h * np.abs(r))
        return (self(r + h, theta, phi) - self(r - h, theta, phi)) / (2.0 * h)

    def angular_gradient(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(dh/dtheta, dh/dphi), exact if supplied, else differenced."""
        if self._da is not None:
            gt, gp = self._da(*_broadcast(r, theta, phi))
            return np.asarray(gt, dtype=float), np.asarray(gp, dtype=float)
        r, theta, phi = _broadcast(r, theta, phi)
        h = self._h
        dt = (self(r, theta + h, phi) - self(r, theta - h, phi)) / (2.0 * h)
        dp = (self(r, theta, phi + h) - self(r, theta, phi - h)) / (2.0 * h)
        return dt, dp

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
    CallableDisplacement.
    """
    if all(hasattr(fn, a) for a in
           ("radial_derivative", "angular_gradient", "knots")):
        if knots or kw:
            raise ValueError(
                f"{fn!r} already declares its knots and derivatives; the "
                "extra arguments would be ignored")
        return fn
    return CallableDisplacement(fn, knots=knots, **kw)


def flattening(f: float, *, rmax: float) -> CallableDisplacement:
    """h = -f r P2(cos theta): the degree-2 shape of flattening `f`.

    Every sphere of the reference body becomes a spheroid of the same
    mean radius with polar radius r (1 - 2f/3)... and equatorial radius
    r (1 + f/3), so that the outer boundary has flattening f to first
    order; `rmax` is the outer radius, where the shape is given.
    """
    f, rmax = float(f), float(rmax)

    def h(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return -f * r * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)

    def dh_dr(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return -f * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0) + 0.0 * r

    def dh_dangles(r: np.ndarray, theta: np.ndarray, phi: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
        return 3.0 * f * r * np.cos(theta) * np.sin(theta), 0.0 * r

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
    gradient; otherwise central differences with step `dstep` are taken.
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
        return self._b

    @property
    def reliefs(self) -> tuple[Relief | None, ...]:
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
        if all(rel is None or hasattr(rel, "angular_gradient")
               for rel in self._reliefs):
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
                    gt, gp = rel.angular_gradient(theta[m], phi[m])
                    dt[m] += w * np.asarray(gt, dtype=float)
                    dp[m] += w * np.asarray(gp, dtype=float)
            return dt, dp
        h = self._h
        dt = (self(r, theta + h, phi) - self(r, theta - h, phi)) / (2.0 * h)
        dp = (self(r, theta, phi + h) - self(r, theta, phi - h)) / (2.0 * h)
        return dt, dp

    def __repr__(self) -> str:
        n = sum(rel is not None for rel in self._reliefs)
        return f"LayerLinear({n} reliefs on {self._b.size} boundaries)"


def layer_linear(skeleton: Skeleton, reliefs: Iterable[Relief | None], *,
                 dstep: float = 1e-6) -> LayerLinear:
    """The displacement interpolating boundary reliefs linearly in r."""
    return LayerLinear(skeleton, reliefs, dstep=dstep)
