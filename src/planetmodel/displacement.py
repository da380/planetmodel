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
to an object carrying every method, exact where `h` supplied it.

Knots make smoothness checkable.  A displacement that is C0 but whose
radial derivative jumps declares the radii where it does, and a
geometry requires every declared knot to lie on one of its boundaries.
A displacement declaring no knots asserts that it is smooth.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["RadialDisplacement", "ZeroDisplacement", "CallableDisplacement",
           "as_displacement"]


@runtime_checkable
class RadialDisplacement(Protocol):
    """h(r, theta, phi): how far a point moves along its radius.

    Required: `__call__(r, theta, phi) -> array`, broadcasting.
    Optional, used when present: `radial_derivative`, `angular_gradient`,
    `knots`, `bounds`.
    """

    def __call__(self, r, theta, phi): ...


def _broadcast(r, theta, phi):
    """The three arguments broadcast to a common float array shape."""
    return np.broadcast_arrays(np.asarray(r, dtype=float),
                               np.asarray(theta, dtype=float),
                               np.asarray(phi, dtype=float))


class ZeroDisplacement:
    """No displacement: the identity mapping's h."""

    knots: tuple[float, ...] = ()

    def __call__(self, r, theta, phi):
        r, _, _ = _broadcast(r, theta, phi)
        return np.zeros(r.shape)

    def radial_derivative(self, r, theta, phi):
        return self(r, theta, phi)

    def angular_gradient(self, r, theta, phi):
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

    def __init__(self, fn, *, knots=(), radial_derivative=None,
                 angular_gradient=None, name: str | None = None,
                 dstep: float = 1e-6) -> None:
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

    def __call__(self, r, theta, phi):
        r, theta, phi = _broadcast(r, theta, phi)
        return np.asarray(self._fn(r, theta, phi), dtype=float)

    def radial_derivative(self, r, theta, phi):
        """dh/dr, exact if supplied, else a central difference."""
        if self._dr is not None:
            return np.asarray(self._dr(*_broadcast(r, theta, phi)), dtype=float)
        r, theta, phi = _broadcast(r, theta, phi)
        h = np.maximum(self._h, self._h * np.abs(r))
        return (self(r + h, theta, phi) - self(r - h, theta, phi)) / (2.0 * h)

    def angular_gradient(self, r, theta, phi):
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


def as_displacement(fn, *, knots=(), **kw) -> RadialDisplacement:
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
