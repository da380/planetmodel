"""displacement.py -- the radial displacement h(r, theta, phi).

A radial mapping is driven by a single function: how far, and in which
direction along the radius, each point moves.  How that function arises
varies completely by circumstance -- interpolated between interface
topographies, prescribed as a volumetric field, drawn from a random
realisation, produced by somebody's inversion -- so the *callable is the
interface* and the constructors below are one way among many to obtain
one.

    class RadialDisplacement(Protocol):
        def __call__(self, r, theta, phi) -> array

Everything else is an optional enrichment, discovered by attribute:
radial_derivative, angular_gradient, knots, bounds.  A bare function is
accepted anywhere a displacement is wanted, with the derivatives taken
numerically; supplying them exactly is how you get better behaviour.
The rule for using one is *adapt, then call*: `as_displacement(h)`
returns an object with every method, exact where `h` supplied it.

Knots and smoothness
--------------------

Both consumers' weak forms are understood with piecewise-continuous
coefficients, so h needs to be C0 globally and smooth on each patch --
not C1 across interfaces.  What matters instead is that the patch
boundaries coincide with element boundaries, so no element straddles a
kink and quadrature stays accurate.

`knots` is how that becomes checkable rather than hoped for: a
displacement declares the radii where dh/dr may jump, and the mesher
verifies each one coincides with a meshed interface.  A displacement
declaring no knots is treated as smooth, at the caller's risk.

Confinement is geometry
-----------------------

To stop a topography propagating deep into the body, put an interface
where the displacement should vanish and attach no surface to it: h
interpolates linearly to zero there and is identically zero below, and
the kink lands on an element boundary.  That is a property of the
geometry, not a parameter of a rule -- which is why layer_linear has
nothing to tune.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..registry import register
from .topography import ZeroTopography, as_topography

__all__ = [
    "RadialDisplacement", "as_displacement", "ZeroDisplacement",
    "CallableDisplacement", "BlendDisplacement", "SumDisplacement",
    "layer_linear",
]


@runtime_checkable
class RadialDisplacement(Protocol):
    """h(r, theta, phi): how far a point moves along its radius.

    Required: __call__(r, theta, phi) -> array, broadcasting.  Optional
    and discovered when present:

        radial_derivative(r, theta, phi)   -> dh/dr
        angular_gradient(r, theta, phi)    -> (dh/dtheta, dh/dphi)
        knots                              -> radii where dh/dr may jump
        bounds()                           -> (min h, max h)
    """

    def __call__(self, r, theta, phi): ...


def _broadcast(r, theta, phi):
    """Broadcast the three arguments to a common shape."""
    return np.broadcast_arrays(np.asarray(r, dtype=float),
                               np.asarray(theta, dtype=float),
                               np.asarray(phi, dtype=float))


class _DisplacementAlgebra:
    """Addition and scaling shared by the concrete displacements."""

    def __add__(self, other):
        if not callable(other):
            return NotImplemented
        return SumDisplacement((self, other))

    def __neg__(self):
        return SumDisplacement((), scale_of=((-1.0, self),))

    def __mul__(self, k):
        if callable(k):
            raise TypeError("displacement * displacement is not defined")
        return SumDisplacement((), scale_of=((float(k), self),))

    __rmul__ = __mul__


class ZeroDisplacement(_DisplacementAlgebra):
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


class CallableDisplacement(_DisplacementAlgebra):
    """Any callable h(r, theta, phi), with numerical enrichments.

    The route for a prescribed volumetric displacement: a field read
    from disk, an analytic test case, an inversion's output.  Declare
    `knots` if dh/dr jumps anywhere, so the mesher can align elements
    with them; leaving them empty asserts smoothness.  A derivative the
    callable carries itself is kept and used in place of a difference.
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
        """dh/dr, analytic if supplied, else a central difference."""
        if self._dr is not None:
            return np.asarray(self._dr(*_broadcast(r, theta, phi)), dtype=float)
        r, theta, phi = _broadcast(r, theta, phi)
        h = np.maximum(self._h, self._h * np.abs(r))
        return (self(r + h, theta, phi) - self(r - h, theta, phi)) / (2.0 * h)

    def angular_gradient(self, r, theta, phi):
        """(dh/dtheta, dh/dphi), analytic if supplied, else differenced."""
        if self._da is not None:
            gt, gp = self._da(*_broadcast(r, theta, phi))
            return np.asarray(gt, dtype=float), np.asarray(gp, dtype=float)
        r, theta, phi = _broadcast(r, theta, phi)
        h = self._h
        dt = (self(r, theta + h, phi) - self(r, theta - h, phi)) / (2.0 * h)
        dp = (self(r, theta, phi + h) - self(r, theta, phi - h)) / (2.0 * h)
        return dt, dp

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"CallableDisplacement({self._fn!r}{nm}, knots={list(self.knots)})"


class SumDisplacement(_DisplacementAlgebra):
    """A linear combination of displacements.

    Each term is adapted on construction, so the sum's derivatives are
    exact wherever the terms' are.
    """

    def __init__(self, terms, *, scale_of=(), name: str | None = None) -> None:
        weighted = ([(1.0, as_displacement(t)) for t in terms]
                    + [(c, as_displacement(t)) for c, t in scale_of])
        if not weighted:
            raise ValueError("SumDisplacement needs at least one term")
        self._terms = tuple(weighted)
        self.name = name

    @property
    def knots(self) -> tuple[float, ...]:
        """Every term's knots: dh/dr may jump wherever any term's does."""
        out: set[float] = set()
        for _, t in self._terms:
            out.update(getattr(t, "knots", ()))
        return tuple(sorted(out))

    def __call__(self, r, theta, phi):
        out = None
        for c, t in self._terms:
            v = c * np.asarray(t(r, theta, phi), dtype=float)
            out = v if out is None else out + v
        return out

    def radial_derivative(self, r, theta, phi):
        out = None
        for c, t in self._terms:
            v = c * np.asarray(t.radial_derivative(r, theta, phi), dtype=float)
            out = v if out is None else out + v
        return out

    def angular_gradient(self, r, theta, phi):
        gt = gp = None
        for c, t in self._terms:
            a, b = t.angular_gradient(r, theta, phi)
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            gt = c * a if gt is None else gt + c * a
            gp = c * b if gp is None else gp + c * b
        return gt, gp

    def __repr__(self) -> str:
        return f"SumDisplacement({len(self._terms)} terms)"


def as_displacement(fn, *, knots=(), **kw) -> RadialDisplacement:
    """Adapt any callable of (r, theta, phi) to the protocol.

    A displacement that already carries its own derivatives and knots is
    returned unchanged, so wrapping never costs exactness; anything else
    is wrapped in a `CallableDisplacement`, which keeps whichever
    derivatives the callable carries and differences the rest.
    """
    if all(hasattr(fn, a) for a in
           ("radial_derivative", "angular_gradient", "knots")):
        if knots or kw:
            raise ValueError(
                f"{fn!r} already declares its knots and derivatives; the "
                "extra arguments would be silently ignored")
        return fn
    return CallableDisplacement(fn, knots=knots, **kw)


class BlendDisplacement(_DisplacementAlgebra):
    """h interpolated linearly in radius between knots carrying relief.

    Built by a displacement rule from a body's skeleton and attached
    surfaces.  Between consecutive knots r_i < r_{i+1} carrying relief
    t_i and t_{i+1},

        u = (r - r_i) / (r_{i+1} - r_i)
        h = (1 - u) t_i + u t_{i+1}

    so h is continuous everywhere, linear -- hence smooth -- within each
    span, and dh/dr is piecewise constant with jumps only at the knots.
    Outside the outermost knots h is zero.

    A taper radius multiplies h by a linear ramp rising from zero there
    to one at the next knot, which keeps the displacement away from the
    origin where dividing by small radii is delicate.
    """

    def __init__(self, radii, reliefs, *, taper_radius=None,
                 name: str | None = None) -> None:
        radii = np.asarray(radii, dtype=float)
        if radii.ndim != 1 or radii.size < 2:
            raise ValueError("a blend needs at least two knot radii")
        if not np.all(np.diff(radii) > 0.0):
            raise ValueError("knot radii must be strictly increasing")
        if len(reliefs) != radii.size:
            raise ValueError(
                f"got {len(reliefs)} reliefs for {radii.size} knots")
        self._r = radii
        self._t = tuple(as_topography(t) for t in reliefs)
        self._taper = None if taper_radius is None else float(taper_radius)
        if self._taper is not None and not radii[0] <= self._taper < radii[-1]:
            raise ValueError(
                f"taper radius {self._taper} must lie in [{radii[0]}, "
                f"{radii[-1]}): the ramp runs from it to the next knot, and "
                "at or beyond the last knot there is no next knot")
        self.name = name

    @property
    def knots(self) -> tuple[float, ...]:
        """Radii where dh/dr may jump: the blend knots, plus any taper."""
        extra = () if self._taper is None else (self._taper,)
        return tuple(sorted(set(map(float, self._r)) | set(extra)))

    def _span(self, r):
        """Per point: the span index, and the interpolation parameter u."""
        r = np.asarray(r, dtype=float)
        i = np.clip(np.searchsorted(self._r, r, side="right") - 1,
                    0, self._r.size - 2)
        lo, hi = self._r[i], self._r[i + 1]
        u = (r - lo) / (hi - lo)
        inside = (r >= self._r[0]) & (r <= self._r[-1])
        return i, u, inside

    def _taper_top(self) -> float:
        """The knot the inner ramp reaches one at."""
        above = self._r[self._r > self._taper]
        return float(above[0]) if above.size else float(self._r[-1])

    def _taper_weight(self, r):
        """The inner ramp: 0 below the taper radius, 1 from the next knot."""
        if self._taper is None:
            return None
        top = self._taper_top()
        r = np.asarray(r, dtype=float)
        return np.clip((r - self._taper) / (top - self._taper), 0.0, 1.0)

    def _relief(self, k, theta, phi):
        """The k-th knot's relief in the given directions."""
        return np.asarray(self._t[k](theta, phi), dtype=float)

    def _blend(self, r, theta, phi, i, u, inside, value):
        """(1 - u) value_k + u value_{k+1} on each span, zero outside.

        `value(k, theta, phi)` gives the k-th knot's quantity -- the
        relief itself or one component of its gradient -- so the blend
        is written once for h and its angular gradient alike.
        """
        out = np.zeros(np.asarray(r).shape)
        for k in np.unique(i):
            m = (i == k) & inside
            if not m.any():
                continue
            lo = value(int(k), theta[m], phi[m])
            hi = value(int(k) + 1, theta[m], phi[m])
            out[m] = (1.0 - u[m]) * lo + u[m] * hi
        return out

    def _untapered(self, r, theta, phi, i, u, inside):
        """h before the taper is applied."""
        return self._blend(r, theta, phi, i, u, inside, self._relief)

    def __call__(self, r, theta, phi):
        """The blended displacement."""
        r, theta, phi = _broadcast(r, theta, phi)
        i, u, inside = self._span(r)
        out = self._untapered(r, theta, phi, i, u, inside)
        w = self._taper_weight(r)
        return out if w is None else out * w

    def radial_derivative(self, r, theta, phi):
        """dh/dr: piecewise constant within a span, exactly.

        With a taper the product rule applies, since the ramp varies
        with radius too.
        """
        r, theta, phi = _broadcast(r, theta, phi)
        i, u, inside = self._span(r)
        slope = np.zeros(r.shape)
        for k in np.unique(i):
            m = (i == k) & inside
            if not m.any():
                continue
            lo = self._relief(int(k), theta[m], phi[m])
            hi = self._relief(int(k) + 1, theta[m], phi[m])
            slope[m] = (hi - lo) / (self._r[int(k) + 1] - self._r[int(k)])
        if self._taper is None:
            return slope
        w = self._taper_weight(r)
        top = self._taper_top()
        dw = np.where((r > self._taper) & (r < top), 1.0 / (top - self._taper),
                      0.0)
        base = self._untapered(r, theta, phi, i, u, inside)
        return slope * w + base * dw

    def angular_gradient(self, r, theta, phi):
        """The same linear blend, applied to the reliefs' gradients."""
        r, theta, phi = _broadcast(r, theta, phi)
        i, u, inside = self._span(r)

        def d_theta(k, th, ph):
            return np.asarray(self._t[k].gradient(th, ph)[0], dtype=float)

        def d_phi(k, th, ph):
            return np.asarray(self._t[k].gradient(th, ph)[1], dtype=float)

        gt = self._blend(r, theta, phi, i, u, inside, d_theta)
        gp = self._blend(r, theta, phi, i, u, inside, d_phi)
        w = self._taper_weight(r)
        if w is not None:
            gt, gp = gt * w, gp * w
        return gt, gp

    def bounds(self) -> tuple[float, float] | None:
        """The extreme displacements, if every relief knows its own."""
        lo = hi = 0.0
        for t in self._t:
            b = getattr(t, "bounds", None)
            if b is None:
                return None
            a, c = b()
            lo, hi = min(lo, a), max(hi, c)
        return lo, hi

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"BlendDisplacement({self._r.size} knots{nm})"


@register("displacement_rule", "layer_linear")
@dataclass(frozen=True)
class layer_linear:
    """Interpolate relief linearly in radius between the body's interfaces.

    A *displacement rule*: call it on a ReferenceBody and it returns a
    RadialDisplacement.  The knots are the skeleton's boundaries, so
    dh/dr jumps only where the mesh already has an element boundary --
    the smoothness requirement satisfied by construction, with nothing
    to tune.

    Interfaces carrying no surface contribute zero relief, which is what
    makes confinement a matter of geometry: put an interface where the
    displacement should vanish, attach nothing to it, and h ramps
    linearly to zero there and is identically zero below.  The same
    applies outward, so a buffer whose outer boundary carries no surface
    gets h = 0 there exactly -- which the exterior coupling requires.

    inner_taper_radius keeps the displacement away from the origin,
    where dividing by small radii is delicate; control_radii add knots
    without adding mesh interfaces, at the price of a kink inside an
    element, and the mesher warns when they are used.
    """

    _: KW_ONLY
    inner_taper_radius: float | None = None
    control_radii: tuple[float, ...] = ()

    def __call__(self, body) -> BlendDisplacement:
        """Build the displacement from a body's skeleton and surfaces."""
        radii = list(map(float, body.skeleton.boundaries))
        for r in self.control_radii:
            r = float(r)
            if not radii[0] <= r <= radii[-1]:
                raise ValueError(
                    f"control radius {r} lies outside the body "
                    f"[{radii[0]}, {radii[-1]}]")
            if r not in radii:
                radii.append(r)
        radii = sorted(set(radii))

        # The ATTACHMENT decides which knot carries the relief: a surface
        # attached to the CMB drives the CMB knot even if its own
        # reference radius bookkeeping says something else.  Matching on
        # the surface's reference_radius instead would silently drop any
        # relief whose placement disagrees with its interface.
        by_radius = {body.interfaces[i].radius: s.topography
                     for i, s in body.surfaces.items()}
        reliefs = [by_radius.get(r, ZeroTopography()) for r in radii]

        return BlendDisplacement(radii, reliefs,
                                 taper_radius=self.inner_taper_radius,
                                 name="layer_linear")

    def __repr__(self) -> str:
        return (f"layer_linear(inner_taper_radius={self.inner_taper_radius}, "
                f"control_radii={list(self.control_radii)})")
