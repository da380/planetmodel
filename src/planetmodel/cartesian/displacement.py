"""The vertical displacement w(depth, x, y) that drives a vertical stretch.

THE SIGN, stated here once for `planetmodel.cartesian`: a vertical
displacement and a relief are HEIGHTS, positive up.  w is how far a
point of the reference body rises, the point at depth d going to
z = w - d, so to the physical depth d - w below the plane z = 0 of the
reference free surface; a relief(x, y) is the height of a surface above
its place in the reference body, so the relief of the free surface is
the topography as a user gives it (a mountain positive, a basin
negative) and a Moho relief of -5 is a root 5 deeper than the reference
Moho.  The layered coordinate is the depth, positive down, so w and the
depth increase in opposite senses: dw/d depth is minus dw/dz, and a
layer keeps a positive thickness exactly where 1 - dw/d depth > 0.

A vertical stretch (`cartesian.VerticalStretch`) is driven by one
scalar function, the flat counterpart of the sphere's radial
displacement (`planetmodel.displacement`).  The callable is the
interface,

    class VerticalDisplacement(Protocol):
        def __call__(self, depth, x, y) -> array

and everything else is an optional enrichment discovered by attribute:

    depth_derivative(depth, x, y)    -> dw/d depth
    lateral_gradient(depth, x, y)    -> (dw/dx, dw/dy)
    knots                            -> depths where dw/d depth may jump
    bounds()                         -> (min w, max w)
    scaled(k)                        -> k w(depth / k, x / k, y / k)

the last the same displacement on a body k times the size, in its own
form, so that a change of length scale keeps a stack of reliefs a
stack.  A bare function is accepted wherever a displacement is wanted,
with the derivatives taken by central differences; supplying them
exactly gives exact deformation gradients.  `as_displacement(w)` adapts
any callable to an object carrying every method, exact where `w`
supplied it.  `layer_linear` gives every boundary of a skeleton of
depths a relief and interpolates linearly in depth between them within
each layer: the free surface raised to its topography, each interface
to its own relief, the base held (or given its own), every layer
stretched between the surfaces that bound it.

A relief is a callable of (x, y) returning heights, which may carry a
`lateral_gradient(x, y)` attribute returning its two derivatives and a
`scaled(k)` one returning k relief(x / k, y / k) in its own form: the
sphere's relief with the lateral coordinates lengths rather than
angles.  Knots make smoothness checkable, as on the sphere: a
displacement that is C0 but whose depth derivative jumps declares the
depths where it does; one declaring none asserts that it is smooth.
Nothing here knows about units: a difference step is relative to the
coordinate it steps, with a floor of the step itself.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from ..frames import SphericalFunction
from ..skeleton import Skeleton

__all__ = ["VerticalDisplacement", "ZeroDisplacement", "CallableDisplacement",
           "as_displacement", "LayerLinear", "layer_linear", "Relief",
           "LateralGradientFunction"]

#: A function of (depth, x, y) returning (dw/dx, dw/dy).
type LateralGradientFunction = Callable[[np.ndarray, np.ndarray, np.ndarray],
                                        tuple[ArrayLike, ArrayLike]]

#: The shape of one surface: a callable of (x, y) returning its height above
#: its reference place (positive up), which may carry a
#: `lateral_gradient(x, y)` attribute returning its two derivatives and a
#: `scaled(k)` one returning k relief(x / k, y / k) in its own form.
type Relief = Callable[[np.ndarray, np.ndarray], ArrayLike]


@runtime_checkable
class VerticalDisplacement(Protocol):
    """w(depth, x, y): how far a point rises, positive up (the module
    docstring states the sign).

    Required: `__call__(depth, x, y) -> array`, broadcasting.  Optional,
    used when present: `depth_derivative`, `lateral_gradient`, `knots`,
    `bounds`, `scaled`.
    """

    def __call__(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """The upward displacement at the broadcast coordinates."""
        ...


def _broadcast(*arrays: ArrayLike) -> tuple[np.ndarray, ...]:
    """The arguments broadcast to a common float array shape."""
    return tuple(np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in arrays)))


def _central(f: Callable[[np.ndarray], ArrayLike], v: np.ndarray, *,
             step: float) -> np.ndarray:
    """df/dv by a central difference of `max(step, step |v|)`."""
    h = np.maximum(step, step * np.abs(v))
    return (np.asarray(f(v + h), dtype=float)
            - np.asarray(f(v - h), dtype=float)) / (2.0 * h)


def _lateral_difference(f: Callable[[np.ndarray, np.ndarray], ArrayLike],
                        x: np.ndarray, y: np.ndarray, *, step: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(df/dx, df/dy) of a function of (x, y) by central differences."""
    return (_central(lambda a: f(a, y), x, step=step),
            _central(lambda b: f(x, b), y, step=step))


class ZeroDisplacement:
    """No displacement: the identity mapping's w."""

    knots: tuple[float, ...] = ()

    def __call__(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Zeros of the broadcast shape."""
        depth, _, _ = _broadcast(depth, x, y)
        return np.zeros(depth.shape)

    def depth_derivative(self, depth: ArrayLike, x: ArrayLike,
                         y: ArrayLike) -> np.ndarray:
        """dw/d depth: zeros of the broadcast shape."""
        return self(depth, x, y)

    def lateral_gradient(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(dw/dx, dw/dy): zeros of the broadcast shape."""
        z = self(depth, x, y)
        return z, z.copy()

    def bounds(self) -> tuple[float, float]:
        """(min w, max w): both zero."""
        return 0.0, 0.0

    def scaled(self, k: float) -> "ZeroDisplacement":
        """No displacement on a body of any size."""
        return self

    def __repr__(self) -> str:
        return "cartesian.ZeroDisplacement()"


class CallableDisplacement:
    """Any callable w(depth, x, y), heights, with its derivatives exact or
    differenced.

    Derivatives passed in, or carried by `fn` as attributes named
    `depth_derivative` and `lateral_gradient`, are used as given;
    otherwise central differences are taken with step `max(dstep,
    dstep |v|)` in each coordinate v.  `knots` declares the depths where
    dw/d depth jumps, by default those `fn` carries; leaving them empty
    asserts smoothness.  `bounds` is `fn`'s where it has one.
    """

    def __init__(self, fn: SphericalFunction, *, knots: Iterable[float] = (),
                 depth_derivative: SphericalFunction | None = None,
                 lateral_gradient: LateralGradientFunction | None = None,
                 name: str | None = None, dstep: float = 1e-6) -> None:
        if not callable(fn):
            raise TypeError(f"expected a callable, got {type(fn).__name__}")
        self._fn = fn
        self._dd = (depth_derivative if depth_derivative is not None
                    else getattr(fn, "depth_derivative", None))
        self._dl = (lateral_gradient if lateral_gradient is not None
                    else getattr(fn, "lateral_gradient", None))
        self._h = float(dstep)
        knots = tuple(knots) or tuple(getattr(fn, "knots", ()))
        self.knots = tuple(sorted(float(k) for k in knots))
        if callable(getattr(fn, "bounds", None)):
            self.bounds = fn.bounds
        self.name = name

    def __call__(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """The wrapped callable at the broadcast coordinates."""
        depth, x, y = _broadcast(depth, x, y)
        return np.asarray(self._fn(depth, x, y), dtype=float) + 0.0 * depth

    def depth_derivative(self, depth: ArrayLike, x: ArrayLike,
                         y: ArrayLike) -> np.ndarray:
        """dw/d depth, exact if supplied, else a central difference."""
        depth, x, y = _broadcast(depth, x, y)
        if self._dd is not None:
            return np.asarray(self._dd(depth, x, y), dtype=float) + 0.0 * depth
        return _central(lambda d: self(d, x, y), depth, step=self._h)

    def lateral_gradient(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(dw/dx, dw/dy), exact if supplied, else differenced."""
        depth, x, y = _broadcast(depth, x, y)
        if self._dl is not None:
            gx, gy = self._dl(depth, x, y)
            return (np.asarray(gx, dtype=float) + 0.0 * depth,
                    np.asarray(gy, dtype=float) + 0.0 * depth)
        return _lateral_difference(lambda a, b: self(depth, a, b), x, y, step=self._h)

    def scaled(self, k: float) -> "CallableDisplacement":
        """k w(depth / k, x / k, y / k): every derivative read at the point
        divided by k, each exact where this displacement's is and
        differenced as it is otherwise; the knots and the bounds
        multiplied by k."""
        k = float(k)

        def w(depth: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return k * self(depth / k, x / k, y / k)

        def dw_dd(depth: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return self.depth_derivative(depth / k, x / k, y / k)

        def dw_dl(depth: np.ndarray, x: np.ndarray, y: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
            return self.lateral_gradient(depth / k, x / k, y / k)

        if callable(getattr(self, "bounds", None)):
            def bounds() -> tuple[float, float]:
                lo, hi = self.bounds()
                return k * float(lo), k * float(hi)
            w.bounds = bounds
        return CallableDisplacement(w, knots=[k * v for v in self.knots],
                                    depth_derivative=dw_dd, lateral_gradient=dw_dl,
                                    name=self.name, dstep=self._h)

    def __repr__(self) -> str:
        fn = getattr(self._fn, "__name__", None) or repr(self._fn)
        nm = f" {self.name!r}" if self.name else ""
        knots = f", knots={list(self.knots)}" if self.knots else ""
        return f"cartesian.CallableDisplacement({fn}{nm}{knots})"


def as_displacement(fn: VerticalDisplacement | SphericalFunction, *,
                    knots: Iterable[float] = (), **kw: object) -> VerticalDisplacement:
    """Adapt any callable of (depth, x, y) to the protocol.

    An object that already carries `depth_derivative`, `lateral_gradient`
    and `knots` is returned unchanged; extra arguments are then refused
    rather than ignored.  Anything else is wrapped in a
    `CallableDisplacement`, which keeps what it carries of the three and
    its `bounds`.
    """
    if all(hasattr(fn, a) for a in ("depth_derivative", "lateral_gradient", "knots")):
        if knots or kw:
            raise ValueError(
                f"{fn!r} already declares its knots and derivatives; the extra "
                "arguments would be ignored")
        return fn
    return CallableDisplacement(fn, knots=knots, **kw)


class LayerLinear:
    """Reliefs on a skeleton's boundaries, interpolated linearly in depth.

    `reliefs` gives one callable `relief(x, y)` per boundary of
    `skeleton`, the free surface (or a hollow body's top) first and the
    base last, each the height of that surface above its reference
    plane, positive up (the module docstring), or None for a surface
    that stays flat: the base held is None.  Within layer i the
    displacement is the linear interpolation in depth between the
    reliefs of its top and its bottom, so w is continuous, its depth
    derivative jumps only at the interior boundaries (the knots), and
    every boundary takes exactly its relief.  Above the top and below
    the base the end layers' interpolations continue.  A relief carrying
    a `lateral_gradient(x, y)` attribute is used for the lateral
    gradient; any other is differenced with step `max(dstep, dstep |v|)`.
    """

    def __init__(self, skeleton: Skeleton, reliefs: Iterable[Relief | None], *,
                 dstep: float = 1e-6) -> None:
        b = np.asarray(skeleton.boundaries, dtype=float)
        reliefs = list(reliefs)
        if len(reliefs) != b.size:
            raise ValueError(
                f"{b.size} boundaries need {b.size} reliefs, the free surface "
                f"first and the base last (None where flat), got {len(reliefs)}")
        for rel in reliefs:
            if rel is not None and not callable(rel):
                raise TypeError(f"a relief is a callable of (x, y) or None, got "
                                f"{type(rel).__name__}")
        self._b = b
        self._reliefs = reliefs
        self._h = float(dstep)
        self.knots = tuple(float(v) for v in b[1:-1])
        self.name = "layer_linear"

    @property
    def boundaries(self) -> np.ndarray:
        """The skeleton's depths the reliefs sit on, the top first."""
        return self._b

    @property
    def reliefs(self) -> tuple[Relief | None, ...]:
        """One relief per boundary, None where a boundary stays flat."""
        return tuple(self._reliefs)

    def _relief(self, j: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rel = self._reliefs[j]
        if rel is None:
            return np.zeros(np.broadcast(x, y).shape)
        return np.asarray(rel(x, y), dtype=float) + 0.0 * x

    def _layer(self, depth: np.ndarray) -> np.ndarray:
        """The layer of every depth, the deeper one on a boundary, the end
        layers continued beyond the skeleton."""
        return np.clip(np.searchsorted(self._b, depth, side="right") - 1,
                       0, self._b.size - 2)

    def _pieces(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike
                ) -> tuple[np.ndarray, ...]:
        """The layer of every point and the reliefs of its top and bottom:
        (depth, x, y, top, bottom, upper, lower)."""
        depth, x, y = _broadcast(depth, x, y)
        i = self._layer(depth)
        top, bottom = self._b[i], self._b[i + 1]
        upper = np.empty(depth.shape)
        lower = np.empty(depth.shape)
        for j in np.unique(i):
            m = i == j
            upper[m] = self._relief(j, x[m], y[m])
            lower[m] = self._relief(j + 1, x[m], y[m])
        return depth, x, y, top, bottom, upper, lower

    def __call__(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """The reliefs of the point's layer, interpolated linearly in depth."""
        depth, x, y, top, bottom, upper, lower = self._pieces(depth, x, y)
        t = (depth - top) / (bottom - top)
        return (1.0 - t) * upper + t * lower

    def depth_derivative(self, depth: ArrayLike, x: ArrayLike,
                         y: ArrayLike) -> np.ndarray:
        """dw/d depth: the slope of the layer's interpolation, exact."""
        depth, x, y, top, bottom, upper, lower = self._pieces(depth, x, y)
        return (lower - upper) / (bottom - top)

    def lateral_gradient(self, depth: ArrayLike, x: ArrayLike, y: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(dw/dx, dw/dy), exact where the reliefs supply theirs."""
        depth, x, y = _broadcast(depth, x, y)
        i = self._layer(depth)
        top, bottom = self._b[i], self._b[i + 1]
        t = (depth - top) / (bottom - top)
        gx = np.zeros(depth.shape)
        gy = np.zeros(depth.shape)
        for j in np.unique(i):
            m = i == j
            for k, weight in ((j, 1.0 - t[m]), (j + 1, t[m])):
                rel = self._reliefs[k]
                if rel is None:
                    continue
                if hasattr(rel, "lateral_gradient"):
                    a, b = rel.lateral_gradient(x[m], y[m])
                else:
                    a, b = _lateral_difference(
                        lambda u, v, k=k: self._relief(k, u, v), x[m], y[m],
                        step=self._h)
                gx[m] += weight * np.asarray(a, dtype=float)
                gy[m] += weight * np.asarray(b, dtype=float)
        return gx, gy

    def scaled(self, k: float) -> "LayerLinear":
        """The same reliefs on a body k times the size: the depths
        multiplied by k and every relief r replaced by k r(x / k, y / k),
        by its own `scaled` where it has one."""
        k = float(k)
        return LayerLinear(Skeleton(k * self._b),
                           [None if rel is None else _scaled_relief(rel, k, self._h)
                            for rel in self._reliefs], dstep=self._h)

    def __repr__(self) -> str:
        n = sum(rel is not None for rel in self._reliefs)
        return f"cartesian.LayerLinear({n} reliefs on {self._b.size} boundaries)"


def _scaled_relief(relief: Relief, k: float, step: float) -> Relief:
    """k relief(x / k, y / k): the relief's own `scaled(k)`, else a
    `_ScaledRelief`."""
    own = getattr(relief, "scaled", None)
    return own(k) if callable(own) else _ScaledRelief(relief, k, step=step)


class _ScaledRelief:
    """k relief(x / k, y / k), with the relief's lateral gradient at
    (x / k, y / k), its own or differenced with `step` as `LayerLinear`
    would difference the relief itself there."""

    def __init__(self, relief: Relief, k: float, *, step: float) -> None:
        self._relief, self._k, self._step = relief, float(k), float(step)

    def __call__(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """k times the relief at (x / k, y / k)."""
        x, y = _broadcast(x, y)
        k = self._k
        return k * (np.asarray(self._relief(x / k, y / k), dtype=float) + 0.0 * x)

    def lateral_gradient(self, x: ArrayLike, y: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(d/dx, d/dy) of k relief(x / k, y / k): the relief's own at
        (x / k, y / k)."""
        x, y = _broadcast(x, y)
        a, b = x / self._k, y / self._k
        own = getattr(self._relief, "lateral_gradient", None)
        if own is not None:
            gx, gy = own(a, b)
        else:
            gx, gy = _lateral_difference(
                lambda u, v: np.asarray(self._relief(u, v), dtype=float) + 0.0 * u,
                a, b, step=self._step)
        return (np.asarray(gx, dtype=float) + 0.0 * x,
                np.asarray(gy, dtype=float) + 0.0 * x)

    def scaled(self, k: float) -> "_ScaledRelief":
        """The relief scaled by this factor times k."""
        return _ScaledRelief(self._relief, self._k * float(k), step=self._step)


def layer_linear(skeleton: Skeleton, reliefs: Iterable[Relief | None], *,
                 dstep: float = 1e-6) -> LayerLinear:
    """The displacement interpolating boundary reliefs (heights, positive
    up) linearly in depth: `LayerLinear`."""
    return LayerLinear(skeleton, reliefs, dstep=dstep)
