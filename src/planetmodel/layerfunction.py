"""Functions of one radius on one interval, exact where they can be.

A layer function is what a field stores per component: a callable of the
radius on one interval that also differentiates and integrates itself,
re-states itself on another interval, and rescales its coordinate and
value.  Two kinds are shipped.  `PolynomialLayer` wraps a scipy `PPoly`
and does everything on the coefficients: sums, products and powers of
polynomials are polynomials, a derivative or an antiderivative is exact,
and the rescaling f(r) -> v f(r / k) is one multiply per coefficient
(breakpoints k b, coefficients v c / k^p).  `NumericLayer` wraps any
callable and supplies the rest honestly: a central difference with a
step relative to the interval's width, and adaptive quadrature.

Arithmetic between layer functions on the same interval follows the
operands: polynomial with polynomial stays polynomial, anything with a
numeric layer becomes a numeric closure.  Beyond its interval a
polynomial continues as the same polynomial and a callable is called as
given; `on_interval` is how that continuation is asked for on purpose,
while the fields built on layer functions refuse radii outside their
own interval.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad
from scipy.interpolate import BSpline, PPoly

__all__ = ["LayerFunction", "PolynomialLayer", "NumericLayer",
           "as_layer_function", "polynomial_layer", "constant_layer",
           "polynomial_fit", "same_interval", "as_values", "as_scalar"]

#: Two intervals are the same when their ends agree to this fraction of
#: the wider one's width.
_INTERVAL_RTOL = 1e-9


def _interval(interval) -> tuple[float, float]:
    lo, hi = (float(x) for x in interval)
    if not hi > lo:
        raise ValueError(f"an interval must increase, got ({lo:g}, {hi:g})")
    return lo, hi


def same_interval(a, b, *, rtol: float = _INTERVAL_RTOL) -> bool:
    """Whether two intervals coincide to `rtol` of the wider width."""
    (a0, a1), (b0, b1) = _interval(a), _interval(b)
    tol = rtol * max(a1 - a0, b1 - b0)
    return abs(a0 - b0) <= tol and abs(a1 - b1) <= tol


def _require_same_interval(a, b) -> tuple[float, float]:
    if not same_interval(a.interval, b.interval):
        raise ValueError(
            f"layer functions on different intervals cannot be combined: "
            f"{a.interval} and {b.interval}; re-state one with on_interval")
    return a.interval


@runtime_checkable
class LayerFunction(Protocol):
    """A function of one radius on one interval.

    `__call__(r)` broadcasts over any shape and returns float64, or
    complex128 for a complex-valued function;
    `derivative(*, nu)` and `integrate(a, b)` (signed) are the
    calculus; `on_interval(lo, hi)` re-states the same function on
    another interval by its own rule; `rescaled(*, k, v)` is
    v f(r / k) on the interval scaled by k.  `as_layer_function` adapts
    a bare callable, a `PPoly` or a `BSpline`.
    """

    interval: tuple[float, float]

    def __call__(self, r): ...

    def derivative(self, *, nu: int = 1): ...

    def integrate(self, a: float, b: float) -> float: ...

    def on_interval(self, lo: float, hi: float): ...

    def rescaled(self, *, k: float, v: float): ...


def _is_layer_function(x) -> bool:
    return hasattr(x, "interval") and callable(x) and hasattr(x, "integrate")


def _is_number(x) -> bool:
    return np.isscalar(x) and not isinstance(x, (str, bytes))


def as_values(x) -> np.ndarray:
    """`x` as a float64 array, or complex128 when it is complex."""
    a = np.asarray(x)
    return a if a.dtype.kind == "c" else a.astype(float)


def as_scalar(x):
    """`x` as a float, or a complex when it is complex."""
    return complex(x) if np.iscomplexobj(x) or isinstance(x, complex) else float(x)


class PolynomialLayer:
    """A piecewise polynomial on one interval, exact in every operation.

    Wraps a scipy `PPoly` whose pieces are local polynomials in
    (r - x[i]) stored highest degree first.  The interval may be given
    separately from the breakpoints, in which case the end pieces
    continue beyond them; by default it is the breakpoint range.
    """

    def __init__(self, ppoly, *, interval=None) -> None:
        if isinstance(ppoly, BSpline):
            ppoly = PPoly.from_spline(ppoly)
        if not isinstance(ppoly, PPoly):
            raise TypeError(f"expected a PPoly or BSpline, got {type(ppoly).__name__}")
        c = as_values(ppoly.c)
        x = np.asarray(ppoly.x, dtype=float)
        self._p = PPoly(c, x, extrapolate=True)
        self._interval = (_interval(interval) if interval is not None
                          else (float(x[0]), float(x[-1])))

    @property
    def interval(self) -> tuple[float, float]:
        return self._interval

    @property
    def ppoly(self) -> PPoly:
        """The scipy piecewise polynomial underneath."""
        return self._p

    @property
    def degree(self) -> int:
        return self._p.c.shape[0] - 1

    def __call__(self, r):
        return as_values(self._p(np.asarray(r, dtype=float)))

    def derivative(self, *, nu: int = 1) -> "PolynomialLayer":
        if nu < 0:
            raise ValueError("nu must be non-negative")
        if nu == 0:
            return self
        return PolynomialLayer(self._p.derivative(nu), interval=self._interval)

    def integrate(self, a: float, b: float) -> float:
        """The signed integral from a to b, exact."""
        return as_scalar(self._p.integrate(float(a), float(b)))

    def on_interval(self, lo: float, hi: float) -> "PolynomialLayer":
        """The same polynomial on [lo, hi]: the end pieces continue."""
        return PolynomialLayer(self._p, interval=(lo, hi))

    def rescaled(self, *, k: float, v: float) -> "PolynomialLayer":
        """v f(r / k) on the interval scaled by k, one multiply per coefficient."""
        k, v = float(k), float(v)
        if k <= 0.0:
            raise ValueError(f"the coordinate factor must be positive, got {k}")
        powers = self.degree - np.arange(self.degree + 1)
        c = self._p.c * (v / k ** powers)[:, None]
        lo, hi = self._interval
        return PolynomialLayer(PPoly(c, self._p.x * k), interval=(k * lo, k * hi))

    def is_zero(self) -> bool:
        """Whether every coefficient is exactly zero."""
        return not np.any(self._p.c)

    # -- the exact algebra --------------------------------------------------

    def _aligned(self, other: "PolynomialLayer"):
        """Both coefficient arrays on the union of the breakpoints."""
        x = np.union1d(self._p.x, other._p.x)
        return x, _refine(self._p, x), _refine(other._p, x)

    def _combine(self, other, op):
        if isinstance(other, PolynomialLayer):
            interval = _require_same_interval(self, other)
            x, a, b = self._aligned(other)
            return PolynomialLayer(PPoly(op(a, b), x), interval=interval)
        if _is_number(other):
            c = np.array(self._p.c)
            return PolynomialLayer(PPoly(op(c, as_scalar(other)), self._p.x),
                                   interval=self._interval)
        if _is_layer_function(other):
            return NotImplemented
        return NotImplemented

    def __add__(self, other):
        if _is_number(other):
            return self._combine(constant_layer(other, self._interval), _add_c)
        return self._combine(other, _add_c)

    __radd__ = __add__

    def __sub__(self, other):
        if _is_number(other):
            return self + (-as_scalar(other))
        return self._combine(other, _sub_c)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return PolynomialLayer(PPoly(-self._p.c, self._p.x), interval=self._interval)

    def __mul__(self, other):
        if isinstance(other, PolynomialLayer):
            interval = _require_same_interval(self, other)
            x, a, b = self._aligned(other)
            return PolynomialLayer(PPoly(_mul_c(a, b), x), interval=interval)
        if _is_number(other):
            return PolynomialLayer(PPoly(self._p.c * as_scalar(other), self._p.x),
                                   interval=self._interval)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if _is_number(other):
            return self * (1.0 / as_scalar(other))
        if isinstance(other, PolynomialLayer):
            interval = _require_same_interval(self, other)
            return NumericLayer(lambda r: self(r) / other(r), interval)
        return NotImplemented

    def __pow__(self, n):
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("a polynomial layer is raised to a non-negative integer")
        out = constant_layer(1.0, self._interval)
        for _ in range(int(n)):
            out = out * self
        return out

    def __repr__(self) -> str:
        lo, hi = self._interval
        return (f"PolynomialLayer(degree {self.degree}, {self._p.c.shape[1]} "
                f"pieces on [{lo:g}, {hi:g}])")


def _refine(p: PPoly, x: np.ndarray) -> np.ndarray:
    """p's coefficients on the breakpoints x, a superset of p.x."""
    n = p.c.shape[0]
    out = np.zeros((n, x.size - 1), dtype=p.c.dtype)
    mid = 0.5 * (x[:-1] + x[1:])
    owner = np.clip(np.searchsorted(p.x, mid, side="right") - 1, 0, p.x.size - 2)
    for j, i in enumerate(owner):
        shift = x[j] - p.x[i]
        if shift == 0.0:
            out[:, j] = p.c[:, i]
        else:
            q = Polynomial(p.c[::-1, i])(Polynomial([shift, 1.0]))
            out[n - q.coef.size:, j] = q.coef[::-1]
    return out


def _pad(a: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, a.shape[1]), dtype=a.dtype)
    out[n - a.shape[0]:] = a
    return out


def _add_c(a, b):
    if _is_number(b):
        a = np.array(a, dtype=np.result_type(a, b))
        a[-1] += b
        return a
    n = max(a.shape[0], b.shape[0])
    dtype = np.result_type(a, b)
    return _pad(a.astype(dtype), n) + _pad(b.astype(dtype), n)


def _sub_c(a, b):
    return _add_c(a, -b)


def _mul_c(a, b):
    from numpy.polynomial.polynomial import polymul
    npiece = a.shape[1]
    cols = [polymul(a[::-1, i], b[::-1, i])[::-1] for i in range(npiece)]
    deg = max(c.size for c in cols)
    out = np.zeros((deg, npiece), dtype=np.result_type(a, b))
    for i, col in enumerate(cols):
        out[deg - col.size:, i] = col
    return out


class NumericLayer:
    """Any callable of the radius on one interval, with honest calculus.

    The derivative is a central difference with step `dstep` times the
    interval's width, or the callable given as `derivative` for the
    first one; the integral is adaptive quadrature.  Arithmetic with
    anything gives another `NumericLayer`.
    """

    def __init__(self, fn, interval, *, derivative=None,
                 dstep: float = 1e-6) -> None:
        if not callable(fn):
            raise TypeError(f"expected a callable, got {type(fn).__name__}")
        self._fn = fn
        self._interval = _interval(interval)
        self._d = derivative
        self._dstep = float(dstep)

    @property
    def interval(self) -> tuple[float, float]:
        return self._interval

    @property
    def fn(self):
        """The callable underneath."""
        return self._fn

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        out = as_values(self._fn(r))
        return np.broadcast_to(out, r.shape).copy() if out.shape != r.shape else out

    def derivative(self, *, nu: int = 1) -> "NumericLayer":
        if nu < 0:
            raise ValueError("nu must be non-negative")
        if nu == 0:
            return self
        if self._d is not None:
            first = NumericLayer(self._d, self._interval, dstep=self._dstep)
        else:
            h = self._dstep * (self._interval[1] - self._interval[0])
            fn = self._fn
            first = NumericLayer(lambda r: (as_values(fn(r + h))
                                            - as_values(fn(r - h))) / (2.0 * h),
                                 self._interval, dstep=self._dstep)
        return first.derivative(nu=nu - 1)

    def integrate(self, a: float, b: float) -> float:
        """The signed integral from a to b by adaptive quadrature; the real
        and imaginary parts of a complex function separately."""
        lo, hi = self._interval
        if np.iscomplexobj(self(np.array(0.5 * (lo + hi)))):
            re, _ = quad(lambda x: float(self(np.array(x)).real), float(a),
                         float(b), limit=200)
            im, _ = quad(lambda x: float(self(np.array(x)).imag), float(a),
                         float(b), limit=200)
            return complex(re, im)
        val, _ = quad(lambda x: float(self(np.array(x))), float(a), float(b),
                      limit=200)
        return float(val)

    def on_interval(self, lo: float, hi: float) -> "NumericLayer":
        """The same callable on [lo, hi]."""
        return NumericLayer(self._fn, (lo, hi), derivative=self._d, dstep=self._dstep)

    def rescaled(self, *, k: float, v: float) -> "NumericLayer":
        k, v = float(k), float(v)
        if k <= 0.0:
            raise ValueError(f"the coordinate factor must be positive, got {k}")
        fn, d = self._fn, self._d
        lo, hi = self._interval
        return NumericLayer(
            lambda r: v * as_values(fn(np.asarray(r, dtype=float) / k)),
            (k * lo, k * hi),
            derivative=(None if d is None else
                        lambda r: (v / k) * as_values(
                            d(np.asarray(r, dtype=float) / k))),
            dstep=self._dstep)

    def _binary(self, other, op):
        if _is_number(other):
            c = as_scalar(other)
            return NumericLayer(lambda r: op(self(r), c), self._interval,
                                dstep=self._dstep)
        if _is_layer_function(other):
            interval = _require_same_interval(self, other)
            return NumericLayer(
                lambda r: op(self(r), as_values(other(r))),
                interval, dstep=self._dstep)
        return NotImplemented

    def __add__(self, other):
        return self._binary(other, np.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return NumericLayer(lambda r: -self(r), self._interval, dstep=self._dstep)

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, np.divide)

    def __rtruediv__(self, other):
        if _is_number(other):
            c = float(other)
            return NumericLayer(lambda r: c / self(r), self._interval,
                                dstep=self._dstep)
        return NotImplemented

    def __pow__(self, n):
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("a layer function is raised to a non-negative integer")
        return NumericLayer(lambda r: self(r) ** int(n), self._interval,
                            dstep=self._dstep)

    def __repr__(self) -> str:
        lo, hi = self._interval
        return f"NumericLayer({self._fn!r} on [{lo:g}, {hi:g}])"


def as_layer_function(fn, interval) -> LayerFunction:
    """Adapt `fn` to the protocol on `interval`.

    A layer function is returned as it is when its interval is the one
    given, and re-stated on it otherwise; a `PPoly` or `BSpline` becomes
    a `PolynomialLayer`, a number a constant, any other callable a
    `NumericLayer`.
    """
    interval = _interval(interval)
    if _is_layer_function(fn):
        return fn if same_interval(fn.interval, interval) else fn.on_interval(*interval)
    if isinstance(fn, (PPoly, BSpline)):
        return PolynomialLayer(fn, interval=interval)
    if _is_number(fn):
        return constant_layer(fn, interval)
    if callable(fn):
        return NumericLayer(fn, interval)
    raise TypeError(f"expected a callable, got {type(fn).__name__}")


def polynomial_layer(coeffs, interval, *, scale: float = 1.0) -> PolynomialLayer:
    """sum_k c_k (r / scale)^k on `interval`, as one exact piece in r."""
    lo, hi = _interval(interval)
    c = as_values(np.asarray(coeffs).ravel())
    if c.size == 0:
        raise ValueError("need at least one coefficient")
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    p = Polynomial(c / scale ** np.arange(c.size))          # in r
    local = p(Polynomial([lo, 1.0]))                         # in (r - lo)
    coef = np.zeros(c.size, dtype=c.dtype)
    coef[:local.coef.size] = local.coef
    return PolynomialLayer(PPoly(coef[::-1][:, None], [lo, hi]))


def constant_layer(value, interval) -> PolynomialLayer:
    """The constant `value` on `interval`, as an exact polynomial."""
    return polynomial_layer([as_scalar(value)], interval)


def polynomial_fit(fn, interval, *, degree: int,
                   n: int | None = None) -> PolynomialLayer:
    """The least-squares polynomial of `degree` through `fn` on `interval`.

    `fn` is sampled at `n` Chebyshev points (2 degree + 1 by default);
    the fit is the way a numeric layer is brought into the exact
    algebra on purpose, and the caller judges its residual.
    """
    lo, hi = _interval(interval)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    n = 2 * degree + 1 if n is None else int(n)
    if n <= degree:
        raise ValueError("need more points than the degree")
    t = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    r = 0.5 * (lo + hi) + 0.5 * (hi - lo) * t
    y = as_values(fn(r))
    p = Polynomial.fit(r, y, degree, domain=[lo, hi]).convert()
    return polynomial_layer(p.coef, (lo, hi))
