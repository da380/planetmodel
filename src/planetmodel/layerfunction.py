"""Functions of one radius on one interval, exact where they can be.

A layer function is what a field stores per component: a callable of the
radius on one interval that also differentiates and integrates itself,
re-states itself on another interval, and rescales its coordinate and
value.  Three kinds are shipped.  `PolynomialLayer` wraps a scipy `PPoly`
and does everything on the coefficients: sums, products and powers of
polynomials are polynomials, a derivative or an antiderivative is exact,
and the rescaling f(r) -> v f(r / k) is one multiply per coefficient
(breakpoints k b, coefficients v c / k^p).  `ExponentialLayer` is
q(r) exp(p(r)) for piecewise polynomials q and p, the form of a deck's
column joined in its logarithm: its derivative (q' + q p') exp(p) is of
the same form, so every derivative is exact, as are products with
polynomials, numbers and its own kind, powers and the rescaling (q by
(k, v), p by (k, 1)); its integral is exact on a piece where p is a line
and q a constant, and adaptive quadrature elsewhere.  `NumericLayer`
wraps any callable and supplies the rest honestly: a central difference
with a step relative to the interval's width, and adaptive quadrature.

A layer function whose natural logarithm is a piecewise polynomial says
so: `logarithm()` gives that logarithm as a `PolynomialLayer`, and None
where there is none.  An exponential layer has one where q is a positive
constant on every piece (log q + p), a polynomial layer where it is a
positive constant on every piece; a model file holds such a function
exactly by a radial rule in the logarithm (`modelfile.LOG_RULES`).

Arithmetic between layer functions on the same interval follows the
operands: polynomial with polynomial stays polynomial, an exponential
layer times a polynomial or an exponential layer stays exponential, and
anything else becomes a numeric closure.  Beyond its interval a
polynomial continues as the same polynomial and a callable is called as
given; `on_interval` is how that continuation is asked for on purpose,
while the fields built on layer functions refuse radii outside their
own interval.
"""
from __future__ import annotations

from collections.abc import Callable
from types import NotImplementedType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.interpolate import BSpline, PPoly

__all__ = ["LayerFunction", "LayerFunctionLike", "PolynomialLayer", "ExponentialLayer",
           "NumericLayer", "as_layer_function", "polynomial_layer", "constant_layer",
           "polynomial_fit", "same_interval", "as_interval", "as_values",
           "as_scalar", "PolynomialLike"]

#: Two intervals are the same when their ends agree to this fraction of
#: the wider one's width.
_INTERVAL_RTOL = 1e-9


def as_interval(interval: tuple[float, float]) -> tuple[float, float]:
    """`interval` as a (lo, hi) pair of floats, refused unless it increases."""
    lo, hi = (float(x) for x in interval)
    if not hi > lo:
        raise ValueError(f"an interval must increase, got ({lo:g}, {hi:g})")
    return lo, hi


def same_interval(a: tuple[float, float], b: tuple[float, float], *,
                  rtol: float = _INTERVAL_RTOL) -> bool:
    """Whether two intervals coincide to `rtol` of the wider width."""
    (a0, a1), (b0, b1) = as_interval(a), as_interval(b)
    tol = rtol * max(a1 - a0, b1 - b0)
    return abs(a0 - b0) <= tol and abs(a1 - b1) <= tol


def _require_same_interval(a: LayerFunction, b: LayerFunction) -> tuple[float, float]:
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

    def __call__(self, r: ArrayLike) -> np.ndarray:
        """The values at `r`, of its shape."""
        ...

    def derivative(self, *, nu: int = 1) -> LayerFunction:
        """The nu-th derivative as a layer function; nu=0 is the function."""
        ...

    def integrate(self, a: float, b: float) -> float | complex:
        """The signed integral from a to b."""
        ...

    def on_interval(self, lo: float, hi: float) -> LayerFunction:
        """The same function restated on [lo, hi]."""
        ...

    def rescaled(self, *, k: float, v: float) -> LayerFunction:
        """v f(r / k) on the interval scaled by k."""
        ...


#: What `as_layer_function` accepts: a layer function, a scipy `PPoly` or
#: `BSpline`, a number for a constant, or any callable of the radius.
type LayerFunctionLike = (LayerFunction | PPoly | BSpline | float | complex
                          | Callable[[np.ndarray], ArrayLike])


def _is_layer_function(x: object) -> bool:
    return hasattr(x, "interval") and callable(x) and hasattr(x, "integrate")


def _is_number(x: object) -> bool:
    return np.isscalar(x) and not isinstance(x, (str, bytes))


def as_values(x: ArrayLike) -> np.ndarray:
    """`x` as a float64 array, or complex128 when it is complex."""
    a = np.asarray(x)
    return a if a.dtype.kind == "c" else a.astype(float)


def as_scalar(x: ArrayLike) -> float | complex:
    """`x` as a float, or a complex when it is complex."""
    return complex(x) if np.iscomplexobj(x) or isinstance(x, complex) else float(x)


class PolynomialLayer:
    """A piecewise polynomial on one interval, exact in every operation.

    Wraps a scipy `PPoly` whose pieces are local polynomials in
    (r - x[i]) stored highest degree first.  The interval is given
    separately from the breakpoints, the end pieces continuing beyond
    them where it is wider; `as_layer_function` takes the breakpoint
    range where none is meant.
    """

    def __init__(self, interval: tuple[float, float],
                 ppoly: PPoly | BSpline) -> None:
        if isinstance(ppoly, BSpline):
            ppoly = PPoly.from_spline(ppoly)
        if not isinstance(ppoly, PPoly):
            raise TypeError(f"expected a PPoly or BSpline, got {type(ppoly).__name__}")
        c = as_values(ppoly.c)
        x = np.asarray(ppoly.x, dtype=float)
        self._p = PPoly(c, x, extrapolate=True)
        self._interval = as_interval(interval)

    @property
    def interval(self) -> tuple[float, float]:
        """The (lo, hi) span of radius the function is stated on."""
        return self._interval

    @property
    def ppoly(self) -> PPoly:
        """The scipy piecewise polynomial underneath."""
        return self._p

    @property
    def degree(self) -> int:
        """The polynomial degree of the pieces."""
        return self._p.c.shape[0] - 1

    def __call__(self, r: ArrayLike) -> np.ndarray:
        """The polynomial at `r`, of its shape."""
        return as_values(self._p(np.asarray(r, dtype=float)))

    def derivative(self, *, nu: int = 1) -> "PolynomialLayer":
        """The nu-th derivative, exact on the coefficients."""
        if nu < 0:
            raise ValueError("nu must be non-negative")
        if nu == 0:
            return self
        return PolynomialLayer(self._interval, self._p.derivative(nu))

    def integrate(self, a: float, b: float) -> float | complex:
        """The signed integral from a to b, exact."""
        return as_scalar(self._p.integrate(float(a), float(b)))

    def on_interval(self, lo: float, hi: float) -> "PolynomialLayer":
        """The same polynomial on [lo, hi]: the end pieces continue."""
        return PolynomialLayer((lo, hi), self._p)

    def rescaled(self, *, k: float, v: float) -> "PolynomialLayer":
        """v f(r / k) on the interval scaled by k, one multiply per coefficient."""
        k, v = float(k), float(v)
        if k <= 0.0:
            raise ValueError(f"the coordinate factor must be positive, got {k}")
        powers = self.degree - np.arange(self.degree + 1)
        c = self._p.c * (v / k ** powers)[:, None]
        lo, hi = self._interval
        return PolynomialLayer((k * lo, k * hi), PPoly(c, self._p.x * k))

    def is_zero(self) -> bool:
        """Whether every coefficient is exactly zero."""
        return not np.any(self._p.c)

    def logarithm(self) -> "PolynomialLayer | None":
        """The natural logarithm, a polynomial layer of the same pieces,
        where the function is a real positive constant on every piece; None
        otherwise, the logarithm of a polynomial that varies being none."""
        c = self._p.c
        if np.iscomplexobj(c) or np.any(c[:-1]) or not np.all(c[-1] > 0.0):
            return None
        return PolynomialLayer(self._interval, PPoly(np.log(c[-1:]), self._p.x))

    # -- the exact algebra --------------------------------------------------

    def _aligned(self, other: "PolynomialLayer"
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Both coefficient arrays on the union of the breakpoints."""
        x = np.union1d(self._p.x, other._p.x)
        return x, _refine(self._p, x), _refine(other._p, x)

    def _combine(self, other: object, op: Callable[[np.ndarray, object], np.ndarray]
                 ) -> "PolynomialLayer | NotImplementedType":
        if isinstance(other, PolynomialLayer):
            interval = _require_same_interval(self, other)
            x, a, b = self._aligned(other)
            return PolynomialLayer(interval, PPoly(op(a, b), x))
        if _is_number(other):
            c = np.array(self._p.c)
            return PolynomialLayer(self._interval,
                                   PPoly(op(c, as_scalar(other)), self._p.x))
        if _is_layer_function(other):
            return NotImplemented
        return NotImplemented

    def __add__(self, other: LayerFunction | float | complex) -> "PolynomialLayer":
        if _is_number(other):
            return self._combine(constant_layer(self._interval, other), _add_c)
        return self._combine(other, _add_c)

    __radd__ = __add__

    def __sub__(self, other: LayerFunction | float | complex) -> "PolynomialLayer":
        if _is_number(other):
            return self + (-as_scalar(other))
        return self._combine(other, _sub_c)

    def __rsub__(self, other: LayerFunction | float | complex) -> "PolynomialLayer":
        return (-self) + other

    def __neg__(self) -> "PolynomialLayer":
        return PolynomialLayer(self._interval, PPoly(-self._p.c, self._p.x))

    def __mul__(self, other: LayerFunction | float | complex) -> "PolynomialLayer":
        if isinstance(other, PolynomialLayer):
            interval = _require_same_interval(self, other)
            x, a, b = self._aligned(other)
            return PolynomialLayer(interval, PPoly(_mul_c(a, b), x))
        if _is_number(other):
            return PolynomialLayer(self._interval,
                                   PPoly(self._p.c * as_scalar(other), self._p.x))
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other: LayerFunction | float | complex
                    ) -> "PolynomialLayer | NumericLayer":
        if _is_number(other):
            return self * (1.0 / as_scalar(other))
        if _is_layer_function(other):
            interval = _require_same_interval(self, other)
            return NumericLayer(interval,
                                lambda r: self(r) / as_values(other(r)))
        return NotImplemented

    def __pow__(self, n: int) -> "PolynomialLayer":
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("a polynomial layer is raised to a non-negative integer")
        out = constant_layer(self._interval, 1.0)
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


def _add_c(a: np.ndarray, b: np.ndarray | float | complex) -> np.ndarray:
    if _is_number(b):
        a = np.array(a, dtype=np.result_type(a, b))
        a[-1] += b
        return a
    n = max(a.shape[0], b.shape[0])
    dtype = np.result_type(a, b)
    return _pad(a.astype(dtype), n) + _pad(b.astype(dtype), n)


def _sub_c(a: np.ndarray, b: np.ndarray | float | complex) -> np.ndarray:
    return _add_c(a, -b)


def _mul_c(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from numpy.polynomial.polynomial import polymul
    npiece = a.shape[1]
    cols = [polymul(a[::-1, i], b[::-1, i])[::-1] for i in range(npiece)]
    deg = max(c.size for c in cols)
    out = np.zeros((deg, npiece), dtype=np.result_type(a, b))
    for i, col in enumerate(cols):
        out[deg - col.size:, i] = col
    return out


#: What an `ExponentialLayer` takes for its exponent or factor: a
#: polynomial layer, or a scipy `PPoly` or `BSpline` stated on its interval.
type PolynomialLike = PolynomialLayer | PPoly | BSpline


def _polynomial_on(interval: tuple[float, float],
                   x: PolynomialLike | float | complex) -> PolynomialLayer:
    """`x` as a polynomial layer on `interval`: a polynomial restated on
    it, a `PPoly` or `BSpline` wrapped, a number a constant."""
    if isinstance(x, PolynomialLayer):
        return PolynomialLayer(interval, x.ppoly)
    if isinstance(x, (PPoly, BSpline)):
        return PolynomialLayer(interval, x)
    if _is_number(x):
        return constant_layer(interval, as_scalar(x))
    raise TypeError(f"expected a polynomial layer, a PPoly, a BSpline or a number, got "
                    f"{type(x).__name__}")


def _piece(p: PPoly, x: float) -> int:
    """The index of the piece of `p` holding `x`, the end pieces continued."""
    return int(np.clip(np.searchsorted(p.x, x, side="right") - 1, 0, p.x.size - 2))


class ExponentialLayer:
    """q(r) exp(p(r)) on one interval, q and p piecewise polynomials.

    `exponent` is p, real, and `factor` q, one by default; each is
    restated on the interval, its end pieces continuing beyond its
    breakpoints.  A deck's column joined in its logarithm is one: p the
    kind through log(y_i / y_0) and q the constant y_0
    (`deck.LOG_KINDS`).  The calculus and the algebra are those of the
    module docstring.
    """

    def __init__(self, interval: tuple[float, float], exponent: PolynomialLike, *,
                 factor: PolynomialLike | float | complex = 1.0) -> None:
        self._interval = as_interval(interval)
        if _is_number(exponent):
            raise TypeError("the exponent of an exponential layer is a polynomial; a "
                            "constant one is a factor")
        self._p = _polynomial_on(self._interval, exponent)
        self._q = _polynomial_on(self._interval, factor)
        if np.iscomplexobj(self._p.ppoly.c):
            raise ValueError("the exponent of an exponential layer is real")

    @property
    def interval(self) -> tuple[float, float]:
        """The (lo, hi) span of radius the function is stated on."""
        return self._interval

    @property
    def exponent(self) -> PolynomialLayer:
        """p, the polynomial in the exponential."""
        return self._p

    @property
    def factor(self) -> PolynomialLayer:
        """q, the polynomial multiplying the exponential."""
        return self._q

    def logarithm(self) -> PolynomialLayer | None:
        """The natural logarithm, log q + p, a polynomial layer on the
        union of their breakpoints, where q is a real positive constant on
        every piece; None otherwise."""
        log_q = self._q.logarithm()
        return None if log_q is None else self._p + log_q

    def __call__(self, r: ArrayLike) -> np.ndarray:
        """q(r) exp(p(r)) at `r`, of its shape."""
        r = np.asarray(r, dtype=float)
        return as_values(self._q(r) * np.exp(self._p(r)))

    def derivative(self, *, nu: int = 1) -> "ExponentialLayer":
        """The nu-th derivative, exact: each one (q' + q p') exp(p)."""
        if nu < 0:
            raise ValueError("nu must be non-negative")
        if nu == 0:
            return self
        q, dp = self._q, self._p.derivative()
        for _ in range(nu):
            q = q.derivative() + q * dp
        return ExponentialLayer(self._interval, self._p, factor=q)

    def integrate(self, a: float, b: float) -> float | complex:
        """The signed integral from a to b, piece by piece between the
        breakpoints of p and q: exact where p is a line and q a constant,
        q(u) exp(p(u)) h expm1(s) / s over a piece [u, u + h] on which p
        rises by s; adaptive quadrature to a relative 1e-12 elsewhere."""
        a, b = float(a), float(b)
        lo, hi = min(a, b), max(a, b)
        cuts = np.union1d(self._p.ppoly.x, self._q.ppoly.x)
        edges = np.unique(np.concatenate([[lo, hi], cuts[(cuts > lo) & (cuts < hi)]]))
        total: float | complex = 0.0
        for u, v in zip(edges[:-1], edges[1:]):
            total += self._integral_of_piece(float(u), float(v))
        return as_scalar(total if b >= a else -total)

    def _integral_of_piece(self, u: float, v: float) -> float | complex:
        """The integral over [u, v], within one piece of p and of q."""
        mid = 0.5 * (u + v)
        p, q = self._p.ppoly, self._q.ppoly
        cp, cq = p.c[:, _piece(p, mid)], q.c[:, _piece(q, mid)]
        if not np.any(cp[:-2]) and not np.any(cq[:-1]):
            s = float(cp[-2]) * (v - u) if cp.size > 1 else 0.0
            ratio = float(np.expm1(s) / s) if s != 0.0 else 1.0
            return as_scalar(self(np.array(u))) * (v - u) * ratio
        probe = np.abs(self(np.linspace(u, v, 9)))
        kw = {"limit": 200, "epsrel": 1e-12,
              "epsabs": 1e-15 * float(np.max(probe)) * (v - u)}
        if np.iscomplexobj(q.c):
            re, _ = quad(lambda x: float(self(np.array(x)).real), u, v, **kw)
            im, _ = quad(lambda x: float(self(np.array(x)).imag), u, v, **kw)
            return complex(re, im)
        val, _ = quad(lambda x: float(self(np.array(x))), u, v, **kw)
        return float(val)

    def on_interval(self, lo: float, hi: float) -> "ExponentialLayer":
        """The same function on [lo, hi]: both polynomials continue."""
        return ExponentialLayer((lo, hi), self._p, factor=self._q)

    def rescaled(self, *, k: float, v: float) -> "ExponentialLayer":
        """v f(r / k) on the interval scaled by k: q rescaled by (k, v) and
        p by (k, 1)."""
        k, v = float(k), float(v)
        if k <= 0.0:
            raise ValueError(f"the coordinate factor must be positive, got {k}")
        lo, hi = self._interval
        return ExponentialLayer((k * lo, k * hi), self._p.rescaled(k=k, v=1.0),
                                factor=self._q.rescaled(k=k, v=v))

    # -- the algebra --------------------------------------------------------

    def _numeric(self, other: object, op: Callable[[np.ndarray, object], np.ndarray]
                 ) -> "NumericLayer | NotImplementedType":
        """op(self, other) as a numeric closure, other a number or a layer
        function on the same interval."""
        if _is_number(other):
            c = as_scalar(other)
            return NumericLayer(self._interval, lambda r: op(self(r), c))
        if _is_layer_function(other):
            interval = _require_same_interval(self, other)
            return NumericLayer(interval, lambda r: op(self(r), as_values(other(r))))
        return NotImplemented

    def _inverse(self) -> "ExponentialLayer | None":
        """1 / f as an exponential layer where q is a non-zero constant on
        every piece, else None."""
        c = self._q.ppoly.c
        if np.any(c[:-1]) or not np.all(c[-1] != 0.0):
            return None
        inverse = PPoly((1.0 / c[-1])[None, :], self._q.ppoly.x)
        return ExponentialLayer(self._interval, -self._p, factor=inverse)

    def __add__(self, other: LayerFunction | float | complex) -> "NumericLayer":
        return self._numeric(other, np.add)

    __radd__ = __add__

    def __sub__(self, other: LayerFunction | float | complex) -> "NumericLayer":
        return self._numeric(other, np.subtract)

    def __rsub__(self, other: LayerFunction | float | complex) -> "NumericLayer":
        return (-self) + other

    def __neg__(self) -> "ExponentialLayer":
        return ExponentialLayer(self._interval, self._p, factor=-self._q)

    def __mul__(self, other: LayerFunction | float | complex
                ) -> "ExponentialLayer | NumericLayer":
        if isinstance(other, ExponentialLayer):
            interval = _require_same_interval(self, other)
            return ExponentialLayer(interval, self._p + other._p,
                                    factor=self._q * other._q)
        if isinstance(other, PolynomialLayer):
            interval = _require_same_interval(self, other)
            return ExponentialLayer(interval, self._p, factor=self._q * other)
        if _is_number(other):
            return ExponentialLayer(self._interval, self._p,
                                    factor=self._q * as_scalar(other))
        return self._numeric(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other: LayerFunction | float | complex
                    ) -> "ExponentialLayer | NumericLayer":
        if _is_number(other):
            return self * (1.0 / as_scalar(other))
        inverse = other._inverse() if isinstance(other, ExponentialLayer) else None
        if inverse is not None:
            return self * inverse
        return self._numeric(other, np.divide)

    def __rtruediv__(self, other: float | complex) -> "ExponentialLayer | NumericLayer":
        if not _is_number(other):
            return NotImplemented
        inverse = self._inverse()
        if inverse is not None:
            return inverse * as_scalar(other)
        c = as_scalar(other)
        return NumericLayer(self._interval, lambda r: c / self(r))

    def __pow__(self, n: int) -> "ExponentialLayer":
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("a layer function is raised to a non-negative integer")
        return ExponentialLayer(self._interval, self._p * int(n),
                                factor=self._q ** int(n))

    def __repr__(self) -> str:
        lo, hi = self._interval
        return (f"ExponentialLayer(exponent of degree {self._p.degree}, factor of "
                f"degree {self._q.degree} on [{lo:g}, {hi:g}])")


def _stencil(offsets: np.ndarray, nu: int) -> np.ndarray:
    """The weights w with sum_k w_k f(x + o_k h) = h^nu f^(nu)(x) to the
    order the offsets allow: the nu-th derivative of the interpolating
    polynomial."""
    n = offsets.size
    V = offsets[None, :] ** np.arange(n)[:, None]
    rhs = np.zeros(n)
    rhs[nu] = float(np.prod(np.arange(1, nu + 1)))
    return np.linalg.solve(V, rhs)


def _differenced(fn: Callable[[np.ndarray], ArrayLike], lo: float, hi: float,
                 h: float, nu: int) -> Callable[[np.ndarray], np.ndarray]:
    """d^nu fn / dr^nu by finite differences of step h on nu + 2 points
    (nu + 3 for even nu, so that the central stencil is symmetric),
    centred where the stencil fits in [lo, hi] and slid inward where it
    would not, so that every evaluation lies in it."""
    n = nu + 2 + (nu + 1) % 2
    half = (n - 1) // 2

    def f(x: np.ndarray) -> np.ndarray:
        return np.broadcast_to(as_values(fn(x)), x.shape)

    def d(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        start = np.full(r.shape, -half)
        start = np.maximum(start, np.ceil((lo - r) / h))
        start = np.minimum(start, np.floor((hi - r) / h) - (n - 1)).astype(int)
        parts = []
        for s in np.unique(start):
            mask = start == s
            offsets = np.arange(s, s + n, dtype=float)
            w = _stencil(offsets, nu)
            x = r[mask]
            parts.append((mask, sum(wk * f(x + ok * h) for wk, ok in zip(w, offsets))
                          / h ** nu))
        out = np.empty(r.shape, dtype=np.result_type(*(v for _, v in parts))
                       if parts else float)
        for mask, values in parts:
            out[mask] = values
        return out
    return d


class NumericLayer:
    """Any callable of the radius on one interval, with honest calculus.

    The derivative is a central difference with step `dstep` times the
    interval's width, or the callable given as `derivative` for the
    first one; the integral is adaptive quadrature.  Arithmetic with
    anything gives another `NumericLayer`.
    """

    def __init__(self, interval: tuple[float, float],
                 fn: Callable[[np.ndarray], ArrayLike], *,
                 derivative: Callable[[np.ndarray], ArrayLike] | None = None,
                 dstep: float = 1e-6) -> None:
        if not callable(fn):
            raise TypeError(f"expected a callable, got {type(fn).__name__}")
        self._fn = fn
        self._interval = as_interval(interval)
        self._d = derivative
        self._dstep = float(dstep)

    @property
    def interval(self) -> tuple[float, float]:
        """The (lo, hi) span of radius the function is stated on."""
        return self._interval

    @property
    def fn(self) -> Callable[[np.ndarray], ArrayLike]:
        """The callable underneath."""
        return self._fn

    def __call__(self, r: ArrayLike) -> np.ndarray:
        """The callable at `r`, broadcast to its shape."""
        r = np.asarray(r, dtype=float)
        out = as_values(self._fn(r))
        return np.broadcast_to(out, r.shape).copy() if out.shape != r.shape else out

    def derivative(self, *, nu: int = 1) -> "NumericLayer":
        """The nu-th derivative: the supplied first derivative and then
        differences of it, else a finite-difference stencil for the nu-th
        derivative, centred inside the interval and slid inward near its
        ends, so that no evaluation leaves it.  The step is `dstep`, or
        eps^(1 / (nu + 2)) where that is larger, times the width: the
        balance of truncation and rounding for a nu-th difference."""
        if nu < 0:
            raise ValueError("nu must be non-negative")
        if nu == 0:
            return self
        if self._d is not None:
            first = NumericLayer(self._interval, self._d, dstep=self._dstep)
            return first.derivative(nu=nu - 1)
        lo, hi = self._interval
        h = max(self._dstep, float(np.finfo(float).eps) ** (1.0 / (nu + 2))) \
            * (hi - lo)
        return NumericLayer(self._interval, _differenced(self._fn, lo, hi, h, nu),
                            dstep=self._dstep)

    def integrate(self, a: float, b: float) -> float | complex:
        """The signed integral from a to b by adaptive quadrature, to a
        relative tolerance of 1e-10 of the integral of |f| as a few
        samples estimate it; the real and imaginary parts of a complex
        function separately."""
        a, b = float(a), float(b)
        probe = np.abs(self(np.linspace(a, b, 9))) if a != b else np.zeros(1)
        floor = 1e-14 * float(np.max(probe)) * abs(b - a)
        kw = {"limit": 200, "epsrel": 1e-10, "epsabs": floor}
        lo, hi = self._interval
        if np.iscomplexobj(self(np.array(0.5 * (lo + hi)))):
            re, _ = quad(lambda x: float(self(np.array(x)).real), a, b, **kw)
            im, _ = quad(lambda x: float(self(np.array(x)).imag), a, b, **kw)
            return complex(re, im)
        val, _ = quad(lambda x: float(self(np.array(x))), a, b, **kw)
        return float(val)

    def on_interval(self, lo: float, hi: float) -> "NumericLayer":
        """The same callable on [lo, hi]."""
        return NumericLayer((lo, hi), self._fn, derivative=self._d, dstep=self._dstep)

    def rescaled(self, *, k: float, v: float) -> "NumericLayer":
        """v f(r / k) as a closure, on the interval scaled by k."""
        k, v = float(k), float(v)
        if k <= 0.0:
            raise ValueError(f"the coordinate factor must be positive, got {k}")
        fn, d = self._fn, self._d
        lo, hi = self._interval
        return NumericLayer(
            (k * lo, k * hi),
            lambda r: v * as_values(fn(np.asarray(r, dtype=float) / k)),
            derivative=(None if d is None else
                        lambda r: (v / k) * as_values(
                            d(np.asarray(r, dtype=float) / k))),
            dstep=self._dstep)

    def _binary(self, other: object, op: Callable[[np.ndarray, object], np.ndarray]
                ) -> "NumericLayer | NotImplementedType":
        if _is_number(other):
            c = as_scalar(other)
            return NumericLayer(self._interval, lambda r: op(self(r), c),
                                dstep=self._dstep)
        if _is_layer_function(other):
            interval = _require_same_interval(self, other)
            return NumericLayer(
                interval,
                lambda r: op(self(r), as_values(other(r))), dstep=self._dstep)
        return NotImplemented

    def __add__(self, other: LayerFunction | float | complex) -> "NumericLayer":
        return self._binary(other, np.add)

    __radd__ = __add__

    def __sub__(self, other: LayerFunction | float | complex) -> "NumericLayer":
        return self._binary(other, np.subtract)

    def __rsub__(self, other: LayerFunction | float | complex) -> "NumericLayer":
        return (-self) + other

    def __neg__(self) -> "NumericLayer":
        return NumericLayer(self._interval, lambda r: -self(r), dstep=self._dstep)

    def __mul__(self, other: LayerFunction | float | complex) -> "NumericLayer":
        return self._binary(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other: LayerFunction | float | complex) -> "NumericLayer":
        return self._binary(other, np.divide)

    def __rtruediv__(self, other: float | complex) -> "NumericLayer":
        if _is_number(other):
            c = as_scalar(other)
            return NumericLayer(self._interval, lambda r: c / self(r),
                                dstep=self._dstep)
        return NotImplemented

    def __pow__(self, n: int) -> "NumericLayer":
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("a layer function is raised to a non-negative integer")
        return NumericLayer(self._interval, lambda r: self(r) ** int(n),
                            dstep=self._dstep)

    def __repr__(self) -> str:
        lo, hi = self._interval
        return f"NumericLayer({self._fn!r} on [{lo:g}, {hi:g}])"


def as_layer_function(interval: tuple[float, float],
                      fn: LayerFunctionLike) -> LayerFunction:
    """Adapt `fn` to the protocol on `interval`.

    A layer function is returned as it is when its interval is the one
    given, and re-stated on it otherwise; a `PPoly` or `BSpline` becomes
    a `PolynomialLayer`, a number a constant, any other callable a
    `NumericLayer`.  Every constructor here takes the interval first.
    """
    interval = as_interval(interval)
    if _is_layer_function(fn):
        return fn if same_interval(fn.interval, interval) else fn.on_interval(*interval)
    if isinstance(fn, (PPoly, BSpline)):
        return PolynomialLayer(interval, fn)
    if _is_number(fn):
        return constant_layer(interval, fn)
    if callable(fn):
        return NumericLayer(interval, fn)
    raise TypeError(f"expected a callable, got {type(fn).__name__}")


def polynomial_layer(interval: tuple[float, float], coeffs: ArrayLike, *,
                     scale: float = 1.0) -> PolynomialLayer:
    """sum_k c_k (r / scale)^k on `interval`, as one exact piece in r."""
    lo, hi = as_interval(interval)
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
    return PolynomialLayer((lo, hi), PPoly(coef[::-1][:, None], [lo, hi]))


def constant_layer(interval: tuple[float, float],
                   value: float | complex) -> PolynomialLayer:
    """The constant `value` on `interval`, as an exact polynomial."""
    return polynomial_layer(interval, [as_scalar(value)])


def polynomial_fit(interval: tuple[float, float],
                   fn: Callable[[np.ndarray], ArrayLike], *, degree: int,
                   n: int | None = None) -> PolynomialLayer:
    """The least-squares polynomial of `degree` through `fn` on `interval`.

    `fn` is sampled at `n` Chebyshev points (2 degree + 1 by default);
    the fit is the way a numeric layer is brought into the exact
    algebra on purpose, and the caller judges its residual.
    """
    lo, hi = as_interval(interval)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    n = 2 * degree + 1 if n is None else int(n)
    if n <= degree:
        raise ValueError("need more points than the degree")
    t = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    r = 0.5 * (lo + hi) + 0.5 * (hi - lo) * t
    y = as_values(fn(r))
    # coefficients in (r - lo), as PPoly holds them: monomials about the
    # centre would lose (lo / width)^degree of the digits on a thin shell
    p = Polynomial.fit(r, y, degree, domain=[lo, hi]).convert(
        domain=[lo, hi], window=[0.0, hi - lo])
    coef = np.zeros(degree + 1, dtype=p.coef.dtype)
    coef[:p.coef.size] = p.coef
    return PolynomialLayer((lo, hi), PPoly(coef[::-1][:, None], [lo, hi]))
