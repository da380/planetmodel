"""layer_function.py -- the layer-function protocol and algebra on it.

A layer function is the smallest unit planetmodel evaluates: something
callable on arrays over one layer's radial interval.  Optionally it also
differentiates and integrates itself exactly, and where it does planetmodel
uses that rather than a numerical stand-in.  scipy's piecewise
polynomials qualify natively; a bare lambda qualifies through
as_layer_function, which supplies the missing operations numerically.

This is the template for every other callable protocol in planetmodel: the
plain function is always accepted, and richer capability is discovered
by attribute rather than demanded up front.

multiply_layer_functions exists for one specific reason.  Deck readers
convert velocities to moduli on load, and the moduli are products
rho * v^2.  PREM's rho and v are exact polynomials, so their products
are exact polynomials too -- but only if the multiplication is done on
the coefficients.  Sampling and refitting would quietly turn PREM from
an exact model into an interpolated one.
"""
from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline, PPoly

__all__ = ["MergedLayerFunction", "LayerFunction", "as_layer_function",
           "multiply_layer_functions", "combine_layer_functions",
           "rescale_layer_function"]

#: Finite-difference step for numerical derivatives of plain callables.
#: Absolute near the origin, relative away from it, so it behaves at both
#: r = 0 and r = 6.371e6.
def _fd_step(r):
    """The differentiation step used by as_layer_function."""
    return np.maximum(1e-6, 1e-8 * np.abs(np.asarray(r, dtype=float)))


@runtime_checkable
class LayerFunction(Protocol):
    """Callable on arrays over one layer; may differentiate and integrate.

    Only __call__ is required.  `derivative(nu=1)` returns another layer
    function and `integrate(a, b)` a float; both are used when present
    and synthesised numerically when not.
    """

    def __call__(self, r): ...


class _NumericLayerFunction:
    """A plain callable, given derivative() and integrate() numerically.

    Both are honest approximations and say so: the derivative is a
    central difference, the integral adaptive quadrature.  Anything
    needing exactness should supply a function that provides its own.
    """

    def __init__(self, fn, *, nu: int = 0) -> None:
        self._fn = fn
        self._nu = nu

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        if self._nu == 0:
            return np.asarray(self._fn(r), dtype=float)
        h = _fd_step(r)
        lo = _NumericLayerFunction(self._fn, nu=self._nu - 1)
        return (lo(r + h) - lo(r - h)) / (2.0 * h)

    def derivative(self, nu: int = 1) -> "_NumericLayerFunction":
        """A further nu derivatives, taken by repeated central differences."""
        return _NumericLayerFunction(self._fn, nu=self._nu + nu)

    def integrate(self, a: float, b: float) -> float:
        """Adaptive quadrature over [a, b]."""
        val, _ = quad(lambda x: float(self(np.array([x]))[0]), float(a), float(b),
                      limit=200)
        return val

    def __repr__(self) -> str:
        d = f", d^{self._nu}" if self._nu else ""
        return f"as_layer_function({self._fn!r}{d})"


def _unwrap(fn):
    """The layer function inside a single-layer field, else `fn` itself.

    `field[i]` is a single-layer field rather than the bare function;
    the helpers here accept either, so
    `multiply_layer_functions(rho[i], vs[i])` reads naturally.
    """
    sk = getattr(fn, "skeleton", None)
    if sk is not None and getattr(sk, "nlayers", 0) == 1 and hasattr(fn, "function"):
        return fn.function
    return fn


def as_layer_function(fn) -> LayerFunction:
    """Adapt any callable to the layer-function protocol.

    A function that already differentiates and integrates itself is
    returned unchanged, so wrapping an exact polynomial never costs it
    its exactness.  A single-layer field is unwrapped to its function.
    """
    fn = _unwrap(fn)
    if hasattr(fn, "derivative") and hasattr(fn, "integrate"):
        return fn
    if not callable(fn):
        raise TypeError(f"expected a callable, got {type(fn).__name__}")
    return _NumericLayerFunction(fn)


def _as_ppoly(f):
    """f as a PPoly if it is exactly one, else None.

    B-splines are piecewise polynomials, so make_interp_spline results
    convert exactly -- which is what lets a spline-fitted deck keep
    exact moduli, not just an analytically defined model like PREM.
    """
    f = _unwrap(f)
    if isinstance(f, PPoly):
        return f
    if isinstance(f, BSpline):
        return PPoly.from_spline(f)
    return None


def _ppoly_product(f: PPoly, g: PPoly) -> PPoly:
    """Exact product of two PPolys sharing breakpoints.

    PPoly stores each piece highest-degree-first in the local variable
    (r - x[i]), so the pieces multiply as ordinary polynomials once both
    coefficient columns are reversed to lowest-first, and the result is
    reversed back.
    """
    from numpy.polynomial.polynomial import polymul

    npiece = f.c.shape[1]
    cols = []
    for i in range(npiece):
        a = f.c[:, i][::-1]                 # lowest-degree-first
        b = g.c[:, i][::-1]
        cols.append(polymul(a, b)[::-1])    # back to highest-first
    deg = max(c.size for c in cols)
    c = np.zeros((deg, npiece))
    for i, col in enumerate(cols):
        c[deg - col.size:, i] = col         # pad the leading (high) degrees
    return PPoly(c, f.x)


def multiply_layer_functions(f, g, *, n: int = 65, kind: str = "cubic",
                             names: tuple[str, str] = ("f", "g")):
    """The product f * g as a layer function, exactly where possible.

    Two PPolys over the same breakpoints multiply on their coefficients
    and the result is exact -- which is what keeps PREM exact when
    velocities are converted to moduli on load.  Anything else is
    sampled on n points and refitted, which is an approximation, so it
    warns and names the operands.
    """
    fp, gp = _as_ppoly(f), _as_ppoly(g)
    if fp is not None and gp is not None and np.array_equal(fp.x, gp.x):
        return _ppoly_product(fp, gp)

    from .radial import make_fitter

    lo, hi = _common_interval(f, g)
    warnings.warn(
        f"multiplying {names[0]} by {names[1]}: no exact product for "
        f"{type(f).__name__} * {type(g).__name__}, so the result is sampled "
        f"on {n} points and refitted ({kind}) and is an approximation",
        stacklevel=2)
    r = np.linspace(lo, hi, n)
    return make_fitter(kind=kind)(r, np.asarray(f(r), dtype=float)
                             * np.asarray(g(r), dtype=float))


def _common_interval(f, g) -> tuple[float, float]:
    """The interval on which both functions are defined.

    Derived from breakpoints where they exist; otherwise the caller is
    multiplying two opaque callables and must have matched them already,
    so fall back to whichever endpoints can be found.
    """
    spans = []
    for h in (f, g):
        q = _as_ppoly(h)
        if q is not None:
            spans.append((float(q.x[0]), float(q.x[-1])))
        elif hasattr(h, "x"):
            spans.append((float(h.x[0]), float(h.x[-1])))
    if not spans:
        raise ValueError(
            "cannot infer an interval for the product: neither operand has "
            "breakpoints, so sample and refit them yourself")
    lo = max(s[0] for s in spans)
    hi = min(s[1] for s in spans)
    if not hi > lo:
        raise ValueError(f"operands do not overlap: {spans}")
    return lo, hi


def combine_layer_functions(terms):
    """The linear combination sum(coeff * f) as a layer function.

    Exact when every operand is a PPoly over the same breakpoints, which
    covers the cases that matter -- promoting an isotropic medium to VTI
    is kappa + 4 mu / 3, and both are polynomials in an exact model.
    Otherwise the result is a closure, evaluated on demand: pointwise
    exact, with derivative and integral supplied numerically.

    terms is an iterable of (coefficient, layer function) pairs.
    """
    terms = [(float(c), f) for c, f in terms]
    if not terms:
        raise ValueError("need at least one term")

    ps = [_as_ppoly(f) for _, f in terms]
    if (all(q is not None for q in ps)
            and all(np.array_equal(q.x, ps[0].x) for q in ps)):
        deg = max(q.c.shape[0] for q in ps)
        c = np.zeros((deg, ps[0].c.shape[1]))
        for (coeff, _), q in zip(terms, ps):
            c[deg - q.c.shape[0]:, :] += coeff * q.c   # align on high degrees
        return PPoly(c, ps[0].x)

    def combined(r):
        r = np.asarray(r, dtype=float)
        out = np.zeros_like(r)
        for coeff, f in terms:
            out = out + coeff * np.asarray(f(r), dtype=float)
        return out

    return as_layer_function(combined)


def rescale_layer_function(f, k: float, vr: float):
    """f re-expressed in a scaled coordinate: g(x) = vr * f(x / k).

    `k` multiplies the coordinate (a radius r becomes x = k r) and `vr`
    multiplies the value.  Exact for piecewise polynomials: on a piece
    with breakpoint b and local coefficients c_m at powers p, the
    substitution x - k b = k (r - b) gives new breakpoints k b and
    coefficients vr * c_m / k^p -- one multiply each, so an exact model
    stays exact through non-dimensionalisation and back.  Anything else
    is wrapped pointwise, with derivatives and integrals supplied
    numerically by the adapter.
    """
    k, vr = float(k), float(vr)
    if k <= 0.0:
        raise ValueError(f"coordinate factor must be positive, got {k}")
    if k == 1.0 and vr == 1.0:
        return f

    q = _as_ppoly(f)
    if q is not None:
        deg = q.c.shape[0] - 1
        powers = deg - np.arange(q.c.shape[0])          # row m has power deg-m
        c = q.c * (vr / k ** powers)[:, None]
        return PPoly(c, q.x * k)

    wrapped = as_layer_function(lambda x: vr * np.asarray(
        f(np.asarray(x, dtype=float) / k), dtype=float))
    return wrapped


class MergedLayerFunction:
    """One layer function dispatching to abutting pieces, by radius.

    What a coarsened body's merged layer carries: the fine layers'
    functions, each answering on its own sub-interval, so the material
    keeps its original resolution while the geometry is coarser.  A
    full LayerFunction: it differentiates piece by piece and integrates
    by splitting at the fine boundaries, so mass and gravity on a
    coarsened body are the fine model's, not a resampling of it.

    `boundaries` are the n + 1 edges of the n pieces.  On a shared
    edge the piece above answers, as `side="upper"` would.
    """

    def __init__(self, funcs, boundaries) -> None:
        self._funcs = tuple(as_layer_function(f) for f in funcs)
        self._b = np.asarray(boundaries, dtype=float)
        if self._b.size != len(self._funcs) + 1:
            raise ValueError(
                f"{len(self._funcs)} pieces need {len(self._funcs) + 1} "
                f"boundaries, got {self._b.size}")

    @property
    def boundaries(self) -> np.ndarray:
        """The edges of the pieces, ascending."""
        return self._b

    @property
    def pieces(self) -> tuple:
        """The piece functions, centre outward."""
        return self._funcs

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        flat = np.atleast_1d(r).ravel()
        idx = np.clip(np.searchsorted(self._b, flat, side="right") - 1,
                      0, len(self._funcs) - 1)
        out = np.empty(flat.shape)
        for i in np.unique(idx):
            m = idx == i
            out[m] = self._funcs[int(i)](flat[m])
        return out.reshape(r.shape) if r.shape else out[0]

    def derivative(self, nu: int = 1) -> "MergedLayerFunction":
        """Differentiate each piece."""
        return MergedLayerFunction([f.derivative(nu) for f in self._funcs],
                                   self._b)

    def integrate(self, a: float, b: float) -> float:
        """Integrate piece by piece over the overlap with [a, b]."""
        total = 0.0
        for i, f in enumerate(self._funcs):
            lo = max(float(a), float(self._b[i]))
            hi = min(float(b), float(self._b[i + 1]))
            if hi > lo:
                total += float(f.integrate(lo, hi))
        return total

    def __repr__(self) -> str:
        return f"MergedLayerFunction({len(self._funcs)} pieces)"

