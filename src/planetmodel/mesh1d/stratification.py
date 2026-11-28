"""The stratification of a spherically symmetric model: the Bullen
parameter and the buoyancy frequency.

A layer in hydrostatic equilibrium is neutrally stratified where its
density follows the adiabat, d rho / d r = -rho^2 g / kappa, and stably
stratified where it falls with radius faster than that.  The Bullen
parameter states which,

    eta(r) = -kappa rho'(r) / (rho^2 g),

one on the adiabat, greater where the layer is stable and less where it
is unstable; its excess over one is the name `bullen_excess` of the
vocabulary.  The square of the buoyancy (Brunt-Vaisala) frequency is the
same statement with the dimensions of a frequency squared,

    N^2(r) = -g (rho' / rho + rho g / kappa) = (rho g^2 / kappa) (eta - 1),

and vanishes with g at the centre whatever the stratification, which is
why the Bullen parameter and not N^2 is what describes a layer there.

Everything is read off the fields: rho and its radial derivative, the
bulk modulus as `materials.kappa_mu` has it, and the gravity of
`mesh1d.gravity`.  The derivative is exact where the density is a
polynomial.  Where g vanishes the Bullen parameter is the limit of the
ratio: where rho' vanishes too (the centre of a density that is smooth
there) it is -kappa rho'' / (rho^2 g'), and elsewhere it is infinite,
with the sign of -rho'.  A radius on an interior boundary is answered
from the layer above it, and the parameter jumps there with the fields.

The statement can be turned round.  Given the Bullen excess e = eta - 1
of a layer and its P velocity, the density is the solution of

    d ln rho / d r = -(1 + e) g / vp^2,
    d m / d r = 4 pi r^2 rho,        g = G (M_below + m) / r^2,

(with rho g / kappa in the place of g / vp^2 where the bulk modulus is
given), fixed by one number: the density at the bottom of the layer, the
density at its top, or its mass.  `stratified_density` integrates it and
returns the density as a piecewise polynomial that agrees with the
solution to a stated relative tolerance, with a breakpoint wherever the
excess or the velocity has one, so that a layer neutral on some pieces
and stratified on others is one layer function.

A layer whose Bullen excess is negative in places is unstable there.
Along an adiabat q = ln rho + int g / vp^2 dr is constant, so q is the
logarithm of a potential density, and the layer is stable exactly where
q does not increase with radius.  `stable_layer` finds the stable layer
nearest to a given one with the same P velocity: the q that does not
increase and is closest to the layer's own in the mean square weighted
by mass, which pools each unstable interval with as much of the fluid
beside it as it takes to make the whole neutral and leaves the rest as
it was.  Holding the mass of the layer and its moment of inertia as
well, the solution is the same pooling of q + s + mu r^2, with the two
numbers fixed by the two constraints.  Among stable layers of one mass
the neutral one has the greatest moment of inertia, so a layer that is
unstable throughout cannot keep its own.

Nothing here is a number with a unit.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import Polynomial, chebyshev
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_simpson, quad, solve_ivp
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import brentq, root_scalar

from .. import materials
from ..layerfunction import (
    LayerFunction,
    LayerFunctionLike,
    PolynomialLayer,
    as_interval,
    as_layer_function,
    polynomial_layer,
)
from .gravity import gravity, gravity_fields, radii_within

if TYPE_CHECKING:
    from ..model import Model

__all__ = ["bullen_parameter", "bullen_excess", "brunt_vaisala_squared",
           "stratified_density", "is_neutral", "stable_layer", "StableLayer",
           "Stabilisation",
           "BULLEN_EXCESS", "DENSITY_DEGREE", "STABILISER_NODES", "STABLE_EXCESS"]

#: The vocabulary name under which a layer holds its Bullen excess.
BULLEN_EXCESS = "bullen_excess"

#: The degree of each polynomial piece of a derived density.
DENSITY_DEGREE = 12

#: The most times a piece of a derived density is halved to reach its
#: tolerance.
_HALVINGS = 12

#: The nodes of the grid on which a layer is stabilised.
STABILISER_NODES = 4001

#: A layer whose Bullen excess is nowhere below minus this is stable.
STABLE_EXCESS = 1e-9


def _on_layers(model: Model, radii: ArrayLike
               ) -> tuple[np.ndarray, list[tuple[np.ndarray, int]]]:
    """The radii held to the model, flattened, and for each layer that
    holds some of them the mask of those it holds and its index."""
    r = radii_within(model, np.asarray(radii, dtype=float))
    flat = r.reshape(-1)
    b = model.skeleton.boundaries
    idx = np.clip(np.searchsorted(b, flat, side="right") - 1, 0, model.nlayers - 1)
    return r, [(idx == i, int(i)) for i in np.unique(idx)]


def _density_and_modulus(model: Model, i: int, r: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """rho, rho', rho'' and kappa of layer `i` at `r`."""
    layer = model.layers[i]
    rho = layer["rho"]
    if not hasattr(rho, "derivative"):
        raise ValueError(
            f"layer {layer.index} ({layer.name!r}) holds a rho that cannot "
            "state its radial derivative, so its stratification cannot be "
            "formed")
    kappa = materials.kappa_mu(layer)[0]
    return (np.asarray(rho(r), dtype=float),
            np.asarray(rho.derivative(nu=1)(r), dtype=float),
            np.asarray(rho.derivative(nu=2)(r), dtype=float),
            np.asarray(kappa(r), dtype=float))


def bullen_parameter(model: Model, radii: ArrayLike) -> np.ndarray:
    """The Bullen parameter eta(r) = -kappa rho' / (rho^2 g) at `radii`.

    `radii` broadcast to any shape and must lie in the model, to the
    geometry's tolerance.  One where the layer is neutrally stratified,
    greater where it is stable, less where it is unstable.  Where g
    vanishes the value is the limit of the ratio (the module docstring):
    finite at the centre of a density that is smooth there, infinite
    where rho' does not vanish with g.
    """
    r, layers = _on_layers(model, radii)
    flat = r.reshape(-1)
    g = gravity(model, flat)
    fields = None
    out = np.empty(flat.shape)
    for mask, i in layers:
        x = flat[mask]
        rho, d1, d2, kappa = _density_and_modulus(model, i, x)
        gx = g[mask]
        still = gx == 0.0
        eta = -kappa * d1 / (rho ** 2 * np.where(still, 1.0, gx))
        if still.any():
            if fields is None:
                fields = gravity_fields(model)
            dg = np.asarray(fields[i].derivative(nu=1)(x), dtype=float)
            flat_density = d1 == 0.0
            limit = -kappa * d2 / (rho ** 2 * np.where(dg != 0.0, dg, 1.0))
            unbounded = np.where(d1 < 0.0, np.inf, -np.inf)
            eta = np.where(still, np.where(flat_density, limit, unbounded), eta)
        out[mask] = eta
    return out.reshape(r.shape)


def bullen_excess(model: Model, radii: ArrayLike) -> np.ndarray:
    """The excess of the Bullen parameter over one at `radii`: zero where
    the layer is neutrally stratified, positive where it is stable and
    negative where it is unstable (`bullen_parameter`)."""
    return bullen_parameter(model, radii) - 1.0


def brunt_vaisala_squared(model: Model, radii: ArrayLike) -> np.ndarray:
    """The square of the buoyancy frequency,
    N^2(r) = -g (rho' / rho + rho g / kappa), at `radii`.

    `radii` broadcast to any shape and must lie in the model.  Positive
    where the layer is stably stratified and negative where it is
    unstable; zero where g vanishes, the centre among them.
    """
    r, layers = _on_layers(model, radii)
    flat = r.reshape(-1)
    g = gravity(model, flat)
    out = np.empty(flat.shape)
    for mask, i in layers:
        rho, d1, _, kappa = _density_and_modulus(model, i, flat[mask])
        out[mask] = -g[mask] * (d1 / rho + rho * g[mask] / kappa)
    return out.reshape(r.shape)


def is_neutral(layer: materials.LayerLike) -> bool:
    """Whether the Bullen excess a layer holds vanishes throughout it, so
    that the layer is neutrally stratified; a layer that does not hold
    one, under `BULLEN_EXCESS`, is refused with KeyError."""
    if BULLEN_EXCESS not in layer:
        raise KeyError(
            f"the layer holds no {BULLEN_EXCESS!r}, so whether it is neutrally "
            "stratified is not stated; `bullen_excess` computes it from the "
            "density")
    return materials.vanishes(layer[BULLEN_EXCESS])


def _pieces(a: float, b: float, fn: Callable[[np.ndarray], np.ndarray], *,
            degree: int, rtol: float, what: str, atol: float = 0.0, depth: int = 0
            ) -> list[tuple[float, np.ndarray]]:
    """`fn` on [a, b] as polynomial pieces of `degree`, each given by its
    left end and its coefficients in (r - left), lowest degree first.

    A piece interpolates `fn` at the Chebyshev-Lobatto points of its
    span, so neighbouring pieces meet, and is halved until it holds `fn`
    to `rtol` of its values plus `atol` at points between them.  A piece
    that starts at the centre has no slope there.
    """
    nodes = np.cos(np.pi * np.arange(degree + 1) / degree)[::-1]   # in [-1, 1]
    checks = np.cos(np.pi * (np.arange(2 * degree + 1) + 0.5) / (2 * degree + 1))
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    values = np.asarray(fn(mid + half * nodes), dtype=float)
    in_t = Polynomial(chebyshev.cheb2poly(chebyshev.chebfit(nodes, values, degree)))
    local = in_t(Polynomial([-1.0, 1.0 / half]))                    # in (r - a)
    coef = np.zeros(degree + 1)
    coef[:local.coef.size] = local.coef
    if a == 0.0:
        # g vanishes at the centre and nothing radial has a slope there
        coef[1] = 0.0
    x = mid + half * checks
    exact = np.asarray(fn(x), dtype=float)
    error = np.abs(Polynomial(coef)(x - a) - exact)
    if np.all(error <= rtol * np.abs(exact) + atol):
        return [(a, coef)]
    if depth == _HALVINGS:
        raise RuntimeError(
            f"{what} on [{a:g}, {b:g}] is not held to rtol = {rtol:g} by a "
            f"polynomial of degree {degree} (error {np.max(error):.1e}); ask for "
            "a looser tolerance")
    return (_pieces(a, mid, fn, degree=degree, rtol=rtol, what=what, atol=atol,
                    depth=depth + 1)
            + _pieces(mid, b, fn, degree=degree, rtol=rtol, what=what, atol=atol,
                      depth=depth + 1))


def _assembled(interval: tuple[float, float],
               pieces: list[tuple[float, np.ndarray]]) -> PolynomialLayer:
    """The pieces of `_pieces`, in order, as one layer function."""
    x = np.array([left for left, _ in pieces] + [interval[1]])
    c = np.array([coef[::-1] for _, coef in pieces]).T
    return PolynomialLayer(interval, PPoly(c, x))


def _knots(interval: tuple[float, float],
           functions: tuple[LayerFunction, ...]) -> np.ndarray:
    """The ends of `interval` and the breakpoints of the piecewise
    polynomials among `functions` that lie strictly inside it."""
    lo, hi = interval
    points = [np.array([lo, hi])]
    for fn in functions:
        pp = getattr(fn, "ppoly", None)
        if pp is not None:
            x = np.asarray(pp.x, dtype=float)
            points.append(x[(x > lo) & (x < hi)])
    return np.unique(np.concatenate(points))


def stratified_density(interval: tuple[float, float], excess: LayerFunctionLike, *,
                       G: float, vp: LayerFunctionLike | None = None,
                       kappa: LayerFunctionLike | None = None,
                       mass_below: float = 0.0, mass: float | None = None,
                       rho_bottom: float | None = None,
                       rho_top: float | None = None, rtol: float = 1e-12,
                       degree: int = DENSITY_DEGREE) -> PolynomialLayer:
    """The density of a layer from its stratification.

    `excess` is the Bullen excess e(r) on `interval` and exactly one of
    `vp` and `kappa` its P velocity or bulk modulus, each a layer
    function, a number or a callable of radius.  `mass_below` is the mass
    inside the layer's inner radius and `G` the gravitational constant.
    Exactly one of `rho_bottom`, `rho_top` and `mass` fixes the solution:
    the density at the inner radius, at the outer radius, or the mass of
    the layer; the last two are found by a search on the first.

    The density is returned as a piecewise polynomial of `degree` that
    agrees with the solution of the equation of the module docstring to
    `rtol`, relatively, and is continuous, with no slope at the centre:
    its breakpoints are those of `excess` and of the velocity or modulus,
    and where a piece needs it, its halves.  An excess of either sign is
    integrated; a layer is stable where it is not negative.
    """
    lo, hi = as_interval(interval)
    if (vp is None) == (kappa is None):
        raise ValueError("give exactly one of vp and kappa")
    anchors = {"rho_bottom": rho_bottom, "rho_top": rho_top, "mass": mass}
    given = [name for name, value in anchors.items() if value is not None]
    if len(given) != 1:
        raise ValueError("give exactly one of rho_bottom, rho_top and mass; "
                         f"got {given if given else 'none'}")
    anchor, target = given[0], float(anchors[given[0]])
    if not target > 0.0:
        raise ValueError(f"{anchor} must be positive, got {target}")
    if mass_below < 0.0 or (lo == 0.0 and mass_below != 0.0):
        raise ValueError("mass_below must not be negative, and is zero for a "
                         "layer that reaches the centre")
    e = as_layer_function((lo, hi), excess)
    stiffness = as_layer_function((lo, hi), vp if kappa is None else kappa)
    by_velocity = kappa is None
    knots = _knots((lo, hi), (e, stiffness))
    volume = 4.0 * np.pi * (hi ** 3 - lo ** 3) / 3.0
    tol = max(0.1 * rtol, 100.0 * np.finfo(float).eps)

    def solve(rho_b: float, *, dense: bool) -> tuple[float, float, list]:
        """The density at the top and the mass of the layer for the
        density `rho_b` at the bottom, and the solution on each span
        between knots.  The state is (ln(rho / rho_b), m / m_ref)."""
        m_ref = rho_b * volume
        ratio_below = mass_below / m_ref

        def rhs(r: float, y: np.ndarray) -> list[float]:
            rho = rho_b * np.exp(y[0])
            g = G * m_ref * (ratio_below + y[1]) / (r * r) if r > 0.0 else 0.0
            x = float(stiffness(r))
            slope = g / (x * x) if by_velocity else rho * g / x
            return [-(1.0 + float(e(r))) * slope,
                    4.0 * np.pi * r * r * rho / m_ref]

        y = np.array([0.0, 0.0])
        spans = []
        for a, b in zip(knots[:-1], knots[1:]):
            out = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=tol,
                            atol=1e-3 * tol, dense_output=dense)
            if not out.success:
                raise RuntimeError(
                    f"the density could not be integrated across [{a:g}, {b:g}]: "
                    f"{out.message}")
            y = out.y[:, -1]
            spans.append(out.sol)
        return rho_b * float(np.exp(y[0])), m_ref * float(y[1]), spans

    if anchor == "rho_bottom":
        rho_b = target
    else:
        def miss(log_rho_b: float) -> float:
            top, m, _ = solve(float(np.exp(log_rho_b)), dense=False)
            return np.log((top if anchor == "rho_top" else m) / target)

        start = np.log(target if anchor == "rho_top" else target / volume)
        found = root_scalar(miss, method="secant", x0=start, x1=start + 0.05,
                            xtol=4.0 * np.finfo(float).eps, rtol=tol, maxiter=60)
        if not found.converged:
            raise RuntimeError(f"no density at the bottom gives the {anchor} asked "
                               f"for: {found.flag}")
        rho_b = float(np.exp(found.root))
    _, _, spans = solve(rho_b, dense=True)

    found_pieces = []
    for a, b, sol in zip(knots[:-1], knots[1:], spans):
        found_pieces += _pieces(
            float(a), float(b),
            lambda x, sol=sol: rho_b * np.exp(sol(x)[0]), degree=degree,
            rtol=rtol, what="the density")
    return _assembled((lo, hi), found_pieces)


@dataclass(frozen=True)
class StableLayer:
    """The stable layer nearest to a given one, as `stable_layer` found it.

    `excess` is its Bullen excess, zero on the `neutral` intervals, and
    `density` the density that excess derives, of the mass of the layer
    given.  `tilt` is the multiplier mu of r^2 (r over the outer radius)
    that holds the moment of inertia, zero where it was not held.
    `density_change` is the largest relative change of the density and
    `inertia_change` the relative change of the moment of inertia of the
    layer about the centre, zero to rounding where it was held.
    """

    excess: PolynomialLayer
    density: PolynomialLayer
    neutral: tuple[tuple[float, float], ...]
    tilt: float
    density_change: float
    inertia_change: float


@dataclass(frozen=True)
class Stabilisation:
    """What stabilising a model did: for each layer it changed, by index,
    the `StableLayer` it was given.  A layer that was stable as it stood
    is in no entry."""

    layers: Mapping[int, StableLayer]

    @property
    def changed(self) -> tuple[int, ...]:
        """The indices of the layers that were changed, from the centre."""
        return tuple(sorted(self.layers))

    def describe(self) -> str:
        """A line for each layer changed: where it is now neutral, and how
        far its density and its moment of inertia moved."""
        if not self.layers:
            return "every layer was stable as it stood"
        lines = []
        for i in self.changed:
            found = self.layers[i]
            spans = ", ".join(f"[{a:g}, {b:g}]" for a, b in found.neutral)
            lines.append(
                f"  layer {i}: neutral on {spans or 'no interval'}; density "
                f"changed by up to {found.density_change:.2e}, moment of inertia "
                f"by {found.inertia_change:.2e}")
        return "\n".join(lines)


def _moment(fn: LayerFunction, power: int) -> float:
    """int fn(r) r^power dr over the function's interval, exact where it
    is a piecewise polynomial."""
    lo, hi = fn.interval
    if hasattr(fn, "ppoly"):
        weight = polynomial_layer((lo, hi), [0.0] * power + [1.0])
        return float((fn * weight).integrate(lo, hi))
    return float(quad(lambda x: float(fn(x)) * x ** power, lo, hi, epsabs=0.0,
                      epsrel=1e-13, limit=200)[0])


def _pooled(y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The non-increasing sequence nearest to `y` in the mean square
    weighted by `w` (pool adjacent violators), and for each entry whether
    it was pooled with a neighbour."""
    values, weights, counts = [], [], []
    for yi, wi in zip(y, w):
        values.append(yi)
        weights.append(wi)
        counts.append(1)
        while len(values) > 1 and values[-2] < values[-1]:
            total = weights[-2] + weights[-1]
            values[-2] = (values[-2] * weights[-2] + values[-1] * weights[-1]) / total
            weights[-2] = total
            counts[-2] += counts[-1]
            values.pop()
            weights.pop()
            counts.pop()
    counts = np.asarray(counts)
    return np.repeat(values, counts), np.repeat(counts > 1, counts)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """The runs of True in `mask`, each as its first and last index."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False]))))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def stable_layer(interval: tuple[float, float], rho: LayerFunctionLike,
                 vp: LayerFunctionLike, *, G: float, mass_below: float = 0.0,
                 inertia: bool = True, neutral_below: float = 0.0,
                 rtol: float = 1e-12, nodes: int = STABILISER_NODES,
                 degree: int = DENSITY_DEGREE) -> StableLayer | None:
    """The stable layer nearest to the one of density `rho` and P velocity
    `vp` on `interval`, or None where that layer is stable as it stands.

    The velocity and the mass of the layer are kept, and with `inertia`
    its moment of inertia about the centre; the density changes, least
    in the mean square of the logarithm of the potential density
    weighted by mass (the module docstring).  Each unstable interval
    becomes neutral together with as much of the fluid beside it as that
    takes, and with the moment of inertia held the rest of the layer
    changes a little too.  `mass_below` is the mass inside the layer's
    inner radius and `G` the gravitational constant.

    With `neutral_below` positive, a stretch of the result whose Bullen
    excess stays under it is made neutral as well, and a layer that is
    stable as it stands is treated too.

    The pooling is done on a grid of `nodes` radii, so the result is the
    nearest stable layer to the accuracy of that grid: at the default
    the part of a layer that is left alone is the layer's own to about
    one part in ten million.  It is stable, and of the layer's mass and
    moment of inertia, without that qualification: the density returned
    is derived from the excess returned (`stratified_density`, to
    `rtol`), which is nowhere negative, and it is that density whose
    mass and moment of inertia are held.

    A layer that cannot keep its moment of inertia, being unstable
    throughout or nearly, is refused: ask for `inertia=False` to hold
    the mass alone.  So is a layer whose density has a slope at the
    centre, its Bullen excess being unbounded there.
    """
    lo, hi = as_interval(interval)
    rho0 = as_layer_function((lo, hi), rho)
    speed = as_layer_function((lo, hi), vp)
    r = np.linspace(lo, hi, nodes)
    x2 = (r / hi) ** 2
    rho_r = np.asarray(rho0(r), dtype=float)
    slope = np.asarray(rho0.derivative(nu=1)(r), dtype=float)
    phi = np.asarray(speed(r), dtype=float) ** 2
    mass0 = 4.0 * np.pi * _moment(rho0, 2)
    inertia0 = 8.0 * np.pi / 3.0 * _moment(rho0, 4)
    centre = lo == 0.0
    cusp = centre and slope[0] != 0.0
    curvature = float(rho0.derivative(nu=2)(lo)) if centre else 0.0

    def gravity_of(density: np.ndarray) -> np.ndarray:
        inside = mass_below + cumulative_simpson(4.0 * np.pi * r * r * density, x=r,
                                                 initial=0.0)
        return np.where(r > 0.0, G * inside / np.where(r > 0.0, r, 1.0) ** 2, 0.0)

    def excess_of(density: np.ndarray, mu: float) -> np.ndarray:
        """-vp^2 q' / g on the grid, with q' = rho0' / rho0 + g / vp^2
        + 2 mu r / hi^2 and g the gravity of `density`; its limit at the
        centre."""
        g = gravity_of(density)
        dq = slope / rho_r + g / phi + 2.0 * mu * r / hi ** 2
        e = -phi * dq / np.where(g > 0.0, g, 1.0)
        if cusp:
            e[0] = np.inf if slope[0] < 0.0 else -np.inf
        elif centre:
            dg = 4.0 * np.pi * G * density[0] / 3.0
            e[0] = -phi[0] * (curvature / rho_r[0] + 2.0 * mu / hi ** 2) / dg - 1.0
        elif g[0] <= 0.0:
            e[0] = e[1]
        return e

    if neutral_below <= 0.0 and np.min(excess_of(rho_r, 0.0)) >= -STABLE_EXCESS:
        return None
    if cusp:
        raise ValueError(
            "the density has a slope at the centre, where its Bullen excess is "
            "unbounded; such a layer is not stabilised")

    cell = np.gradient(r)
    weights = np.maximum(r * r * rho_r * cell, np.finfo(float).tiny)

    def inertia_of(density: np.ndarray) -> float:
        return 8.0 * np.pi / 3.0 * float(
            cumulative_simpson(r ** 4 * density, x=r)[-1])

    def solved(target: float, density: np.ndarray
               ) -> tuple[np.ndarray, list[tuple[float, float]], float]:
        """The density, the pooled intervals and the tilt of the nearest
        stable layer of the mass `mass0` and, with `inertia`, of the
        moment of inertia `target` as the grid measures it; `density`
        starts the iteration on gravity."""
        mu = 0.0
        for _ in range(60):
            g = gravity_of(density)
            q = cumulative_simpson(slope / rho_r + g / phi, x=r, initial=0.0)

            def candidate(tilt: float, *, q: np.ndarray = q
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                tilted = q + tilt * x2
                fitted, pooled = _pooled(tilted, weights)
                trial = rho_r * np.exp(fitted - q)
                mass = 4.0 * np.pi * float(cumulative_simpson(r * r * trial, x=r)[-1])
                return trial * (mass0 / mass), pooled, tilted, fitted

            if inertia:
                def miss(tilt: float) -> float:
                    return inertia_of(candidate(tilt)[0]) / target - 1.0

                at_zero = miss(mu)
                step = 1e-6
                other = mu
                for _ in range(60):
                    other = mu + step if at_zero < 0.0 else mu - step
                    if miss(other) * at_zero <= 0.0:
                        break
                    if at_zero < 0.0 and candidate(other)[1].all():
                        raise ValueError(
                            "the layer cannot keep its moment of inertia: made "
                            "neutral throughout, which gives a stable layer of its "
                            "mass the greatest moment of inertia it can have, it "
                            f"still falls short by {-miss(other):.2e} of it; pass "
                            "inertia=False to hold the mass alone")
                    step *= 4.0
                else:
                    raise RuntimeError("no tilt holds the moment of inertia")
                if at_zero != 0.0:
                    mu = brentq(miss, min(mu, other), max(mu, other), xtol=1e-16,
                                rtol=1e-14)
            new, pooled, tilted, fitted = candidate(mu)
            change = float(np.max(np.abs(new / density - 1.0)))
            density = new
            if change < 1e-13:
                break
        # each pooled run of nodes, its ends moved off the nodes to where
        # the tilted q crosses the level the run was pooled to
        intervals = []
        for a, b in _runs(pooled):
            level = fitted[a]
            left, right = float(r[a]), float(r[b])
            if a > 0:
                above, below = tilted[a - 1] - level, tilted[a] - level
                if above > below:
                    left = float(r[a - 1] + (r[a] - r[a - 1]) * above / (above - below))
            if b < nodes - 1:
                above, below = tilted[b] - level, tilted[b + 1] - level
                if above > below:
                    right = float(r[b] + (r[b + 1] - r[b]) * above / (above - below))
            intervals.append((left, right))
        return density, intervals, mu

    def neutral_intervals(density: np.ndarray, pooled: list[tuple[float, float]],
                          mu: float) -> tuple[tuple[float, float], ...]:
        """The intervals of the solution that are neutral: the pooled
        ones, joined across a stretch between them too short to describe
        and, with a threshold, across one whose excess stays under it."""
        e = excess_of(density, mu)
        step = (hi - lo) / (nodes - 1)
        edges = [lo, *[x for interval in pooled for x in interval], hi]
        keep = []
        for k in range(0, len(edges), 2):               # the stretches between
            a, b = edges[k], edges[k + 1]
            if b <= a:
                continue
            inside = (r >= a) & (r <= b)
            short = b - a < 3.0 * step and (a > lo or b < hi)
            weak = neutral_below > 0.0 and (not inside.any()
                                            or np.max(e[inside]) < neutral_below)
            if not (short or weak):
                keep.append((a, b))
        out, start = [], lo
        for a, b in keep:
            if a > start:
                out.append((start, a))
            start = b
        if hi > start:
            out.append((start, hi))
        return tuple(out)

    def described(density: np.ndarray, intervals: tuple[tuple[float, float], ...],
                  mu: float) -> PolynomialLayer:
        """The Bullen excess of the solution as a layer function, zero on
        the neutral `intervals`: on the rest it is that of the tilt `mu`
        under the gravity of `density`."""
        g_of = CubicSpline(r, gravity_of(density))
        d1, d2 = rho0.derivative(nu=1), rho0.derivative(nu=2)

        def excess_at(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            g = g_of(x)
            dq = d1(x) / rho0(x) + g / speed(x) ** 2 + 2.0 * mu * x / hi ** 2
            out = -speed(x) ** 2 * dq / np.where(x > 0.0, g, 1.0)
            if centre:
                limit = (-speed(0.0) ** 2 * (d2(0.0) / rho0(0.0) + 2.0 * mu / hi ** 2)
                         / (4.0 * np.pi * G * density[0] / 3.0) - 1.0)
                out = np.where(x > 0.0, out, limit)
            return out

        pieces = []
        knots = _knots((lo, hi), (rho0, speed))
        edges = sorted({lo, hi, *[x for iv in intervals for x in iv]})
        for a, b in zip(edges[:-1], edges[1:]):
            if any(n0 <= a and b <= n1 for n0, n1 in intervals):
                pieces.append((a, np.zeros(degree + 1)))
                continue
            inner = [a, *[float(k) for k in knots if a < k < b], b]
            for c, d in zip(inner[:-1], inner[1:]):
                for left, coef in _pieces(c, d, excess_at, degree=degree, rtol=1e-9,
                                          atol=1e-11, what="the Bullen excess"):
                    pieces.append((left, coef))
        layer = _assembled((lo, hi), pieces)
        # a piece that dips below zero by rounding is lifted to it
        x = layer.ppoly.x
        c = layer.ppoly.c.copy()
        sample = 0.5 * (1.0 - np.cos(np.pi * np.arange(33) / 32.0))
        for k in range(x.size - 1):
            low = float(np.min(layer(x[k] + (x[k + 1] - x[k]) * sample)))
            if low < 0.0:
                c[-1, k] -= low
        return PolynomialLayer((lo, hi), PPoly(c, x))

    density, pooled, mu = solved(inertia_of(rho_r), rho_r)
    neutral = neutral_intervals(density, pooled, mu)

    def derived_for(tilt: float) -> tuple[PolynomialLayer, PolynomialLayer, float]:
        """The excess of the tilt, the density it derives and the relative
        change of the moment of inertia."""
        excess = described(density, neutral, tilt)
        derived = stratified_density((lo, hi), excess, vp=speed, G=G,
                                     mass_below=mass_below, mass=mass0, rtol=rtol)
        change = 8.0 * np.pi / 3.0 * _moment(derived, 4) / inertia0 - 1.0
        return excess, derived, change

    excess, derived, change = derived_for(mu)
    if inertia and abs(change) > 1e-12:
        # the grid holds the moment of inertia to its own accuracy; the
        # tilt is corrected against the density the excess derives, the
        # neutral intervals staying where the grid put them
        if neutral == ((lo, hi),):
            raise ValueError(
                "the layer is neutral throughout once its weak stratification is "
                "made neutral, and a neutral layer of a given mass has one moment "
                f"of inertia, which differs by {change:.2e} of the layer's own; "
                "pass inertia=False to hold the mass alone")
        found = root_scalar(lambda t: derived_for(t)[2], method="secant", x0=mu,
                            x1=mu + max(1e-3 * abs(mu), 1e-9), xtol=1e-18,
                            rtol=1e-12, maxiter=12)
        mu = float(found.root)
        excess, derived, change = derived_for(mu)
    return StableLayer(
        excess=excess, density=derived, neutral=neutral, tilt=float(mu),
        density_change=float(np.max(np.abs(np.asarray(derived(r)) / rho_r - 1.0))),
        inertia_change=float(change))
