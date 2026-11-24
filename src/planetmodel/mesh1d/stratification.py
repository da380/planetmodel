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

Nothing here is a number with a unit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import Polynomial, chebyshev
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.interpolate import PPoly
from scipy.optimize import root_scalar

from .. import materials
from ..layerfunction import (
    LayerFunction,
    LayerFunctionLike,
    PolynomialLayer,
    as_interval,
    as_layer_function,
)
from .gravity import _within, gravity, gravity_fields

if TYPE_CHECKING:
    from ..model import Model

__all__ = ["bullen_parameter", "bullen_excess", "brunt_vaisala_squared",
           "stratified_density", "is_neutral", "BULLEN_EXCESS", "DENSITY_DEGREE"]

#: The vocabulary name under which a layer holds its Bullen excess.
BULLEN_EXCESS = "bullen_excess"

#: The degree of each polynomial piece of a derived density.
DENSITY_DEGREE = 12

#: The most times a piece of a derived density is halved to reach its
#: tolerance.
_HALVINGS = 12


def _on_layers(model: Model, radii: ArrayLike
               ) -> tuple[np.ndarray, list[tuple[np.ndarray, int]]]:
    """The radii held to the model, flattened, and for each layer that
    holds some of them the mask of those it holds and its index."""
    r = _within(model, np.asarray(radii, dtype=float))
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

    nodes = np.cos(np.pi * np.arange(degree + 1) / degree)[::-1]   # Lobatto, in [-1, 1]
    checks = np.cos(np.pi * (np.arange(2 * degree + 1) + 0.5) / (2 * degree + 1))

    def pieces(a: float, b: float, sol: object, depth: int
               ) -> list[tuple[float, np.ndarray]]:
        """The polynomial pieces on [a, b], each as its left end and its
        coefficients in (r - a), lowest degree first."""
        half, mid = 0.5 * (b - a), 0.5 * (a + b)
        values = rho_b * np.exp(sol(mid + half * nodes)[0])
        in_t = Polynomial(chebyshev.cheb2poly(chebyshev.chebfit(nodes, values, degree)))
        local = in_t(Polynomial([-1.0, 1.0 / half]))                # in (r - a)
        coef = np.zeros(degree + 1)
        coef[:local.coef.size] = local.coef
        if a == 0.0:
            # g vanishes at the centre and the density has no slope there
            coef[1] = 0.0
        x = mid + half * checks
        exact = rho_b * np.exp(sol(x)[0])
        error = np.max(np.abs(Polynomial(coef)(x - a) - exact) / exact)
        if error <= rtol:
            return [(a, coef)]
        if depth == _HALVINGS:
            raise RuntimeError(
                f"the density on [{a:g}, {b:g}] is not held to rtol = {rtol:g} by "
                f"a polynomial of degree {degree} (error {error:.1e}); ask for a "
                "looser tolerance")
        return pieces(a, mid, sol, depth + 1) + pieces(mid, b, sol, depth + 1)

    found_pieces = []
    for a, b, sol in zip(knots[:-1], knots[1:], spans):
        found_pieces += pieces(float(a), float(b), sol, 0)
    x = np.array([left for left, _ in found_pieces] + [hi])
    c = np.array([coef[::-1] for _, coef in found_pieces]).T
    return PolynomialLayer((lo, hi), PPoly(c, x))
