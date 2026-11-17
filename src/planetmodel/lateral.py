"""Values between the nodes of an angular grid: the lateral rules.

A field that varies with direction is stored at the nodes of an
`AngularGrid` and read between them by one of three rules, the same
here and in every reader of the library's files::

    linear     bilinear in (theta, phi) on the padded grid
    cubic      the tensor-product cubic spline on the padded grid,
               periodic in longitude and clamped at the poles
    harmonic   analysis on the grid and synthesis at the point, exact
               for a field band-limited to the grid's band (to the
               accuracy of `harmonics.synthesise`); only on a grid of a
               shipped kind with a band

**The padded grid.**  The local rules are rules on a rectangle of
(theta, phi), so the sphere is first laid out as one.  The longitudes
phi_0 < ... < phi_{n-1} gain the wrap column phi_0 + 2 pi, a copy of the
first, and the colatitudes gain a row at each pole the grid lacks as a
node, holding the field's value at that pole across every column; a
pole that is a node keeps its own row.  A point is read at its
colatitude, which must lie in [0, pi], and at its longitude reduced
into [phi_0, phi_0 + 2 pi).  The bilinear rule is the tensor-product
spline of degree one on that grid.  The bicubic rule is the
tensor-product cubic spline on it with these end conditions:

- in longitude, periodic, so that the interpolant and its first two
  derivatives are continuous across the seam at phi_0;
- in colatitude, clamped at each pole to a slope s(phi) = a cos phi +
  b sin phi, with (a, b) fitted by least squares, over the grid's
  longitudes, to the slopes that the not-a-knot spline through each
  meridian's values has at that pole.  A function with a gradient at a
  pole has exactly such slopes leaving it, the gradient's two
  components; with them the interpolant has a gradient there too, to
  the accuracy of the spline in longitude between the grid's columns.

Both local rules interpolate the nodes and the polar values, and both
are linear in the values, the bicubic one through the fitted slopes as
well.

**The polar values** have one source each.  A pole that is a node of
the grid is its own row, which must hold one value, constant along the
longitudes to `rtol` of the values' largest: a row that varies there
describes no field, and making it single-valued is for whoever read
the data.  A pole the grid lacks takes its value from `poles` (north
then south), as a model file stores it, evaluated exactly by the
writer; `poles` always has both entries, the one for a pole that is a
node either agreeing with its row or NaN, meaning not given.  Without
`poles`, a grid of a shipped kind takes the values from its own
expansion, those of the field band-limited to the grid, and any other
grid lacking a pole is refused, so that no polar value is ever guessed.
`polar_values` is that rule.  A scalar has one value at a pole; a
field with components is interpolated component by component in the
Cartesian frame, where it too is single-valued at a pole, and the
rotation into that frame belongs to the field, not to this module.
The harmonic rule's expansion defines its own values at the poles, so
it checks `poles` like any rule but does not use them.

**Accuracy.**  A local rule wants an oversampled grid.  For a field of
band 16 with coefficients falling as 1/(l + 1), on Gauss-Legendre
grids, the largest error relative to the field's largest value is
about 0.09 for the bicubic rule at the field's own band, 2e-4 at four
times it and 1e-5 at eight, and 0.3, 2e-2 and 5e-3 for the bilinear
one: fourth order against second (the cases of
`tests/test_lateral.py`).  Analysis on an oversampled grid is still
exact to the field's band, so a grid of about four times the band
serves a local reader and a spectral one alike.  The bicubic rule is
continuously differentiable everywhere, so a relief read by it gives a
mapping whose deformation gradient is single-valued on the polar axis
and on the seam; its second derivatives may jump at a pole.

`LateralInterpolant` builds a rule once over any array whose two
trailing axes are a grid's, and gives its values and its two angular
derivatives at any points, and its mean over the sphere: exactly, the
mean of the interpolant itself, so that a mean is consistent with the
rule that reads the values (for the local rules the coefficients
against the integrals of the B-splines, sin(theta) dtheta and dphi; for
the harmonic rule the degree-zero term); `interpolate` builds one and
evaluates it.  `periodic_cubic_spline` is the spline in longitude, on
any nodes of any period: the flat rules of `cartesian.lateral` take it
along both of a box's directions.
Everything is numbers.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import BSpline, NdBSpline, make_interp_spline
from scipy.sparse import csc_array
from scipy.sparse.linalg import SuperLU, splu

from .frames import wrapped
from .harmonics import analyse_grid, synthesise, synthesise_gradient

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["METHODS", "LateralInterpolant", "interpolate", "polar_values",
           "check_rule", "periodic_cubic_spline"]

#: The lateral rules, in the order of the module docstring.
METHODS = ("linear", "cubic", "harmonic")


def check_rule(grid: AngularGrid, method: str) -> None:
    """Refuse, by name, a rule the grid cannot carry: an unknown method;
    the harmonic rule on a grid that is not of a shipped kind with a
    band; the cubic rule on fewer than four rows once the poles the grid
    lacks are added, or fewer than three longitudes."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "harmonic" and (grid.kind == "custom" or grid.lmax is None):
        raise ValueError("the harmonic rule needs a grid of a shipped kind, "
                         f"whose transforms analyse it; this grid is {grid.kind}")
    if method == "cubic":
        rows = grid.ntheta + sum(not node for node in grid.pole_nodes)
        if rows < 4 or grid.nphi < 3:
            raise ValueError(
                f"the cubic rule needs at least 4 rows on the padded grid and "
                f"3 longitudes; this grid gives {rows} rows and has "
                f"{grid.nphi} longitudes")

#: The poles in the order of every (..., 2) array of polar values.
_POLES = ("north", "south")

#: Gauss-Legendre points per knot interval for the integrals of the
#: B-splines: exact against dphi, and against sin(theta) dtheta to
#: rounding on intervals of a grid's spacing.
_POINTS = 12


def _basis_integrals(t: np.ndarray, k: int, lo: float, hi: float,
                     weight: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """integral from lo to hi of B_a(x) weight(x) dx for every B-spline of
    degree k on the knots t, by Gauss-Legendre on each knot interval."""
    xq, wq = np.polynomial.legendre.leggauss(_POINTS)
    u = np.unique(np.clip(t, lo, hi))
    a, b = u[:-1, None], u[1:, None]
    x = (0.5 * (b - a) * xq + 0.5 * (a + b)).ravel()
    w = (0.5 * (b - a) * wq).ravel()
    return BSpline.design_matrix(x, t, k).T @ (w * weight(x))


def _grid_values(values: ArrayLike, grid: AngularGrid) -> np.ndarray:
    """Finite real values with the grid's two axes last, or the error."""
    from .sampling import AngularGrid
    if not isinstance(grid, AngularGrid):
        raise TypeError(f"grid must be an AngularGrid, got {type(grid).__name__}")
    if np.iscomplexobj(values):
        raise TypeError("the lateral rules take real values")
    v = np.asarray(values, dtype=float)
    if v.ndim < 2 or v.shape[-2:] != (grid.ntheta, grid.nphi):
        raise ValueError(f"values must have trailing shape "
                         f"{(grid.ntheta, grid.nphi)}, got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("the lateral rules take finite values")
    return v


def _from_expansion(v: np.ndarray, grid: AngularGrid) -> np.ndarray:
    """The polar values of the field band-limited to the grid: at a pole
    only the zonal harmonics are not zero, Y_l0 = sqrt((2l + 1) / 4 pi)
    at the north pole and (-1)^l times that at the south."""
    c = analyse_grid(v, grid)
    l = np.arange(c.shape[1])
    zonal = np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi))
    north = np.einsum("l...,l->...", c[0, :, 0], zonal)
    south = np.einsum("l...,l->...", c[0, :, 0], zonal * (-1.0) ** l)
    return np.stack([north, south], axis=-1)


def polar_values(values: ArrayLike, grid: AngularGrid, *,
                 poles: ArrayLike | None = None, rtol: float = 1e-9) -> np.ndarray:
    """The values at the north and south poles, by the rule of the module
    docstring: shape (..., 2) for values of shape (..., ntheta, nphi).

    A pole that is a node gives its row, checked to hold one value to
    `rtol` of the values' largest.  A pole the grid lacks gives its entry
    of `poles`, which must be finite; an entry of `poles` for a pole that
    is a node must agree with its row or be NaN.  With `poles` None, the
    poles the grid lacks come from its expansion, which needs a grid of
    a shipped kind with a band; any other grid lacking a pole is
    refused.
    """
    v = _grid_values(values, grid)
    lead = v.shape[:-2]
    scale = float(np.max(np.abs(v))) if v.size else 0.0
    tol = float(rtol) * scale
    rows = {0: 0, 1: -1}
    out = np.empty(lead + (2,))
    nodes = grid.pole_nodes
    for k, is_node in enumerate(nodes):
        if is_node:
            row = v[..., rows[k], :]
            spread = float(np.max(np.abs(row - row[..., :1]))) if row.size else 0.0
            if spread > tol:
                raise ValueError(
                    f"the {_POLES[k]} pole is a node of the grid, but its row "
                    f"varies along the longitudes by {spread:.3g}, more than "
                    f"rtol = {rtol:g} of the values' largest: a field has one "
                    "value at a pole (components in the Cartesian frame), and "
                    "the row must be made single-valued before it is read")
            out[..., k] = row[..., 0]
    if all(nodes):
        given = None if poles is None else np.asarray(poles, dtype=float)
    elif poles is None:
        if grid.kind == "custom" or grid.lmax is None:
            raise ValueError(
                "the grid lacks a pole among its nodes and is not of a shipped "
                "kind, whose expansion would find the field there, so the polar "
                "values must be given: pass "
                f"poles=, the values at the north and south poles, shape {lead + (2,)}")
        found = _from_expansion(v, grid)
        for k, is_node in enumerate(nodes):
            if not is_node:
                out[..., k] = found[..., k]
        return out
    else:
        given = np.asarray(poles, dtype=float)
    if given is not None:
        if given.shape != lead + (2,):
            raise ValueError(f"poles must have shape {lead + (2,)}, the values' "
                             f"leading shape and (north, south); got {given.shape}")
        for k, is_node in enumerate(nodes):
            entry = given[..., k]
            if is_node:
                stated = np.isfinite(entry)
                if np.any(np.abs(entry[stated] - out[..., k][stated]) > tol):
                    raise ValueError(
                        f"the {_POLES[k]} pole is a node of the grid, and the "
                        f"value given for it in poles disagrees with its row; "
                        "give the row's value or NaN")
            else:
                if not np.all(np.isfinite(entry)):
                    raise ValueError(
                        f"the grid lacks the {_POLES[k]} pole, so its entry in "
                        "poles is the field's value there and must be finite")
                out[..., k] = entry
    return out


def _padded(v: np.ndarray, grid: AngularGrid, poles: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(colatitudes, longitudes, values) of the padded grid of the module
    docstring, the values of shape (..., rows, columns), the wrap column
    last."""
    theta, phi = grid.colatitudes, grid.longitudes
    phi_p = np.append(phi, phi[0] + 2.0 * np.pi)
    v = np.concatenate([v, v[..., :1]], axis=-1)
    north, south = grid.pole_nodes
    theta_p, rows = [theta], [v]
    if not north:
        theta_p.insert(0, [0.0])
        rows.insert(0, np.broadcast_to(poles[..., 0, None, None],
                                       v.shape[:-2] + (1, phi_p.size)))
    if not south:
        theta_p.append([np.pi])
        rows.append(np.broadcast_to(poles[..., 1, None, None],
                                    v.shape[:-2] + (1, phi_p.size)))
    return np.concatenate(theta_p), phi_p, np.concatenate(rows, axis=-2)


@lru_cache(maxsize=16)
def _periodic_cubic(nodes: bytes, period: float) -> tuple[np.ndarray, SuperLU]:
    """The knots of the periodic cubic spline through nodes (their float64
    bytes; the nodes extended by whole periods) and the factorised
    collocation matrix of its n free coefficients, coefficient j + n
    being coefficient j: built once for each set of nodes, and solved for
    every column of values at once."""
    x = np.frombuffer(nodes, dtype=float)
    n = x.size
    j = np.arange(-3, n + 4)
    t = x[j % n] + period * (j // n)
    D = BSpline.design_matrix(x, t, 3).tocoo()
    return t, splu(csc_array((D.data, (D.row, D.col % n)), shape=(n, n)))


def periodic_cubic_spline(values: ArrayLike, nodes: ArrayLike, period: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """(knots, coefficients) of the periodic cubic spline through `values`
    of shape (n, ...) at `nodes`, along axis 0.

    The n nodes increase strictly within one `period`, n at least three;
    the knots are the nodes extended by whole periods, and the n + 3
    coefficients repeat with period n, so that the spline and its first
    two derivatives are continuous across the seam.  It is read by
    `scipy.interpolate.BSpline` (or `NdBSpline`) of degree three at a
    point reduced into [nodes[0], nodes[0] + period).  The rule in
    longitude of the bicubic lateral rule, and in both directions of the
    flat bicubic rule (`cartesian.lateral`).
    """
    x = np.ascontiguousarray(nodes, dtype=float)
    C = np.asarray(values, dtype=float)
    t, lu = _periodic_cubic(x.tobytes(), float(period))
    n = C.shape[0]
    u = lu.solve(C.reshape(n, -1)).reshape(C.shape)
    return t, u[np.arange(n + 3) % n]


def _periodic_spline(C: np.ndarray, grid: AngularGrid
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(knots, coefficients) of the periodic cubic spline through the
    values C of shape (nphi, ...) at the grid's longitudes, along axis 0."""
    return periodic_cubic_spline(C, grid.longitudes, 2.0 * np.pi)


def _pole_slopes(theta: np.ndarray, phi: np.ndarray, V: np.ndarray
                 ) -> list[np.ndarray]:
    """The clamped slopes at the north and south poles, each of the shape
    of one row of V (columns first): a cos phi + b sin phi fitted to the
    slopes of the not-a-knot spline through every column."""
    d = make_interp_spline(theta, V, k=3, axis=0).derivative()
    A = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    out = []
    for pole in (0.0, np.pi):
        s = d(pole)
        ab = np.linalg.lstsq(A, s.reshape(phi.size, -1), rcond=None)[0]
        out.append((A @ ab).reshape(s.shape))
    return out


class LateralInterpolant:
    """One lateral rule, built once over an array of values on a grid.

    `values` has shape (..., ntheta, nphi), the grid's two axes last, and
    must be finite and real; the leading shape is `shape`, and calling
    the interpolant at colatitudes and longitudes that broadcast to a
    shape P gives an array of shape `shape + P`.  `method` is one of
    `METHODS`.  `poles` and `rtol` are those of `polar_values`, whose
    rule fixes the values at the poles: `poles` has shape `shape + (2,)`,
    north then south, and the harmonic rule checks it without using it.
    """

    def __init__(self, values: ArrayLike, grid: AngularGrid, *, method: str,
                 poles: ArrayLike | None = None, rtol: float = 1e-9) -> None:
        check_rule(grid, method)
        if poles is not None and np.iscomplexobj(poles):
            raise TypeError("the lateral rules take real values")
        v = _grid_values(values, grid)
        self._grid, self._method, self._shape = grid, method, v.shape[:-2]
        if method == "harmonic":
            if poles is not None:
                polar_values(v, grid, poles=poles, rtol=rtol)
            self._coeffs = analyse_grid(v, grid)
            return
        P = polar_values(v, grid, poles=poles, rtol=rtol)
        theta_p, phi_p, V = _padded(v, grid, P)
        V = np.moveaxis(V, (-2, -1), (0, 1))            # (rows, columns, ...)
        if method == "linear":
            along_theta = make_interp_spline(theta_p, V, k=1, axis=0)
            along_phi = make_interp_spline(
                phi_p, np.swapaxes(along_theta.c, 0, 1), k=1, axis=0)
            t_phi, c_phi, k = along_phi.t, along_phi.c, 1
        else:
            # the meridians through the grid's columns, clamped at the
            # poles; then the periodic spline through the columns
            V = V[:, :-1]
            north, south = _pole_slopes(theta_p, grid.longitudes, V)
            along_theta = make_interp_spline(theta_p, V, k=3, axis=0,
                                             bc_type=([(1, north)], [(1, south)]))
            t_phi, c_phi = _periodic_spline(np.swapaxes(along_theta.c, 0, 1), grid)
            k = 3
        self._spline = NdBSpline((along_theta.t, t_phi), np.swapaxes(c_phi, 0, 1), k)
        self._degree = k

    @property
    def grid(self) -> AngularGrid:
        """The angular grid the values are on."""
        return self._grid

    @property
    def method(self) -> str:
        """The lateral rule, one of `METHODS`."""
        return self._method

    @property
    def shape(self) -> tuple[int, ...]:
        """The leading shape of the values, carried through to every result."""
        return self._shape

    def _points(self, theta: ArrayLike, phi: ArrayLike
                ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        """The points flattened and checked, and their broadcast shape."""
        theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                         np.asarray(phi, dtype=float))
        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(phi))):
            raise ValueError("the points must be finite")
        if np.any(theta < 0.0) or np.any(theta > np.pi):
            raise ValueError("colatitudes must lie in [0, pi]")
        return theta.ravel(), phi.ravel(), theta.shape

    def _local(self, t: np.ndarray, p: np.ndarray,
               nu: tuple[int, int]) -> np.ndarray:
        """The spline, or one of its derivatives, at flat points: shape
        `shape + (P,)`."""
        p = wrapped(p, start=float(self._grid.longitudes[0]))
        return np.moveaxis(self._spline(np.stack([t, p], axis=-1), nu=nu), 0, -1)

    def __call__(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The values at colatitudes `theta` in [0, pi] and longitudes `phi`
        (any range), broadcast together: shape `shape` followed by theirs."""
        t, p, points = self._points(theta, phi)
        if self._method == "harmonic":
            out = synthesise(self._coeffs, t, p)
        else:
            out = self._local(t, p, (0, 0))
        return out.reshape(self._shape + points)

    def gradient(self, theta: ArrayLike, phi: ArrayLike
                 ) -> tuple[np.ndarray, np.ndarray]:
        """(d/dtheta, d/dphi) of the interpolant at the points, each of the
        shape `__call__` gives: the derivatives of the spline, piecewise
        constant for the bilinear rule, or of the expansion."""
        t, p, points = self._points(theta, phi)
        if self._method == "harmonic":
            dt, dp = synthesise_gradient(self._coeffs, t, p)
        else:
            dt, dp = self._local(t, p, (1, 0)), self._local(t, p, (0, 1))
        return dt.reshape(self._shape + points), dp.reshape(self._shape + points)

    def mean(self) -> np.ndarray:
        """The mean over the sphere of the interpolant itself, of shape
        `shape`: exact for the local rules, whose B-spline coefficients are
        contracted with the integrals of the B-splines against
        sin(theta) dtheta and dphi over one turn, and the degree-zero
        term of the expansion for the harmonic rule.  No weights of the
        grid are used, so any grid has a mean."""
        if self._method == "harmonic":
            return np.asarray(self._coeffs[0, 0, 0] / np.sqrt(4.0 * np.pi))
        tt, tp = self._spline.t
        k = self._degree
        phi0 = float(self._grid.longitudes[0])
        it = _basis_integrals(tt, k, 0.0, np.pi, np.sin)
        ip = _basis_integrals(tp, k, phi0, phi0 + 2.0 * np.pi, np.ones_like)
        return np.einsum("ab...,a,b->...", self._spline.c, it, ip) / (4.0 * np.pi)

    def __repr__(self) -> str:
        return f"LateralInterpolant({self._method}, {self._shape} on {self._grid!r})"


def interpolate(values: ArrayLike, grid: AngularGrid, theta: ArrayLike,
                phi: ArrayLike, *, method: str, poles: ArrayLike | None = None,
                rtol: float = 1e-9) -> np.ndarray:
    """Values on a grid read at points by a lateral rule.

    `LateralInterpolant(values, grid, method=method, poles=poles,
    rtol=rtol)` evaluated at (theta, phi): the leading shape of `values`
    followed by the broadcast shape of the points.  Build the
    interpolant itself to read the same values more than once.
    """
    return LateralInterpolant(values, grid, method=method, poles=poles,
                              rtol=rtol)(theta, phi)
