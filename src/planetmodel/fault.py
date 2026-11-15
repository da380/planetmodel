"""Faults: surfaces inside a body, and the slip on them in time.

A fault is a surface and a vector field on it, both evaluable anywhere:
a tuple of patches, each a map x(u, v) of the parameter square into
space carrying its own slip, with one time axis for all of them.  What
a consumer needs is read from it at points of its own choosing: the
position, the unit normal, the area element and the slip at any (u, v)
and time, a quadrature of any density, and where on it a point near it
lies (`locate`), with the slip there (`Fault.slip_at`).  The fault holds no
rigidity and forms no moment: at each point a consumer contracts its
own elastic tensor with the slip du and the normal n (isotropic:
lambda (du.n) I + mu (du n + n du)) and multiplies by the weight.

**Frames.**  A fault's positions are in one frame.  `GeographicFrame`
places them on a body whose reference surface is a sphere of stated
radius: a position is (latitude, longitude, radius), the angles in
degrees by `frames.geographic` (longitude east, any range on input,
[0, 360) on output), the radius in the length scale; Cartesian points
are those of `frames.cartesian_points`, z along the polar axis and x
through longitude zero; the ambient basis at a point is (east, north,
up) = (e_phi, -e_theta, e_r), a point on the polar axis taking the
frame `frames.spherical_coordinates` gives it.  `CartesianFrame` is
local: a position is (x, y, z) with x east, y north and z up, and the
ambient basis is (x, y, z) everywhere; its points are those of the
flat-layered bodies of `planetmodel.cartesian`, its depths read
through their system (`cartesian.FLAT`), so that a fault in it is
built into a box's mesh as it stands (`mesh3d.EmbeddedFault`), as one
in a geographic frame is into a sphere's.  A frame's reference surface,
the sphere or the plane z = `surface`, is what a depth is measured
from (`below`); a depth is below that surface, not below any
topography on it.

**The parameter square.**  Every patch maps the square `SQUARE` x
`SQUARE`, u and v each in [-1/2, 1/2], u along strike and v down dip.
With x_u and x_v the tangents, the area element is the length of
x_u x x_v, the unit normal n is x_v x x_u over that length, pointing
from the footwall into the hanging wall, and the fault basis at (u, v)
is

    strike  s = x_u / norm(x_u),   up dip  a = n x s,   normal  n,

right-handed; down dip, -a, is the direction of increasing v within
the tangent plane, orthogonal to the strike.  Seen from the hanging
wall (looking along -n), the rake turns anticlockwise from s to a.

**A plane** is a flat rectangle in space, oriented at its centroid by
strike and dip as a finite-fault plane is (Aki & Richards 2002,
Fig. 4.13): not a surface following the sphere, so its points away
from the centroid lie at other radii.  With (east, north, up) the
frame's basis at the centroid c, the strike phi measured clockwise
from north (in a Cartesian frame from +y towards +x) and the dip delta
from the horizontal, down to the right of the strike, 0 to 90 degrees,

    s = sin phi east + cos phi north      r = cos phi east - sin phi north
    d = cos delta r - sin delta up        n = sin delta r + cos delta up
    x(u, v) = c + u L s + v W d,

L the length along strike and W the width down dip, so (0, 0) is the
centroid, the area element is L W everywhere and the fault basis is
(s, -d, n) at every point.  A dip of zero is a horizontal plane (a
sill), its strike orienting u.

**A structured patch** holds control
positions at the nodes of a uniform n_u by n_v grid over the square,
corners included, in the frame's coordinates; they are turned into
Cartesian points and interpolated coordinate by coordinate, bilinearly
(`linear`, the tensor-product spline of degree one) or bicubically
(`cubic`, the tensor-product cubic spline with not-a-knot ends, at
least four nodes each way).  Being Cartesian, the rule is indifferent
to the date line and the poles, and a plane's own points give the
plane back, bilinearly exactly and bicubically to rounding.  The
rules are scipy's interpolating B-splines, as the lateral rules are;
`lateral`'s own are rules on the sphere, periodic in longitude and
closed at the poles, and do not apply to a plain square.

**The slip** is the displacement of the hanging wall relative to the
footwall, held and interpolated in fault components: (strike-slip,
dip-slip, opening) in the fault basis (s, a, n), dip-slip positive up
dip (a reverse fault) and opening along the normal, so that a rake
lambda and a magnitude m are (m cos lambda, m sin lambda) (Aki &
Richards 2002, eq. 4.88) and a dyke or a sill is a patch like any
other.  A `Slip` holds the three components on a grid of its own over
the square with a rule of `SLIP_RULES`: `cell`, constant on each of
n_u by n_v equal cells (a point on a boundary between cells reads the
cell above it, the last cell closed at 1/2), or `linear` or `cubic`
between the nodes of a uniform grid as above, each component read by
the rule.  The vector at a point is the components there on the fault
basis there, then given in any of `BASES`: `fault` (the components
themselves), `ambient` (the frame's basis at the point) or
`cartesian`.  So a slip without opening lies in the surface at every
point of any patch; a uniform (strike-slip, dip-slip) on a plane is
one Cartesian vector everywhere; on a curved patch the rake is
measured from its u-lines point by point, and it is the components
that are uniform, near a pole or across the date line as anywhere.
`Slip.from_vectors` turns vectors given at a grid's points of a patch
in the ambient or Cartesian basis into its components there.  Slip
need not match where patches meet.

**Time** follows the surface file (`surfacefile`): the slip's values
carry a leading axis of slices, and the fault states one support of
`surfacefile.TIME_SUPPORTS` for them all, with its axis (`time` for
`node` and `interval`, `instant` for `instant`) and its rule between
nodes; `surfacefile.time_rows` gives the slices a date reads and
their weights, a date within `time_rtol` of a stored time (relative to
the scale of the axis, its span or the magnitude of its first or last
time, whichever is largest; `surfacefile.TIME_RTOL` by default) being
that time.  Outside the span a `node` slip read `linear` or `previous`
and an `interval` slip refuse a date, or hold their nearest slice where
the caller asks; a `node` slip read `none` and an `instant` slip refuse
any date that is not one of their stored times, held or not.  A
`static` slip is one slice and takes no date.  The fault file
(`faultfile`) holds the `static` and `node` supports, the slip at a
node the cumulative slip.

**Quadrature.**  Each patch is cut into n_u by n_v equal cells of the
square with `points` Gauss-Legendre points each way in each cell (one
point is the midpoint rule); a point's weight is the area it stands
for, the area element there times its weight in the square.  A patch's
own quadrature (`PatchQuadrature`) carries the slip of one slice; a
fault's carries each point's patch beside it, with the slip at a date
(`Fault.quadrature`, a `Quadrature`) or with the slip of every slice,
the geometry formed once (`Fault.slice_quadrature`, a
`SliceQuadrature`, whose slices the weights of `Fault.rows` combine
into the slip at a date, `SliceQuadrature.slip_of`, as
`Fault.quadrature` gives it).  A plane's area comes out exactly at any
density, and for a smooth integrand on a smooth patch the error falls
as the cell size to the power twice `points`.
A cell straddling a jump of a `cell` slip, or a knot where a bilinear
patch bends, converges more slowly: the densities that suit a
consumer's own discretisation are its business.

**Units.**  Nothing here knows any: lengths (positions, the plane's
length and width, the slip) are numbers in the length of the fault's
`Scales`, areas in its square, times in its time; angles (latitude,
longitude, strike, dip, rake) are degrees.

**Locating points.**  `locate` gives, for points near a patch, the
parameters of each one's nearest point x over the closed square, the
length of p - x signed as (p - x).n (positive on the hanging-wall
side), and whether x is inside: not held on the square's boundary with
p beyond it.  A plane does this in closed form.  Any other patch starts
from the nearest point of a coarse sample of itself and takes projected
Newton steps on half the squared distance from p to x(u, v) (the full
Hessian where it is positive definite, Gauss-Newton otherwise; an
Armijo line search allowing for the rounding of the distance; a bound
held where the gradient points out of the square), until a full step
moves u and v by at most `rtol`, a fraction of the square.  A bilinear
patch bends at the edges of its cells, so every cell whose bounding
sphere comes within the nearest sample's distance is searched and the
nearest answer kept.  The answer is the nearest point wherever that is
unique, as it is for a point closer to a smooth patch than its radius
of curvature; a search that has not converged is refused, or flagged
where the caller asks.
`Fault.locate` takes the nearest patch, and `Fault.slip_at` reads its
slip there, refusing a point farther than a stated distance.

**Patches for large faults.**  `StructuredPatch.swept` sweeps a dip
profile down from a `Trace` (`GreatCircle`, `SmallCircle`, `Polyline`)
on the reference surface.  At each trace point, with t the horizontal
direction of travel there, the surface descends from depth `top` in
the vertical plane through the point normal to t, on the right of the
direction of travel (the dip to the right of the strike), along the
curve that makes the angle dip(z) with the local horizontal at depth z
below the reference surface:

    dX/dw = cos dip(z) (t x up(X)) - sin dip(z) up(X),

w the length down dip.  On a sphere that plane holds the centre and the
curve follows the sphere: dz/dw = sin dip, and the angle turned about
the centre grows as cos dip / (R - z), so a constant dip is a
logarithmic spiral and a small circle swept at a constant dip is the
surface of revolution of that spiral about the circle's pole.  A
straight trace at a constant dip is therefore a plane in a Cartesian
frame alone, and on a sphere departs from `Plane` at first order in its
size over the radius.  u runs along the trace by the trace's own
parameter and v down dip by length, so rake on such a patch is measured
from the local direction of the trace.  `StructuredPatch.from_depths`
takes an interface as depths below the reference surface on a
logically rectangular grid (Slab2's form), checking its orientation and
refusing NaN (`finite_block` finds a crop).  Both give an ordinary
`StructuredPatch`, whose slip is any `Slip`.

**The spline form.**  A plane and a structured patch give their map as
a `TensorSpline` (`spline`), the tensor-product B-spline a CAD kernel
takes a surface in: a plane the bilinear spline through its corners, a
structured patch its own rule's knots and coefficients, either one the
patch exactly.  `TensorSpline.extended` continues it beyond the square,
its end pieces' polynomials carried on, which is how a mesher carries a
fault's edge across a boundary that is to trim it.

**Pulled back.**  A mesh is built on a reference body and its nodes are
then moved by a mapping m (`planetmodel.mapping`), so a fault, defined
where it physically is, enters the mesh as its preimage: `pull_back`
inverts the mapping at the patch's points on a grid of control nodes (a
root-find along the radius for a radial mapping, down the column for a
flat vertical stretch) and passes the bicubic
structured patch through the results, which m takes back onto the patch
exactly at the nodes and to the interpolation error between them,
measured and returned as the preimage's `offset`.  Under the identity
a patch is its own preimage.

A kind of patch provides its map and the map's first and second
derivatives (`cartesian`, `tangents`, `second_derivatives`, on any
(u, v) in the square); everything else is `PatchBase`'s, built on those
three, and `testing.check_patch` holds a kind to the protocol `Patch`.
"""
from __future__ import annotations

import numbers
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import NdBSpline, make_interp_spline
from scipy.spatial import cKDTree

from .frames import (cartesian_points, from_geographic, geographic,
                     spherical_coordinates, spherical_frame)
from .surfacefile import (TIME_RTOL, TimeRows, check_time_rule, leading_shape,
                          time_axis, time_rows)
from .units import Scales

if TYPE_CHECKING:
    from .mapping import Mapping

__all__ = ["SQUARE", "INTERPOLATIONS", "SLIP_RULES", "BASES", "SLIP_COMPONENTS",
           "MIN_NODES", "POINT_FORMS", "Frame", "GeographicFrame",
           "CartesianFrame", "Slip", "Patch", "PatchBase", "Plane",
           "StructuredPatch", "PatchQuadrature", "Quadrature", "SliceQuadrature",
           "Location", "Fault", "parameter_rule",
           "DipProfile", "Trace", "GreatCircle", "SmallCircle", "Polyline",
           "finite_block", "TensorSpline", "PULL_BACK_CONTROLS", "Preimage",
           "pull_back"]

#: The interval u and v each run over: the parameter square is SQUARE x SQUARE,
#: (0, 0) at its centre, u along strike and v down dip.
SQUARE = (-0.5, 0.5)

#: The rules between the nodes of a grid over the square: the tensor-product
#: splines of degree one and three (not-a-knot), as named by the lateral rules.
INTERPOLATIONS = ("linear", "cubic")

#: The rules of a slip grid: constant on each cell, or one of INTERPOLATIONS.
SLIP_RULES = ("cell",) + INTERPOLATIONS

#: The bases a slip vector is given or asked for in: the frame's ambient
#: basis at the point, Cartesian, or the fault basis at the point, whose
#: components (strike-slip, dip-slip positive up dip, opening) a `Slip` holds.
BASES = ("ambient", "cartesian", "fault")

#: The names of a `Slip`'s three components, in order.
SLIP_COMPONENTS = ("strike_slip", "dip_slip", "opening")

#: How points are given to `locate`: Cartesian, or in the frame's coordinates.
POINT_FORMS = ("cartesian", "frame")

#: The fewest nodes (cells, for `cell`) each way a rule of `SLIP_RULES` needs,
#: and so `INTERPOLATIONS`.
MIN_NODES = {"cell": 1, "linear": 2, "cubic": 4}

#: The sufficient decrease a step of `locate`'s search must make (Armijo), and
#: the halvings of the step it tries before giving up on an iteration.
_ARMIJO = 1e-4
_HALVINGS = 40


def _real(x: object, what: str) -> float:
    """A finite real number, or the refusal naming `what`."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real) \
            or not np.isfinite(float(x)):
        raise ValueError(f"{what} must be a finite number, got {x!r}")
    return float(x)


def _readonly(a: ArrayLike) -> np.ndarray:
    """A read-only float64 copy."""
    out = np.array(a, dtype=float)
    out.setflags(write=False)
    return out


def _points(coordinates: ArrayLike) -> np.ndarray:
    """Finite coordinates of shape (..., 3), as float64."""
    c = np.asarray(coordinates, dtype=float)
    if c.ndim == 0 or c.shape[-1] != 3:
        raise ValueError(f"expected coordinates of shape (..., 3), got {c.shape}")
    if not np.all(np.isfinite(c)):
        raise ValueError("coordinates must be finite")
    return c


def _nodes(n: int) -> np.ndarray:
    """n equally spaced nodes over SQUARE, both ends included."""
    return np.linspace(SQUARE[0], SQUARE[1], n)


def _edges(n: int) -> np.ndarray:
    """The n + 1 edges of n equal cells of SQUARE."""
    return np.linspace(SQUARE[0], SQUARE[1], n + 1)


def _parameters(u: ArrayLike, v: ArrayLike
                ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """(u, v) broadcast, checked to lie in the square, flattened; and
    their broadcast shape."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float),
                               np.asarray(v, dtype=float))
    lo, hi = SQUARE
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise ValueError("(u, v) must be finite")
    if np.any(u < lo) or np.any(u > hi) or np.any(v < lo) or np.any(v > hi):
        raise ValueError(f"(u, v) must lie in the parameter square "
                         f"[{lo}, {hi}] x [{lo}, {hi}]")
    return u.ravel(), v.ravel(), u.shape


def _unit(x: np.ndarray) -> np.ndarray:
    """x / norm(x) along the last axis."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The dot products of two stacks of vectors along the last axis."""
    return np.einsum("...i,...i->...", a, b)


def _fault_basis(xu: np.ndarray, xv: np.ndarray) -> np.ndarray:
    """(strike, up dip, normal) as columns, (..., 3, 3), from the tangents
    (module docstring)."""
    s, n = _unit(xu), _unit(np.cross(xv, xu))
    return np.stack([s, np.cross(n, s), n], axis=-1)


def _on_basis(F: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Fault components (P, 3) on the fault bases F (P, 3, 3), Cartesian."""
    return np.einsum("pij,pj->pi", F, components)


def _positive(x: object, what: str) -> float:
    """A positive finite number, or the refusal naming `what`."""
    value = _real(x, what)
    if not value > 0.0:
        raise ValueError(f"{what} must be positive, got {x!r}")
    return value


def _count(n: object, what: str, *, least: int = 1) -> int:
    """An integer of at least `least`, or the refusal naming `what`."""
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < least:
        raise ValueError(f"{what} must be an integer of at least {least}, "
                         f"got {n!r}")
    return int(n)


def _given_points(frame: Frame, points: ArrayLike,
                  given: str) -> tuple[np.ndarray, tuple[int, ...]]:
    """Points given Cartesian or in the frame's coordinates (`POINT_FORMS`),
    as flat Cartesian points (N, 3), and their shape."""
    if given not in POINT_FORMS:
        raise ValueError(f"given is one of {POINT_FORMS}, got {given!r}")
    P = _points(points) if given == "cartesian" else frame.cartesian(points)
    return P.reshape(-1, 3), P.shape[:-1]


def _depth(frame: Frame, points: np.ndarray) -> np.ndarray:
    """The depth of Cartesian points below the frame's reference surface."""
    c = frame.coordinates(points)
    return frame.below(c[..., 0], c[..., 1], 0.0)[..., 2] - c[..., 2]


# -- frames -----------------------------------------------------------------------

@runtime_checkable
class Frame(Protocol):
    """Where a fault's positions are and what basis its vectors use.

    Coordinates have shape (..., 3) in the frame's own terms; Cartesian
    points have shape (..., 3); the basis at a position is (..., 3, 3)
    with the ambient basis vectors as columns, in Cartesian components.
    """

    def cartesian(self, coordinates: ArrayLike) -> np.ndarray:
        """The Cartesian points of positions in the frame's coordinates."""
        ...

    def coordinates(self, points: ArrayLike) -> np.ndarray:
        """The frame's coordinates of Cartesian points."""
        ...

    def basis(self, coordinates: ArrayLike) -> np.ndarray:
        """The ambient basis at positions, columns in Cartesian components."""
        ...

    def below(self, a: ArrayLike, b: ArrayLike, depth: ArrayLike) -> np.ndarray:
        """The coordinates of the point `depth` below the reference
        surface at the horizontal position (a, b)."""
        ...


@dataclass(frozen=True)
class GeographicFrame:
    """Positions as (latitude, longitude, radius) about a reference
    sphere of `radius`, the ambient basis (east, north, up) (module
    docstring)."""

    #: The frame's name, as a fault file states it.
    kind: ClassVar[str] = "geographic"

    radius: float

    def __post_init__(self) -> None:
        radius = _real(self.radius, "the reference radius")
        if not radius > 0.0:
            raise ValueError(f"the reference radius must be positive, got {radius}")
        object.__setattr__(self, "radius", radius)

    def cartesian(self, coordinates: ArrayLike) -> np.ndarray:
        """X = r e_r(latitude, longitude), shape (..., 3); latitudes in
        [-90, 90], longitudes in any range."""
        c = _points(coordinates)
        theta, phi = from_geographic(c[..., 0], c[..., 1])
        return cartesian_points(c[..., 2], theta, phi)

    def coordinates(self, points: ArrayLike) -> np.ndarray:
        """(latitude, longitude in [0, 360), radius) of Cartesian points."""
        r, theta, phi, _ = spherical_coordinates(_points(points))
        latitude, longitude = geographic(theta, phi)
        return np.stack([latitude, longitude, r], axis=-1)

    def basis(self, coordinates: ArrayLike) -> np.ndarray:
        """(east, north, up) = (e_phi, -e_theta, e_r) as columns, at the
        positions' latitudes and longitudes."""
        c = _points(coordinates)
        theta, phi = from_geographic(c[..., 0], c[..., 1])
        R = spherical_frame(theta, phi)
        return np.stack([R[..., 2], -R[..., 1], R[..., 0]], axis=-1)

    def below(self, a: ArrayLike, b: ArrayLike, depth: ArrayLike) -> np.ndarray:
        """(latitude a, longitude b, radius - depth): depth below the
        reference sphere, not below any topography on it."""
        a, b, depth = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                            for x in (a, b, depth)))
        return np.stack([a, b, self.radius - depth], axis=-1)


@dataclass(frozen=True)
class CartesianFrame:
    """Positions as (x, y, z), x east, y north and z up, the ambient basis
    (x, y, z) everywhere; the reference surface is the plane z =
    `surface`.

    The points are those of `planetmodel.cartesian`, whose flat system
    `cartesian.FLAT` places a depth d at z = -d below its free surface
    z = 0: a depth below this frame's reference surface is read through
    it (`below`), so that with `surface` 0 the two depths are one, and a
    fault in this frame is built into a box of a flat geometry as it
    stands."""

    #: The frame's name, as a fault file states it.
    kind: ClassVar[str] = "cartesian"

    _: KW_ONLY
    #: The height z of the reference plane depths are measured from.
    surface: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface", _real(self.surface, "surface"))

    def cartesian(self, coordinates: ArrayLike) -> np.ndarray:
        """The coordinates themselves."""
        return _points(coordinates).copy()

    def coordinates(self, points: ArrayLike) -> np.ndarray:
        """The points themselves."""
        return _points(points).copy()

    def basis(self, coordinates: ArrayLike) -> np.ndarray:
        """The identity at every position."""
        c = _points(coordinates)
        return np.broadcast_to(np.eye(3), c.shape[:-1] + (3, 3)).copy()

    def below(self, a: ArrayLike, b: ArrayLike, depth: ArrayLike) -> np.ndarray:
        """(x a, y b, surface - depth): the point of `cartesian.FLAT` at the
        depth `depth - surface` below its free surface."""
        from .cartesian.coordinates import FLAT
        a, b, depth = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                            for x in (a, b, depth)))
        X = FLAT.points(depth - self.surface, a, b)
        X[..., 2] += 0.0        # z on the reference plane is +0.0, not -0.0
        return X


# -- values on a grid over the square ---------------------------------------------

class _GridRule:
    """Values at a uniform grid over the square read by one rule: constant
    a cell, or the tensor-product spline (degree one, or three with
    not-a-knot ends) through the nodes; values of shape (n_u, n_v) +
    trailing, the trailing axes carried through."""

    def __init__(self, values: np.ndarray, rule: str) -> None:
        self._rule = rule
        n_u, n_v = values.shape[:2]
        if rule == "cell":
            self._values = values
            self._edges = (_edges(n_u), _edges(n_v))
            return
        k = 1 if rule == "linear" else 3
        along_u = make_interp_spline(_nodes(n_u), values, k=k, axis=0)
        along_v = make_interp_spline(_nodes(n_v), np.swapaxes(along_u.c, 0, 1),
                                     k=k, axis=0)
        self._t = (along_u.t, along_v.t)
        self._c = np.swapaxes(along_v.c, 0, 1)
        self._k = k

    def __call__(self, u: np.ndarray, v: np.ndarray, *,
                 nu: tuple[int, int] = (0, 0), index: int | None = None
                 ) -> np.ndarray:
        """The values, or a derivative, at flat (u, v): shape (P,) +
        trailing, or (P,) + trailing[1:] with the first trailing axis at
        `index`."""
        if self._rule == "cell":
            eu, ev = self._edges
            i = np.clip(np.searchsorted(eu, u, side="right") - 1, 0, eu.size - 2)
            j = np.clip(np.searchsorted(ev, v, side="right") - 1, 0, ev.size - 2)
            values = self._values if index is None else self._values[:, :, index]
            out = values[i, j]
            return out if nu == (0, 0) else np.zeros_like(out)
        c = self._c if index is None else self._c[:, :, index]
        return NdBSpline(self._t, c, self._k)(np.stack([u, v], axis=-1), nu=nu)


def _check_grid(shape: tuple[int, ...], rule: str, what: str) -> None:
    """Refuse a rule unknown, or a grid too small for it."""
    if rule not in SLIP_RULES:
        raise ValueError(f"{what}: the rule must be one of {SLIP_RULES}, got "
                         f"{rule!r}")
    least = MIN_NODES[rule]
    if min(shape) < least:
        raise ValueError(f"{what}: the {rule!r} rule needs at least {least} "
                         f"{'cells' if rule == 'cell' else 'nodes'} each way, "
                         f"got {shape[0]} by {shape[1]}")


# -- slip ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Slip:
    """The slip on one patch: its fault components on a grid over the
    square, and the rule between them (module docstring).

    `values` has shape (n_u, n_v, 3) for one slice or (nslices, n_u,
    n_v, 3), and is held as (nslices, n_u, n_v, 3), read-only: the
    components (strike-slip, dip-slip, opening) (`SLIP_COMPONENTS`) at
    the centres of n_u by n_v equal cells for the `cell` rule, otherwise
    at the nodes of a uniform grid, corners included.  `rule` is one of
    `SLIP_RULES`.
    """

    values: ArrayLike
    _: KW_ONLY
    rule: str = "cell"

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.values):
            raise TypeError("slip is real")
        v = np.asarray(self.values, dtype=float)
        if v.ndim == 3:
            v = v[None]
        if v.ndim != 4 or v.shape[-1] != 3 or v.shape[0] < 1:
            raise ValueError(f"slip values have shape (n_u, n_v, 3) or "
                             f"(nslices, n_u, n_v, 3), got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("slip values must be finite")
        _check_grid(v.shape[1:3], self.rule, "the slip")
        object.__setattr__(self, "values", _readonly(v))

    @classmethod
    def from_rake(cls, rake: ArrayLike, magnitude: ArrayLike, *,
                  opening: ArrayLike = 0.0, rule: str = "cell") -> Slip:
        """Slip of `magnitude` in the fault plane at `rake` degrees, with
        `opening` along the normal: the components (m cos rake,
        m sin rake, opening).  The three broadcast to (n_u, n_v) or
        (nslices, n_u, n_v); scalars alone are one cell."""
        rake, m, o = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                           for x in (rake, magnitude, opening)))
        if rake.ndim == 0 and rule == "cell":
            rake, m, o = (x.reshape(1, 1) for x in (rake, m, o))
        if rake.ndim not in (2, 3):
            raise ValueError(f"rake, magnitude and opening broadcast to "
                             f"(n_u, n_v) or (nslices, n_u, n_v), got {rake.shape}")
        lam = np.deg2rad(rake)
        return cls(np.stack([m * np.cos(lam), m * np.sin(lam), o], axis=-1),
                   rule=rule)

    @classmethod
    def from_vectors(cls, patch: Patch, vectors: ArrayLike, *, rule: str = "cell",
                     basis: str = "ambient") -> Slip:
        """The slip whose vectors at its grid's points on `patch` (the cell
        centres for the `cell` rule, the nodes otherwise) are `vectors`,
        given in `basis` (one of `BASES`) there: each turned into its
        fault components by the patch's fault basis at its point, an
        orthonormal change of basis, so that a component along the normal
        becomes opening.  `vectors` has shape (n_u, n_v, 3) or (nslices,
        n_u, n_v, 3); only the patch's map is read, not its own slip."""
        if not isinstance(patch, Patch):
            raise TypeError(f"expected a Patch, got {type(patch).__name__}")
        if basis not in BASES:
            raise ValueError(f"basis is one of {BASES}, got {basis!r}")
        given = cls(vectors, rule=rule)
        if basis == "fault":
            return given
        U, V = np.meshgrid(*given.nodes(), indexing="ij")
        cart = given.values
        if basis == "ambient":
            B = patch.frame.basis(patch.position(U, V))         # (n_u, n_v, 3, 3)
            cart = np.einsum("uvij,suvj->suvi", B, cart)
        F = patch.fault_basis(U, V)
        return cls(np.einsum("uvji,suvj->suvi", F, cart), rule=rule)

    @property
    def nslices(self) -> int:
        """The number of slices in time."""
        return int(self.values.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """(n_u, n_v): the cells or nodes each way."""
        return int(self.values.shape[1]), int(self.values.shape[2])

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """The u and v at which the values sit: the cell centres for the
        `cell` rule, the nodes otherwise."""
        if self.rule == "cell":
            eu, ev = _edges(self.shape[0]), _edges(self.shape[1])
            return 0.5 * (eu[1:] + eu[:-1]), 0.5 * (ev[1:] + ev[:-1])
        return _nodes(self.shape[0]), _nodes(self.shape[1])

    def __repr__(self) -> str:
        n_u, n_v = self.shape
        return (f"Slip({self.rule}, {self.nslices} slice"
                f"{'' if self.nslices == 1 else 's'} of {n_u} x {n_v})")


# -- patches ------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class PatchQuadrature:
    """Points on one patch, each with what it stands for
    (`PatchBase.quadrature`).

    `points` are Cartesian, (P, 3), and `positions` the same points in
    the frame's coordinates; `normals` are unit, Cartesian, from the
    footwall into the hanging wall; `weights` are the areas the points
    stand for, in the length scale squared; `slip` is Cartesian, (P, 3);
    `u`, `v` are the points' parameters, in `parameter_rule`'s order.
    """

    points: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    slip: np.ndarray
    u: np.ndarray
    v: np.ndarray

    @property
    def size(self) -> int:
        """The number of points."""
        return int(self.weights.size)


@dataclass(frozen=True, eq=False)
class Quadrature(PatchQuadrature):
    """Points on a fault (`Fault.quadrature`): a `PatchQuadrature` over
    every patch, the points running over the patches in order, with
    `patch`, the index of each point's patch."""

    patch: np.ndarray


@dataclass(frozen=True, eq=False)
class SliceQuadrature:
    """Points on a fault with the slip of every slice
    (`Fault.slice_quadrature`): `points`, `positions`, `normals`,
    `weights`, `u`, `v` and `patch` as `Quadrature` holds them, formed
    once, and `slip`, (nslices, P, 3), the Cartesian slip of each slice
    at every point."""

    points: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    u: np.ndarray
    v: np.ndarray
    patch: np.ndarray
    slip: np.ndarray

    @property
    def size(self) -> int:
        """The number of points."""
        return int(self.weights.size)

    @property
    def nslices(self) -> int:
        """The number of slices."""
        return int(self.slip.shape[0])

    def slip_of(self, rows: TimeRows) -> np.ndarray:
        """The Cartesian slip at every point under a date's rows
        ((k0, w0), (k1, w1), as `Fault.rows` gives them), (P, 3): w0 times
        slice k0, plus w1 times slice k1 where w1 is not zero, as
        `Fault.quadrature` gives the slip at that date."""
        (k0, w0), (k1, w1) = rows
        for k in (k0, k1):
            if isinstance(k, bool) or not isinstance(k, numbers.Integral) \
                    or not 0 <= k < self.nslices:
                raise IndexError(f"slice {k!r} of a quadrature of {self.nslices}")
        out = w0 * self.slip[k0]
        if w1 != 0.0:
            out = out + w1 * self.slip[k1]
        return out


def parameter_rule(n_u: int, n_v: int, *, points: int = 1
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, w): `points` by `points` Gauss-Legendre points in each of
    n_u by n_v equal cells of the square, with weights summing to one,
    the square's area; flat, u slowest, each direction running over its
    cells in order and over the points within each."""
    for n, what in ((n_u, "n_u"), (n_v, "n_v"), (points, "points")):
        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
            raise ValueError(f"{what} must be a positive integer, got {n!r}")
    x, w = np.polynomial.legendre.leggauss(int(points))

    def composite(n: int) -> tuple[np.ndarray, np.ndarray]:
        e = _edges(int(n))
        half, mid = 0.5 * np.diff(e), 0.5 * (e[1:] + e[:-1])
        return ((mid[:, None] + half[:, None] * x).ravel(),
                (half[:, None] * w).ravel())

    pu, wu = composite(n_u)
    pv, wv = composite(n_v)
    U, V = np.meshgrid(pu, pv, indexing="ij")
    return U.ravel(), V.ravel(), np.outer(wu, wv).ravel()


@dataclass(frozen=True, eq=False)
class _Geometry:
    """A patch's quadrature points with what each stands for but its slip,
    as `PatchQuadrature` holds them, and `basis`, the fault basis at each
    (`_geometry`)."""

    points: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    u: np.ndarray
    v: np.ndarray
    basis: np.ndarray

    def slip(self, patch: Patch, index: int) -> np.ndarray:
        """The Cartesian slip of slice `index` of `patch` at the points: its
        components there on the fault basis there."""
        return _on_basis(self.basis, patch.slip_vector(self.u, self.v, index=index,
                                                       basis="fault"))


def _geometry(patch: Patch, n_u: int, n_v: int, points: int) -> _Geometry:
    """`parameter_rule(n_u, n_v, points=points)` on `patch`, each point with
    what it stands for but its slip and with the fault basis there, the
    tangents formed once: what a patch's quadrature and a fault's are
    built from."""
    u, v, w = parameter_rule(n_u, n_v, points=points)
    X = patch.cartesian(u, v)
    xu, xv = patch.tangents(u, v)
    c = np.cross(xv, xu)
    area = np.linalg.norm(c, axis=-1)
    return _Geometry(points=X, positions=patch.frame.coordinates(X),
                     normals=c / area[:, None], weights=area * w, u=u, v=v,
                     basis=_fault_basis(xu, xv))


def _joined(parts: Sequence[_Geometry]) -> dict[str, np.ndarray]:
    """The patches' points, with what each stands for but its slip, in
    turn, and `patch`, each point's patch: a fault's quadrature but its
    slip."""
    out = {name: np.concatenate([getattr(g, name) for g in parts])
           for name in ("points", "positions", "normals", "weights", "u", "v")}
    out["patch"] = np.concatenate([np.full(g.u.size, k, dtype=int)
                                   for k, g in enumerate(parts)])
    return out


# -- the spline form ----------------------------------------------------------------

def _greville(t: np.ndarray, k: int) -> np.ndarray:
    """The Greville abscissae of a knot vector of degree k: each the mean of
    the k knots inside its basis function's support, held to the span of
    the spline against the rounding of the mean at a repeated end."""
    n = t.size - k - 1
    x = np.array([np.mean(t[i + 1:i + k + 1]) for i in range(n)])
    return np.clip(x, t[k], t[n])


def _continued(t: np.ndarray, c: np.ndarray, k: int, below: float,
               above: float) -> tuple[np.ndarray, np.ndarray]:
    """Along the first axis of c: the knots and coefficients of the spline
    (t, c, k) continued `below` and `above` its ends, its end pieces'
    polynomials carried on, the new end knots clamped there.  The
    continuation lies in the spline space of the new knots, so
    interpolation at their Greville abscissae gives it exactly."""
    from scipy.interpolate import BSpline
    new = t.copy()
    new[:k + 1] = t[0] - below
    new[-(k + 1):] = t[-1] + above
    x = _greville(new, k)
    values = BSpline(t, c, k, axis=0, extrapolate=True)(x)
    return new, make_interp_spline(x, values, k=k, t=new, axis=0).c


@dataclass(frozen=True, eq=False)
class TensorSpline:
    """A patch's map as a tensor-product B-spline: the form a CAD kernel
    takes a surface in (`Plane.spline`, `StructuredPatch.spline`).

    `degree` is (k_u, k_v); `knots` the two knot vectors, each end
    repeated k + 1 times, so that the ends are the bounds of the
    parameters; `coefficients` the Cartesian control points, (n_u, n_v,
    3), with n = len(knots) - k - 1 each way.  Beyond its bounds the map
    continues its end pieces' polynomials.  The arrays are held
    read-only.
    """

    degree: tuple[int, int]
    knots: tuple[np.ndarray, np.ndarray]
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        ku, kv = (_count(k, "a spline's degree") for k in self.degree)
        tu, tv = (np.array(t, dtype=float) for t in self.knots)
        c = _points(self.coefficients)
        for t, k, n, what in ((tu, ku, c.shape[0], "u"), (tv, kv, c.shape[1], "v")):
            if t.ndim != 1 or t.size != n + k + 1:
                raise ValueError(f"the {what} knots number {t.size}, and a spline "
                                 f"of degree {k} with {n} coefficients has "
                                 f"{n + k + 1}")
            if np.any(np.diff(t) < 0.0) or not t[k + 1] > t[k] \
                    or np.ptp(t[:k + 1]) or np.ptp(t[-(k + 1):]):
                raise ValueError(f"the {what} knots must not decrease and must "
                                 f"repeat each end {k + 1} times")
        if c.ndim != 3:
            raise ValueError(f"coefficients have shape (n_u, n_v, 3), got {c.shape}")
        object.__setattr__(self, "degree", (ku, kv))
        object.__setattr__(self, "knots", (_readonly(tu), _readonly(tv)))
        object.__setattr__(self, "coefficients", _readonly(c))

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """((u_min, u_max), (v_min, v_max)): the ends of the knot vectors."""
        tu, tv = self.knots
        return (float(tu[0]), float(tu[-1])), (float(tv[0]), float(tv[-1]))

    def __call__(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """x(u, v), Cartesian, (..., 3); continued beyond the bounds."""
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float),
                                   np.asarray(v, dtype=float))
        spline = NdBSpline(self.knots, self.coefficients, self.degree,
                           extrapolate=True)
        return spline(np.stack([u.ravel(), v.ravel()], axis=-1)).reshape(
            u.shape + (3,))

    def extended(self, *, u: tuple[float, float] = (0.0, 0.0),
                 v: tuple[float, float] = (0.0, 0.0)) -> TensorSpline:
        """The same map over bounds widened by u[0] below and u[1] above
        the u bounds, and likewise v: its end pieces continued, so that
        within the old bounds nothing changes (to rounding) and beyond
        them it is what `__call__` gives."""
        widths = [float(x) for x in (*u, *v)]
        if not all(np.isfinite(x) and x >= 0.0 for x in widths):
            raise ValueError(f"the extensions must be non-negative, got u={u!r}, "
                             f"v={v!r}")
        (ku, kv), (tu, tv) = self.degree, self.knots
        tu, c = _continued(tu, self.coefficients, ku, widths[0], widths[1])
        tv, c = _continued(tv, np.swapaxes(c, 0, 1), kv, widths[2], widths[3])
        return TensorSpline((ku, kv), (tu, tv), np.swapaxes(c, 0, 1))

    def __repr__(self) -> str:
        (a, b), (c, d) = self.bounds
        n_u, n_v = self.coefficients.shape[:2]
        return (f"TensorSpline(degree {self.degree}, {n_u} x {n_v} control points "
                f"over [{a:g}, {b:g}] x [{c:g}, {d:g}])")


# -- locating points --------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Location:
    """Where points lie relative to a patch or a fault (`locate`).

    Every array has the points' shape.  `u`, `v` are the parameters of
    the point's nearest point x of the patch over the closed square (of
    the nearest patch, on a fault, whose index is `patch`); `distance` is
    the length of p - x signed as (p - x).n, positive on the hanging-wall
    side and zero counted positive; `inside` is whether x is not held on
    the square's boundary with p beyond it, true for the foot of a
    perpendicular and for a nearest point on a crease where a bilinear
    patch bends; `converged` is whether the search met its tolerance.
    `on_u_low`, `on_u_high`, `on_v_low` and `on_v_high` say which edge
    of the square x lies on (two of them at a corner): the edge u = -1/2,
    u = 1/2, v = -1/2 (up dip) or v = 1/2 (down dip).  A point held on
    an edge has that parameter exactly at the bound, so a point not
    `inside` reads which edge it lies beyond; one whose foot falls on an
    edge to rounding may read either way.
    """

    u: np.ndarray
    v: np.ndarray
    distance: np.ndarray
    inside: np.ndarray
    converged: np.ndarray
    patch: np.ndarray

    @property
    def on_u_low(self) -> np.ndarray:
        """Whether the nearest point lies on the edge u = -1/2."""
        return self.u == SQUARE[0]

    @property
    def on_u_high(self) -> np.ndarray:
        """Whether the nearest point lies on the edge u = 1/2."""
        return self.u == SQUARE[1]

    @property
    def on_v_low(self) -> np.ndarray:
        """Whether the nearest point lies on the edge v = -1/2, up dip."""
        return self.v == SQUARE[0]

    @property
    def on_v_high(self) -> np.ndarray:
        """Whether the nearest point lies on the edge v = 1/2, down dip."""
        return self.v == SQUARE[1]

    @property
    def on_edge(self) -> np.ndarray:
        """Whether the nearest point lies on any edge of the square."""
        return self.on_u_low | self.on_u_high | self.on_v_low | self.on_v_high


def _second_order(patch: Patch, P: np.ndarray, w: np.ndarray, *,
                  lo: np.ndarray | None = None, hi: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """At flat parameters w (N, 2): the residuals r = P - x, the gradient
    of |r|^2 / 2 in (u, v), its Hessian and its Gauss-Newton part J^T J.
    On the upper edge `hi` of a box [lo, hi] inside the square the
    derivatives are those of the box itself, read a hair inside it: a
    spline gives the next piece's on a knot."""
    u, v = w[:, 0], w[:, 1]
    r = P - patch.cartesian(u, v)
    if hi is not None:
        inner = (w >= hi) & (hi < SQUARE[1])
        if np.any(inner):
            w = np.where(inner, hi - 1e-9 * (hi - lo), w)
            u, v = w[:, 0], w[:, 1]
    xu, xv = patch.tangents(u, v)
    xuu, xuv, xvv = patch.second_derivatives(u, v)
    g = -np.stack([_dot(r, xu), _dot(r, xv)], axis=-1)
    A = np.empty((w.shape[0], 2, 2))
    A[:, 0, 0], A[:, 1, 1] = _dot(xu, xu), _dot(xv, xv)
    A[:, 0, 1] = A[:, 1, 0] = _dot(xu, xv)
    H = A.copy()
    H[:, 0, 0] -= _dot(r, xuu)
    H[:, 1, 1] -= _dot(r, xvv)
    H[:, 0, 1] -= _dot(r, xuv)
    H[:, 1, 0] = H[:, 0, 1]
    return r, g, H, A


def _newton_step(g: np.ndarray, H: np.ndarray, A: np.ndarray,
                 held: np.ndarray) -> np.ndarray:
    """The projected Newton step (N, 2): -M^-1 g on the coordinates not
    held at a bound, zero on the held ones, M the Hessian there where it is
    positive definite and the Gauss-Newton matrix otherwise."""
    free = ~held
    out = np.zeros_like(g)
    both = free[:, 0] & free[:, 1]
    if np.any(both):
        Hb = H[both]
        det = Hb[:, 0, 0] * Hb[:, 1, 1] - Hb[:, 0, 1] ** 2
        definite = (Hb[:, 0, 0] > 0.0) & (det > 0.0)
        M = np.where(definite[:, None, None], Hb, A[both])
        out[both] = -np.linalg.solve(M, g[both][..., None])[..., 0]
    for k in (0, 1):
        one = free[:, k] & ~free[:, 1 - k]
        if np.any(one):
            m = np.where(H[one, k, k] > 0.0, H[one, k, k], A[one, k, k])
            out[one, k] = -g[one, k] / m
    return out


def _descend(patch: Patch, P: np.ndarray, start: np.ndarray, lo: np.ndarray,
             hi: np.ndarray, *, rtol: float,
             iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Projected Newton with an Armijo line search on |P - x(u, v)|^2 / 2
    over the boxes [lo, hi] (each (N, 2)), from `start`, a step accepted
    where f does not rise beyond its own rounding: the parameters reached
    and whether the full step fell to `rtol` within `iterations`."""
    w = start.copy()
    converged = np.zeros(w.shape[0], dtype=bool)
    todo = np.arange(w.shape[0])
    for _ in range(iterations):
        if todo.size == 0:
            break
        p, wt, l, h = P[todo], w[todo], lo[todo], hi[todo]
        r, g, H, A = _second_order(patch, p, wt, lo=l, hi=h)
        held = ((wt <= l) & (g > 0.0)) | ((wt >= h) & (g < 0.0))
        step = _newton_step(g, H, A, held)
        trial = np.clip(wt + step, l, h)
        done = np.max(np.abs(trial - wt), axis=-1) <= rtol
        f0 = 0.5 * _dot(r, r)
        # the rounding of f itself, below which a decrease cannot be seen
        noise = 8.0 * np.finfo(float).eps * (_dot(p, p) + f0)
        alpha = np.ones(todo.size)
        for _ in range(_HALVINGS):
            rt = p - patch.cartesian(trial[:, 0], trial[:, 1])
            fails = (0.5 * _dot(rt, rt) > f0 + noise
                     + _ARMIJO * np.sum(g * (trial - wt), axis=-1)) & ~done
            if not np.any(fails):
                break
            alpha[fails] *= 0.5
            trial[fails] = np.clip(wt[fails] + alpha[fails, None] * step[fails],
                                   l[fails], h[fails])
        w[todo] = trial
        converged[todo[done]] = True
        todo = todo[~done]
    return w, converged


def _location(patch: Patch, P: np.ndarray, w: np.ndarray, inside: np.ndarray,
              converged: np.ndarray, shape: tuple[int, ...],
              strict: bool) -> Location:
    """The `Location` of flat points P at their nearest parameters w,
    refusing a search that did not converge where `strict`."""
    if strict and not np.all(converged):
        bad = int(np.count_nonzero(~converged))
        raise ValueError(f"locate did not converge for {bad} of {converged.size} "
                         "points; raise iterations, or pass strict=False to "
                         "flag them in Location.converged")
    r = P - patch.cartesian(w[:, 0], w[:, 1])
    sign = np.where(_dot(r, patch.normal(w[:, 0], w[:, 1])) < 0.0, -1.0, 1.0)
    return Location(u=w[:, 0].reshape(shape), v=w[:, 1].reshape(shape),
                    distance=(sign * np.linalg.norm(r, axis=-1)).reshape(shape),
                    inside=inside.reshape(shape),
                    converged=converged.reshape(shape),
                    patch=np.zeros(shape, dtype=int))


@runtime_checkable
class Patch(Protocol):
    """One patch of a fault: a map of the parameter square, its frame,
    and its slip (module docstring).  `PatchBase` gives every method but
    `cartesian`, `tangents` and `second_derivatives` from those three;
    `testing.check_patch` is the contract."""

    @property
    def frame(self) -> Frame:
        """The frame the positions are in."""
        ...

    @property
    def slip(self) -> Slip:
        """The slip, its fault components on its grid."""
        ...

    def cartesian(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """x(u, v), Cartesian, (..., 3)."""
        ...

    def tangents(self, u: ArrayLike, v: ArrayLike
                 ) -> tuple[np.ndarray, np.ndarray]:
        """(x_u, x_v), Cartesian, each (..., 3)."""
        ...

    def second_derivatives(self, u: ArrayLike, v: ArrayLike
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x_uu, x_uv, x_vv), Cartesian, each (..., 3)."""
        ...

    def position(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """x(u, v) in the frame's coordinates, (..., 3)."""
        ...

    def normal(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """The unit normal, Cartesian, (..., 3)."""
        ...

    def area_element(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """The length of x_u x x_v, of the parameters' shape."""
        ...

    def fault_basis(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """(strike, up dip, normal) as columns, Cartesian, (..., 3, 3)."""
        ...

    def slip_vector(self, u: ArrayLike, v: ArrayLike, *, index: int = 0,
                    basis: str = "ambient") -> np.ndarray:
        """The slip of slice `index` at (u, v) in `basis`, (..., 3)."""
        ...

    def quadrature(self, n_u: int, n_v: int, *, points: int = 1,
                   index: int = 0) -> PatchQuadrature:
        """The patch's quadrature, with the slip of slice `index`."""
        ...

    def locate(self, points: ArrayLike, *, given: str = "cartesian",
               rtol: float = 1e-10, iterations: int = 50,
               strict: bool = True) -> Location:
        """Each point's nearest point of the patch and its distance."""
        ...


class PatchBase:
    """Everything a patch derives from its map (module docstring): its
    positions, normal, area element, fault basis and the location of
    points from `cartesian`, `tangents` and `second_derivatives`, which
    the kind supplies, and the slip read by its rule.  A kind sets its
    frame and calls `_attach` with its slip once its map is ready."""

    _frame: Frame
    _slip: Slip
    _slip_rule: _GridRule

    @property
    def frame(self) -> Frame:
        """The frame the positions are in."""
        return self._frame

    @property
    def slip(self) -> Slip:
        """The slip, its fault components on its grid."""
        return self._slip

    def cartesian(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """x(u, v); the kind supplies it."""
        raise NotImplementedError

    def tangents(self, u: ArrayLike, v: ArrayLike
                 ) -> tuple[np.ndarray, np.ndarray]:
        """(x_u, x_v); the kind supplies them."""
        raise NotImplementedError

    def second_derivatives(self, u: ArrayLike, v: ArrayLike
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x_uu, x_uv, x_vv); the kind supplies them."""
        raise NotImplementedError

    def _pieces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray] | None:
        """None where the map is smooth on the whole square; otherwise the
        edges in u and in v of the pieces on each of which it is, and a
        sphere holding each piece, its centre (n_u - 1, n_v - 1, 3) and
        radius (n_u - 1, n_v - 1)."""
        return None

    def _samples(self) -> tuple[int, int]:
        """The nodes each way of the coarse sample `locate` starts from."""
        return 17, 17

    def _attach(self, slip: Slip) -> None:
        """Hold `slip`, and the rule that reads its components."""
        if not isinstance(slip, Slip):
            raise TypeError(f"slip must be a Slip, got {type(slip).__name__}")
        self._slip = slip
        self._slip_rule = _GridRule(np.moveaxis(slip.values, 0, 2), slip.rule)

    def position(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """x(u, v) in the frame's coordinates, (..., 3)."""
        return self._frame.coordinates(self.cartesian(u, v))

    def normal(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """n, x_v x x_u over its length, Cartesian, (..., 3): from the
        footwall into the hanging wall."""
        xu, xv = self.tangents(u, v)
        return _unit(np.cross(xv, xu))

    def area_element(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """The length of x_u x x_v, of the parameters' shape: the area per
        unit of u v."""
        xu, xv = self.tangents(u, v)
        return np.linalg.norm(np.cross(xu, xv), axis=-1)

    def fault_basis(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """(strike, up dip, normal) as columns, Cartesian, (..., 3, 3):
        s = x_u / norm(x_u), n the normal, a = n x s; right-handed, and the
        basis of the `fault` components."""
        return _fault_basis(*self.tangents(u, v))

    def slip_vector(self, u: ArrayLike, v: ArrayLike, *, index: int = 0,
                    basis: str = "ambient") -> np.ndarray:
        """The slip of slice `index` at (u, v), (..., 3): its fault
        components read by the slip's rule, and given in `basis`, one of
        `BASES`, through the fault basis at the point."""
        if basis not in BASES:
            raise ValueError(f"basis is one of {BASES}, got {basis!r}")
        n = self._slip.nslices
        if isinstance(index, bool) or not isinstance(index, numbers.Integral) \
                or not 0 <= index < n:
            raise IndexError(f"slice {index!r} of a slip of {n}")
        fu, fv, shape = _parameters(u, v)
        comp = self._slip_rule(fu, fv, index=int(index))
        if basis == "fault":
            return comp.reshape(shape + (3,))
        cart = _on_basis(self.fault_basis(fu, fv), comp)
        if basis == "ambient":
            B = self._frame.basis(self.position(fu, fv))
            cart = np.einsum("pji,pj->pi", B, cart)
        return cart.reshape(shape + (3,))

    def quadrature(self, n_u: int, n_v: int, *, points: int = 1,
                   index: int = 0) -> PatchQuadrature:
        """`parameter_rule(n_u, n_v, points=points)` on this patch, with
        the slip of slice `index`; a fault's quadratures say which patch
        each point is on (`Fault.quadrature`), which a patch alone does
        not know."""
        g = _geometry(self, n_u, n_v, points)
        return PatchQuadrature(points=g.points, positions=g.positions,
                               normals=g.normals, weights=g.weights,
                               slip=g.slip(self, index), u=g.u, v=g.v)

    def locate(self, points: ArrayLike, *, given: str = "cartesian",
               rtol: float = 1e-10, iterations: int = 50,
               strict: bool = True) -> Location:
        """Each point's nearest point of the patch, its signed distance and
        whether it is inside (`Location`; module docstring).

        `points` (..., 3) are Cartesian or in the frame's coordinates, as
        `given` says (`POINT_FORMS`).  The search starts from the nearest
        point of a coarse sample of the patch and takes projected Newton
        steps with a line search, over the whole square where the map is
        smooth, and otherwise on every piece of it (a bilinear patch's
        cell) whose bounding sphere comes within the nearest sample's
        distance of the point; it has converged when a full step moves u
        and v by at most `rtol`, a fraction of the square, which places
        the point to about `rtol` of the patch's size.  A search
        that has not converged within `iterations` steps is refused where
        `strict`, and flagged in `Location.converged` otherwise.
        """
        P, shape = _given_points(self._frame, points, given)
        rtol = _positive(rtol, "rtol")
        iterations = _count(iterations, "iterations")
        su, sv = self._samples()
        U, V = np.meshgrid(_nodes(su), _nodes(sv), indexing="ij")
        bound, nearest = cKDTree(self.cartesian(U, V).reshape(-1, 3)).query(P)
        start = np.stack([U.ravel()[nearest], V.ravel()[nearest]], axis=-1)
        pieces = self._pieces()
        if pieces is None:
            lo = np.broadcast_to(np.array([SQUARE[0]] * 2), start.shape)
            hi = np.broadcast_to(np.array([SQUARE[1]] * 2), start.shape)
            w, converged = _descend(self, P, np.clip(start, lo, hi), lo, hi,
                                    rtol=rtol, iterations=iterations)
        else:
            # every piece whose sphere comes within the nearest sample's
            # distance of the point may hold its nearest point
            ku, kv, centres, radii = pieces
            C, rho = centres.reshape(-1, 3), radii.ravel()
            reach = bound * (1.0 + 1e-12) + 1e-12 * float(np.max(np.abs(C)))
            near = cKDTree(C).query_ball_point(P, reach + float(np.max(rho)))
            pi = np.repeat(np.arange(P.shape[0]), [len(n) for n in near])
            pk = np.concatenate([np.asarray(n, dtype=int) for n in near])
            keep = np.linalg.norm(P[pi] - C[pk], axis=-1) - rho[pk] <= reach[pi]
            pi, pk = pi[keep], pk[keep]
            iu, iv = np.divmod(pk, kv.size - 1)
            lo = np.stack([ku[iu], kv[iv]], axis=-1)
            hi = np.stack([ku[iu + 1], kv[iv + 1]], axis=-1)
            found, ok = _descend(self, P[pi], np.clip(start[pi], lo, hi), lo, hi,
                                 rtol=rtol, iterations=iterations)
            r = P[pi] - self.cartesian(found[:, 0], found[:, 1])
            order = np.lexsort((_dot(r, r), pi))
            first = order[np.r_[True, pi[order][1:] != pi[order][:-1]]]
            w = np.empty((P.shape[0], 2))
            converged = np.zeros(P.shape[0], dtype=bool)
            w[pi[first]], converged[pi[first]] = found[first], ok[first]
        # inside unless held on the square's edge with the unconstrained
        # foot of the perpendicular beyond it
        _, g, H, A = _second_order(self, P, w)
        free = w + _newton_step(g, H, A, np.zeros_like(w, dtype=bool))
        lo, hi = SQUARE
        outside = ((w <= lo) & (free < lo - rtol)) | ((w >= hi) & (free > hi + rtol))
        return _location(self, P, w, ~np.any(outside, axis=-1), converged, shape,
                         strict)


class Plane(PatchBase):
    """A flat rectangle by its centroid, strike, dip, length and width
    (module docstring), carrying `slip`.

    `centroid` is in the frame's coordinates; `strike` and `dip` are
    degrees, the dip from 0 to 90; `length` (along strike) and `width`
    (down dip) are positive, in the length scale.
    """

    #: The patch's kind, as a fault file states it.
    kind: ClassVar[str] = "plane"

    def __init__(self, frame: Frame, centroid: ArrayLike, *, strike: float,
                 dip: float, length: float, width: float, slip: Slip) -> None:
        if not isinstance(frame, Frame):
            raise TypeError(f"expected a Frame, got {type(frame).__name__}")
        c = _points(centroid)
        if c.shape != (3,):
            raise ValueError(f"the centroid is one position, (3,), got {c.shape}")
        self._frame = frame
        self._centroid = _readonly(c)
        self._strike = _real(strike, "strike")
        self._dip = _real(dip, "dip")
        if not 0.0 <= self._dip <= 90.0:
            raise ValueError(f"dip must lie in [0, 90] degrees, got {dip!r}")
        self._length = _real(length, "length")
        self._width = _real(width, "width")
        if not (self._length > 0.0 and self._width > 0.0):
            raise ValueError(f"length and width must be positive, got "
                             f"{length!r} and {width!r}")
        east, north, up = np.moveaxis(frame.basis(c), -1, 0)
        phi, delta = np.deg2rad(self._strike), np.deg2rad(self._dip)
        s = np.sin(phi) * east + np.cos(phi) * north
        r = np.cos(phi) * east - np.sin(phi) * north
        self._origin = frame.cartesian(c)
        self._xu = self._length * s
        self._xv = self._width * (np.cos(delta) * r - np.sin(delta) * up)
        self._attach(slip)

    @classmethod
    def at_depth(cls, frame: Frame, a: float, b: float, depth: float, *,
                 strike: float, dip: float, length: float, width: float,
                 slip: Slip) -> Plane:
        """The plane whose centroid lies `depth` below the frame's
        reference surface at the horizontal position (a, b): latitude and
        longitude, or x and y (`Frame.below`)."""
        return cls(frame, frame.below(a, b, depth), strike=strike, dip=dip,
                   length=length, width=width, slip=slip)

    @property
    def centroid(self) -> np.ndarray:
        """The centroid in the frame's coordinates."""
        return self._centroid

    @property
    def strike(self) -> float:
        """Degrees clockwise from north (from +y towards +x)."""
        return self._strike

    @property
    def dip(self) -> float:
        """Degrees down from the horizontal, to the right of the strike."""
        return self._dip

    @property
    def length(self) -> float:
        """The extent along strike."""
        return self._length

    @property
    def width(self) -> float:
        """The extent down dip."""
        return self._width

    def cartesian(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """c + u L s + v W d, (..., 3)."""
        fu, fv, shape = _parameters(u, v)
        X = self._origin + fu[:, None] * self._xu + fv[:, None] * self._xv
        return X.reshape(shape + (3,))

    def tangents(self, u: ArrayLike, v: ArrayLike
                 ) -> tuple[np.ndarray, np.ndarray]:
        """(L s, W d) at every point, each (..., 3)."""
        _, _, shape = _parameters(u, v)
        return (np.broadcast_to(self._xu, shape + (3,)).copy(),
                np.broadcast_to(self._xv, shape + (3,)).copy())

    def second_derivatives(self, u: ArrayLike, v: ArrayLike
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zero: the plane is flat."""
        _, _, shape = _parameters(u, v)
        zero = np.zeros(shape + (3,))
        return zero, zero.copy(), zero.copy()

    def locate(self, points: ArrayLike, *, given: str = "cartesian",
               rtol: float = 1e-10, iterations: int = 50,
               strict: bool = True) -> Location:
        """`PatchBase.locate` in closed form: u = (p - c).s / L and
        v = (p - c).d / W, inside where both lie in the square to `rtol`,
        clamped to it otherwise (the nearest point of a rectangle, its
        sides being orthogonal); `iterations` and `strict` are not needed."""
        P, shape = _given_points(self._frame, points, given)
        rtol = _positive(rtol, "rtol")
        r = P - self._origin
        w = np.stack([r @ self._xu / self._length ** 2,
                      r @ self._xv / self._width ** 2], axis=-1)
        lo, hi = SQUARE
        inside = np.all((w >= lo - rtol) & (w <= hi + rtol), axis=-1)
        return _location(self, P, np.clip(w, lo, hi), inside,
                         np.ones(P.shape[0], dtype=bool), shape, strict)

    def spline(self) -> TensorSpline:
        """The plane as a bilinear B-spline through its four corners, which
        is the plane exactly."""
        ends = np.array([SQUARE[0], SQUARE[0], SQUARE[1], SQUARE[1]])
        U, V = np.meshgrid(SQUARE, SQUARE, indexing="ij")
        return TensorSpline((1, 1), (ends, ends), self.cartesian(U, V))

    def __repr__(self) -> str:
        a, b, c = self._centroid
        return (f"Plane(strike={self._strike:g}, dip={self._dip:g}, "
                f"length={self._length:g}, width={self._width:g} at "
                f"({a:g}, {b:g}, {c:g}), {self._slip!r})")


class StructuredPatch(PatchBase):
    """A patch through control positions on a uniform grid over the square
    (module docstring), carrying `slip`.

    `positions` has shape (n_u, n_v, 3) in the frame's coordinates, node
    (i, j) at u = -1/2 + i / (n_u - 1), v = -1/2 + j / (n_v - 1);
    `interpolation` is one of `INTERPOLATIONS`.  The area element must be
    positive at every control node.
    """

    #: The patch's kind, as a fault file states it.
    kind: ClassVar[str] = "structured"

    def __init__(self, frame: Frame, positions: ArrayLike, *, slip: Slip,
                 interpolation: str = "cubic") -> None:
        if not isinstance(frame, Frame):
            raise TypeError(f"expected a Frame, got {type(frame).__name__}")
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation is one of {INTERPOLATIONS}, got "
                             f"{interpolation!r}")
        p = _points(positions)
        if p.ndim != 3:
            raise ValueError(f"control positions have shape (n_u, n_v, 3), got "
                             f"{p.shape}")
        _check_grid(p.shape[:2], interpolation, "the control positions")
        self._frame = frame
        self._positions = _readonly(p)
        self._interpolation = interpolation
        self._control = _readonly(frame.cartesian(p))
        self._rule = _GridRule(self._control, interpolation)
        U, V = np.meshgrid(_nodes(p.shape[0]), _nodes(p.shape[1]), indexing="ij")
        if not np.all(self.area_element(U, V) > 0.0):
            raise ValueError("the patch is degenerate: its tangents vanish or "
                             "are parallel at a control node")
        self._attach(slip)

    @property
    def positions(self) -> np.ndarray:
        """The control positions in the frame's coordinates, (n_u, n_v, 3)."""
        return self._positions

    @property
    def interpolation(self) -> str:
        """The rule between the control positions, one of `INTERPOLATIONS`."""
        return self._interpolation

    @classmethod
    def swept(cls, frame: Frame, trace: Trace, *, dip: DipProfile, slip: Slip,
              n_u: int, n_v: int, bottom: float | None = None,
              width: float | None = None, top: float = 0.0,
              interpolation: str = "cubic", rtol: float = 1e-11
              ) -> StructuredPatch:
        """The patch swept down from a trace by a dip profile (module
        docstring): its control positions at `n_u` points evenly along the
        trace by its own parameter and `n_v` evenly down dip by the length
        along the surface, from `top` below the reference surface to
        `bottom`, or for a down-dip `width`, exactly one of the two given.
        The profile is integrated to `rtol` (relative) by scipy's DOP853.
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"expected a Frame, got {type(frame).__name__}")
        n_u = _count(n_u, "n_u", least=2)
        n_v = _count(n_v, "n_v", least=2)
        if not isinstance(trace, Trace):
            raise TypeError(f"expected a Trace, got {type(trace).__name__}")
        radians = _dip_profile(dip)
        top = _real(top, "top")
        rtol = _positive(rtol, "rtol")
        if (bottom is None) == (width is None):
            raise ValueError("give exactly one of bottom (the greatest depth) "
                             "and width (the length down dip)")
        if width is not None:
            length = _positive(width, "width")
        else:
            length = _width_to(radians, top, _real(bottom, "bottom"), rtol=rtol)
        X0, t = trace.surface(frame, _nodes(n_u) - SQUARE[0])
        c0 = frame.coordinates(X0)
        start = frame.cartesian(frame.below(c0[:, 0], c0[:, 1], top))

        def descent(_w: float, y: np.ndarray) -> np.ndarray:
            X = y.reshape(-1, 3)
            up = frame.basis(frame.coordinates(X))[..., 2]
            delta = radians(_depth(frame, X))
            across = _unit(np.cross(t, up))
            return (np.cos(delta)[:, None] * across
                    - np.sin(delta)[:, None] * up).ravel()

        w = np.linspace(0.0, length, n_v)
        scale = max(float(np.max(np.abs(start))), length)
        solved = solve_ivp(descent, (0.0, length), start.ravel(), method="DOP853",
                           t_eval=w, rtol=rtol, atol=rtol * scale)
        if not solved.success:
            raise ValueError(f"the dip profile could not be integrated: "
                             f"{solved.message}")
        X = solved.y.T.reshape(n_v, n_u, 3).transpose(1, 0, 2).copy()
        X[:, 0] = start
        return cls(frame, frame.coordinates(X), slip=slip,
                   interpolation=interpolation)

    @classmethod
    def from_depths(cls, frame: Frame, a: ArrayLike, b: ArrayLike,
                    depth: ArrayLike, *, slip: Slip, strike_axis: int = 0,
                    interpolation: str = "cubic") -> StructuredPatch:
        """The patch through an interface given by its depth below the
        reference surface (positive down) at the nodes of a logically
        rectangular grid of horizontal positions (latitude and longitude,
        or x and y), not necessarily regular: Slab2's form.

        `a`, `b` and `depth` broadcast to one (m, n) grid; `strike_axis`
        names the grid axis that runs along strike, which becomes u.  The
        grid is checked, not reordered: depth must increase along the
        other axis on average, by more than a billionth of its range (v
        runs down dip), and the normal must point
        up on average (the dip to the right of the strike, so that the
        normal runs from the footwall into the hanging wall); either
        failing is refused, naming the axis to reverse.  NaN is refused:
        crop to a finite block first (`finite_block`).
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"expected a Frame, got {type(frame).__name__}")
        a, b, z = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                        for x in (a, b, depth)))
        if a.ndim != 2:
            raise ValueError(f"the grid is two-dimensional, got shape {a.shape}")
        missing = int(np.count_nonzero(~(np.isfinite(a) & np.isfinite(b)
                                         & np.isfinite(z))))
        if missing:
            raise ValueError(f"{missing} of the grid's {a.size} nodes are not "
                             "finite: crop the grid to a finite block first "
                             "(finite_block)")
        if strike_axis not in (0, 1):
            raise ValueError(f"strike_axis is 0 or 1, got {strike_axis!r}")
        if strike_axis == 1:
            a, b, z = a.T, b.T, z.T
        if a.shape[0] < 2 or a.shape[1] < 2:
            raise ValueError(f"the grid needs two nodes each way, got {a.shape}")
        dip_axis = 1 - strike_axis
        if not np.mean(np.diff(z, axis=1)) > 1e-9 * np.ptp(z):
            raise ValueError(f"depth does not increase along axis {dip_axis} on "
                             "average: v runs down dip, so reverse that axis, or "
                             "the strike runs along the other")
        positions = frame.below(a, b, z)
        X = frame.cartesian(positions)
        xu, xv = np.gradient(X, axis=0), np.gradient(X, axis=1)
        up = frame.basis(positions)[..., 2]
        if np.mean(_dot(_unit(np.cross(xv, xu)), up)) < 0.0:
            raise ValueError(f"the interface dips to the left of the strike as "
                             f"axis {strike_axis} runs: reverse that axis, so "
                             "that the normal runs from the footwall into the "
                             "hanging wall")
        return cls(frame, positions, slip=slip, interpolation=interpolation)

    def cartesian(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """x(u, v), the control points' rule, (..., 3)."""
        fu, fv, shape = _parameters(u, v)
        return self._rule(fu, fv).reshape(shape + (3,))

    def tangents(self, u: ArrayLike, v: ArrayLike
                 ) -> tuple[np.ndarray, np.ndarray]:
        """(x_u, x_v), the derivatives of the rule, each (..., 3)."""
        fu, fv, shape = _parameters(u, v)
        return (self._rule(fu, fv, nu=(1, 0)).reshape(shape + (3,)),
                self._rule(fu, fv, nu=(0, 1)).reshape(shape + (3,)))

    def second_derivatives(self, u: ArrayLike, v: ArrayLike
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x_uu, x_uv, x_vv), the second derivatives of the rule, each
        (..., 3): x_uu and x_vv are zero under the bilinear rule."""
        fu, fv, shape = _parameters(u, v)
        return tuple(self._rule(fu, fv, nu=nu).reshape(shape + (3,))
                     for nu in ((2, 0), (1, 1), (0, 2)))

    def spline(self) -> TensorSpline:
        """The map as the tensor-product B-spline its rule is: degree one
        or three each way, the rule's own knots and coefficients, so the
        patch exactly."""
        rule = self._rule
        return TensorSpline((rule._k, rule._k), rule._t, rule._c)

    def _pieces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray] | None:
        """Under the bilinear rule, the cells between the control nodes,
        where it bends, each within the sphere about the mean of its four
        corners through the farthest (a bilinear cell lies in the hull of
        its corners); None under the bicubic rule, smooth throughout."""
        if self._interpolation != "linear":
            return None
        X = self._control
        corners = np.stack([X[:-1, :-1], X[1:, :-1], X[:-1, 1:], X[1:, 1:]])
        centres = corners.mean(axis=0)
        radii = np.max(np.linalg.norm(corners - centres, axis=-1), axis=0)
        n_u, n_v = self._positions.shape[:2]
        return _nodes(n_u), _nodes(n_v), centres, radii

    def _samples(self) -> tuple[int, int]:
        """The nodes and the midpoints between them, 9 to 129 each way."""
        n_u, n_v = self._positions.shape[:2]
        return (int(np.clip(2 * n_u - 1, 9, 129)), int(np.clip(2 * n_v - 1, 9, 129)))

    def __repr__(self) -> str:
        n_u, n_v = self._positions.shape[:2]
        return (f"StructuredPatch({self._interpolation}, {n_u} x {n_v} control "
                f"positions, {self._slip!r})")


# -- traces and dip profiles -----------------------------------------------------------

#: A dip as a function of depth below the reference surface: a number of
#: degrees, or a callable taking an array of depths and giving degrees.
type DipProfile = float | Callable[[np.ndarray], ArrayLike]


def _dip_profile(dip: DipProfile) -> Callable[[np.ndarray], np.ndarray]:
    """The dip in radians at an array of depths, checked to lie in
    [0, 90] degrees."""
    if callable(dip):
        f = dip
    else:
        value = _real(dip, "dip")
        f = lambda z: np.full(np.shape(z), value)

    def radians(z: np.ndarray) -> np.ndarray:
        d = np.broadcast_to(np.asarray(f(z), dtype=float), np.shape(z))
        if not (np.all(np.isfinite(d)) and np.all((d >= 0.0) & (d <= 90.0))):
            raise ValueError("the dip must lie in [0, 90] degrees at every "
                             "depth swept")
        return np.deg2rad(d)

    return radians


def _width_to(radians: Callable[[np.ndarray], np.ndarray], top: float,
              bottom: float, *, rtol: float) -> float:
    """The length down dip from depth `top` to `bottom`: the integral of
    dz / sin(dip)."""
    if not bottom > top:
        raise ValueError(f"bottom ({bottom!r}) must lie below top ({top!r})")
    z = np.linspace(top, bottom, 257)
    if not np.all(np.sin(radians(z)) > 0.0):
        raise ValueError("the dip vanishes between top and bottom, which it "
                         "then never reaches: give the width instead")
    if np.ptp(radians(z)) == 0.0:
        return float((bottom - top) / np.sin(radians(z[:1]))[0])
    value, _ = quad(lambda x: 1.0 / np.sin(radians(np.asarray(x))), top, bottom,
                    epsabs=0.0, epsrel=max(rtol, 1e-13), limit=200)
    return float(value)


def _sphere_radius(frame: Frame, what: str) -> float:
    """The radius of the frame's reference sphere, or the refusal."""
    radius = getattr(frame, "radius", None)
    if radius is None:
        raise TypeError(f"{what} lies on a reference sphere, and "
                        f"{type(frame).__name__} has none")
    return float(radius)


def _direction(point: ArrayLike, what: str) -> np.ndarray:
    """The unit vector of a (latitude, longitude) in degrees."""
    p = np.asarray(point, dtype=float)
    if p.shape != (2,) or not np.all(np.isfinite(p)):
        raise ValueError(f"{what} is one (latitude, longitude), got {point!r}")
    return cartesian_points(1.0, *from_geographic(p[0], p[1]))


@runtime_checkable
class Trace(Protocol):
    """A curve on a frame's reference surface, travelled from its start to
    its end: the upper edge, at the surface, of a swept patch."""

    def surface(self, frame: Frame, s: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """(X, t) at the fractions `s` in [0, 1] of the way along, by the
        trace's own parameter: the Cartesian points on the frame's
        reference surface and the unit horizontal directions of travel
        there, each (len(s), 3)."""
        ...


class GreatCircle:
    """The shorter great-circle arc from `start` to `end`, each (latitude,
    longitude) in degrees, on a geographic frame's reference sphere; the
    fraction along is the fraction of its length."""

    def __init__(self, start: ArrayLike, end: ArrayLike) -> None:
        self._e1 = _direction(start, "start")
        self._e2 = _direction(end, "end")
        self._angle = float(np.arctan2(np.linalg.norm(np.cross(self._e1, self._e2)),
                                       self._e1 @ self._e2))
        if not 0.0 < self._angle < np.pi:
            raise ValueError("a great-circle arc needs two distinct points that "
                             "are not antipodal")

    def surface(self, frame: Frame, s: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """The arc on the frame's reference sphere, and its direction."""
        radius = _sphere_radius(frame, "a great circle")
        s = np.asarray(s, dtype=float)[:, None]
        o = self._angle
        e = (np.sin((1.0 - s) * o) * self._e1 + np.sin(s * o) * self._e2) / np.sin(o)
        t = _unit(-np.cos((1.0 - s) * o) * self._e1 + np.cos(s * o) * self._e2)
        return radius * e, t

    def __repr__(self) -> str:
        return f"GreatCircle({np.rad2deg(self._angle):g} degrees)"


class SmallCircle:
    """An arc of the small circle `radius` degrees from `pole` (latitude,
    longitude), from the azimuth `start` to the azimuth `end` about the
    pole, in degrees clockwise from north there (north at the pole as
    `GeographicFrame.basis` gives it), travelled from start to end: the
    dip side, to the right, faces the pole when the azimuth increases.
    The fraction along is the fraction of the turn."""

    def __init__(self, pole: ArrayLike, radius: float, start: float,
                 end: float) -> None:
        self._pole = _direction(pole, "pole")
        lat, lon = (float(x) for x in np.asarray(pole, dtype=float))
        basis = GeographicFrame(1.0).basis([lat, lon, 1.0])
        self._east, self._north = basis[:, 0], basis[:, 1]
        self._radius = np.deg2rad(_real(radius, "radius"))
        if not 0.0 < self._radius < np.pi:
            raise ValueError(f"radius lies strictly between 0 and 180 degrees, "
                             f"got {radius!r}")
        self._start = np.deg2rad(_real(start, "start"))
        self._end = np.deg2rad(_real(end, "end"))
        if self._start == self._end:
            raise ValueError("an arc needs two distinct azimuths")

    @classmethod
    def through(cls, pole: ArrayLike, start: ArrayLike, end: ArrayLike, *,
                rtol: float = 1e-9) -> SmallCircle:
        """The arc about `pole` from the point `start` to the point `end`
        (each (latitude, longitude)), the shorter way round; the two must
        lie equally far from the pole, to `rtol` of that distance."""
        p, e1, e2 = (_direction(x, w) for x, w in ((pole, "pole"),
                                                  (start, "start"), (end, "end")))
        g1, g2 = (float(np.arctan2(np.linalg.norm(np.cross(p, e)), p @ e))
                  for e in (e1, e2))
        if abs(g1 - g2) > _positive(rtol, "rtol") * g1:
            raise ValueError(f"start and end lie {np.rad2deg(g1):.9g} and "
                             f"{np.rad2deg(g2):.9g} degrees from the pole: not "
                             "on one small circle about it")
        lat, lon = (float(x) for x in np.asarray(pole, dtype=float))
        basis = GeographicFrame(1.0).basis([lat, lon, 1.0])
        a1, a2 = (float(np.rad2deg(np.arctan2(e @ basis[:, 0], e @ basis[:, 1])))
                  for e in (e1, e2))
        a2 = a1 + (a2 - a1 + 180.0) % 360.0 - 180.0
        return cls(pole, float(np.rad2deg(0.5 * (g1 + g2))), a1, a2)

    def surface(self, frame: Frame, s: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """The arc on the frame's reference sphere, and its direction."""
        radius = _sphere_radius(frame, "a small circle")
        alpha = (self._start + np.asarray(s, dtype=float)
                 * (self._end - self._start))[:, None]
        sb, cb = np.sin(self._radius), np.cos(self._radius)
        e = cb * self._pole + sb * (np.cos(alpha) * self._north
                                    + np.sin(alpha) * self._east)
        turn = np.sign(self._end - self._start)
        t = turn * (-np.sin(alpha) * self._north + np.cos(alpha) * self._east)
        return radius * e, t

    def __repr__(self) -> str:
        return (f"SmallCircle({np.rad2deg(self._radius):g} degrees from the "
                f"pole, azimuth {np.rad2deg(self._start):g} to "
                f"{np.rad2deg(self._end):g})")


class Polyline:
    """A smooth curve through points given in a frame's horizontal
    coordinates (latitude and longitude, or x and y), in order: the points
    are placed on the frame's reference surface, the interpolating spline
    through them in Cartesian space against cumulative chord length (cubic,
    not-a-knot, from four points; the parabola through three; the chord
    between two) is taken back onto the reference surface, and the
    fraction along is the fraction of the chord length."""

    def __init__(self, points: ArrayLike) -> None:
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 2:
            raise ValueError(f"a polyline is two or more points (n, 2), got "
                             f"shape {p.shape}")
        if not np.all(np.isfinite(p)):
            raise ValueError("a polyline's points must be finite")
        self._points = _readonly(p)

    @property
    def points(self) -> np.ndarray:
        """The points, in the frame's horizontal coordinates."""
        return self._points

    def surface(self, frame: Frame, s: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """The curve on the frame's reference surface, and its direction."""
        P = frame.cartesian(frame.below(self._points[:, 0], self._points[:, 1], 0.0))
        chord = np.linalg.norm(np.diff(P, axis=0), axis=-1)
        if not np.all(chord > 0.0):
            raise ValueError("a polyline's consecutive points must differ")
        knots = np.concatenate([[0.0], np.cumsum(chord)]) / np.sum(chord)
        spline = make_interp_spline(knots, P, k=min(3, P.shape[0] - 1), axis=0)
        s = np.asarray(s, dtype=float)
        c = frame.coordinates(spline(s))
        on = frame.below(c[:, 0], c[:, 1], 0.0)
        up = frame.basis(on)[..., 2]
        d = spline.derivative()(s)
        return frame.cartesian(on), _unit(d - _dot(d, up)[:, None] * up)

    def __repr__(self) -> str:
        return f"Polyline({self._points.shape[0]} points)"


def finite_block(values: ArrayLike) -> tuple[slice, slice]:
    """The largest block (rows, columns) of a two-dimensional array that
    holds no NaN or infinity, by its number of entries, the first found
    scanning down the rows where several tie: the crop of a grid with
    holes, for
    `StructuredPatch.from_depths`.  A block of one entry may be all there
    is; none at all is refused."""
    ok = np.isfinite(np.asarray(values, dtype=float))
    if ok.ndim != 2:
        raise ValueError(f"expected a two-dimensional array, got {ok.shape}")
    rows, cols = ok.shape
    heights = np.zeros(cols, dtype=int)
    best, block = 0, None
    for i in range(rows):
        heights = np.where(ok[i], heights + 1, 0)
        stack: list[int] = []
        for j in range(cols + 1):
            h = heights[j] if j < cols else 0
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = int(heights[top]) * (j - left)
                if area > best:
                    best = area
                    block = (slice(i - int(heights[top]) + 1, i + 1), slice(left, j))
            stack.append(j)
    if block is None:
        raise ValueError("the array holds no finite value")
    return block


# -- the fault ------------------------------------------------------------------------

class Fault:
    """Patches in one frame, their slip on one time axis (module
    docstring).

    `support` is one of `surfacefile.TIME_SUPPORTS` and
    `time_interpolation` the rule between nodes, by the surface file's
    rules; `time` is the axis of a `node` or `interval` support and
    `instant` that of an `instant` support, and an axis the support
    does not use is refused.  Every patch's slip has the slices the
    support implies: one for `static`, which takes no date.  A date
    within `time_rtol` of a stored time, relative to the scale of the
    axis, is that time (`surfacefile.time_rows`).  `scales` says what the
    numbers are in, and is not consulted.
    """

    def __init__(self, patches: Sequence[Patch], *, support: str = "static",
                 time_interpolation: str = "none", time: ArrayLike | None = None,
                 instant: ArrayLike | None = None,
                 scales: Scales = Scales.SI) -> None:
        patches = tuple(patches)
        if not patches:
            raise ValueError("a fault has at least one patch")
        for k, p in enumerate(patches):
            if not isinstance(p, Patch):
                raise TypeError(f"patch {k} is not a Patch: {type(p).__name__}")
        frame = patches[0].frame
        if any(p.frame != frame for p in patches[1:]):
            raise ValueError("every patch of a fault is in the same frame")
        if not isinstance(scales, Scales):
            raise TypeError(f"expected Scales, got {type(scales).__name__}")
        check_time_rule(support, time_interpolation, name="slip")
        t, s = time_axis(time, "time"), time_axis(instant, "instant")
        uses = {"static": (), "node": ("time",), "interval": ("time",),
                "instant": ("instant",)}[support]
        for axis, given in (("time", t), ("instant", s)):
            if given is not None and axis not in uses:
                raise ValueError(f"a {support!r} slip uses no {axis} axis")
        lead = leading_shape(support, time=t, instant=s, name="slip")
        nslices = lead[0] if lead else 1
        for k, p in enumerate(patches):
            if p.slip.nslices != nslices:
                raise ValueError(f"patch {k} has {p.slip.nslices} slices of "
                                 f"slip; a {support!r} fault has {nslices}")
        self._patches, self._frame = patches, frame
        self._support, self._rule = support, time_interpolation
        self._time = None if t is None else _readonly(t)
        self._instant = None if s is None else _readonly(s)
        self._nslices = nslices
        self._scales = scales

    @property
    def patches(self) -> tuple[Patch, ...]:
        """The patches, in order."""
        return self._patches

    @property
    def frame(self) -> Frame:
        """The frame every patch is in."""
        return self._frame

    @property
    def support(self) -> str:
        """How the slices sit in time, one of `surfacefile.TIME_SUPPORTS`."""
        return self._support

    @property
    def time_interpolation(self) -> str:
        """The rule between nodes, one of `surfacefile.TIME_METHODS`."""
        return self._rule

    @property
    def time(self) -> np.ndarray | None:
        """The time nodes of a `node` or `interval` slip."""
        return self._time

    @property
    def instant(self) -> np.ndarray | None:
        """The instants of an `instant` slip."""
        return self._instant

    @property
    def nslices(self) -> int:
        """The slices every patch's slip has."""
        return self._nslices

    @property
    def scales(self) -> Scales:
        """What one unit of each number is, in SI."""
        return self._scales

    def rows(self, date: float | None, *, extrapolation: str = "refuse",
             time_rtol: float = TIME_RTOL) -> TimeRows:
        """The slices a date reads and their weights, ((k0, w0), (k1, w1)),
        by `surfacefile.time_rows`, `time_rtol` the date's tolerance on a
        stored time; a static slip reads its one slice, and refuses a
        date."""
        if self._support == "static" and date is not None:
            raise ValueError("the slip is static, the same at every date, and "
                             f"takes no date; got {date!r}")
        rows = time_rows(self._support, self._rule, date, time=self._time,
                         instant=self._instant, extrapolation=extrapolation,
                         name="slip", time_rtol=time_rtol)
        return ((0, 1.0), (0, 0.0)) if rows is None else rows

    def slip_vector(self, patch: int, u: ArrayLike, v: ArrayLike, *,
                    date: float | None = None, extrapolation: str = "refuse",
                    basis: str = "ambient",
                    time_rtol: float = TIME_RTOL) -> np.ndarray:
        """The slip of patch `patch` at (u, v) and `date`, in `basis`
        (one of `BASES`), (..., 3)."""
        (k0, w0), (k1, w1) = self.rows(date, extrapolation=extrapolation,
                                       time_rtol=time_rtol)
        p = self._patches[patch]
        out = w0 * p.slip_vector(u, v, index=k0, basis=basis)
        if w1 != 0.0:
            out = out + w1 * p.slip_vector(u, v, index=k1, basis=basis)
        return out

    def _counts(self, n_u: int | Sequence[int], n_v: int | Sequence[int]
                ) -> list[list[int]]:
        """The cells each way on each patch: one count for every patch, or
        one a patch."""
        counts = []
        for n, what in ((n_u, "n_u"), (n_v, "n_v")):
            seq = list(n) if isinstance(n, Sequence) else [n] * len(self._patches)
            if len(seq) != len(self._patches):
                raise ValueError(f"{what} gives {len(seq)} counts for "
                                 f"{len(self._patches)} patches")
            counts.append(seq)
        return counts

    def quadrature(self, n_u: int | Sequence[int], n_v: int | Sequence[int], *,
                   points: int = 1, date: float | None = None,
                   extrapolation: str = "refuse",
                   time_rtol: float = TIME_RTOL) -> Quadrature:
        """Every patch's quadrature (`parameter_rule`), with the slip at
        `date`: `n_u` and `n_v` are the cells each way on every patch,
        or one count a patch."""
        (k0, w0), (k1, w1) = self.rows(date, extrapolation=extrapolation,
                                       time_rtol=time_rtol)
        counts = self._counts(n_u, n_v)
        parts, slips = [], []
        for p, nu, nv in zip(self._patches, *counts):
            g = _geometry(p, nu, nv, points)
            slip = w0 * g.slip(p, k0)
            if w1 != 0.0:
                slip = slip + w1 * g.slip(p, k1)
            parts.append(g)
            slips.append(slip)
        return Quadrature(**_joined(parts), slip=np.concatenate(slips))

    def slice_quadrature(self, n_u: int | Sequence[int], n_v: int | Sequence[int],
                         *, points: int = 1) -> SliceQuadrature:
        """Every patch's quadrature (`parameter_rule`) with the slip of
        every slice: the points, normals, weights, parameters and patches
        once, as `quadrature` gives them, and each slice's Cartesian slip
        at them, (nslices, P, 3).  The slip at a date is
        `SliceQuadrature.slip_of(self.rows(date))`, what `quadrature`
        gives at that date.  `n_u` and `n_v` are as `quadrature` takes
        them; the geometry and the fault basis are formed once a point,
        every slice's components read on them."""
        counts = self._counts(n_u, n_v)
        parts, slips = [], []
        for p, nu, nv in zip(self._patches, *counts):
            g = _geometry(p, nu, nv, points)
            parts.append(g)
            slips.append(np.stack([g.slip(p, s) for s in range(self._nslices)]))
        return SliceQuadrature(**_joined(parts), slip=np.concatenate(slips, axis=1))

    def locate(self, points: ArrayLike, *, given: str = "cartesian",
               rtol: float = 1e-10, iterations: int = 50,
               strict: bool = True) -> Location:
        """Each point's nearest patch (the smallest distance, the first
        patch on a tie) and its location on it (`PatchBase.locate`, whose
        arguments these are); a search that did not converge on the chosen
        patch is refused where `strict`, and flagged otherwise."""
        found = [p.locate(points, given=given, rtol=rtol, iterations=iterations,
                          strict=False) for p in self._patches]
        k = np.argmin(np.stack([np.abs(f.distance) for f in found]), axis=0)

        def chosen(name: str) -> np.ndarray:
            stack = np.stack([getattr(f, name) for f in found])
            return np.take_along_axis(stack, k[None], axis=0)[0]

        converged = chosen("converged")
        if strict and not np.all(converged):
            bad = int(np.count_nonzero(~converged))
            raise ValueError(f"locate did not converge for {bad} of "
                             f"{converged.size} points; raise iterations, or "
                             "pass strict=False to flag them")
        return Location(u=chosen("u"), v=chosen("v"), distance=chosen("distance"),
                        inside=chosen("inside"), converged=converged, patch=k)

    def slip_at(self, points: ArrayLike, *, max_distance: float,
                given: str = "cartesian", date: float | None = None,
                extrapolation: str = "refuse", basis: str = "ambient",
                rtol: float = 1e-10, time_rtol: float = TIME_RTOL) -> np.ndarray:
        """The slip at points near the fault, (..., 3): each point located
        (`locate`) and the slip of its nearest patch read there, at `date`
        and in `basis` (one of `BASES`, the ambient one at the nearest
        point of the patch).  A point farther than `max_distance` (in the
        length scale) from the fault is refused, all of them with it; a
        point within it but beyond an edge reads the edge.  `rtol` is
        `locate`'s tolerance, `time_rtol` the date's (`rows`)."""
        limit = _real(max_distance, "max_distance")
        if limit < 0.0:
            raise ValueError(f"max_distance must not be negative, got "
                             f"{max_distance!r}")
        where = self.locate(points, given=given, rtol=rtol)
        far = np.abs(where.distance) > limit
        if np.any(far):
            raise ValueError(f"{int(np.count_nonzero(far))} of {far.size} points "
                             f"lie farther than max_distance = {limit:g} from the "
                             f"fault, the farthest at "
                             f"{float(np.max(np.abs(where.distance))):g}")
        out = np.empty(where.u.shape + (3,))
        for k in np.unique(where.patch):
            m = where.patch == k
            out[m] = self.slip_vector(int(k), where.u[m], where.v[m], date=date,
                                      extrapolation=extrapolation, basis=basis,
                                      time_rtol=time_rtol)
        return out

    def __repr__(self) -> str:
        n = len(self._patches)
        return (f"Fault({n} patch{'' if n == 1 else 'es'}, {self._support}"
                f"{'' if self._support == 'static' else f' {self._rule}'}, "
                f"{self._nslices} slice{'' if self._nslices == 1 else 's'}, "
                f"{self._frame!r})")


# -- under a mapping ----------------------------------------------------------------

#: The control grids `pull_back` tries in turn, nodes each way, when none is given.
PULL_BACK_CONTROLS = (5, 9, 17, 33, 65, 129)


@dataclass(frozen=True, eq=False)
class Preimage:
    """A patch pulled back through a mapping (`pull_back`): the surface of
    the reference body that the mapping takes onto the patch.

    `patch` is the preimage, its positions in the patch's frame but
    reference ones, carrying no slip (the slip stays on the fault);
    `controls` the control nodes (n_u, n_v) it was built through, None
    where the mapping is the identity and the patch is its own preimage;
    `offset` the largest distance from the patch of the preimage mapped
    forward found at the control nodes and between them (`pull_back`),
    in the patch's lengths: zero under the identity.
    """

    patch: Patch
    controls: tuple[int, int] | None
    offset: float


#: The points per cell each way at which `pull_back` measures the offset.
_OFFSET_SAMPLES = 4


def _offset(patch: Patch, preimage: Patch, mapping: Mapping, n_u: int,
            n_v: int) -> float:
    """The largest distance from `patch` of `preimage` mapped forward, at
    the nodes of an n_u by n_v grid and `_OFFSET_SAMPLES` points each way
    across each cell between them."""
    k = _OFFSET_SAMPLES
    U, V = np.meshgrid(_nodes(k * (n_u - 1) + 1), _nodes(k * (n_v - 1) + 1),
                       indexing="ij")
    x = np.asarray(mapping(preimage.cartesian(U, V).reshape(-1, 3)), dtype=float)
    return float(np.max(np.abs(patch.locate(x, strict=False).distance)))


def pull_back(patch: Patch, mapping: Mapping, *,
              controls: int | tuple[int, int] | None = None,
              rtol: float = 1e-6) -> Preimage:
    """The preimage of a patch under a mapping m of a reference body
    (`planetmodel.mapping`): the bicubic `StructuredPatch` through
    X = m^-1(x(u, v)) at the nodes of a uniform grid of `controls` by
    `controls` nodes over the square (or (n_u, n_v)), at least four each
    way, by the mapping's own `inverse`.

    Mapped forward the preimage is the patch at the control nodes, to
    the inverse's tolerance, and between them to the interpolation
    error, which falls as the fourth power of the spacing for a smooth
    patch under a smooth mapping; its `offset` is measured, at the
    nodes and on a grid four times as fine between them, mapped forward
    and located on the patch: the largest distance found, not a bound.
    Without `controls` the grids of `PULL_BACK_CONTROLS` are tried in
    turn until the offset is at most `rtol` times the patch's size (the
    longer diagonal between its corners); none meeting it is refused.
    A patch crossing a knot of the mapping, where its gradient jumps (an
    interface of a stack of reliefs whose slope changes there), has a
    preimage creased along the crossing, which the bicubic follows only
    to first or second order in the spacing: it meets a small `rtol`
    with many control nodes or none of these, and is given a larger
    `rtol` or its `controls`.  Under the identity (`is_identity`) the
    patch is its own preimage, exactly.  A mapping that does not invert
    itself is refused.
    """
    if not isinstance(patch, Patch):
        raise TypeError(f"expected a Patch, got {type(patch).__name__}")
    if bool(getattr(mapping, "is_identity", False)):
        return Preimage(patch=patch, controls=None, offset=0.0)
    rtol = _positive(rtol, "rtol")
    inverse = getattr(mapping, "inverse", None)
    probe = patch.cartesian(0.0, 0.0)[None]
    try:
        if not callable(inverse):
            raise NotImplementedError
        inverse(probe)
    except NotImplementedError:
        raise ValueError(f"the mapping {mapping!r} does not invert itself, so a "
                         "fault cannot be pulled back through it") from None
    target = size = None
    if controls is None:
        grids = [(n, n) for n in PULL_BACK_CONTROLS]
        lo, hi = SQUARE
        corners = patch.cartesian(np.array([lo, hi, lo, hi]),
                                  np.array([lo, hi, hi, lo]))
        size = max(float(np.linalg.norm(corners[1] - corners[0])),
                   float(np.linalg.norm(corners[3] - corners[2])))
        target = rtol * size
    else:
        pair = (controls, controls) if isinstance(controls, numbers.Integral) \
            else tuple(controls)
        if len(pair) != 2:
            raise ValueError(f"controls is a count or (n_u, n_v), got {controls!r}")
        grids = [tuple(_count(n, "controls", least=MIN_NODES["cubic"])
                       for n in pair)]
    zero = Slip(np.zeros((1, 1, 3)))
    for n_u, n_v in grids:
        U, V = np.meshgrid(_nodes(n_u), _nodes(n_v), indexing="ij")
        X = np.asarray(inverse(patch.cartesian(U, V)), dtype=float)
        built = StructuredPatch(patch.frame, patch.frame.coordinates(X), slip=zero,
                                interpolation="cubic")
        offset = _offset(patch, built, mapping, n_u, n_v)
        if target is None or offset <= target:
            return Preimage(patch=built, controls=(n_u, n_v), offset=offset)
    raise ValueError(f"the preimage is {offset:.3g} off the patch with {n_u} by "
                     f"{n_v} control nodes, short of rtol = {rtol:g} of its size "
                     f"({size:.3g}); give controls, or a larger rtol")
