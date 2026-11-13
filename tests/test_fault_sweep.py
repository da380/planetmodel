"""Structured patches built for large curved faults: swept from a trace
by a dip profile, or through a grid of depths.

The oracles: a straight trace with a constant dip against `Plane`,
exactly in a Cartesian frame and to first order in the length over the
radius on a sphere; a small-circle trace with a constant dip against the
closed form of its sphere's cone (the surface of revolution about the
pole of the logarithmic spiral that keeps a constant angle with the
local horizontal): positions, normal and area; a steepening dip against
an integration of the profile in (depth, angle) written apart; the
signs of a thrust on a subduction interface; a polyline through points
of a small circle against the small circle; and the depth-grid
constructor against a patch built directly from the same points.
"""
import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from planetmodel import testing
from planetmodel.fault import (CartesianFrame, GeographicFrame, GreatCircle, Plane,
                               Polyline, Slip, SmallCircle, StructuredPatch,
                               finite_block)

R = 6371e3
EARTH = GeographicFrame(R)
LOCAL = CartesianFrame()
ZERO = Slip(np.zeros((1, 1, 3)))
RNG = np.random.default_rng(14)
U_, V_ = RNG.uniform(-0.5, 0.5, (2, 400))


def test_a_straight_line_with_constant_dip_is_the_plane():
    a, b = np.array([10.0, -5.0]), np.array([40.0, 25.0])
    dip, width, top = 35.0, 20.0, 2.0
    patch = StructuredPatch.swept(LOCAL, Polyline([a, b]), dip=dip, width=width,
                                  top=top, n_u=5, n_v=4, slip=ZERO)
    strike = np.rad2deg(np.arctan2(*(b - a)))
    # the plane's centroid, by hand: the line's middle at depth `top`, and
    # half the width down dip, to the right of the strike
    phi, delta = np.deg2rad(strike), np.deg2rad(dip)
    right = np.array([np.cos(phi), -np.sin(phi), 0.0])
    down = np.cos(delta) * right - np.sin(delta) * np.array([0.0, 0.0, 1.0])
    centroid = np.r_[(a + b) / 2, -top] + 0.5 * width * down
    plane = Plane(LOCAL, centroid, strike=strike, dip=dip,
                  length=np.linalg.norm(b - a), width=width, slip=ZERO)
    # measured: 2.8e-14 on coordinates of tens, normals 3.3e-15
    assert np.allclose(patch.cartesian(U_, V_), plane.cartesian(U_, V_), rtol=0,
                       atol=1e-12)
    assert np.allclose(patch.normal(U_, V_), plane.normal(U_, V_), rtol=0,
                       atol=1e-13)
    with_bottom = StructuredPatch.swept(LOCAL, Polyline([a, b]), dip=dip,
                                        bottom=top + width * np.sin(delta),
                                        top=top, n_u=5, n_v=4, slip=ZERO)
    assert np.allclose(with_bottom.cartesian(U_, V_), plane.cartesian(U_, V_),
                       rtol=0, atol=1e-12)


def test_a_great_circle_is_the_plane_to_first_order():
    # along the equator, east, dipping 30 degrees south: (0, 0) has east
    # +y, south -z and up +x
    errors = []
    for L in (10e3, 20e3, 40e3, 80e3):
        width = L / 2
        half = np.rad2deg(L / R) / 2
        patch = StructuredPatch.swept(EARTH, GreatCircle([0.0, -half], [0.0, half]),
                                      dip=30.0, width=width, n_u=9, n_v=9,
                                      slip=ZERO)
        down = np.array([-0.5, 0.0, -np.sqrt(3.0) / 2.0])
        centroid = np.array([R, 0.0, 0.0]) + 0.5 * width * down
        plane = Plane(EARTH, EARTH.coordinates(centroid), strike=90.0, dip=30.0,
                      length=L, width=width, slip=ZERO)
        errors.append([
            np.max(np.abs(patch.cartesian(U_, V_) - plane.cartesian(U_, V_))) / L,
            np.max(np.abs(patch.normal(U_, V_) - plane.normal(U_, V_))),
            abs(patch.quadrature(8, 8, points=4).weights.sum() / (L * width) - 1)])
    errors = np.array(errors) / (np.array([10e3, 20e3, 40e3, 80e3]) / R)[:, None]
    # measured, over L / R: positions 0.166 L, normals 0.432, the area
    # 0.125 (the length along strike shrinking with depth, L / 8R): first
    # order, the sweep following the sphere where the plane does not
    assert np.allclose(errors, errors[0], rtol=0.02)
    assert errors[0, 0] < 0.2 and errors[0, 1] < 0.45
    assert abs(errors[0, 2] - 0.125) < 0.002


# a trench arc: 10 degrees from a pole at (40 N, 145 E), azimuths 200 to
# 260 (increasing: the interface dips towards the pole), 20 degrees to
# 60 km
_POLE, _BETA, _A1, _A2, _DIP, _BOTTOM = (40.0, 145.0), 10.0, 200.0, 260.0, 20.0, 60e3


def _cone(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions and normals of the swept small circle in closed form: at
    length w down dip the radius is R - w sin(dip) and the angle turned
    from the trace, about the centre, is -ln(1 - w sin(dip) / R) / tan(dip),
    towards the pole."""
    d, beta = np.deg2rad(_DIP), np.deg2rad(_BETA)
    p = EARTH.cartesian([_POLE[0], _POLE[1], 1.0])
    B = GeographicFrame(1.0).basis([_POLE[0], _POLE[1], 1.0])
    alpha = np.deg2rad(_A1 + (u + 0.5) * (_A2 - _A1))[..., None]
    w = (v + 0.5) * _BOTTOM / np.sin(d)
    theta = -np.log(1.0 - w * np.sin(d) / R) / np.tan(d)
    g = (beta - theta)[..., None]
    ring = np.cos(alpha) * B[:, 1] + np.sin(alpha) * B[:, 0]
    e = np.cos(g) * p + np.sin(g) * ring
    towards = np.sin(g) * p - np.cos(g) * ring
    return (R - w * np.sin(d))[..., None] * e, np.sin(d) * towards + np.cos(d) * e


def _cone_area() -> float:
    """The area: the turn times the integral of R^2 exp(-2 theta tan(dip))
    sin(beta - theta) / cos(dip) d theta."""
    d, beta = np.deg2rad(_DIP), np.deg2rad(_BETA)
    top = -np.log(1.0 - _BOTTOM / R) / np.tan(d)
    a, b = -2.0 * np.tan(d), -1.0

    def F(t: float) -> float:
        return (np.exp(a * t) * (a * np.sin(b * t + beta) - b * np.cos(b * t + beta))
                / (a * a + b * b))

    return np.deg2rad(_A2 - _A1) / np.cos(d) * R ** 2 * (F(top) - F(0.0))


def test_a_small_circle_with_constant_dip_is_its_cone():
    trace = SmallCircle(_POLE, _BETA, _A1, _A2)
    errors = []
    for n_u, n_v in ((9, 5), (17, 9), (33, 17)):
        patch = StructuredPatch.swept(EARTH, trace, dip=_DIP, bottom=_BOTTOM,
                                      n_u=n_u, n_v=n_v, slip=ZERO)
        U, V = np.meshgrid(np.linspace(-0.5, 0.5, n_u), np.linspace(-0.5, 0.5, n_v),
                           indexing="ij")
        # the control nodes are the cone's: measured 9.5e-16 of R
        assert np.allclose(patch.cartesian(U, V), _cone(U, V)[0], rtol=0,
                           atol=1e-12 * R)
        X, n = _cone(U_, V_)
        width = _BOTTOM / np.sin(np.deg2rad(_DIP))
        errors.append([np.max(np.abs(patch.cartesian(U_, V_) - X)) / width,
                       np.max(np.abs(patch.normal(U_, V_) - n)),
                       abs(patch.quadrature(n_u - 1, n_v - 1, points=4).weights.sum()
                           / _cone_area() - 1.0)])
    errors = np.array(errors)
    orders = np.log2(errors[:-1] / errors[1:])
    # measured: positions 4.5e-5, 2.8e-6, 1.6e-7 of the width (order 4),
    # normals 5.2e-5, 6.3e-6, 7.8e-7 (order 3), area 1.3e-6, 1.7e-8, 1e-9
    assert np.all(orders[:, 0] > 3.8) and np.all(orders[:, 1] > 2.8)
    assert errors[-1, 2] < 1e-8
    # the same arc through its end points
    ends = [EARTH.coordinates(_cone(np.array(s), np.array(-0.5))[0])[:2]
            for s in (-0.5, 0.5)]
    again = StructuredPatch.swept(EARTH, SmallCircle.through(_POLE, *ends),
                                  dip=_DIP, bottom=_BOTTOM, n_u=9, n_v=5, slip=ZERO)
    assert np.allclose(again.cartesian(U_, V_), _cone(U_, V_)[0], rtol=0,
                       atol=1e-4 * width)


def _steepening(z: np.ndarray) -> np.ndarray:
    """Ten degrees at the trench, steepening towards 35 below 100 km."""
    return 10.0 + 25.0 * np.tanh(np.asarray(z) / 80e3)


def test_a_steepening_dip_against_its_profile_integrated_apart():
    trace = GreatCircle([38.0, 143.0], [36.0, 141.5])
    patch = StructuredPatch.swept(EARTH, trace, dip=_steepening, bottom=200e3,
                                  n_u=5, n_v=9, slip=ZERO)
    # the profile in (depth, angle about the centre) along the length w:
    # dz/dw = sin(dip), d theta/dw = cos(dip) / (R - z)
    width = quad(lambda z: 1.0 / np.sin(np.deg2rad(_steepening(z))), 0.0, 200e3,
                 epsabs=0, epsrel=1e-13)[0]
    solved = solve_ivp(lambda w, y: [np.sin(np.deg2rad(_steepening(y[0]))),
                                     np.cos(np.deg2rad(_steepening(y[0])))
                                     / (R - y[0])],
                       (0.0, width), [0.0, 0.0], method="DOP853", rtol=1e-13,
                       atol=1e-9, t_eval=np.linspace(0.0, width, 9))
    z, theta = solved.y
    X0, t = trace.surface(EARTH, np.linspace(0.0, 1.0, 5))
    up = X0 / R
    right = np.cross(t, up)
    want = (R - z)[None, :, None] * (np.cos(theta)[None, :, None] * up[:, None]
                                     + np.sin(theta)[None, :, None] * right[:, None])
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 9),
                       indexing="ij")
    # measured: 9.3e-12 of R (6e-5 m), the bottom reached to 1e-6 m
    assert np.allclose(patch.cartesian(U, V), want, rtol=0, atol=1e-10 * R)
    assert abs(R - np.linalg.norm(patch.cartesian(0.0, 0.5)) - 200e3) < 1e-10 * R
    testing.check_patch(patch)


def test_a_thrust_on_a_subduction_interface():
    trace = GreatCircle([38.0, 143.0], [36.0, 141.5])
    patch = StructuredPatch.swept(EARTH, trace, dip=_steepening, bottom=150e3,
                                  n_u=5, n_v=7, slip=Slip.from_rake(
                                      90.0, np.ones((5, 7)), rule="linear"))
    u, v = np.linspace(-0.4, 0.4, 5), np.linspace(-0.4, 0.4, 5)
    X = patch.cartesian(u, v)
    up = EARTH.basis(EARTH.coordinates(X))[..., 2]
    # the trench is up dip: towards the trace from below, horizontally
    towards_trench = -np.cross(patch.fault_basis(u, v)[..., 0], up)
    slip = patch.slip_vector(u, v, basis="cartesian")
    assert np.all(np.einsum("pi,pi->p", patch.normal(u, v), up) > 0.0)
    assert np.all(np.einsum("pi,pi->p", slip, up) > 0.0)
    assert np.all(np.einsum("pi,pi->p", slip, towards_trench) > 0.0)
    # rake is measured from the u-lines, here the local trace direction,
    # and the rule reads fault components: pure dip-slip everywhere, on a
    # grid of nodes or in one cell, and no opening, the vector turning
    # with the surface down the steepening dip and staying in it
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 41), np.linspace(-0.5, 0.5, 41),
                       indexing="ij")
    single = StructuredPatch.swept(EARTH, trace, dip=_steepening, bottom=150e3,
                                   n_u=5, n_v=7, slip=Slip.from_rake(90.0, 1.0))
    for p in (patch, single):
        assert np.allclose(p.slip_vector(U, V, basis="fault"), [0.0, 1.0, 0.0],
                           rtol=0, atol=1e-15)
        cart = p.slip_vector(U, V, basis="cartesian")
        # measured: 1.1e-16
        assert np.max(np.abs(np.einsum("...i,...i->...", cart,
                                       p.normal(U, V)))) < 1e-15
        assert np.allclose(np.linalg.norm(cart, axis=-1), 1.0, rtol=0,
                           atol=1e-15)
    cart = single.slip_vector(0.0, np.array([-0.5, 0.5]), basis="cartesian")
    assert abs(np.degrees(np.arccos(cart[0] @ cart[1]))) > 10.0


def test_a_polyline_through_a_small_circle_tends_to_it():
    trace = SmallCircle(_POLE, _BETA, _A1, _A2)
    exact = StructuredPatch.swept(EARTH, trace, dip=_DIP, bottom=_BOTTOM, n_u=9,
                                  n_v=5, slip=ZERO)
    errors = []
    for m in (5, 9, 17, 33):
        X0, _ = trace.surface(EARTH, np.linspace(0.0, 1.0, m))
        points = EARTH.coordinates(X0)[:, :2]
        points[:, 1] -= 360.0 * (points[:, 1] > 180.0)       # any longitudes
        patch = StructuredPatch.swept(EARTH, Polyline(points), dip=_DIP,
                                      bottom=_BOTTOM, n_u=9, n_v=5, slip=ZERO)
        errors.append(np.max(np.abs(patch.cartesian(U_, V_)
                                    - exact.cartesian(U_, V_))))
    errors = np.array(errors)
    # measured at the same (u, v): 315, 41, 5.3 and 0.67 m, order 3, the
    # chord-length parameter differing from the arc's
    assert np.all(np.diff(errors) < 0.0) and errors[-1] < 1.0
    assert np.all(np.log2(errors[:-1] / errors[1:]) > 2.8)


def test_through_a_grid_of_depths():
    trace = SmallCircle(_POLE, _BETA, _A1, _A2)
    swept = StructuredPatch.swept(EARTH, trace, dip=_steepening, bottom=150e3,
                                  n_u=9, n_v=6, slip=ZERO)
    lat, lon, r = np.moveaxis(swept.positions, -1, 0)
    depth = R - r
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    patch = StructuredPatch.from_depths(EARTH, lat, lon, depth, slip=ZERO)
    direct = StructuredPatch(EARTH, EARTH.below(lat, lon, depth), slip=ZERO)
    assert np.allclose(patch.cartesian(U_, V_), direct.cartesian(U_, V_), rtol=0,
                       atol=1e-9)
    assert np.allclose(patch.cartesian(U_, V_), swept.cartesian(U_, V_), rtol=0,
                       atol=1e-6)
    turned = StructuredPatch.from_depths(EARTH, lat.T, lon.T, depth.T, slip=ZERO,
                                         strike_axis=1)
    assert np.allclose(turned.cartesian(U_, V_), patch.cartesian(U_, V_), rtol=0,
                       atol=1e-9)
    # checked, not reordered: each wrong order refused, naming the axis
    with pytest.raises(ValueError, match="does not increase along axis 1"):
        StructuredPatch.from_depths(EARTH, lat[:, ::-1], lon[:, ::-1],
                                    depth[:, ::-1], slip=ZERO)
    with pytest.raises(ValueError, match="left of the strike as axis 0 runs"):
        StructuredPatch.from_depths(EARTH, lat[::-1], lon[::-1], depth[::-1],
                                    slip=ZERO)
    with pytest.raises(ValueError, match="does not increase along axis 0"):
        StructuredPatch.from_depths(EARTH, lat, lon, depth, slip=ZERO,
                                    strike_axis=1)
    holed = depth.copy()
    holed[0, 0] = holed[7, 4] = np.nan
    with pytest.raises(ValueError, match="2 of the grid's 54 nodes"):
        StructuredPatch.from_depths(EARTH, lat, lon, holed, slip=ZERO)
    rows, cols = finite_block(holed)
    assert (rows, cols) == (slice(1, 7), slice(0, 6))      # 36 of the 52 left
    cropped = StructuredPatch.from_depths(EARTH, lat[rows, cols], lon[rows, cols],
                                          holed[rows, cols], slip=ZERO)
    testing.check_patch(cropped)


def test_finite_block():
    a = np.ones((5, 6))
    assert finite_block(a) == (slice(0, 5), slice(0, 6))
    a[2, 3] = np.nan
    a[0, 0] = np.inf
    rows, cols = finite_block(a)
    assert np.all(np.isfinite(a[rows, cols]))
    assert (rows.stop - rows.start) * (cols.stop - cols.start) == 12
    with pytest.raises(ValueError, match="no finite"):
        finite_block(np.full((2, 2), np.nan))


def test_refusals():
    trace = GreatCircle([38.0, 143.0], [36.0, 141.5])
    kw = dict(slip=ZERO, n_u=4, n_v=4)
    with pytest.raises(ValueError, match="exactly one"):
        StructuredPatch.swept(EARTH, trace, dip=20.0, **kw)
    with pytest.raises(ValueError, match="exactly one"):
        StructuredPatch.swept(EARTH, trace, dip=20.0, bottom=1e3, width=1e3, **kw)
    with pytest.raises(ValueError, match="never reaches"):
        StructuredPatch.swept(EARTH, trace, dip=lambda z: 0.0 * z, bottom=1e3, **kw)
    with pytest.raises(ValueError, match="below top"):
        StructuredPatch.swept(EARTH, trace, dip=20.0, bottom=1e3, top=2e3, **kw)
    with pytest.raises(ValueError, match="dip must lie"):
        StructuredPatch.swept(EARTH, trace, dip=lambda z: 95.0 + 0.0 * z,
                              width=1e3, **kw)
    with pytest.raises(TypeError, match="reference sphere"):
        StructuredPatch.swept(LOCAL, trace, dip=20.0, width=1.0, **kw)
    with pytest.raises(TypeError, match="Trace"):
        StructuredPatch.swept(EARTH, [(0, 0), (1, 1)], dip=20.0, width=1.0, **kw)
    with pytest.raises(ValueError, match="antipodal"):
        GreatCircle([0.0, 0.0], [0.0, 180.0])
    with pytest.raises(ValueError, match="not on one small circle"):
        SmallCircle.through(_POLE, (30.0, 140.0), (20.0, 150.0))
    with pytest.raises(ValueError, match="radius"):
        SmallCircle(_POLE, 0.0, 10.0, 20.0)
    with pytest.raises(ValueError, match="consecutive"):
        StructuredPatch.swept(LOCAL, Polyline([(0, 0), (0, 0), (1, 1)]), dip=20.0,
                              width=1.0, **kw)
    with pytest.raises(ValueError, match="two or more"):
        Polyline([(0.0, 0.0)])
