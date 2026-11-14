"""planetmodel.fault: frames, planes, structured patches, slip in time,
quadrature.

The oracles: a plane's corners against a construction by hand from the
geographic unit vectors; Aki & Richards (2002) for the signs of strike,
dip and rake, their unit normal and slip vector (eq. 4.88, axes north,
east, down) and the moment tensor of their Box 4.4; central differences
of every patch's map (`testing.check_patch`); a structured patch
through a plane's own points giving the plane back; the area and the
integral of a smooth slip over a twisted quadrilateral against scipy's
adaptive quadrature of the closed-form area element; a piece of a
sphere against its closed-form area; the surface file's own reading of
the same slices in time; a plane over a pole and a patch across the
date line.
"""
import numpy as np
import pytest
from scipy.integrate import dblquad

from planetmodel import Scales, testing
from planetmodel.fault import (BASES, SLIP_COMPONENTS, SQUARE, CartesianFrame,
                               Fault, GeographicFrame, Plane, Slip,
                               StructuredPatch, parameter_rule)
from planetmodel.modelfile import FileVariable
from planetmodel.sampling import AngularGrid
from planetmodel.surfacefile import SCHEMA, SurfaceFile, SurfaceHistory

R = 6371e3
EARTH = GeographicFrame(R)
LOCAL = CartesianFrame()


def _enu(latitude: float, longitude: float) -> np.ndarray:
    """East, north and up at a latitude and longitude, written out by hand
    (not through `frames`), as rows."""
    p, l = np.deg2rad(latitude), np.deg2rad(longitude)
    return np.array([[-np.sin(l), np.cos(l), 0.0],
                     [-np.sin(p) * np.cos(l), -np.sin(p) * np.sin(l), np.cos(p)],
                     [np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)]])


def _grid_positions(frame_lat: np.ndarray, frame_lon: np.ndarray,
                    radius: float) -> np.ndarray:
    """Control positions (n_u, n_v, 3): u along the longitudes, v along the
    latitudes."""
    return np.stack(np.broadcast_arrays(frame_lat[None, :], frame_lon[:, None],
                                        radius), axis=-1)


# -- frames -----------------------------------------------------------------------

def test_geographic_frame_by_hand():
    f = GeographicFrame(2.0)
    c = np.array([[30.0, -170.0, 1.5], [-90.0, 10.0, 2.0], [0.0, 0.0, 1.0]])
    X = f.cartesian(c)
    assert np.allclose(X[2], [1.0, 0.0, 0.0], rtol=0, atol=1e-15)
    assert np.allclose(X[1], [0.0, 0.0, -2.0], rtol=0, atol=1e-15)
    lat, lon = np.deg2rad(30.0), np.deg2rad(-170.0)
    assert np.allclose(X[0], 1.5 * np.array([np.cos(lat) * np.cos(lon),
                                             np.cos(lat) * np.sin(lon),
                                             np.sin(lat)]), rtol=0, atol=1e-15)
    back = f.coordinates(X)
    assert np.allclose(back[0], [30.0, 190.0, 1.5], rtol=0, atol=1e-12)
    assert back[1, 0] == -90.0 and np.all((back[:, 1] >= 0) & (back[:, 1] < 360))
    for latitude, longitude in ((0.0, 0.0), (0.0, 90.0), (37.0, 23.0),
                                (-62.0, 211.0)):
        B = f.basis([latitude, longitude, 1.0])
        assert np.allclose(B.T, _enu(latitude, longitude), rtol=0, atol=1e-15)
    assert np.allclose(f.below(10.0, 20.0, 0.5), [10.0, 20.0, 1.5])
    for bad in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError, match="radius"):
            GeographicFrame(bad)
    with pytest.raises(ValueError, match="latitudes"):
        f.cartesian([91.0, 0.0, 1.0])


def test_cartesian_frame():
    f = CartesianFrame(surface=3.0)
    X = np.array([[1.0, 2.0, -3.0], [0.5, 0.0, 9.0]])
    assert np.array_equal(f.cartesian(X), X) and np.array_equal(f.coordinates(X), X)
    assert np.array_equal(f.basis(X), np.broadcast_to(np.eye(3), (2, 3, 3)))
    assert np.allclose(f.below(1.0, 2.0, 4.0), [1.0, 2.0, -1.0])
    assert CartesianFrame() == LOCAL and CartesianFrame(surface=1.0) != LOCAL


# -- a plane --------------------------------------------------------------------------

def _corners(patch: Plane) -> np.ndarray:
    """The corners at (u, v) = (-, -), (+, -), (-, +), (+, +)."""
    lo, hi = SQUARE
    return patch.cartesian([lo, hi, lo, hi], [lo, lo, hi, hi])


def test_plane_corners_by_hand():
    # strike 90 dip 30 at (0, 0): east +y, north +z, up +x; the right of
    # the strike is south, -z; down dip cos30 (-z) - sin30 (+x)
    L, W, depth = 40e3, 20e3, 10e3
    p = Plane.at_depth(EARTH, 0.0, 0.0, depth, strike=90.0, dip=30.0, length=L,
                       width=W, slip=Slip.from_rake(90.0, 1.0))
    c = np.array([R - depth, 0.0, 0.0])
    s = np.array([0.0, 1.0, 0.0])
    d = np.array([-0.5, 0.0, -np.sqrt(3.0) / 2.0])
    want = [c + a * L * s + b * W * d for a, b in
            ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))]
    assert np.allclose(_corners(p), want, rtol=0, atol=1e-9)
    assert np.allclose(_corners(p)[0], [R - depth + 5e3, -20e3, 5e3 * np.sqrt(3.0)],
                       rtol=0, atol=1e-9)
    assert np.allclose(p.normal(0.0, 0.0), [np.sqrt(3.0) / 2.0, 0.0, -0.5],
                       rtol=0, atol=1e-15)

    # a generic place, its basis written out by hand
    lat, lon, strike, dip = 37.0, 23.0, 200.0, 35.0
    p = Plane.at_depth(EARTH, lat, lon, depth, strike=strike, dip=dip, length=L,
                       width=W, slip=Slip.from_rake(0.0, 1.0))
    east, north, up = _enu(lat, lon)
    f, g = np.deg2rad(strike), np.deg2rad(dip)
    s = np.sin(f) * east + np.cos(f) * north
    right = np.cos(f) * east - np.sin(f) * north
    d = np.cos(g) * right - np.sin(g) * up
    c = (R - depth) * up
    want = [c + a * L * s + b * W * d for a, b in
            ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))]
    assert np.allclose(_corners(p), want, rtol=0, atol=1e-8)
    assert np.allclose(p.fault_basis(0.3, -0.2),
                       np.stack([s, -d, np.sin(g) * right + np.cos(g) * up], -1),
                       rtol=0, atol=1e-14)

    # Cartesian: strike from +y towards +x, z up
    p = Plane.at_depth(LOCAL, 100.0, 200.0, 50.0, strike=30.0, dip=60.0,
                       length=8.0, width=4.0, slip=Slip.from_rake(0.0, 1.0))
    h = np.sqrt(3.0) / 2.0
    s = np.array([0.5, h, 0.0])
    d = np.array([0.5 * h, -0.25, -h])
    n = np.array([0.75, -0.5 * h, 0.5])
    want = [np.array([100.0, 200.0, -50.0]) + a * 8.0 * s + b * 4.0 * d
            for a, b in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))]
    assert np.allclose(_corners(p), want, rtol=0, atol=1e-13)
    assert np.allclose(p.normal(0.1, 0.4), n, rtol=0, atol=1e-15)
    assert np.allclose(p.position(0.0, 0.0), p.centroid)


def test_plane_area_is_exact_at_any_density():
    p = Plane.at_depth(EARTH, 37.0, 23.0, 15e3, strike=200.0, dip=35.0,
                       length=120e3, width=40e3, slip=Slip.from_rake(70.0, 3.0))
    for n_u, n_v, points in ((1, 1, 1), (3, 7, 2), (10, 4, 3), (5, 5, 4)):
        q = p.quadrature(n_u, n_v, points=points)
        assert q.size == n_u * n_v * points ** 2
        # measured: 4.4e-16 at worst
        assert abs(q.weights.sum() / (120e3 * 40e3) - 1.0) < 4e-15
        assert np.allclose(q.normals, p.normal(0.0, 0.0), rtol=0, atol=1e-15)
    assert np.allclose(p.area_element([0.1, -0.5], [0.5, 0.2]), 120e3 * 40e3,
                       rtol=1e-15, atol=0)


# -- the signs of strike, dip and rake ------------------------------------------------

def _ned(vector: np.ndarray, latitude: float, longitude: float) -> np.ndarray:
    """A Cartesian vector's (north, east, down) components."""
    east, north, up = _enu(latitude, longitude)
    return np.array([vector @ north, vector @ east, -(vector @ up)])


def test_aki_richards_signs():
    lat, lon = 37.0, 23.0
    for strike in (0.0, 37.0, 90.0, 200.0, 315.0):
        for dip in (0.0, 15.0, 45.0, 72.0, 90.0):
            for rake in (-90.0, 0.0, 30.0, 90.0, 180.0, -135.0):
                p = Plane.at_depth(EARTH, lat, lon, 10e3, strike=strike, dip=dip,
                                   length=10e3, width=5e3,
                                   slip=Slip.from_rake(rake, 1.0))
                f, g, l = np.deg2rad([strike, dip, rake])
                # Aki & Richards (2002), eq. (4.88): x north, y east, z down
                n_ar = np.array([-np.sin(g) * np.sin(f), np.sin(g) * np.cos(f),
                                 -np.cos(g)])
                d_ar = np.array([np.cos(l) * np.cos(f)
                                 + np.cos(g) * np.sin(l) * np.sin(f),
                                 np.cos(l) * np.sin(f)
                                 - np.cos(g) * np.sin(l) * np.cos(f),
                                 -np.sin(l) * np.sin(g)])
                n = _ned(p.normal(0.0, 0.0), lat, lon)
                d = _ned(p.slip_vector(0.0, 0.0, basis="cartesian"), lat, lon)
                assert np.allclose(n, n_ar, rtol=0, atol=1e-14), (strike, dip)
                assert np.allclose(d, d_ar, rtol=0, atol=1e-14), (strike, dip, rake)
                # their Box 4.4: the moment tensor of unit moment, n d + d n
                M = np.outer(n, d) + np.outer(d, n)
                sd, cd, s2d, c2d = np.sin(g), np.cos(g), np.sin(2 * g), np.cos(2 * g)
                sl, cl = np.sin(l), np.cos(l)
                box = {(0, 0): -(sd * cl * np.sin(2 * f) + s2d * sl * np.sin(f) ** 2),
                       (0, 1): sd * cl * np.cos(2 * f) + 0.5 * s2d * sl * np.sin(2 * f),
                       (0, 2): -(cd * cl * np.cos(f) + c2d * sl * np.sin(f)),
                       (1, 1): sd * cl * np.sin(2 * f) - s2d * sl * np.cos(f) ** 2,
                       (1, 2): -(cd * cl * np.sin(f) - c2d * sl * np.cos(f)),
                       (2, 2): s2d * sl}
                for (i, j), value in box.items():
                    assert abs(M[i, j] - value) < 1e-14, (strike, dip, rake, i, j)


def test_known_mechanisms():
    lat, lon = 37.0, 23.0

    def ned(strike: float, dip: float, rake: float) -> tuple[np.ndarray, np.ndarray]:
        p = Plane.at_depth(EARTH, lat, lon, 10e3, strike=strike, dip=dip,
                           length=10e3, width=5e3, slip=Slip.from_rake(rake, 2.0))
        return (_ned(p.slip_vector(0.0, 0.0, basis="cartesian"), lat, lon),
                _ned(p.normal(0.0, 0.0), lat, lon))

    h = np.sqrt(3.0) / 2.0
    # pure thrust on an east-striking plane dipping 30 degrees south: the
    # hanging wall (south, above) moves north and up; the normal points
    # from the footwall into it, south and up
    slip, normal = ned(90.0, 30.0, 90.0)
    assert np.allclose(slip, 2.0 * np.array([h, 0.0, -0.5]), rtol=0, atol=1e-14)
    assert np.allclose(normal, [-0.5, 0.0, -h], rtol=0, atol=1e-14)
    # left-lateral on a vertical north-striking plane: the hanging wall is
    # the east side, and moves north
    slip, normal = ned(0.0, 90.0, 0.0)
    assert np.allclose(slip, [2.0, 0.0, 0.0], rtol=0, atol=1e-14)
    assert np.allclose(normal, [0.0, 1.0, 0.0], rtol=0, atol=1e-14)
    # normal faulting on a north-striking plane dipping 60 degrees east:
    # the hanging wall moves east and down, down dip
    slip, _ = ned(0.0, 60.0, -90.0)
    assert np.allclose(slip, 2.0 * np.array([0.0, 0.5, h]), rtol=0, atol=1e-14)


def test_fault_components_and_rake():
    p = Plane.at_depth(EARTH, -20.0, 300.0, 5e3, strike=123.0, dip=50.0,
                       length=30e3, width=10e3,
                       slip=Slip.from_rake([[10.0, -100.0], [170.0, 45.0],
                                            [0.0, 90.0]],
                                           [[1.0, 2.0], [3.0, 0.5], [1.5, 2.5]],
                                           opening=[[0.0, 0.1], [0.2, 0.0],
                                                    [0.0, 0.0]]))
    u, v = p.slip.nodes()
    U, V = np.meshgrid(u, v, indexing="ij")
    comp = p.slip_vector(U, V, basis="fault")
    rake = np.rad2deg(np.arctan2(comp[..., 1], comp[..., 0]))
    assert np.allclose(rake, [[10.0, -100.0], [170.0, 45.0], [0.0, 90.0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(np.hypot(comp[..., 0], comp[..., 1]),
                       [[1.0, 2.0], [3.0, 0.5], [1.5, 2.5]], rtol=0, atol=1e-14)
    assert np.allclose(comp[..., 2], [[0.0, 0.1], [0.2, 0.0], [0.0, 0.0]],
                       rtol=0, atol=1e-14)
    # a cell reads one value anywhere in it; a boundary reads the cell above
    assert np.allclose(p.slip_vector(-0.4, 0.49, basis="fault"), comp[0, 1])
    assert np.allclose(p.slip_vector(-0.5 + 1 / 3, -0.1, basis="fault"), comp[1, 0])
    # the three bases are one vector, and round trip
    rng = np.random.default_rng(3)
    a, b = rng.uniform(-0.5, 0.5, (2, 50))
    cart = p.slip_vector(a, b, basis="cartesian")
    for basis, M in (("ambient", EARTH.basis(p.position(a, b))),
                     ("fault", p.fault_basis(a, b))):
        got = p.slip_vector(a, b, basis=basis)
        assert np.allclose(np.einsum("pij,pj->pi", M, got), cart, rtol=0,
                           atol=1e-14)
    # fault components are held as given; vectors given in another basis
    # at the grid's points are turned into them there, and read back
    given = rng.normal(size=(3, 4, 3))
    q = Plane(EARTH, [10.0, 20.0, R - 3e3], strike=0.0, dip=80.0, length=1e3,
              width=1e3, slip=Slip(given, rule="linear"))
    assert np.array_equal(q.slip.values[0], given)
    U, V = np.meshgrid(*q.slip.nodes(), indexing="ij")
    for basis in BASES:
        slip = Slip.from_vectors(q, given, rule="linear", basis=basis)
        back = Plane(EARTH, [10.0, 20.0, R - 3e3], strike=0.0, dip=80.0,
                     length=1e3, width=1e3, slip=slip)
        assert np.allclose(back.slip_vector(U, V, basis=basis), given, rtol=0,
                           atol=1e-15), basis


# -- a structured patch -----------------------------------------------------------

def test_a_plane_through_its_own_points_is_the_plane():
    plane = Plane.at_depth(EARTH, 37.0, 23.0, 15e3, strike=200.0, dip=35.0,
                           length=120e3, width=40e3,
                           slip=Slip.from_rake(70.0, 3.0))
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 6), np.linspace(-0.5, 0.5, 5),
                       indexing="ij")
    rng = np.random.default_rng(1)
    u, v = rng.uniform(-0.5, 0.5, (2, 200))
    for interpolation in ("linear", "cubic"):
        patch = StructuredPatch(EARTH, plane.position(U, V),
                                slip=Slip.from_rake(70.0, 3.0),
                                interpolation=interpolation)
        # measured: positions to 4.4e-16 of R; tangents to 3e-13 of their
        # length; normal, basis and area element to 3.2e-13; slip to 2e-13
        assert np.allclose(patch.cartesian(u, v), plane.cartesian(u, v), rtol=0,
                           atol=1e-14 * R)
        for t, want in zip(patch.tangents(u, v), plane.tangents(u, v)):
            assert np.allclose(t, want, rtol=0, atol=3e-12 * 120e3)
        assert np.allclose(patch.normal(u, v), plane.normal(u, v), rtol=0,
                           atol=3e-12)
        assert np.allclose(patch.fault_basis(u, v), plane.fault_basis(u, v),
                           rtol=0, atol=3e-12)
        assert np.allclose(patch.area_element(u, v), plane.area_element(u, v),
                           rtol=3e-12, atol=0)
        assert np.allclose(patch.slip_vector(u, v, basis="cartesian"),
                           plane.slip_vector(u, v, basis="cartesian"), rtol=0,
                           atol=3e-12)


# a hyperbolic paraboloid: x = (2 a u, 2 b v, 4 c u v), the bilinear patch
# through four corners that are not coplanar, and a bilinear slip on it
_A, _B, _C = 1.5, 1.0, 0.8
_SLIP = np.array([[[1.0, -0.5, 0.2], [0.3, 0.7, -0.4]],
                  [[-0.6, 0.1, 0.9], [0.5, 0.5, 0.5]]])


def _twisted() -> StructuredPatch:
    pos = np.array([[[2 * _A * u, 2 * _B * v, 4 * _C * u * v] for v in SQUARE]
                    for u in SQUARE])
    return StructuredPatch(LOCAL, pos, slip=Slip(_SLIP, rule="linear"),
                           interpolation="linear")


def _area_element(v: float, u: float) -> float:
    """|x_u x x_v| of the twisted quadrilateral, written out."""
    return np.sqrt(64 * (_B * _C * v) ** 2 + 64 * (_A * _C * u) ** 2
                   + 16 * (_A * _B) ** 2)


def _slip(u: float, v: float) -> np.ndarray:
    """The slip vector, Cartesian: its bilinear fault components, written
    out, on the fault basis built from the tangents (2a, 0, 4cv) and
    (0, 2b, 4cu)."""
    lu, lv = np.array([0.5 - u, 0.5 + u]), np.array([0.5 - v, 0.5 + v])
    comp = np.einsum("i,j,ijk->k", lu, lv, _SLIP)
    xu, xv = np.array([2 * _A, 0.0, 4 * _C * v]), np.array([0.0, 2 * _B, 4 * _C * u])
    s = xu / np.linalg.norm(xu)
    n = np.cross(xv, xu) / np.linalg.norm(np.cross(xv, xu))
    return comp[0] * s + comp[1] * np.cross(n, s) + comp[2] * n


def _orders(errors: np.ndarray) -> np.ndarray:
    return np.log2(errors[:-1] / errors[1:])


def test_quadrature_converges_at_the_rule_order():
    patch = _twisted()
    rng = np.random.default_rng(5)
    u, v = rng.uniform(-0.5, 0.5, (2, 20))
    assert np.allclose(patch.area_element(u, v),
                       [_area_element(b, a) for a, b in zip(u, v)], rtol=1e-14,
                       atol=0)
    area = dblquad(_area_element, -0.5, 0.5, -0.5, 0.5, epsabs=0, epsrel=1e-13)[0]
    total = np.array([dblquad(lambda v, u, k=k: _slip(u, v)[k] * _area_element(v, u),
                              -0.5, 0.5, -0.5, 0.5, epsabs=1e-13,
                              epsrel=1e-13)[0]
                      for k in range(3)])
    cells = np.array([1, 2, 4, 8, 16])
    for points, order in ((1, 2.0), (2, 4.0), (3, 6.0)):
        ea, es = [], []
        for n in cells:
            q = patch.quadrature(int(n), int(n), points=points)
            ea.append(abs(q.weights.sum() / area - 1.0))
            es.append(np.max(np.abs(q.weights @ q.slip - total)) / np.max(total))
        ea, es = np.array(ea), np.array(es)
        # measured orders: midpoint 2.19, 2.02, 2.01, 2.00; two points
        # 5.2, 3.9, 4.01, 4.00; three points 7.8, 8.4, 5.7, 6.0 (the last
        # at 4e-11); the slip integral the same to three figures
        for e in (ea, es):
            assert np.all(_orders(e)[-2:] > order - 0.3), (points, _orders(e))
    assert ea[-1] < 1e-10


def test_a_piece_of_a_sphere_at_the_interpolation_order():
    # latitudes 10 to 30, longitudes 40 to 70 on the unit sphere: u east
    # along the longitudes, v south along the latitudes, the normal outward
    exact = (np.sin(np.deg2rad(30.0)) - np.sin(np.deg2rad(10.0))) * np.deg2rad(30.0)
    frame = GeographicFrame(1.0)
    zero = Slip(np.zeros((1, 1, 3)))
    nodes = np.array([5, 9, 17, 33])
    errors = {}
    for interpolation in ("linear", "cubic"):
        e = []
        for n in nodes:
            P = _grid_positions(np.linspace(30.0, 10.0, n), np.linspace(40.0, 70.0, n),
                                1.0)
            patch = StructuredPatch(frame, P, slip=zero, interpolation=interpolation)
            q = patch.quadrature(int(n) - 1, int(n) - 1, points=4)
            assert np.all(np.einsum("pi,pi->p", q.normals, q.points) > 0.0)
            e.append(abs(q.weights.sum() / exact - 1.0))
        errors[interpolation] = np.array(e)
    # measured: bilinear 1.9e-3 to 3.0e-5, orders 2.00; bicubic 2.5e-6 to
    # 5e-11 (orders 5.2, 6.3, 4.1: its error changes sign near 30 nodes),
    # 4.45 over the whole range to 129 nodes
    assert np.all(np.abs(_orders(errors["linear"]) - 2.0) < 0.02)
    slope = np.log2(errors["cubic"][0] / errors["cubic"][-1]) / np.log2(32 / 4)
    assert slope > 3.8 and errors["cubic"][-1] < 1e-10

    # the quadrature's own order on a fixed bicubic piece, against its own
    # finest rule; measured: midpoint 2.00, two points 4.00
    P = _grid_positions(np.linspace(30.0, 10.0, 9), np.linspace(40.0, 70.0, 9), 1.0)
    patch = StructuredPatch(frame, P, slip=zero)
    ref = patch.quadrature(64, 64, points=6).weights.sum()
    for points, order in ((1, 2.0), (2, 4.0)):
        e = np.array([abs(patch.quadrature(m, m, points=points).weights.sum() / ref
                          - 1.0) for m in (8, 16, 32)])
        assert np.all(np.abs(_orders(e) - order) < 0.05)


# -- time ------------------------------------------------------------------------------

def _timed(support: str, rule: str, values: np.ndarray, **axes: object) -> Fault:
    plane = Plane(LOCAL, [0.0, 0.0, -5.0], strike=10.0, dip=40.0, length=6.0,
                  width=3.0, slip=Slip(values))
    return Fault([plane], support=support, time_interpolation=rule, **axes)


def test_time_by_the_surface_files_rules():
    rng = np.random.default_rng(11)
    time, instant = np.array([0.0, 1.0, 3.0]), np.array([2.0, 5.0])
    nodes = rng.normal(size=(3, 2, 3, 3))
    grid = AngularGrid(np.array([0.5, 1.5]), np.array([0.0, 2.0, 4.0]))
    cases = {"linear": ("node", "linear", nodes, {"time": time}),
             "previous": ("node", "previous", nodes, {"time": time}),
             "none": ("node", "none", nodes, {"time": time}),
             "interval": ("interval", "none", nodes[:2], {"time": time}),
             "instant": ("instant", "none", nodes[:2], {"instant": instant}),
             "static": ("static", "none", nodes[0], {})}
    variables = {}
    for name, (support, rule, values, _) in cases.items():
        for k in range(3):
            variables[f"{name}{k}"] = FileVariable(
                np.asarray(values[..., k]),
                {"time_support": support, "time_interpolation": rule})
    history = SurfaceHistory(SurfaceFile(SCHEMA, grid, variables, time=time,
                                         instant=instant))
    dates = (-1.0, 0.0, 0.25, 1.0, 2.0, 2.999, 3.0, 5.0, 7.5)
    for name, (support, rule, values, axes) in cases.items():
        fault = _timed(support, rule, values, **axes)
        U, V = np.meshgrid(*fault.patches[0].slip.nodes(), indexing="ij")
        F = fault.patches[0].fault_basis(0.0, 0.0)
        for date in dates:
            for extrapolation in ("refuse", "hold"):
                try:
                    want = np.stack([history.values(f"{name}{k}", date=date,
                                                     extrapolation=extrapolation)
                                     for k in range(3)], axis=-1)
                except ValueError as error:
                    with pytest.raises(ValueError) as caught:
                        fault.slip_vector(0, U, V, date=date,
                                          extrapolation=extrapolation,
                                          basis="fault")
                    assert str(caught.value) == str(error).replace(
                        f"'{name}0'", "'slip'")
                    continue
                at = date
                if support == "static":
                    # a static field serves any date; a static fault takes none
                    with pytest.raises(ValueError, match="static.*takes no date"):
                        fault.slip_vector(0, U, V, date=date, basis="fault")
                    with pytest.raises(ValueError, match="static.*takes no date"):
                        fault.rows(date)
                    at = None
                got = fault.slip_vector(0, U, V, date=at,
                                        extrapolation=extrapolation, basis="fault")
                assert np.allclose(got, want, rtol=0, atol=1e-15), (name, date)
                q = fault.quadrature(2, 3, date=at, extrapolation=extrapolation)
                assert np.allclose(q.slip.reshape(2, 3, 3), want @ F.T, rtol=0,
                                   atol=1e-14)
    with pytest.raises(ValueError, match="needs a date"):
        _timed("node", "linear", nodes, time=time).quadrature(1, 1)


def test_a_fault_of_two_patches_with_afterslip():
    time = np.array([0.0, 10.0, 100.0])
    a = Plane.at_depth(EARTH, 35.0, 140.0, 20e3, strike=200.0, dip=15.0,
                       length=100e3, width=50e3,
                       slip=Slip.from_rake(90.0, [[[1.0]], [[1.5]], [[1.8]]]))
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 4), np.linspace(-0.5, 0.5, 4),
                       indexing="ij")
    b = StructuredPatch(EARTH, Plane.at_depth(
        EARTH, 34.2, 139.7, 30e3, strike=210.0, dip=25.0, length=60e3, width=40e3,
        slip=Slip(np.zeros((1, 1, 3)))).position(U, V),
        slip=Slip.from_rake(80.0, np.linspace(0.5, 1.0, 3)[:, None, None]
                            * np.ones((3, 3, 3)), rule="linear"))
    fault = Fault([a, b], support="node", time_interpolation="linear", time=time,
                  scales=Scales(length=1e3))
    assert fault.nslices == 3 and fault.frame == EARTH and fault.scales.length == 1e3
    q = fault.quadrature([4, 3], [2, 3], points=2, date=55.0)
    assert q.size == 4 * 2 * 4 + 3 * 3 * 4
    assert np.array_equal(np.unique(q.patch), [0, 1])
    assert np.isclose(q.weights[q.patch == 0].sum(), 100e3 * 50e3, rtol=1e-14)
    comp = fault.slip_vector(0, 0.2, 0.3, date=55.0, basis="fault")
    assert np.allclose(comp, [0.0, 1.65, 0.0], rtol=0, atol=1e-14)
    comp = fault.slip_vector(1, 0.0, 0.0, date=10.0, basis="fault")
    m = np.hypot(comp[0], comp[1])
    assert np.isclose(m, 0.75, rtol=1e-12)
    for p in fault.patches:
        testing.check_patch(p)



def _slices_case(support: str, rule: str) -> Fault:
    """A fault of a plane, a bicubic and a bilinear patch, on four nodes
    (or static), every slip rule among them."""
    rng = np.random.default_rng(31)
    n = 1 if support == "static" else 4
    lead = () if support == "static" else (n,)
    a = Plane.at_depth(EARTH, 35.0, 140.0, 20e3, strike=200.0, dip=15.0,
                       length=100e3, width=50e3,
                       slip=Slip(rng.normal(size=lead + (3, 2, 3)), rule="cell"))
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 4),
                       indexing="ij")
    base = Plane.at_depth(EARTH, 34.2, 139.7, 30e3, strike=210.0, dip=25.0,
                          length=60e3, width=40e3, slip=Slip(np.zeros((1, 1, 3))))
    bend = np.stack([1e-3 * U * V, 2e-3 * U ** 2, -4e3 * V ** 2], axis=-1)
    b = StructuredPatch(EARTH, base.position(U, V) + bend,
                        slip=Slip(rng.normal(size=lead + (4, 5, 3)), rule="cubic"),
                        interpolation="cubic")
    c = StructuredPatch(EARTH, base.position(U, V)[:3, :3],
                        slip=Slip(rng.normal(size=lead + (2, 3, 3)), rule="linear"),
                        interpolation="linear")
    if support == "static":
        return Fault([a, b, c])
    return Fault([a, b, c], support=support, time_interpolation=rule,
                 time=np.array([0.0, 1.0, 2.5, 6.0]))


@pytest.mark.parametrize("support, rule", [("static", "none"), ("node", "linear"),
                                           ("node", "previous"), ("node", "none")])
def test_the_slice_quadrature_is_the_quadrature_at_every_date(support, rule):
    """The geometry once and every slice's slip, combined by the rows of a
    date, is the quadrature at that date, number for number."""
    fault = _slices_case(support, rule)
    counts = ([2, 3, 1], [3, 1, 2])
    sq = fault.slice_quadrature(*counts, points=2)
    assert sq.nslices == fault.nslices and sq.slip.shape == (fault.nslices, sq.size, 3)
    sizes = [nu * nv * 4 for nu, nv in zip(*counts)]
    assert np.array_equal(sq.patch, np.repeat([0, 1, 2], sizes))
    dates: list = [None]
    extra = {}
    if support == "node":
        t = fault.time
        dates = list(t) + [t[1] + 1e-13, t[-1] + 1e-12]
        if rule != "none":
            dates += [0.4, 2.0, 5.9, 7.0, -3.0]
            extra = {"extrapolation": "hold"}
    for date in dates:
        q = fault.quadrature(*counts, points=2, date=date, **extra)
        for name in ("points", "positions", "normals", "weights", "u", "v", "patch"):
            assert np.array_equal(getattr(sq, name), getattr(q, name)), (date, name)
        rows = fault.rows(date, **extra)
        assert np.array_equal(sq.slip_of(rows), q.slip), date
    for k in range(fault.nslices):
        assert np.array_equal(sq.slip_of(((k, 1.0), (k, 0.0))), sq.slip[k])
    with pytest.raises(IndexError):
        sq.slip_of(((fault.nslices, 1.0), (0, 0.0)))
    same = fault.slice_quadrature(2, 3)
    assert same.size == 3 * 2 * 3 and np.array_equal(same.patch,
                                                      np.repeat([0, 1, 2], 6))


def test_the_quadratures_form_a_patch_s_geometry_once(monkeypatch):
    """Each patch's tangents are formed once for all its points and every
    slice, and the numbers are, to the bit, the patch's own evaluations
    point by point: the normal and the weight from the tangents there,
    each slice's slip as `slip_vector` reads it."""
    fault = _slices_case("node", "linear")
    calls = []
    for kind in {type(p) for p in fault.patches}:
        def counted(self, u, v, _tangents=kind.tangents):
            calls.append(self)
            return _tangents(self, u, v)
        monkeypatch.setattr(kind, "tangents", counted)
    sq = fault.slice_quadrature(3, 2, points=2)
    assert len(calls) == len(fault.patches)
    calls.clear()
    q = fault.quadrature(3, 2, points=2, date=1.7)
    assert len(calls) == len(fault.patches)
    calls.clear()
    own = fault.patches[1].quadrature(3, 2, points=2, index=2)
    assert len(calls) == 1
    monkeypatch.undo()
    (k0, w0), (k1, w1) = fault.rows(1.7)
    for k, p in enumerate(fault.patches):
        m = q.patch == k
        u, v = q.u[m], q.v[m]
        xu, xv = p.tangents(u, v)
        c = np.cross(xv, xu)
        area = np.linalg.norm(c, axis=-1)
        assert np.array_equal(q.normals[m], c / area[:, None])
        assert np.array_equal(q.weights[m], area * parameter_rule(3, 2, points=2)[2])
        assert np.array_equal(q.points[m], p.cartesian(u, v))
        want = w0 * p.slip_vector(u, v, index=k0, basis="cartesian")
        if w1 != 0.0:
            want = want + w1 * p.slip_vector(u, v, index=k1, basis="cartesian")
        assert np.array_equal(q.slip[m], want)
        for s in range(fault.nslices):
            assert np.array_equal(sq.slip[s][m],
                                  p.slip_vector(u, v, index=s, basis="cartesian"))
    assert np.array_equal(own.slip, sq.slip[2][sq.patch == 1])
    assert np.array_equal(own.normals, sq.normals[sq.patch == 1])


def test_a_patch_s_quadrature_names_no_patch_and_a_fault_s_names_each():
    from planetmodel.fault import PatchQuadrature, Quadrature
    fault = _slices_case("node", "linear")
    q = fault.patches[1].quadrature(2, 2, index=3)
    assert type(q) is PatchQuadrature and not hasattr(q, "patch")
    whole = fault.quadrature(2, 2, date=6.0)
    assert isinstance(whole, Quadrature) and isinstance(whole, PatchQuadrature)
    assert np.array_equal(whole.patch, np.repeat([0, 1, 2], 4))
    assert np.array_equal(whole.slip[whole.patch == 1], q.slip)


def test_a_static_fault_takes_no_date():
    fault = _slices_case("static", "none")
    assert fault.rows(None) == ((0, 1.0), (0, 0.0))
    for call in (lambda: fault.rows(0.0),
                 lambda: fault.slip_vector(0, 0.0, 0.0, date=1.0),
                 lambda: fault.quadrature(1, 1, date=1.0),
                 lambda: fault.slip_at(fault.patches[0].cartesian(0.0, 0.0),
                                       max_distance=1.0, date=1.0)):
        with pytest.raises(ValueError, match="static, the same at every date"):
            call()


def test_a_date_within_the_tolerance_of_a_stored_time_is_that_time():
    """The tolerance is time_rtol times the span (here 3, so 3e-12 by
    default), or the magnitude of a lone time; zero is exact matching."""
    from planetmodel.surfacefile import TIME_RTOL, time_rows
    assert TIME_RTOL == 1e-12
    t, s = np.array([0.0, 1.0, 3.0]), np.array([2.0, 5.0])

    def rows(support, rule, date, **kw):
        return time_rows(support, rule, date, time=t, instant=s, **kw)

    near, far = 1e-12, 1e-11
    one = ((1, 1.0), (1, 0.0))
    # served at the stored times alone
    assert rows("node", "none", 1.0 + near) == one
    assert rows("node", "none", 1.0 - near) == one
    for bad in (dict(), dict(time_rtol=0.0)):
        date = 1.0 + (far if not bad else near)
        with pytest.raises(ValueError, match="stored nodes alone"):
            rows("node", "none", date, **bad)
    assert rows("instant", "none", 5.0 + near) == one
    with pytest.raises(ValueError, match="stored instants alone"):
        rows("instant", "none", 5.0 + near, time_rtol=0.0)
    # read from a node a rounding error below it
    assert rows("node", "previous", 1.0 - near) == one
    assert rows("node", "previous", 1.0 - near, time_rtol=0.0) == ((0, 1.0), (0, 0.0))
    assert rows("interval", "none", 1.0 - near) == one
    assert rows("interval", "none", 1.0 - near, time_rtol=0.0) == ((0, 1.0), (0, 0.0))
    # a node's own values alone, weights exact
    assert rows("node", "linear", 1.0 + near) == one
    (k0, w0), (k1, w1) = rows("node", "linear", 1.0 + near, time_rtol=0.0)
    assert (k0, k1) == (1, 2) and 0.0 < w1 < 1e-11
    # the ends of the span
    assert rows("node", "linear", 3.0 + near) == ((2, 1.0), (2, 0.0))
    assert rows("node", "linear", -near) == ((0, 1.0), (0, 0.0))
    with pytest.raises(ValueError, match="outside"):
        rows("node", "linear", 3.0 + far)
    with pytest.raises(ValueError, match="outside"):
        rows("node", "linear", 3.0 + near, time_rtol=0.0)
    assert rows("node", "linear", 3.0 + far, extrapolation="hold") == (
        (2, 1.0), (2, 0.0))
    # a lone time: its magnitude is the scale
    lone = np.array([4.0])
    assert time_rows("node", "none", 4.0 + 3e-12, time=lone) == ((0, 1.0), (0, 0.0))
    with pytest.raises(ValueError, match="stored nodes alone"):
        time_rows("node", "none", 4.0 + 5e-12, time=lone)
    # the nearest within the tolerance, the earlier on a tie
    assert time_rows("node", "none", 0.5, time=np.array([0.0, 1.0]),
                     time_rtol=0.5) == ((0, 1.0), (0, 0.0))
    assert time_rows("node", "none", 0.6, time=np.array([0.0, 1.0]),
                     time_rtol=0.5) == ((1, 1.0), (1, 0.0))
    for bad in (-1e-12, np.inf, np.nan):
        with pytest.raises(ValueError, match="time_rtol"):
            rows("node", "linear", 1.0, time_rtol=bad)
    # the fault and the surface file take it alike
    fault = _slices_case("node", "none")
    assert fault.rows(2.5 + 5e-12) == ((2, 1.0), (2, 0.0))
    with pytest.raises(ValueError, match="stored nodes alone"):
        fault.rows(2.5 + 5e-12, time_rtol=0.0)


def test_the_tolerance_is_relative_to_the_times_as_well_as_the_span():
    """A clock whose zero is far from the file's reaches a time stored at
    -999 as -998.99999999999511, 4.9e-12 off: on a span of 1 the span
    alone would refuse it; the scale is the largest of the span and the
    magnitudes of the first and last times, here 1000."""
    from planetmodel.surfacefile import time_rows
    axis = np.array([-1000.0, -999.0])
    reached = -998.99999999999511
    assert reached - -999.0 > 4.8e-12
    one, zero = ((1, 1.0), (1, 0.0)), ((0, 1.0), (0, 0.0))
    for rule in ("linear", "previous", "none"):
        assert time_rows("node", rule, reached, time=axis) == one
        assert time_rows("node", rule, -1000.0 - 4.9e-12, time=axis) == zero
        # 1e-12 of the span alone, and exact matching, refuse it
        for tight in (1e-15, 0.0):
            with pytest.raises(ValueError):
                time_rows("node", rule, reached, time=axis, time_rtol=tight)
        # beyond 1e-12 of the scale, 1e-9, it is not the node
        with pytest.raises(ValueError):
            time_rows("node", rule, -999.0 + 2e-9, time=axis)
    assert time_rows("node", "linear", -999.0 + 2e-9, time=axis,
                     extrapolation="hold") == one
    with pytest.raises(ValueError, match="stored nodes alone"):
        time_rows("node", "none", -999.0 + 2e-9, time=axis, extrapolation="hold")
    assert time_rows("node", "linear", -999.5, time=axis) == ((0, 0.5), (1, 0.5))
    assert time_rows("node", "none", 999.0 - 4.9e-12,
                     time=np.array([999.0, 1000.0])) == zero
    assert time_rows("instant", "none", reached, instant=axis) == one
    # a fault on such an axis reads it alike: nodes at -1006 to -1000, a
    # date 8e-12 past the last within 1e-12 of 1006 but not of the span 6
    fault = _slices_case("node", "previous")
    late = Fault(fault.patches, support="node", time_interpolation="previous",
                 time=fault.time - 1006.0)
    assert late.rows(-1000.0 + 8e-12) == ((3, 1.0), (3, 0.0))
    with pytest.raises(ValueError, match="outside"):
        late.rows(-1000.0 + 8e-12, time_rtol=1e-12 * 6.0 / 1006.0)


def test_an_axis_of_the_one_time_zero_matches_exactly():
    """Its scale is zero: the date 0 is that time and any other is not."""
    from planetmodel.surfacefile import time_rows
    zero = np.array([0.0])
    for rule in ("linear", "previous", "none"):
        for date in (0.0, -0.0):
            assert time_rows("node", rule, date, time=zero) == ((0, 1.0), (0, 0.0))
        with pytest.raises(ValueError):
            time_rows("node", rule, 1e-300, time=zero)
    assert time_rows("node", "linear", 1e-300, time=zero,
                     extrapolation="hold") == ((0, 1.0), (0, 0.0))
    assert time_rows("node", "previous", -1.0, time=zero,
                     extrapolation="hold") == ((0, 1.0), (0, 0.0))
    with pytest.raises(ValueError, match="stored nodes alone"):
        time_rows("node", "none", 1e-300, time=zero, extrapolation="hold")

# -- the poles and the date line ------------------------------------------------------

def test_a_plane_over_a_pole():
    # 100 km along strike north across the pole, 5.6 km from it
    p = Plane.at_depth(EARTH, 89.95, 30.0, 10e3, strike=0.0, dip=60.0,
                       length=100e3, width=20e3, slip=Slip.from_rake(45.0, 3.0))
    q = p.quadrature(20, 4, points=2)
    position = q.positions
    assert np.all(position[:, 0] <= 90.0)
    assert np.all((position[:, 1] >= 0.0) & (position[:, 1] < 360.0))
    assert np.ptp(position[:, 1]) > 170.0     # both sides of the pole
    assert np.allclose(q.slip, q.slip[0], rtol=0, atol=1e-14)
    comp = p.slip_vector(q.u, q.v, basis="fault")
    assert np.allclose(comp, [3 / np.sqrt(2), 3 / np.sqrt(2), 0.0], rtol=0,
                       atol=1e-14)
    ambient = p.slip_vector(q.u, q.v)
    assert np.ptp(ambient[:, 1]) > 2.0         # north turns over the pole
    assert abs(q.weights.sum() / (100e3 * 20e3) - 1.0) < 1e-14
    testing.check_patch(p)
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 7), np.linspace(-0.5, 0.5, 4),
                       indexing="ij")
    patch = StructuredPatch(EARTH, p.position(U, V),
                            slip=Slip.from_rake(45.0, 3.0))
    assert np.allclose(patch.cartesian(q.u, q.v), q.points, rtol=0, atol=1e-14 * R)
    # the slip on the patch's own fault basis, the plane's to 4e-13
    # (measured: 1.0e-12 of the slip's 3)
    assert np.allclose(patch.slip_vector(q.u, q.v, basis="cartesian"), q.slip,
                       rtol=0, atol=1e-12 * 3.0)
    testing.check_patch(patch)


def test_a_patch_across_the_date_line():
    lat = np.linspace(5.0, -5.0, 5)
    lon = np.linspace(175.0, 185.0, 6)
    given = np.where(lon > 180.0, lon - 360.0, lon)       # written as -177 ...
    rng = np.random.default_rng(2)
    slip = Slip(rng.normal(size=(4, 4, 3)), rule="cubic")
    across = StructuredPatch(EARTH, _grid_positions(lat, given, R - 10e3), slip=slip)
    unwrapped = StructuredPatch(EARTH, _grid_positions(lat, lon, R - 10e3),
                                slip=slip)
    home = StructuredPatch(EARTH, _grid_positions(lat, lon - 180.0, R - 10e3),
                           slip=slip)
    u, v = rng.uniform(-0.5, 0.5, (2, 100))
    X = across.cartesian(u, v)
    assert np.allclose(X, unwrapped.cartesian(u, v), rtol=0, atol=1e-14 * R)
    flip = np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(X, home.cartesian(u, v) @ flip, rtol=0, atol=1e-14 * R)
    assert np.allclose(across.area_element(u, v), home.area_element(u, v),
                       rtol=1e-12, atol=0)
    assert np.allclose(across.normal(u, v), home.normal(u, v) @ flip, rtol=0,
                       atol=1e-12)
    # the fault components are what the rule reads: the same on both, and
    # so the ambient components too, the two bases turning together
    assert np.array_equal(across.slip_vector(u, v, basis="fault"),
                          home.slip_vector(u, v, basis="fault"))
    assert np.allclose(across.slip_vector(u, v), home.slip_vector(u, v), rtol=0,
                       atol=1e-12)
    lon_read = across.position(u, v)[:, 1]
    assert np.all((lon_read >= 175.0 - 1e-9) & (lon_read <= 185.0 + 1e-9))
    testing.check_patch(across)


# -- the contract and the refusals ---------------------------------------------------

def _patches() -> list:
    rng = np.random.default_rng(4)
    plane = Plane.at_depth(EARTH, -33.0, 287.0, 25e3, strike=15.0, dip=20.0,
                           length=200e3, width=80e3,
                           slip=Slip.from_rake(rng.uniform(60, 120, (2, 5, 4)),
                                               rng.uniform(0, 5, (2, 5, 4)),
                                               rule="cubic"))
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 6),
                       indexing="ij")
    wavy = plane.position(U, V) + np.stack(
        [0.05 * np.sin(3 * U) * np.cos(2 * V), 0.04 * U * V,
         3e3 * np.cos(2 * U + V)], axis=-1)
    local = np.stack(np.broadcast_arrays(3.0 * U, 2.0 * V,
                                         -4.0 - 0.3 * V ** 2 + 0.2 * U * V), -1)
    vertical = Plane(LOCAL, [1.0, 2.0, -3.0], strike=250.0, dip=90.0, length=2.0,
                     width=1.0, slip=Slip(np.zeros((1, 1, 3))))
    return [plane,
            Plane(LOCAL, [1.0, 2.0, -3.0], strike=250.0, dip=90.0, length=2.0,
                  width=1.0, slip=Slip.from_vectors(
                      vertical, rng.normal(size=(3, 2, 3)), rule="linear",
                      basis="cartesian")),
            Plane(LOCAL, [0.0, 0.0, -1.0], strike=30.0, dip=0.0, length=2.0,
                  width=2.0, slip=Slip.from_rake(0.0, 0.0, opening=0.4)),
            StructuredPatch(EARTH, wavy, slip=Slip(rng.normal(size=(2, 2, 3)),
                                                   rule="cell")),
            StructuredPatch(EARTH, wavy, interpolation="linear",
                            slip=Slip.from_rake(30.0, rng.uniform(0, 2, (4, 4)),
                                                rule="cubic")),
            StructuredPatch(LOCAL, local, slip=Slip(rng.normal(size=(3, 3, 3)),
                                                    rule="linear"))]


@pytest.mark.parametrize("index", range(6))
def test_every_kind_passes_the_contract(index):
    testing.check_patch(_patches()[index])


def test_a_sill_is_a_horizontal_plane():
    sill = _patches()[2]
    assert np.allclose(sill.normal(0.2, -0.1), [0.0, 0.0, 1.0], rtol=0, atol=1e-15)
    assert np.allclose(sill.slip_vector(0.0, 0.0), [0.0, 0.0, 0.4], rtol=0,
                       atol=1e-15)


def test_refusals():
    one = Slip.from_rake(0.0, 1.0)
    kw = dict(strike=0.0, length=1.0, width=1.0, slip=one)
    for dip in (-1.0, 90.5, np.nan):
        with pytest.raises(ValueError, match="dip"):
            Plane(LOCAL, [0.0, 0.0, 0.0], dip=dip, **kw)
    with pytest.raises(ValueError, match="positive"):
        Plane(LOCAL, [0.0, 0.0, 0.0], dip=10.0, strike=0.0, length=0.0, width=1.0,
              slip=one)
    with pytest.raises(ValueError, match="one position"):
        Plane(LOCAL, [[0.0, 0.0, 0.0]], dip=10.0, **kw)
    with pytest.raises(TypeError, match="Frame"):
        Plane("local", [0.0, 0.0, 0.0], dip=10.0, **kw)
    with pytest.raises(ValueError, match="shape"):
        Slip(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="at least 2 nodes"):
        Slip(np.zeros((1, 3, 3)), rule="linear")
    with pytest.raises(ValueError, match="at least 4 nodes"):
        Slip(np.zeros((4, 3, 3)), rule="cubic")
    with pytest.raises(ValueError, match="finite"):
        Slip(np.full((1, 1, 3), np.nan))
    with pytest.raises(ValueError, match="rule"):
        Slip(np.zeros((1, 1, 3)), rule="nearest")
    with pytest.raises(ValueError, match="basis"):
        Slip.from_vectors(Plane(LOCAL, [0.0, 0.0, 0.0], dip=10.0, **kw),
                          np.zeros((1, 1, 3)), basis="spherical")
    with pytest.raises(TypeError, match="Patch"):
        Slip.from_vectors("plane", np.zeros((1, 1, 3)))
    with pytest.raises(TypeError, match="real"):
        Slip(np.zeros((1, 1, 3), dtype=complex))
    with pytest.raises(ValueError, match="broadcast"):
        Slip.from_rake([0.0, 1.0], 1.0)
    pos = np.zeros((3, 3, 3))
    pos[..., 0], pos[..., 1] = np.meshgrid([0, 1, 2], [0, 1, 2], indexing="ij")
    with pytest.raises(ValueError, match="at least 4 nodes"):
        StructuredPatch(LOCAL, pos, slip=one)
    with pytest.raises(ValueError, match="interpolation"):
        StructuredPatch(LOCAL, pos, slip=one, interpolation="quadratic")
    flat = pos.copy()
    flat[..., 1] = 0.0
    with pytest.raises(ValueError, match="degenerate"):
        StructuredPatch(LOCAL, flat, slip=one, interpolation="linear")
    p = StructuredPatch(LOCAL, pos, slip=one, interpolation="linear")
    for u, v in ((0.6, 0.0), (0.0, -0.51), (np.nan, 0.0)):
        with pytest.raises(ValueError, match="square|finite"):
            p.cartesian(u, v)
    with pytest.raises(IndexError):
        p.slip_vector(0.0, 0.0, index=1)
    with pytest.raises(ValueError, match="basis"):
        p.slip_vector(0.0, 0.0, basis="enu")
    with pytest.raises(ValueError, match="positive integer"):
        parameter_rule(0, 1)
    with pytest.raises(ValueError, match="positive integer"):
        p.quadrature(1, 1, points=0)
    with pytest.raises(ValueError, match="at least one"):
        Fault([])
    q = Plane(EARTH, [0.0, 0.0, R - 1e3], dip=10.0, **kw)
    with pytest.raises(ValueError, match="same frame"):
        Fault([p, q])
    with pytest.raises(ValueError, match="slices"):
        Fault([p], support="node", time_interpolation="linear", time=[0.0, 1.0])
    with pytest.raises(ValueError, match="uses no time"):
        Fault([p], time=[0.0])
    with pytest.raises(ValueError, match="time_interpolation"):
        Fault([p], support="interval", time_interpolation="linear",
              time=[0.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        Fault([p], support="node", time_interpolation="linear", time=[1.0, 1.0])
    with pytest.raises(ValueError, match="2 counts for 1"):
        Fault([p]).quadrature([1, 2], 1)
    with pytest.raises(ValueError, match="extrapolation"):
        Fault([p]).quadrature(1, 1, extrapolation="clamp")


def test_the_bases_and_the_square_are_one_statement():
    assert BASES == ("ambient", "cartesian", "fault")
    assert SLIP_COMPONENTS == ("strike_slip", "dip_slip", "opening")
    assert SQUARE == (-0.5, 0.5)
    u, v, w = parameter_rule(3, 2, points=2)
    assert np.isclose(w.sum(), 1.0, rtol=1e-15)
    assert u.min() > SQUARE[0] and u.max() < SQUARE[1]
    assert np.all(np.diff(u.reshape(6, 4)[:, 0]) > 0.0)      # u slowest
