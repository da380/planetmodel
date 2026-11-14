"""Locating points against a fault: the nearest point of a patch, the
signed distance off it, and whether its foot lies in the square.

The oracles: a plane's closed form against the Newton search run on a
structured patch through the plane's own points; round trips (u, v) ->
x + d n -> (u, v, d) on genuinely curved patches (a piece of the unit
sphere, a twisted quadrilateral) near edges and corners and at distances
up to about half the radius of curvature; the nearest point of a dense
sample of the patch, which the search may never be farther than; points
beyond an edge or a corner; and a search cut short, which is refused or
flagged, never answered silently.
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from planetmodel.fault import (SQUARE, CartesianFrame, Fault, GeographicFrame,
                               Plane, Slip, StructuredPatch)

R = 6371e3
EARTH = GeographicFrame(R)
LOCAL = CartesianFrame()
ZERO = Slip(np.zeros((1, 1, 3)))


def _sphere(interpolation: str, n: int = 9) -> StructuredPatch:
    """Latitudes 30 to 10, longitudes 40 to 70 on the unit sphere: the
    normal outward, the radius of curvature one."""
    lat, lon = np.linspace(30.0, 10.0, n), np.linspace(40.0, 70.0, n)
    P = np.stack(np.broadcast_arrays(lat[None, :], lon[:, None], 1.0), axis=-1)
    return StructuredPatch(GeographicFrame(1.0), P, slip=ZERO,
                           interpolation=interpolation)


def _twisted() -> StructuredPatch:
    """x = (3 u, 2 v, 3.2 u v): the saddle z = 0.53 X Y, whose radius of
    curvature is 1.9 at its centre and more elsewhere."""
    pos = np.array([[[3.0 * u, 2.0 * v, 3.2 * u * v] for v in SQUARE]
                    for u in SQUARE])
    return StructuredPatch(LOCAL, pos, slip=ZERO, interpolation="linear")


def _off(patch: StructuredPatch, u: np.ndarray, v: np.ndarray,
         d: np.ndarray) -> np.ndarray:
    return patch.cartesian(u, v) + d[:, None] * patch.normal(u, v)


def _near_edges(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Parameters at random, on and a hair inside the edges, and at the
    corners."""
    lo, hi = SQUARE
    u, v = rng.uniform(lo, hi, (2, n))
    u[: n // 4] = rng.choice([lo, hi, lo + 1e-7, hi - 1e-7], n // 4)
    v[n // 4: n // 2] = rng.choice([lo, hi, lo + 1e-7, hi - 1e-7], n // 2 - n // 4)
    u[-4:], v[-4:] = [lo, hi, lo, hi], [lo, lo, hi, hi]
    return u, v


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_the_plane_in_closed_form_and_by_newton(interpolation):
    plane = Plane.at_depth(EARTH, 37.0, 23.0, 15e3, strike=200.0, dip=35.0,
                           length=120e3, width=40e3, slip=ZERO)
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 4),
                       indexing="ij")
    patch = StructuredPatch(EARTH, plane.position(U, V), slip=ZERO,
                            interpolation=interpolation)
    rng = np.random.default_rng(8)
    # points around the plane, many beyond its edges and corners
    u, v = rng.uniform(-0.9, 0.9, (2, 400))
    d = rng.uniform(-30e3, 30e3, 400)
    origin = plane.cartesian(0.0, 0.0)
    xu, xv = plane.tangents(0.0, 0.0)
    P = origin + u[:, None] * xu + v[:, None] * xv + d[:, None] * plane.normal(0, 0)
    exact, newton = plane.locate(P), patch.locate(P)
    inside = (np.abs(u) <= 0.5) & (np.abs(v) <= 0.5)
    assert np.array_equal(exact.inside, inside)
    assert np.allclose(exact.u, np.clip(u, -0.5, 0.5), rtol=0, atol=1e-12)
    assert np.allclose(exact.distance[inside], d[inside], rtol=0, atol=1e-6)
    # measured: u and v to 2.2e-13, distances to 3.8e-9 m, the same inside
    assert np.array_equal(newton.inside, exact.inside)
    assert np.allclose(newton.u, exact.u, rtol=0, atol=1e-12)
    assert np.allclose(newton.v, exact.v, rtol=0, atol=1e-12)
    assert np.allclose(newton.distance, exact.distance, rtol=0, atol=1e-6)


@pytest.mark.parametrize("patch, reach", [(_sphere("cubic"), 0.5),
                                          (_twisted(), 0.9)])
def test_round_trips_on_curved_patches(patch, reach):
    rng = np.random.default_rng(9)
    u, v = _near_edges(rng, 600)
    d = rng.uniform(-reach, reach, u.size)
    found = patch.locate(_off(patch, u, v, d))
    # measured: u, v to 1.1e-14 and d to 5e-16 on the sphere, 2e-16 on
    # the saddle, every foot inside (those on an edge included), within
    # four Newton steps
    assert np.all(found.converged) and np.all(found.inside)
    assert np.allclose(found.u, u, rtol=0, atol=1e-10)
    assert np.allclose(found.v, v, rtol=0, atol=1e-10)
    assert np.allclose(found.distance, d, rtol=0, atol=1e-10)
    assert np.array_equal(found.patch, np.zeros(u.size, dtype=int))


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_never_farther_than_a_dense_sample(interpolation):
    # a faceted (bilinear) piece of sphere has, below its creases, points
    # with several nearly equal nearest points: every cell that could hold
    # a nearer one is searched
    patch = _sphere(interpolation)
    rng = np.random.default_rng(10)
    u, v = rng.uniform(-0.7, 0.7, (2, 1500))
    d = rng.uniform(-0.5, 0.5, u.size)
    uc, vc = np.clip(u, -0.5, 0.5), np.clip(v, -0.5, 0.5)
    xu, xv = patch.tangents(uc, vc)
    P = (_off(patch, uc, vc, d) + (u - uc)[:, None] * xu + (v - vc)[:, None] * xv)
    found = patch.locate(P)
    g = np.linspace(-0.5, 0.5, 301)
    G, H = np.meshgrid(g, g, indexing="ij")
    dense, _ = cKDTree(patch.cartesian(G, H).reshape(-1, 3)).query(P)
    # measured: never farther, at most 1.5e-4 nearer (the sample's spacing;
    # 2.1e-5 with 801 a side); a search confined to the cells around its
    # start was 4.2e-4 farther on the bilinear patch
    assert np.all(np.abs(found.distance) <= dense + 1e-13)
    assert np.all(dense - np.abs(found.distance) < 1e-3)


def test_beyond_the_edges_and_corners():
    patch = _sphere("cubic")
    lo, hi = SQUARE
    rng = np.random.default_rng(11)
    v = rng.uniform(-0.4, 0.4, 20)
    for edge, axis in ((hi, 0), (lo, 0), (hi, 1), (lo, 1)):
        a = np.full(20, edge)
        u_, v_ = (a, v) if axis == 0 else (v, a)
        t = patch.tangents(u_, v_)[axis]
        t = np.sign(edge) * t / np.linalg.norm(t, axis=-1)[:, None]
        for lift in (0.0, 0.1, -0.1):
            P = (patch.cartesian(u_, v_) + 0.05 * t
                 + lift * patch.normal(u_, v_))
            found = patch.locate(P)
            assert not np.any(found.inside) and np.all(found.converged)
            held = found.u if axis == 0 else found.v
            assert np.all(held == edge)
            assert np.all(np.abs(found.distance) <= np.hypot(0.05, lift) + 1e-12)
            if lift:
                assert np.all(np.sign(found.distance) == np.sign(lift))
    # beyond a corner along its diagonal: the corner itself
    for cu, cv in ((lo, lo), (hi, lo), (lo, hi), (hi, hi)):
        xu, xv = patch.tangents(cu, cv)
        out = (np.sign(cu) * xu / np.linalg.norm(xu)
               + np.sign(cv) * xv / np.linalg.norm(xv))
        found = patch.locate(patch.cartesian(cu, cv) + 0.2 * out)
        assert found.u == cu and found.v == cv and not found.inside


def test_a_cut_short_search_is_refused_or_flagged():
    patch = _sphere("cubic")
    P = _off(patch, np.array([0.33, -0.21]), np.array([0.1, 0.4]),
             np.array([0.45, -0.45]))
    with pytest.raises(ValueError, match="did not converge"):
        patch.locate(P, iterations=1)
    flagged = patch.locate(P, iterations=1, strict=False)
    assert not np.any(flagged.converged)
    assert np.all(patch.locate(P).converged)
    with pytest.raises(ValueError, match="given"):
        patch.locate(P, given="spherical")
    with pytest.raises(ValueError, match="rtol"):
        patch.locate(P, rtol=0.0)


def test_points_given_in_the_frame_and_any_shape():
    patch = _sphere("cubic")
    rng = np.random.default_rng(12)
    u, v = rng.uniform(-0.5, 0.5, (2, 3, 4))
    P = patch.cartesian(u, v) + 0.2 * patch.normal(u, v)
    found = patch.locate(GeographicFrame(1.0).coordinates(P), given="frame")
    assert found.u.shape == (3, 4) and found.distance.shape == (3, 4)
    assert np.allclose(found.u, u, rtol=0, atol=1e-12)
    assert np.allclose(found.distance, 0.2, rtol=0, atol=1e-12)


def _two_patch_fault() -> Fault:
    time = np.array([0.0, 1.0])
    a = Plane.at_depth(EARTH, 35.0, 140.0, 20e3, strike=200.0, dip=15.0,
                       length=100e3, width=50e3,
                       slip=Slip.from_rake(90.0, [[[1.0]], [[2.0]]]))
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 4),
                       indexing="ij")
    base = Plane.at_depth(EARTH, 34.0, 139.0, 30e3, strike=210.0, dip=40.0,
                          length=60e3, width=40e3, slip=ZERO)
    bend = base.position(U, V) + np.stack([0.0 * U, 0.0 * U, 2e3 * U ** 2], -1)
    b = StructuredPatch(EARTH, bend, slip=Slip.from_rake(
        0.0, np.stack([np.full((2, 2), 3.0), np.full((2, 2), 4.0)]), rule="linear"))
    return Fault([a, b], support="node", time_interpolation="linear", time=time)


def test_a_fault_locates_on_its_nearest_patch_and_reads_its_slip():
    fault = _two_patch_fault()
    a, b = fault.patches
    rng = np.random.default_rng(13)
    u, v = rng.uniform(-0.45, 0.45, (2, 30))
    d = rng.uniform(-500.0, 500.0, 30)
    P = np.concatenate([a.cartesian(u, v) + d[:, None] * a.normal(u, v),
                        b.cartesian(u, v) + d[:, None] * b.normal(u, v)])
    found = fault.locate(P)
    assert np.array_equal(found.patch, np.repeat([0, 1], 30))
    assert np.allclose(found.u, np.tile(u, 2), rtol=0, atol=1e-9)
    assert np.allclose(found.distance, np.tile(d, 2), rtol=0, atol=1e-6)
    slip = fault.slip_at(P, max_distance=600.0, date=0.5, basis="fault")
    assert np.allclose(slip[:30], [0.0, 1.5, 0.0], rtol=0, atol=1e-12)
    # a bilinear slip on a bent patch is read in Cartesian components, so
    # between its nodes it is a little shorter: measured 3.4981 to 3.4996
    assert np.allclose(np.linalg.norm(slip[30:], axis=-1), 3.5, rtol=1e-3)
    want = fault.slip_vector(1, found.u[30:], found.v[30:], date=0.5)
    assert np.allclose(fault.slip_at(P[30:], max_distance=600.0, date=0.5), want,
                       rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="farther than max_distance"):
        fault.slip_at(P, max_distance=400.0, date=0.5)
    # beyond an edge, within the distance: the edge's slip
    edge = a.cartesian(0.5, 0.0) + 300.0 * a.fault_basis(0.5, 0.0)[:, 0]
    assert np.allclose(fault.slip_at(edge, max_distance=400.0, date=1.0,
                                     basis="fault"), [0.0, 2.0, 0.0], atol=1e-12)
    with pytest.raises(ValueError, match="max_distance"):
        fault.slip_at(P, max_distance=-1.0, date=0.5)


def test_the_edge_a_point_lies_beyond_is_named():
    """Beyond each edge the flag of that edge alone, beyond a corner its
    two, and inside none; for a plane in closed form, a curved patch by
    the search, and a fault of the two."""
    lo, hi = SQUARE
    plane = Plane(GeographicFrame(1.0), [30.0, 40.0, 0.8], strike=20.0, dip=50.0,
                  length=0.3, width=0.2, slip=Slip(np.zeros((1, 1, 3))))
    for patch in (plane, _sphere("cubic")):
        names = ("on_u_low", "on_u_high", "on_v_low", "on_v_high")
        for (cu, cv), want in (((lo, 0.1), {"on_u_low"}), ((hi, -0.2), {"on_u_high"}),
                               ((0.3, lo), {"on_v_low"}), ((-0.1, hi), {"on_v_high"}),
                               ((lo, hi), {"on_u_low", "on_v_high"}),
                               ((hi, lo), {"on_u_high", "on_v_low"})):
            xu, xv = patch.tangents(cu, cv)
            out = (np.sign(cu) * (abs(cu) == hi) * xu / np.linalg.norm(xu)
                   + np.sign(cv) * (abs(cv) == hi) * xv / np.linalg.norm(xv))
            scale = max(np.linalg.norm(xu), np.linalg.norm(xv))
            found = patch.locate(patch.cartesian(cu, cv) + 0.05 * scale * out)
            assert not found.inside and found.on_edge
            assert {n for n in names if getattr(found, n)} == want, (patch, cu, cv)
        inner = patch.locate(patch.cartesian(np.array([0.1, -0.3]),
                                             np.array([0.2, 0.4])))
        assert not np.any(inner.on_edge) and np.all(inner.inside)
    fault = Fault([plane])
    found = fault.locate(plane.cartesian(hi, 0.0) + 0.05 * plane.tangents(hi, 0.0)[0])
    assert found.on_u_high and not (found.on_u_low or found.on_v_low or found.on_v_high)
