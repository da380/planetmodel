"""A fault in a Cartesian frame, the frame of a box: its points are those of
`planetmodel.cartesian`, z up, its depths `FLAT`'s (depth = -z below a
reference plane at z = 0); it evaluates, lays a quadrature, locates
points near it and round-trips through the fault file, as a fault in a
geographic frame does on a sphere."""
import numpy as np
import pytest

from planetmodel import testing
from planetmodel.cartesian import FLAT
from planetmodel.fault import (CartesianFrame, Fault, Plane, Polyline, Slip,
                               StructuredPatch)

LOCAL = CartesianFrame()
RAISED = CartesianFrame(surface=0.25)
DIP = 40.0


def plane(frame=LOCAL, **kw):
    """A plane striking 30 degrees east of y and dipping 40 degrees, its
    centroid 0.6 below the frame's reference plane, slipping 2 at a rake
    of 60 degrees."""
    kw = {"strike": 30.0, "dip": DIP, "length": 0.4, "width": 0.2,
          "slip": Slip.from_rake(60.0, 2.0)} | kw
    return Plane.at_depth(frame, 0.3, -0.2, 0.6, **kw)


def grid():
    """A curved patch from a grid of depths in x and y (Slab2's form): an
    interface steepening down dip, bicubic, its slip varying by cell."""
    x, y = np.meshgrid(np.linspace(0.1, 0.5, 6), np.linspace(-0.3, 0.3, 5),
                       indexing="ij")
    depth = 0.2 + 0.4 * x + 0.5 * x ** 2 + 0.02 * y ** 2
    slip = Slip(np.random.default_rng(3).normal(size=(1, 3, 2, 3)), rule="cell")
    return StructuredPatch.from_depths(LOCAL, x.T, y.T, depth.T, slip=slip,
                                       strike_axis=0)


def swept():
    """A patch swept down a bent trace at a dip steepening with depth."""
    return StructuredPatch.swept(LOCAL, Polyline([(0.0, -0.3), (0.2, 0.0),
                                                  (0.1, 0.3)]),
                                 dip=lambda z: 20.0 + 40.0 * z, width=0.3, top=0.05,
                                 n_u=6, n_v=5, slip=Slip.from_rake(90.0, 1.0))


def test_a_depth_is_flat_s_depth():
    """`below` places a depth through `FLAT`: with the reference plane at
    z = 0 the two depths are one to the bit, and a raised plane adds its
    height; a plane given at a depth has its centroid there and runs down
    dip, its normal up into the hanging wall."""
    rng = np.random.default_rng(0)
    a, b, d = rng.uniform(-1.0, 1.0, (3, 50))
    depth, x, y, _ = FLAT.split(LOCAL.cartesian(LOCAL.below(a, b, d)))
    assert np.array_equal(depth, d) and np.array_equal(x, a) and np.array_equal(y, b)
    raised = FLAT.split(RAISED.cartesian(RAISED.below(a, b, d)))[0]
    assert np.allclose(raised, d - 0.25, rtol=0.0, atol=1e-15)
    assert np.array_equal(FLAT.points(d, a, b), LOCAL.cartesian(LOCAL.below(a, b, d)))

    p = plane()
    assert FLAT.split(p.cartesian(0.0, 0.0))[0] == pytest.approx(0.6, abs=1e-15)
    assert FLAT.split(plane(RAISED).cartesian(0.0, 0.0))[0] == pytest.approx(
        0.35, abs=1e-15)
    down = FLAT.split(p.cartesian(0.0, np.linspace(-0.5, 0.5, 5)))[0]
    assert np.all(np.diff(down) > 0.0)
    assert down[-1] - down[0] == pytest.approx(0.2 * np.sin(np.deg2rad(DIP)),
                                               abs=1e-15)
    n = p.normal(0.0, 0.0)
    assert n[2] == pytest.approx(np.cos(np.deg2rad(DIP)), abs=1e-15)
    # the grid's depths are FLAT's at its nodes
    patch = grid()
    nodes = patch.cartesian(*np.meshgrid(np.linspace(-0.5, 0.5, 5),
                                         np.linspace(-0.5, 0.5, 6), indexing="ij"))
    depth, x, y, _ = FLAT.split(nodes)
    assert np.allclose(depth, 0.2 + 0.4 * x + 0.5 * x ** 2 + 0.02 * y ** 2,
                       rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("make", [plane, lambda: plane(RAISED), grid, swept])
def test_a_cartesian_patch_keeps_the_contract(make):
    testing.check_patch(make(), rng=np.random.default_rng(1))


def test_a_cartesian_fault_lays_its_quadrature():
    """A plane's area is its length times its width at any density and its
    slip the one vector everywhere; a curved patch's area converges at the
    rule's order to the integral of its area element."""
    p = plane()
    for n_u, n_v, points in ((1, 1, 1), (3, 2, 2), (5, 4, 3)):
        q = p.quadrature(n_u, n_v, points=points)
        assert q.weights.sum() == pytest.approx(0.4 * 0.2, rel=1e-14)
        assert np.allclose(q.slip, q.slip[0], rtol=0.0, atol=1e-15)
        assert np.allclose(np.linalg.norm(q.slip, axis=1), 2.0, rtol=1e-15)
        on = p.locate(q.points)
        assert np.all(on.inside) and np.abs(on.distance).max() < 1e-15
    patch = grid()
    ref = patch.quadrature(48, 48, points=4).weights.sum()
    errors = [abs(patch.quadrature(n, n, points=2).weights.sum() / ref - 1.0)
              for n in (2, 4, 8)]
    assert errors[2] < errors[1] < errors[0] and errors[2] < 1e-6
    # the fault's quadrature carries each point's patch, the slip by its rows
    fault = Fault([p, patch])
    q = fault.quadrature([2, 3], [2, 3], points=2)
    assert set(q.patch.tolist()) == {0, 1}
    assert q.weights.sum() == pytest.approx(0.08 + ref, rel=1e-5)


def test_a_cartesian_fault_locates_points_near_it():
    """Points set off along the normal from known parameters come back to
    those parameters and that distance, signed by the hanging wall; the
    slip read there is the patch's."""
    rng = np.random.default_rng(2)
    for patch in (plane(), grid()):
        u, v = rng.uniform(-0.45, 0.45, (2, 40))
        off = rng.uniform(-0.02, 0.02, 40)
        P = patch.cartesian(u, v) + off[:, None] * patch.normal(u, v)
        found = Fault([patch]).locate(P)
        assert np.allclose(found.u, u, rtol=0.0, atol=1e-10)
        assert np.allclose(found.v, v, rtol=0.0, atol=1e-10)
        assert np.allclose(found.distance, off, rtol=0.0, atol=1e-12)
        assert np.all(found.inside)
        slip = Fault([patch]).slip_at(P, max_distance=0.05, basis="cartesian")
        assert np.allclose(slip, patch.slip_vector(u, v, basis="cartesian"),
                           rtol=0.0, atol=1e-9)
    # and given in the frame's own coordinates, the same points
    p = plane()
    P = p.cartesian(0.1, -0.2) + 0.01 * p.normal(0.1, -0.2)
    a = p.locate(P[None])
    b = p.locate(LOCAL.coordinates(P)[None], given="frame")
    assert np.allclose([a.u, a.v, a.distance], [b.u, b.v, b.distance], rtol=0.0,
                       atol=1e-15)


@pytest.mark.netcdf
def test_a_cartesian_fault_round_trips_through_the_fault_file(tmp_path):
    """Written and read back, a fault of a plane and a curved patch in a
    Cartesian frame, its reference plane raised or not, evaluates as it
    did: its frame `cartesian` with the plane's height."""
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.faultfile import read_fault_file, write_fault

    for frame, name in ((LOCAL, "local"), (RAISED, "raised")):
        fault = Fault([plane(frame), StructuredPatch(
            frame, frame.coordinates(grid().cartesian(*np.meshgrid(
                np.linspace(-0.5, 0.5, 4), np.linspace(-0.5, 0.5, 5),
                indexing="ij"))),
            slip=Slip(np.ones((1, 2, 2, 3)), rule="cell"), interpolation="cubic")])
        path = write_fault(tmp_path / f"{name}.nc", fault, source="test")
        testing.check_fault_file(path, fault, rng=np.random.default_rng(4))
        back = read_fault_file(path).fault
        assert back.frame == frame
        U, V = np.meshgrid(np.linspace(-0.5, 0.5, 7), np.linspace(-0.5, 0.5, 7),
                           indexing="ij")
        for got, want in zip(back.patches, fault.patches):
            assert np.allclose(got.cartesian(U, V), want.cartesian(U, V), rtol=0.0,
                               atol=1e-14)
            assert np.allclose(FLAT.split(got.cartesian(U, V))[0],
                               FLAT.split(want.cartesian(U, V))[0], rtol=0.0,
                               atol=1e-14)
