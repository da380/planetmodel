"""`planetmodel.fourier` against pygeoinf's periodic and bounded boxes,
whose conventions it states: the labels and the order of the components,
the coefficients of grid values and the values of coefficients, the basis,
values, gradients and adjoints at points, the derivative, the transfer
between grids, the rfftn interchange, the grid of a bounded domain, its
mask and its taper, in one to three dimensions with even and odd sizes.
Skipped where pygeoinf is not importable: it depends on this library, so
it is never one of its dependencies."""
import numpy as np
import pytest

from planetmodel.cartesian import Box as CartesianBox
from planetmodel.fourier import PaddedDomain, Transform

symmetric_space = pytest.importorskip("pygeoinf.symmetric_space")

PERIODIC = [((8,), (2.0 * np.pi,)), ((7,), (3.0,)), ((6, 4), (1.0, 2.0)),
            ((5, 5), (1.0, 1.0)), ((8, 9), (2.0, 1.5)), ((4, 3, 5), (1.0, 2.0, 3.0)),
            ((4, 4, 4), (1.0, 1.0, 1.0))]
BOUNDED = [((16,), ((0.0, 1.0),), (0.25,)), ((15,), ((2.0, 5.0),), (0.5,)),
           ((16, 12), ((0.0, 1.0), (-1.0, 1.0)), (0.1, 0.2)),
           ((9, 10), ((5.0, 6.0), (0.0, 2.0)), (0.3, 0.0)),
           ((6, 5, 4), ((0.0, 1.0), (0.0, 1.0), (-2.0, -1.0)), (0.2, 0.1, 0.3))]


def close(a, b, *, rel=1e-12):
    """Agreement to rounding, relative to the larger of the two."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b), initial=0.0) <= rel * max(np.max(np.abs(b),
                                                                 initial=0.0), 1.0)


def pairs(case, kind):
    """(pygeoinf's space, planetmodel's transform) of one case."""
    if kind == "periodic":
        shape, lengths = case
        return symmetric_space.PeriodicBox(shape, lengths=lengths), Transform(shape,
                                                                             lengths)
    shape, bounds, padding = case
    return (symmetric_space.Box(shape, bounds=bounds, padding=padding),
            Transform(shape, PaddedDomain(bounds, padding=padding)))


EVERY = ([pytest.param(c, "periodic", id=f"torus{c[0]}") for c in PERIODIC]
         + [pytest.param(c, "bounded", id=f"box{c[0]}") for c in BOUNDED])


def inside_points(space, rng, count):
    """Points of the domain proper (the whole box for a torus)."""
    if hasattr(space, "bounds"):
        lo, hi = np.array(space.bounds).T
    else:
        lo, hi = np.zeros(len(space.lengths)), np.asarray(space.lengths)
    return rng.uniform(lo, hi, (count, lo.size))


@pytest.mark.parametrize("case, kind", EVERY)
def test_the_labels_and_the_order_are_pygeoinfs(case, kind):
    space, t = pairs(case, kind)
    assert t.size == space.dim
    assert np.array_equal(t.wavevectors, space.wavevectors)
    assert np.array_equal(t.phases, space.phases)
    assert np.array_equal(t.degrees, space.degrees)
    close(t.laplacian_eigenvalues, space.laplacian_eigenvalues)
    assert np.array_equal(t.component_of(t.wavevectors, phase=t.phases),
                          space.component_of(space.wavevectors, phase=space.phases))
    for ours, theirs in zip(t.grid_axes, space.grid_axes):
        close(ours, theirs, rel=1e-14)


@pytest.mark.parametrize("case, kind", EVERY)
def test_coefficients_and_grid_values_are_pygeoinfs(case, kind):
    rng = np.random.default_rng(0)
    space, t = pairs(case, kind)
    x = rng.standard_normal(t.shape)
    close(t.analyse_grid(x), space.to_components(x))
    c = rng.standard_normal(t.size)
    close(t.synthesise_grid(c), space.from_components(c))
    plain = space.to_coefficients(x)
    close(t.to_rfft(t.analyse_grid(x)).real, plain.real)
    close(t.to_rfft(t.analyse_grid(x)).imag, plain.imag)
    close(t.from_rfft(plain), space.to_components(space.from_coefficients(plain)))


@pytest.mark.parametrize("case, kind", EVERY)
def test_values_gradients_and_adjoints_at_points_are_pygeoinfs(case, kind):
    rng = np.random.default_rng(1)
    space, t = pairs(case, kind)
    x = rng.standard_normal(t.shape)
    c = t.analyse_grid(x)
    points = inside_points(space, rng, 25)
    for p in points[:3]:
        close(t.basis(p), space.basis_at(p))
    # both through finufft at 1e-12: the two agree far inside that
    close(t.synthesise(c, points), space.evaluate(x, points), rel=1e-11)
    close(t.synthesise(c, points, route="direct"), space.evaluate(x, points),
          rel=1e-11)
    close(t.synthesise_gradient(c, points), space.evaluate_gradient(x, points),
          rel=1e-11)
    w = rng.standard_normal(25)
    close(t.accumulate(w, points), space.accumulate(w, points), rel=1e-11)


@pytest.mark.parametrize("case, kind", EVERY)
def test_the_derivative_is_pygeoinfs(case, kind):
    rng = np.random.default_rng(2)
    space, t = pairs(case, kind)
    x = rng.standard_normal(t.shape)
    for axis in range(t.dimension):
        close(t.synthesise_grid(t.derivative(t.analyse_grid(x), axis)),
              space.derivative_operator(axis=axis)(x))


@pytest.mark.parametrize("coarse, fine", [((16,), (32,)), ((7,), (12,)),
                                          ((8, 6), (16, 12)), ((4, 4, 4), (6, 5, 4))])
def test_the_transfer_between_grids_is_pygeoinfs(coarse, fine):
    rng = np.random.default_rng(3)
    lengths = (1.0, 2.0, 3.0)[:len(coarse)]
    for a, b in ((coarse, fine), (fine, coarse)):
        here = symmetric_space.PeriodicBox(a, lengths=lengths)
        there = symmetric_space.PeriodicBox(b, lengths=lengths)
        ta, tb = Transform(a, lengths), Transform(b, lengths)
        x = rng.standard_normal(a)
        close(tb.synthesise_grid(ta.transfer(ta.analyse_grid(x), tb)),
              here.degree_transfer_operator(there)(x))


@pytest.mark.parametrize("case", BOUNDED, ids=[str(c[0]) for c in BOUNDED])
def test_the_mask_and_the_taper_are_pygeoinfs(case):
    space, t = pairs(case, "bounded")
    assert np.array_equal(t.mask, space.interior_mask)
    close(t.taper, space._taper, rel=1e-14)
    close(t.lengths, space.lengths, rel=1e-15)
    assert t.domain.domain_volume == pytest.approx(space.domain_volume, rel=1e-15)


def test_a_cartesian_box_is_pygeoinfs_box():
    rng = np.random.default_rng(4)
    box = CartesianBox([(-3.0, 5.0), (1.0, 2.5)], padding=(0.5, 0.25))
    space = symmetric_space.Box((16, 12), bounds=box.bounds, padding=box.padding)
    t = Transform((16, 12), box)
    x = rng.standard_normal(t.shape)
    points = inside_points(space, rng, 20)
    close(t.synthesise(t.analyse_grid(x), points), space.evaluate(x, points),
          rel=1e-11)
    assert np.array_equal(t.mask, space.interior_mask)
    for ours, theirs in zip(t.grid_axes, space.grid_axes):
        close(ours, theirs, rel=1e-14)


def test_four_axes_by_the_direct_route_are_pygeoinfs():
    rng = np.random.default_rng(5)
    shape = (3, 4, 2, 3)
    space = symmetric_space.PeriodicBox(shape)
    t = Transform(shape, [2.0 * np.pi] * 4)
    x = rng.standard_normal(shape)
    points = inside_points(space, rng, 6)
    close(t.synthesise(t.analyse_grid(x), points, route="direct"),
          space.evaluate(x, points))
    w = rng.standard_normal(6)
    close(t.accumulate(w, points, route="direct"), space.accumulate(w, points))
