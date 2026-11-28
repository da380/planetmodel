"""The real Fourier series of `planetmodel.fourier`: the packing, the basis
and its Nyquist rule, the grid transforms against scipy's rfftn, the
transforms at points through finufft against the direct sum over the
basis (the oracle), the gradient against finite differences, the adjoint
identity, the transforms at prepared points against those at points given
on each call, and the padded domain, in one to three dimensions with even
and odd sizes."""
import copy
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.fft import irfftn, rfftn

from planetmodel.cartesian import Box, Rectangle
from planetmodel.fourier import (
    POINT_EPSILON,
    POSITION_RTOL,
    Packing,
    PaddedDomain,
    PointTransform,
    Transform,
)

# Even and odd sizes, mixed parity, unequal periods, and padded domains whose
# box does not start at the origin.
CASES = [
    ((8,), [2.0 * np.pi]),
    ((7,), 3.0),
    ((6, 4), [1.0, 2.0]),
    ((5, 5), PaddedDomain([(0.0, 1.0), (-1.0, 2.0)], padding=0.3)),
    ((8, 9), PaddedDomain([(2.0, 3.0), (5.0, 6.0)], padding=(0.25, 0.1))),
    ((4, 3, 5), [1.0, 2.0, 3.0]),
    ((4, 4, 4), PaddedDomain([(0.0, 1.0), (0.0, 1.0), (-1.0, 0.0)], padding=0.2)),
]
IDS = [f"{shape}" for shape, _ in CASES]


def transform(case):
    shape, domain = case
    return Transform(shape, domain)


def random_points(t, rng, count=40, *, spill=0.0):
    """Points spread over the box, `spill` of a period beyond it each side."""
    lo = np.asarray(t.origin) - spill * np.asarray(t.lengths)
    hi = np.asarray(t.origin) + (1.0 + spill) * np.asarray(t.lengths)
    return rng.uniform(lo, hi, (count, t.dimension))


def scale(t, c):
    """The field's size: the sum of its coefficients' moduli times the
    larger basis amplitude, bounding it everywhere."""
    return np.abs(c).sum(axis=-1).max() * np.sqrt(2.0 / t.volume)


def close(a, b, *, rel=1e-12):
    """Agreement to rounding, relative to the largest entry of `b` or one."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b), initial=0.0) <= rel * np.max(np.abs(b), initial=1.0)


# -- the packing -------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1,), (2,), (8,), (7,), (1, 4), (6, 4), (5, 5),
                                   (8, 9), (4, 3, 5), (4, 4, 4), (3, 4, 2, 3)])
def test_the_packing_counts_one_component_per_point(shape):
    p = Packing(shape)
    assert p.size == int(np.prod(shape)) == p.wavevectors.shape[1] == p.phases.size
    assert p.rfft_shape == shape[:-1] + (shape[-1] // 2 + 1,)
    labels = {(tuple(k), s) for k, s in zip(p.wavevectors.T.tolist(), p.phases)}
    assert len(labels) == p.size
    # the self-conjugate modes: every wavenumber zero or an even axis's Nyquist
    count = p.fixed_indices.size
    assert count == 2 ** sum(n % 2 == 0 for n in shape)
    fixed = p.wavevectors[:, :count]
    nyquist = np.array([n // 2 if n % 2 == 0 else 0 for n in shape])[:, None]
    assert np.all((fixed == 0) | (fixed == nyquist))
    assert np.all(p.phases[:count] == 0)
    assert np.array_equal(p.phases[count:], np.tile([0, 1], (p.size - count) // 2))
    # a pair's cosine and sine share their wavevector; wavenumbers are signed,
    # with the Nyquist at +n / 2
    assert np.array_equal(p.wavevectors[:, count::2], p.wavevectors[:, count + 1::2])
    upper = np.array([n // 2 for n in shape])[:, None]
    lower = np.array([-((n - 1) // 2) for n in shape])[:, None]
    assert np.all((p.wavevectors >= lower) & (p.wavevectors <= upper))
    assert np.array_equal(p.degrees, np.floor(np.linalg.norm(p.wavevectors, axis=0)))
    assert np.array_equal(p.measures, np.where(np.arange(p.size) < count, 1.0, 0.5))


@pytest.mark.parametrize("shape", [(8,), (7,), (6, 4), (5, 5), (4, 3, 5)])
def test_component_of_finds_every_label_modulo_the_grid(shape):
    p = Packing(shape)
    every = np.arange(p.size)
    assert np.array_equal(p.component_of(p.wavevectors, phase=p.phases), every)
    shifted = p.wavevectors + 3 * np.array(shape)[:, None]
    assert np.array_equal(p.component_of(shifted, phase=p.phases), every)
    i = p.size - 1
    assert p.component_of(p.wavevectors[:, i], phase=p.phases[i]) == i
    with pytest.raises(ValueError, match="no component"):
        p.component_of(np.zeros(len(shape), dtype=int), phase=1)
    with pytest.raises(ValueError, match="components"):
        p.component_of(np.zeros(len(shape) + 1, dtype=int))


@pytest.mark.parametrize("shape", [(8,), (7,), (6, 4), (5, 5), (4, 3, 5)])
def test_a_cosine_is_found_under_either_member_of_its_pair_and_a_sine_under_one(shape):
    """cos(k . x) is cos(-k . x), so the conjugate of a pair's representative
    names the same cosine: the grid's cosine of -k, analysed, is that one
    component, sqrt(V / 2) times its amplitude.  A sine changes sign, so it
    is named by the representative alone."""
    t = Transform(shape, np.arange(1.0, len(shape) + 1.0))
    p = t.packing
    sines = np.flatnonzero(p.phases == 1)
    k = p.wavevectors[:, sines]
    assert np.array_equal(p.component_of(-k), sines - 1)
    # an array of both members and of self-conjugate modes, each found
    every = np.concatenate([p.wavevectors, -p.wavevectors], axis=1)
    cosines = np.where(p.phases == 1, np.arange(p.size) - 1, np.arange(p.size))
    assert np.array_equal(p.component_of(every), np.tile(cosines, 2))
    x = t.grid_points
    for j in sines[::max(1, sines.size // 4)]:
        minus = -p.wavevectors[:, j]
        theta = sum(2.0 * np.pi * minus[i] * x[..., i] / t.lengths[i]
                    for i in range(t.dimension))
        c = t.analyse_grid(np.cos(theta))
        assert p.component_of(minus) == j - 1 == p.component_of(minus.tolist())
        assert np.isclose(c[p.component_of(minus)], np.sqrt(0.5 * t.volume))
        assert np.isclose(np.sum(c ** 2), 0.5 * t.volume)
        # the sine of -k is minus the representative's, which a position
        # cannot say: refused, naming the representative
        named = str(p.wavevectors[:, j].tolist())
        with pytest.raises(ValueError, match=f"conjugate {named}".replace(
                "[", r"\[").replace("]", r"\]")):
            p.component_of(minus, phase=1)
    with pytest.raises(ValueError, match="self-conjugate mode has no sine"):
        p.component_of(np.zeros(len(shape), dtype=int), phase=1)


def test_the_packing_is_built_once_per_shape_and_read_only():
    a, b = Transform((6, 4), [1.0, 1.0]), Transform((6, 4), [2.0, 3.0])
    assert a.packing is b.packing
    with pytest.raises(ValueError):
        a.wavevectors[0, 0] = 5


# -- the basis ---------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_basis_is_orthonormal_under_the_grid_quadrature(case):
    t = transform(case)
    b = t.basis(t.grid_points).reshape(t.size, t.size)
    gram = (t.volume / t.size) * b.T @ b
    assert np.allclose(gram, np.identity(t.size), atol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_nyquist_rule(case):
    """Over the continuous box -- a grid fine enough that the quadrature is
    exact for every product -- the basis is orthonormal but for the
    self-conjugate modes with a Nyquist wavenumber, whose squared norm is a
    half: the grid's norm is the one kept."""
    t = transform(case)
    fine = Transform(tuple(2 * n + 1 for n in t.shape), t.domain)
    b = t.basis(fine.grid_points).reshape(-1, t.size)
    gram = (fine.volume / fine.size) * b.T @ b
    count = t.packing.fixed_indices.size
    nyquist = np.any(t.wavevectors[:, :count] != 0, axis=0)
    want = np.ones(t.size)
    want[:count][nyquist] = 0.5
    assert np.allclose(gram, np.diag(want), atol=1e-12)
    # a Nyquist cosine on the grid is one component, sqrt(V) times its amplitude
    if np.any(nyquist):
        k = t.wavevectors[:, np.flatnonzero(nyquist)[0]]
        x = t.grid_points
        theta = sum(2.0 * np.pi * k[i] * (x[..., i] - t.origin[i]) / t.lengths[i]
                    for i in range(t.dimension))
        c = t.analyse_grid(np.cos(theta))
        assert np.isclose(c[t.component_of(k)], np.sqrt(t.volume))
        assert np.isclose(np.sum(c ** 2), t.volume)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_basis_at_the_grid_points_is_the_field(case):
    rng = np.random.default_rng(0)
    t = transform(case)
    x = rng.standard_normal(t.shape)
    values = t.basis(t.grid_points) @ t.analyse_grid(x)
    assert np.allclose(values, x, atol=1e-12)


def test_the_basis_off_the_grid_takes_the_representative_s_wavevector():
    """A pair at a leading axis's Nyquist (+n / 2, k) is not its fold
    (-n / 2, k) off the grid: the representative's sign is the basis's."""
    t = Transform((8, 8), [1.0, 1.0])
    i = t.component_of([4, 1])
    point = np.array([0.137, 0.421])
    want = np.sqrt(2.0) * np.cos(2.0 * np.pi * (4 * point[0] + point[1]))
    assert np.isclose(t.basis(point)[i], want)
    assert not np.isclose(want, np.sqrt(2.0) * np.cos(
        2.0 * np.pi * (-4 * point[0] + point[1])))


def test_the_circle_of_unit_radius_and_the_disk_s_ring_convention():
    """On a circle of period 2 pi with an odd number of angles, the basis is
    the disk's ring modes (1 / sqrt(2 pi), cos(m t) / sqrt(pi), sin(m t) /
    sqrt(pi)) but for the sign of the sines: a ring coefficient array (2,
    mmax + 1) is the components [c0, c[0, 1], -c[1, 1], c[0, 2], ...]."""
    mmax = 4
    t = Transform((2 * mmax + 1,), 2.0 * np.pi)
    angles = np.random.default_rng(10).uniform(0.0, 2.0 * np.pi, 7)
    m = np.arange(1, mmax + 1)[:, None]
    ring_cos = np.cos(m * angles) / np.sqrt(np.pi)
    ring_sin = np.sin(m * angles) / np.sqrt(np.pi)
    b = t.basis(angles[:, None]).T
    assert np.allclose(b[0], 1.0 / np.sqrt(2.0 * np.pi))
    order = t.wavevectors[0, 1::2]
    assert np.array_equal(order, np.arange(1, mmax + 1))
    assert np.allclose(b[1::2], ring_cos) and np.allclose(b[2::2], -ring_sin)


# -- the grid transforms -----------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_analysis_and_synthesis_invert_each_other(case):
    rng = np.random.default_rng(1)
    t = transform(case)
    x = rng.standard_normal((3, 2) + t.shape)                 # batch axes first
    c = t.analyse_grid(x)
    assert c.shape == (3, 2, t.size)
    assert np.allclose(t.synthesise_grid(c), x, atol=1e-13)
    c = rng.standard_normal((4, t.size))
    assert np.allclose(t.analyse_grid(t.synthesise_grid(c)), c, atol=1e-13)
    # Parseval: the components' sum of squares is the grid's quadrature
    grid = tuple(range(-t.dimension, 0))
    assert np.allclose(np.sum(t.analyse_grid(x) ** 2, axis=-1),
                       t.volume / t.size * np.sum(x ** 2, axis=grid))


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_rfft_interchange(case):
    rng = np.random.default_rng(2)
    t = transform(case)
    x = rng.standard_normal((2,) + t.shape)
    axes = tuple(range(-t.dimension, 0))
    plain = rfftn(x, s=t.shape, axes=axes)
    c = t.analyse_grid(x)
    assert np.allclose(t.to_rfft(c), plain, atol=1e-12)
    assert np.array_equal(t.from_rfft(plain), c)
    assert np.allclose(irfftn(t.to_rfft(c), s=t.shape, axes=axes), x, atol=1e-13)
    assert np.allclose(t.from_rfft(t.to_rfft(c)), c, atol=1e-14)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_one_vector_in_the_rfft_interchange_is_a_batch_s_row(case):
    """One array or vector without batch axes goes by its own steps, the
    batched ones' elementwise arithmetic on the same entries, so the values
    are a batch's rows exactly."""
    rng = np.random.default_rng(11)
    t = transform(case)
    axes = tuple(range(-t.dimension, 0))
    plain = rfftn(rng.standard_normal((3,) + t.shape), s=t.shape, axes=axes)
    batch = t.from_rfft(plain)
    for i in range(3):
        one = t.from_rfft(plain[i])
        assert one.shape == (t.size,) and np.array_equal(one, batch[i])
    c = rng.standard_normal((3, t.size))
    batch = t.to_rfft(c)
    for i in range(3):
        one = t.to_rfft(c[i])
        assert one.shape == t.packing.rfft_shape and np.array_equal(one, batch[i])
    # a batch of one is a batch
    assert t.from_rfft(plain[:1]).shape == (1, t.size)
    assert t.to_rfft(c[:1]).shape == (1,) + t.packing.rfft_shape


# -- at points ---------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_synthesis_at_points_is_the_direct_sum(case):
    rng = np.random.default_rng(3)
    t = transform(case)
    c = rng.standard_normal((2, t.size))
    x = random_points(t, rng).reshape(5, 8, t.dimension)
    direct = t.synthesise(c, x, route="direct")
    assert direct.shape == (2, 5, 8)
    assert np.allclose(direct, np.einsum("pqn,bn->bpq", t.basis(x), c), atol=1e-13)
    fast = t.synthesise(c, x)
    assert np.max(np.abs(fast - direct)) < 100 * POINT_EPSILON * scale(t, c)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_a_point_outside_the_box_is_its_periodic_image(case):
    rng = np.random.default_rng(4)
    t = transform(case)
    c = rng.standard_normal(t.size)
    x = random_points(t, rng, 10)
    shift = rng.integers(-3, 4, (10, t.dimension)) * np.asarray(t.lengths)
    for route in ("direct", "nufft"):
        assert np.allclose(t.synthesise(c, x + shift, route=route),
                           t.synthesise(c, x, route=route), atol=1e-11)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_gradient_is_the_finite_difference(case):
    rng = np.random.default_rng(5)
    t = transform(case)
    c = rng.standard_normal(t.size)
    x = random_points(t, rng, 12)
    direct = t.synthesise_gradient(c, x, route="direct")
    assert direct.shape == (12, t.dimension)
    for axis in range(t.dimension):
        h = 1e-5 * t.lengths[axis]
        step = np.zeros(t.dimension)
        step[axis] = h
        fd = (t.synthesise(c, x + step, route="direct")
              - t.synthesise(c, x - step, route="direct")) / (2.0 * h)
        assert np.allclose(direct[:, axis], fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max())
    kmax = np.abs(t.wavenumbers).max()
    fast = t.synthesise_gradient(c, x)
    assert np.max(np.abs(fast - direct)) < 100 * POINT_EPSILON * kmax * scale(t, c)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_adjoint_identity(case):
    """<synthesise(c), w> = <c, accumulate(w)>: to rounding on each route,
    finufft's type 1 being the adjoint of its type 2."""
    rng = np.random.default_rng(6)
    t = transform(case)
    c = rng.standard_normal((3, t.size))
    x = random_points(t, rng, 30, spill=0.5).reshape(6, 5, t.dimension)
    w = rng.standard_normal((3, 6, 5))
    for route in ("nufft", "direct"):
        values = t.synthesise(c, x, route=route)
        back = t.accumulate(w, x, route=route)
        assert back.shape == (3, t.size)
        lhs, rhs = np.sum(values * w, axis=(1, 2)), np.sum(c * back, axis=1)
        assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12 * np.abs(lhs).max())
    direct = t.accumulate(w, x, route="direct")
    fast = t.accumulate(w, x)
    size = np.abs(w).sum(axis=(1, 2)).max() * np.sqrt(2.0 / t.volume)
    assert np.max(np.abs(fast - direct)) < 100 * POINT_EPSILON * size


def test_no_points_give_nothing_and_add_nothing():
    t = Transform((6, 4), [1.0, 2.0])
    c = np.ones((2, t.size))
    for route in ("nufft", "direct"):
        assert t.synthesise(c, np.zeros((0, 2)), route=route).shape == (2, 0)
        gradient = t.synthesise_gradient(c, np.zeros((0, 2)), route=route)
        assert gradient.shape == (2, 0, 2)
        assert np.array_equal(t.accumulate(np.zeros((2, 0)), np.zeros((0, 2)),
                                           route=route), np.zeros((2, t.size)))


def test_a_single_point_and_a_single_field():
    t = Transform((7,), 2.0)
    c = np.random.default_rng(7).standard_normal(t.size)
    v = t.synthesise(c, [0.3])
    assert v.shape == () and np.isclose(v, t.basis([0.3]) @ c)
    assert t.synthesise_gradient(c, [0.3]).shape == (1,)
    assert t.accumulate(2.0, [0.3]).shape == (t.size,)


# -- prepared points ---------------------------------------------------------------

@pytest.mark.parametrize("route", ["nufft", "direct"])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_prepared_points_give_the_transforms_at_points(case, route):
    """Applied again and again, to batches of different sizes and to one
    field, a preparation gives what the transform gives with the points."""
    rng = np.random.default_rng(12)
    t = transform(case)
    x = random_points(t, rng, 30, spill=0.5).reshape(6, 5, t.dimension)
    prepared = t.at(x, route=route)
    assert isinstance(prepared, PointTransform)
    assert prepared.transform is t and prepared.route == route
    assert prepared.shape == (6, 5)
    for batch in [(3,), (), (2, 2), (3,)]:
        c = rng.standard_normal(batch + (t.size,))
        w = rng.standard_normal(batch + (6, 5))
        values = prepared.synthesise(c)
        assert values.shape == batch + (6, 5)
        close(values, t.synthesise(c, x, route=route))
        gradient = prepared.synthesise_gradient(c)
        assert gradient.shape == batch + (6, 5, t.dimension)
        close(gradient, t.synthesise_gradient(c, x, route=route))
        back = prepared.accumulate(w)
        assert back.shape == batch + (t.size,)
        close(back, t.accumulate(w, x, route=route))
        # the adjoint identity, to rounding of the sum's terms
        lhs = np.sum(values * w, axis=(-2, -1))
        rhs = np.sum(c * back, axis=-1)
        terms = np.sum(np.abs(values * w), axis=(-2, -1))
        assert np.all(np.abs(lhs - rhs) <= 1e-12 * np.max(terms))


def test_prepared_points_copy_and_pickle_without_their_plans():
    rng = np.random.default_rng(13)
    t = Transform((8, 9), PaddedDomain([(2.0, 3.0), (5.0, 6.0)], padding=0.1))
    x = random_points(t, rng, 50)
    prepared = t.at(x)
    c, w = rng.standard_normal(t.size), rng.standard_normal(50)
    values, back = prepared.synthesise(c), prepared.accumulate(w)
    for other in (pickle.loads(pickle.dumps(prepared)), copy.copy(prepared),
                  copy.deepcopy(prepared)):
        close(other.synthesise(c), values)
        close(other.accumulate(w), back)
        close(other.synthesise_gradient(c), prepared.synthesise_gradient(c))
    assert "points of shape (50,)" in repr(prepared)


def test_prepared_points_from_several_threads():
    """Applications from several threads take turns on finufft's plans."""
    rng = np.random.default_rng(14)
    t = Transform((16, 12), [1.0, 2.0])
    x = random_points(t, rng, 400)
    prepared = t.at(x)
    c = rng.standard_normal((12, t.size))
    w = rng.standard_normal((12, 400))
    with ThreadPoolExecutor(max_workers=3) as pool:
        values = list(pool.map(prepared.synthesise, c))
        back = list(pool.map(prepared.accumulate, w))
    for i in range(12):
        close(values[i], t.synthesise(c[i], x))
        close(back[i], t.accumulate(w[i], x))


def test_prepared_points_of_none_one_and_more_axes():
    t = Transform((6, 4), [1.0, 2.0])
    c = np.ones((2, t.size))
    for route in ("nufft", "direct"):
        none = t.at(np.zeros((0, 2)), route=route)
        assert none.synthesise(c).shape == (2, 0)
        assert none.synthesise_gradient(c).shape == (2, 0, 2)
        assert np.array_equal(none.accumulate(np.zeros((2, 0))), np.zeros((2, t.size)))
    one = Transform((7,), 2.0)
    c = np.random.default_rng(15).standard_normal(one.size)
    single = one.at([0.3])
    assert single.shape == ()
    assert single.synthesise(c).shape == () and np.isclose(single.synthesise(c),
                                                          one.basis([0.3]) @ c)
    assert single.synthesise_gradient(c).shape == (1,)
    assert single.accumulate(2.0).shape == (one.size,)
    four = Transform((3, 4, 2, 3), [1.0, 2.0, 1.0, 1.0])
    p = random_points(four, np.random.default_rng(16), 5)
    c = np.random.default_rng(17).standard_normal(four.size)
    close(four.at(p, route="direct").synthesise(c),
          four.synthesise(c, p, route="direct"))
    with pytest.raises(ValueError, match="route='direct'"):
        four.at(p)


def test_prepared_points_refuse_what_is_wrong():
    t = Transform((6, 4), [1.0, 2.0])
    with pytest.raises(ValueError, match="route must be one of"):
        t.at(np.zeros((3, 2)), route="fast")
    with pytest.raises(ValueError, match="2 coordinates each"):
        t.at(np.zeros((3, 3)))
    prepared = t.at(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="24 entries"):
        prepared.synthesise(np.zeros(23))
    with pytest.raises(ValueError, match="24 entries"):
        prepared.synthesise_gradient(np.zeros((2, 25)))
    with pytest.raises(ValueError, match="trailing shape"):
        prepared.accumulate(np.zeros(4))


# -- derivative and transfer -------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_derivative_is_the_gradient_without_the_nyquist_modes(case):
    rng = np.random.default_rng(8)
    t = transform(case)
    c = rng.standard_normal(t.size)
    count = t.packing.fixed_indices.size
    c[1:count] = 0.0                    # the Nyquist cosines' slopes are not held
    x = random_points(t, rng, 10)
    gradient = t.synthesise_gradient(c, x, route="direct")
    for axis in range(t.dimension):
        d = t.derivative(c, axis)
        assert np.all(d[:count] == 0.0)
        assert np.allclose(t.synthesise(d, x, route="direct"), gradient[:, axis],
                           atol=1e-11 * np.abs(gradient).max())


def test_a_transfer_keeps_a_nyquist_cosine():
    coarse, fine = Transform((16,), 2.0 * np.pi), Transform((32,), 2.0 * np.pi)
    theta, fine_theta = coarse.grid_axes[0], fine.grid_axes[0]
    up = coarse.transfer(coarse.analyse_grid(np.cos(8.0 * theta)), fine)
    assert np.allclose(fine.synthesise_grid(up), np.cos(8.0 * fine_theta))
    down = fine.transfer(fine.analyse_grid(np.cos(8.0 * fine_theta)
                                           + np.cos(3.0 * fine_theta)), coarse)
    assert np.allclose(coarse.synthesise_grid(down),
                       np.cos(8.0 * theta) + np.cos(3.0 * theta))
    torus, finer = Transform((8, 6), [1.0, 1.0]), Transform((16, 12), [1.0, 1.0])
    x, y = np.meshgrid(*torus.grid_axes, indexing="ij")
    xf, yf = np.meshgrid(*finer.grid_axes, indexing="ij")
    field = np.cos(2.0 * np.pi * (4.0 * x + 3.0 * y))
    moved = torus.transfer(torus.analyse_grid(field), finer)
    assert np.allclose(finer.synthesise_grid(moved),
                       np.cos(2.0 * np.pi * (4.0 * xf + 3.0 * yf)))
    # the same field, the same grid: the identity
    c = np.arange(torus.size, dtype=float)
    assert np.array_equal(torus.transfer(c, Transform((8, 6), [1.0, 1.0])), c)
    with pytest.raises(ValueError, match="one domain"):
        torus.transfer(c, Transform((8, 6), [1.0, 2.0]))


# -- the padded domain -------------------------------------------------------------

def test_a_cartesian_box_and_rectangle_are_read_through_their_bounds_and_padding():
    box = Box([(-3.0, 5.0), (1.0, 2.5)], padding=(0.5, 0.25))
    t = Transform((16, 12), box)
    assert t.domain.bounds == box.bounds and t.domain.padding == box.padding
    assert t.domain.extent == box.extent and t.lengths == box.lengths
    assert t.origin == (box.extent[0][0], box.extent[1][0])
    assert np.isclose(t.grid_axes[0][0], -3.5) and np.isclose(t.grid_axes[1][0], 0.75)
    rect = Rectangle((0.0, 4.0), padding=1.0)
    r = Transform((24,), rect)
    assert r.domain == PaddedDomain([(0.0, 4.0)], padding=1.0)
    assert r.domain.bounds == rect.bounds and r.domain.padding == rect.padding
    assert r.domain.extent == rect.extent and r.lengths == rect.lengths
    assert r.origin == (rect.extent[0][0],)
    # each is its padded domain, held the same way
    for domain in (box, rect):
        assert PaddedDomain.of(domain) == PaddedDomain(domain.bounds,
                                                       padding=domain.padding)
    model = Transform((16, 12), Box([(0.0, 2.0), (0.0, 1.0)]))
    assert model.domain == PaddedDomain.of([2.0, 1.0])          # its box, its period
    assert model.mask.all() and np.array_equal(model.taper, np.ones((16, 12)))


def test_a_torus_is_its_box_from_the_origin():
    d = PaddedDomain.of([2.0, 3.0])
    assert d.bounds == ((0.0, 2.0), (0.0, 3.0)) and d.padding == (0.0, 0.0)
    assert d.lengths == (2.0, 3.0) and d.origin == (0.0, 0.0) and d.volume == 6.0
    assert PaddedDomain.of(2.5) == PaddedDomain([(0.0, 2.5)])
    assert PaddedDomain.of(d) is d
    t = Transform((4,), 2.0)
    assert np.array_equal(t.grid_axes[0], np.arange(4) * 2.0 / 4)


def test_the_mask_and_the_taper():
    d = PaddedDomain([(0.0, 1.0)], padding=0.25)
    t = Transform((12,), d)
    axis = t.grid_axes[0]                    # -0.25 + 0.125 j
    assert np.allclose(axis, -0.25 + 0.125 * np.arange(12))
    assert np.array_equal(np.flatnonzero(t.mask), np.arange(2, 11))   # 0 ... 1
    assert np.all(t.taper[t.mask] == 1.0)
    outside = t.taper[~t.mask]
    assert np.all((outside >= 0.0) & (outside < 1.0))
    assert t.taper[0] == pytest.approx(0.0, abs=1e-15)        # the far edge
    assert t.taper[1] == pytest.approx(0.5) and t.taper[11] == pytest.approx(0.5)
    # in two dimensions, the product of the axes'
    d2 = PaddedDomain([(0.0, 1.0), (0.0, 2.0)], padding=(0.25, 0.5))
    t2 = Transform((12, 12), d2)
    one = [Transform((12,), PaddedDomain([b], padding=p))
           for b, p in zip(d2.bounds, d2.padding)]
    assert np.allclose(t2.taper, np.multiply.outer(one[0].taper, one[1].taper))
    assert np.array_equal(t2.mask, np.logical_and.outer(one[0].mask, one[1].mask))
    assert d2.domain_volume == 2.0 and d2.volume == pytest.approx(1.5 * 3.0)


def test_contains_and_checked():
    d = PaddedDomain([(0.0, 1.0), (5.0, 6.0)], padding=0.1)
    edge = np.array([[1.0, 6.0], [0.0, 5.0], [1.0 + 0.5 * POSITION_RTOL, 5.5]])
    assert d.contains(edge).all()
    assert np.array_equal(d.checked(edge), edge)
    out = np.array([[0.5, 5.5], [0.5, 4.95]])
    assert d.contains(out).tolist() == [True, False]
    message = (r"every point must lie in the domain, \[0, 1\] x \[5, 6\]; one is "
               r"at \(0.5, 4.95\)")
    with pytest.raises(ValueError, match=message):
        d.checked(out)
    with pytest.raises(ValueError, match="2 coordinates"):
        d.contains([0.5, 5.5, 0.0])


# -- more axes, and refusals -------------------------------------------------------

def test_four_axes_take_the_direct_route():
    rng = np.random.default_rng(9)
    t = Transform((3, 4, 2, 3), [1.0, 2.0, 1.0, 1.0])
    x = rng.standard_normal(t.shape)
    c = t.analyse_grid(x)
    assert np.allclose(t.synthesise_grid(c), x, atol=1e-13)
    assert np.allclose(t.synthesise(c, t.grid_points, route="direct"), x, atol=1e-12)
    w = rng.standard_normal(5)
    p = random_points(t, rng, 5)
    assert np.isclose(np.dot(t.synthesise(c, p, route="direct"), w),
                      np.dot(c, t.accumulate(w, p, route="direct")))
    with pytest.raises(ValueError, match="route='direct'"):
        t.synthesise(c, p)


def test_refusals_name_what_is_wrong():
    t = Transform((6, 4), [1.0, 2.0])
    with pytest.raises(ValueError, match="positive sizes"):
        Transform((6, 0), [1.0, 2.0])
    with pytest.raises(ValueError, match="2 sizes"):
        Transform((6,), [1.0, 2.0])
    with pytest.raises(ValueError, match="positive lengths"):
        Transform((6,), [-1.0])
    with pytest.raises(ValueError, match="must be finite and increase"):
        PaddedDomain([(1.0, 0.0)])
    with pytest.raises(ValueError, match="non-negative"):
        PaddedDomain([(0.0, 1.0)], padding=-0.1)
    with pytest.raises(ValueError, match="one padding or 2"):
        PaddedDomain([(0.0, 1.0), (0.0, 1.0)], padding=(0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="24 entries"):
        t.synthesise_grid(np.zeros(23))
    with pytest.raises(ValueError, match="trailing shape"):
        t.analyse_grid(np.zeros((4, 6)))
    with pytest.raises(ValueError, match="trailing shape"):
        t.from_rfft(np.zeros((6, 4), dtype=complex))
    with pytest.raises(ValueError, match="2 coordinates each"):
        t.synthesise(np.zeros(24), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="route must be one of"):
        t.synthesise(np.zeros(24), np.zeros((3, 2)), route="fast")
    with pytest.raises(ValueError, match="trailing shape"):
        t.accumulate(np.zeros(4), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="no axis 2"):
        t.derivative(np.zeros(24), 2)


def test_the_module_imports_without_finufft_until_a_transform_at_points():
    """`import planetmodel` loads neither finufft nor the module; the
    module and its grid transforms load no finufft; a transform at points
    does."""
    code = (
        "import sys, numpy as np, planetmodel\n"
        "assert 'finufft' not in sys.modules\n"
        "assert 'planetmodel.fourier' not in sys.modules\n"
        "from planetmodel import fourier\n"
        "t = fourier.Transform((6, 4), [1.0, 2.0])\n"
        "c = t.analyse_grid(np.ones((6, 4)))\n"
        "t.synthesise(c, np.zeros((1, 2)), route='direct')\n"
        "assert 'finufft' not in sys.modules, 'loaded early'\n"
        "t.synthesise(c, np.zeros((1, 2)))\n"
        "assert 'finufft' in sys.modules\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
