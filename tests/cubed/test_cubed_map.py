"""The map from the nested cubes to the reference body: every sphere at
its radius, equal angles across a block, the cube and the blend, and
its symmetries."""
import itertools

import numpy as np
import pytest

from planetmodel.mesh3d._cubed import (corner_jacobians, cubed_map,
                                       cubed_topology)

SPHERES = np.array([0.4, 0.55, 0.8, 1.0])
BLEND = np.array([0.0, 0.3, 0.7, 1.0])
A = 0.45 * SPHERES[0]


def mapped(d, hollow, *, edge=4):
    """A topology with the spheres above, and its points mapped."""
    levels = SPHERES.size - 1 + (0 if hollow else BLEND.size - 1)
    t = cubed_topology(edge, levels, dimension=d, hollow=hollow)
    x = (cubed_map(t.points, spheres=SPHERES) if hollow
         else cubed_map(t.points, spheres=SPHERES, blend=BLEND, half_width=A))
    return t, x


@pytest.mark.parametrize("d", [3, 2])
@pytest.mark.parametrize("hollow", [False, True])
def test_every_sphere_is_at_its_radius(d, hollow):
    t, x = mapped(d, hollow)
    first = 0 if hollow else BLEND.size - 1
    for k, radius in enumerate(SPHERES):
        r = np.linalg.norm(x[t.nodes_at(first + k)], axis=1)
        np.testing.assert_allclose(r, radius, rtol=4e-16, atol=0)


@pytest.mark.parametrize("d", [3, 2])
@pytest.mark.parametrize("hollow", [False, True])
def test_the_cells_are_right_handed(d, hollow):
    for edge in (1, 2, 5):
        t, x = mapped(d, hollow, edge=edge)
        assert corner_jacobians(x, t.cells).min() > 0.0


@pytest.mark.parametrize("d", [3, 2])
def test_the_angles_across_a_block_are_equal(d):
    """On the block above the face x = +1, a point's angles from the x
    axis are pi/4 times its lateral parameters over s."""
    t, x = mapped(d, True, edge=6)
    P = t.points
    s = np.abs(P).max(axis=1)
    on = P[:, 0] == s
    for axis in range(1, d):
        angle = np.arctan2(x[on, axis], x[on, 0])
        np.testing.assert_allclose(angle, 0.25 * np.pi * P[on, axis] / s[on],
                                   atol=1e-15)


def test_the_cube_and_the_blend():
    rng = np.random.default_rng(0)
    inside = rng.uniform(-1.0, 1.0, (50, 3))
    np.testing.assert_allclose(
        cubed_map(inside, spheres=SPHERES, blend=BLEND, half_width=A),
        A * np.tan(0.25 * np.pi * inside), atol=1e-15)
    corner = cubed_map([[1.0, 1.0, 1.0]], spheres=SPHERES, blend=BLEND, half_width=A)
    np.testing.assert_allclose(corner, [[A, A, A]], atol=1e-15)
    # along the ray through a face's centre the blend's levels sit at
    # their fractions between the cube's half-width and the first sphere
    ray = np.zeros((BLEND.size, 3))
    ray[:, 0] = 1.0 + np.arange(BLEND.size)
    r = cubed_map(ray, spheres=SPHERES, blend=BLEND, half_width=A)[:, 0]
    np.testing.assert_allclose(r, A + BLEND * (SPHERES[0] - A), atol=1e-15)
    # and above it the radius is linear in the level between two spheres
    first = BLEND.size
    mid = cubed_map([[first + 0.5, 0.0, 0.0], [first + 1.25, 0.0, 0.0]],
                    spheres=SPHERES, blend=BLEND, half_width=A)[:, 0]
    np.testing.assert_allclose(
        mid, [0.5 * (SPHERES[0] + SPHERES[1]),
              0.75 * SPHERES[1] + 0.25 * SPHERES[2]], atol=1e-15)


def test_the_map_is_continuous_where_its_rule_changes():
    rng = np.random.default_rng(1)
    direction = rng.uniform(-1.0, 1.0, (40, 3))
    direction /= np.abs(direction).max(axis=1, keepdims=True)
    eps = 1e-9
    for s in (1.0, float(BLEND.size)):         # the cube's surface, the first sphere
        lo = cubed_map((s - eps) * direction, spheres=SPHERES, blend=BLEND,
                       half_width=A)
        hi = cubed_map((s + eps) * direction, spheres=SPHERES, blend=BLEND,
                       half_width=A)
        assert np.abs(hi - lo).max() < 1e-8


@pytest.mark.parametrize("hollow", [False, True])
def test_the_map_has_the_cube_s_symmetries(hollow):
    rng = np.random.default_rng(2)
    P = rng.uniform(-4.0, 4.0, (60, 3))
    if hollow:
        P = P[np.abs(P).max(axis=1) >= 1.0]
    kwargs = ({"spheres": SPHERES} if hollow
              else {"spheres": SPHERES, "blend": BLEND, "half_width": A})
    x = cubed_map(P, **kwargs)
    for perm in itertools.permutations(range(3)):
        np.testing.assert_allclose(cubed_map(P[:, perm], **kwargs), x[:, perm],
                                   atol=1e-14)
    for signs in itertools.product((1.0, -1.0), repeat=3):
        np.testing.assert_allclose(cubed_map(P * signs, **kwargs), x * signs,
                                   atol=1e-14)


def test_the_disc_is_the_ball_s_section():
    rng = np.random.default_rng(3)
    P = np.zeros((40, 3))
    P[:, :2] = rng.uniform(-5.0, 5.0, (40, 2))
    ball = cubed_map(P, spheres=SPHERES, blend=BLEND, half_width=A)
    disc = cubed_map(P[:, :2], spheres=SPHERES, blend=BLEND, half_width=A)
    np.testing.assert_allclose(ball[:, 2], 0.0, atol=1e-15)
    np.testing.assert_allclose(ball[:, :2], disc, atol=1e-15)


def test_a_single_sphere():
    """A ball of one layer: the cube, the blend and nothing above."""
    out = cubed_map([[2.0, 0.0, 0.0], [2.0, 2.0, 2.0]], spheres=[1.0],
                    blend=[0.0, 1.0], half_width=0.45)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-15)
