"""The radial knots of a cubed sphere: the target size, the cells a span
takes at equal shares of it, the spheres of a mesh and the blend's cap;
and the measures of a cell they are judged by."""
import numpy as np
import pytest

from planetmodel.mesh3d import InterfaceSizing
from planetmodel.mesh3d._cubed import (SHARE_NODES, aspect_ratios, blend_elongation,
                                       blend_fractions, cell_lengths, cell_shapes,
                                       corner_jacobians, cubed_map, cubed_topology,
                                       equal_shares, shared_faces, size_at,
                                       sphere_knots)

CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
SQUARE = CUBE[:4, :2]


def shares(knots, radii, sizings):
    """The integral of dr / h across each cell between `knots`."""
    out = []
    for lo, hi in zip(knots[:-1], knots[1:]):
        r = np.linspace(lo, hi, 20001)
        f = 1.0 / size_at(r, radii, sizings)
        out.append(float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(r))))
    return np.array(out)


def test_the_size_at_a_distance():
    sizing = InterfaceSizing(0.1, 0.5, 2.0)
    np.testing.assert_allclose(sizing.at([0.0, 1.0, 2.0, 7.0]), [0.1, 0.3, 0.5, 0.5])
    assert sizing.at(0.5).shape == ()
    fine = InterfaceSizing(0.02, 0.5, 1.0)
    got = size_at([0.0, 0.5, 1.0], [0.0, 1.0], [sizing, fine])
    np.testing.assert_allclose(got, [0.1, 0.2, 0.02])
    with pytest.raises(ValueError, match="one sizing for each interface"):
        size_at([0.0], [0.0, 1.0], [sizing])


def test_a_span_takes_the_cells_the_target_asks_for():
    flat = [InterfaceSizing(0.1, 0.1, 1.0)]
    np.testing.assert_allclose(equal_shares(0.2, 0.5, [0.5], flat),
                               [0.2, 0.3, 0.4, 0.5], atol=1e-13)
    assert equal_shares(0.2, 0.52, [0.5], flat).size == 5      # 3.2 sizes: 4 cells
    assert equal_shares(0.2, 0.23, [0.5], flat).tolist() == [0.2, 0.23]
    np.testing.assert_allclose(equal_shares(0.0, 1.0, [0.5], flat, count=4),
                               [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-13)
    with pytest.raises(ValueError, match="runs upward"):
        equal_shares(0.5, 0.5, [0.5], flat)


def test_the_cells_are_equal_shares_of_the_target():
    radii = [0.3, 1.0]
    sizings = [InterfaceSizing(0.05, 0.3, 0.4), InterfaceSizing(0.01, 0.3, 0.2)]
    knots = equal_shares(0.3, 1.0, radii, sizings)
    assert knots[0] == 0.3 and knots[-1] == 1.0 and np.all(np.diff(knots) > 0.0)
    each = shares(knots, radii, sizings)
    np.testing.assert_allclose(each, each.mean(), rtol=1e-4)    # the quadrature
    assert each.mean() <= 1.0 and each.sum() > knots.size - 2    # no cell too many
    # fine by the surface, coarser below it: thinnest at the top
    assert np.argmin(np.diff(knots)) == knots.size - 2
    assert SHARE_NODES > 1000


def test_the_spheres_of_a_mesh():
    sizings = [InterfaceSizing(0.1, 0.1, 1.0)] * 3
    full = sphere_knots([0.0, 0.4, 0.8, 1.0], [0.4, 0.8, 1.0], sizings)
    np.testing.assert_allclose(full, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], atol=1e-13)
    hollow = sphere_knots([0.5, 0.8, 1.0], [0.5, 0.8, 1.0], sizings)
    np.testing.assert_allclose(hollow, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], atol=1e-13)
    one = sphere_knots([0.0, 1.0], [1.0], sizings[:1])
    assert one.tolist() == [1.0]
    for knots, boundaries in ((full, [0.4, 0.8, 1.0]), (hollow, [0.5, 0.8, 1.0])):
        assert set(boundaries) <= set(knots.tolist())      # exactly, not nearly


@pytest.mark.parametrize("d", [3, 2])
def test_the_blend_s_elongation_is_the_mesh_s_own(d):
    """Measured on one block, it is what `cell_lengths` gives over the
    blend's cells of the whole mesh."""
    w = np.array([0.0, 0.2, 0.7, 1.0])
    t = cubed_topology(5, w.size - 1, dimension=d)
    x = cubed_map(t.points, spheres=[1.0], blend=w, half_width=0.4)
    lengths = cell_lengths(x, t.cells[t.cell_levels > 0])
    whole = float(np.max(lengths[:, -1] / lengths[:, :-1].min(axis=1)))
    got = blend_elongation(5, w, half_width=0.4, radius=1.0, dimension=d)
    assert got == pytest.approx(whole, rel=1e-12)


def test_one_cell_across_the_blend_is_a_needle():
    for edge, want in ((8, 3.82), (16, 7.70), (32, 15.44)):
        got = blend_elongation(edge, [0.0, 1.0], half_width=0.45, radius=1.0)
        assert got == pytest.approx(want, abs=0.01)
    # scale leaves it alone
    assert blend_elongation(8, [0.0, 1.0], half_width=4.5, radius=10.0) == (
        pytest.approx(blend_elongation(8, [0.0, 1.0], half_width=0.45, radius=1.0)))


@pytest.mark.parametrize("d", [3, 2])
@pytest.mark.parametrize("edge", [4, 16, 32])
def test_the_blend_is_divided_until_its_cells_are_under_the_cap(d, edge):
    kwargs = {"half_width": 0.45, "radius": 1.0, "dimension": d}
    assert blend_fractions(edge, **kwargs).tolist() == [0.0, 1.0]
    w = blend_fractions(edge, elongation=4.0, **kwargs)
    count = w.size - 1
    np.testing.assert_allclose(w, np.linspace(0.0, 1.0, count + 1))
    assert blend_elongation(edge, w, **kwargs) <= 4.0
    if count > 1:       # and with one cell fewer it is over it
        fewer = np.linspace(0.0, 1.0, count)
        assert blend_elongation(edge, fewer, **kwargs) > 4.0


def test_the_blend_follows_the_sizing():
    radii, sizings = [1.0], [InterfaceSizing(0.02, 0.2, 0.3)]
    w = blend_fractions(4, half_width=0.45, radius=1.0, radii=radii, sizings=sizings)
    knots = 0.45 + w * 0.55
    np.testing.assert_allclose(knots, equal_shares(0.45, 1.0, radii, sizings),
                               atol=1e-14)
    assert np.argmin(np.diff(w)) == w.size - 2           # fine by the first sphere
    # the cap adds cells where the sizing's leave a needle, at equal shares still
    capped = blend_fractions(64, half_width=0.45, radius=1.0, radii=radii,
                             sizings=sizings, elongation=2.0)
    assert capped.size > w.size
    each = shares(0.45 + capped * 0.55, radii, sizings)
    np.testing.assert_allclose(each, each.mean(), rtol=1e-4)    # the quadrature
    assert blend_elongation(64, capped, half_width=0.45, radius=1.0) <= 2.0


def test_what_the_blend_refuses():
    with pytest.raises(ValueError, match="half-width must lie between"):
        blend_fractions(4, half_width=1.0, radius=1.0)
    with pytest.raises(ValueError, match="must be positive"):
        blend_fractions(4, half_width=0.4, radius=1.0, elongation=0.0)


def test_the_lengths_of_a_cell():
    box = CUBE * [1.0, 2.0, 4.0]
    np.testing.assert_allclose(cell_lengths(box, [np.arange(8)]), [[1.0, 2.0, 4.0]])
    np.testing.assert_allclose(aspect_ratios(box, [np.arange(8)]), [4.0])
    np.testing.assert_allclose(aspect_ratios(CUBE, [np.arange(8)]), [1.0])
    np.testing.assert_allclose(cell_lengths(SQUARE * [3.0, 0.5], [np.arange(4)]),
                               [[3.0, 0.5]])
    with pytest.raises(ValueError, match="quadrilaterals or hexahedra"):
        cell_lengths(CUBE, [[0, 1, 2]])


def test_the_jacobian_at_a_cell_s_corners():
    np.testing.assert_allclose(corner_jacobians(CUBE, [np.arange(8)]), 1.0)
    np.testing.assert_allclose(corner_jacobians(SQUARE, [np.arange(4)]), 1.0)
    np.testing.assert_allclose(corner_jacobians(CUBE * [1.0, 2.0, 4.0],
                                                [np.arange(8)]), 8.0)
    flipped = [[4, 5, 6, 7, 0, 1, 2, 3]]
    assert np.all(corner_jacobians(CUBE, flipped) < 0.0)
    assert np.all(corner_jacobians(SQUARE, [[0, 3, 2, 1]]) < 0.0)
    # a cell folded at one corner is negative there alone
    folded = CUBE.copy()
    folded[6] = [0.2, 0.2, 0.2]
    dets = corner_jacobians(folded, [np.arange(8)])[0]
    assert dets[6] < 0.0 and dets[0] > 0.0


def test_the_shape_of_a_cell():
    one = [np.arange(8)]
    np.testing.assert_allclose(cell_shapes(CUBE, one), 1.0)
    np.testing.assert_allclose(cell_shapes(7.0 * CUBE, one), 1.0)      # no scale
    sides = np.array([1.0, 0.5, 0.25])
    want = 3.0 * np.prod(sides) ** (2.0 / 3.0) / np.sum(sides ** 2)
    np.testing.assert_allclose(cell_shapes(CUBE * sides, one), want)
    np.testing.assert_allclose(cell_shapes(SQUARE * [1.0, 0.5], [np.arange(4)]),
                               2.0 * 0.5 / 1.25)
    sheared = CUBE + np.outer(CUBE[:, 2], [0.5, 0.0, 0.0])
    assert 0.0 < cell_shapes(sheared, one)[0] < 1.0
    assert cell_shapes(CUBE, [[4, 5, 6, 7, 0, 1, 2, 3]])[0] == pytest.approx(-1.0)
    folded = CUBE.copy()
    folded[6] = [0.2, 0.2, 0.2]
    assert cell_shapes(folded, one)[0] < 0.0


def test_a_measure_of_many_cells_is_that_of_each(monkeypatch):
    """Cells are measured a chunk at a time; the chunks change nothing."""
    from planetmodel.mesh3d import _cubed
    t = cubed_topology(3, 2)
    x = cubed_map(t.points, spheres=[0.6, 1.0], blend=[0.0, 1.0], half_width=0.25)
    whole = (cell_lengths(x, t.cells), corner_jacobians(x, t.cells),
             cell_shapes(x, t.cells))
    monkeypatch.setattr(_cubed, "CHUNK", 7)
    parts = (cell_lengths(x, t.cells), corner_jacobians(x, t.cells),
             cell_shapes(x, t.cells))
    for a, b in zip(whole, parts):
        np.testing.assert_array_equal(a, b)


def test_the_cells_a_face_belongs_to():
    points = np.concatenate([CUBE, CUBE[4:] + [0.0, 0.0, 1.0]])
    cells = [np.arange(8), [4, 5, 6, 7, 8, 9, 10, 11]]
    assert sorted(shared_faces(cells).tolist()) == [1] * 10 + [2]
    assert corner_jacobians(points, cells).min() > 0.0
    assert shared_faces([[0, 1, 2, 3], [1, 4, 5, 2]]).tolist().count(2) == 1
