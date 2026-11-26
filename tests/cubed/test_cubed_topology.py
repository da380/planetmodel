"""The cells of a cubed sphere in its parameter space: their number, that
they conform and are right-handed, and the faces of each level."""
import numpy as np
import pytest

from planetmodel.mesh3d._cubed import (_FACES, corner_jacobians, cubed_topology,
                                       shared_faces)

CASES = [(d, hollow, edge, levels)
         for d in (3, 2) for hollow in (False, True)
         for edge in (1, 2, 3, 5) for levels in (1, 3)]
CASES += [(d, False, edge, 0) for d in (3, 2) for edge in (1, 4)]


def surface_points(edge, d):
    return (edge + 1) ** d - (edge - 1) ** d


@pytest.mark.parametrize(("d", "hollow", "edge", "levels"), CASES)
def test_the_counts(d, hollow, edge, levels):
    t = cubed_topology(edge, levels, dimension=d, hollow=hollow)
    block = 2 * d * edge ** (d - 1)
    assert t.cells.shape == ((0 if hollow else edge ** d) + block * levels, 2 ** d)
    p = surface_points(edge, d)
    assert t.points.shape == ((levels + 1) * p if hollow
                              else (edge + 1) ** d + levels * p, d)
    assert t.surface.size == p and t.ring.shape == (block, 2 ** (d - 1))
    assert t.block == edge ** (d - 1)
    assert np.bincount(t.cell_levels, minlength=levels + 1).tolist() == (
        [0 if hollow else edge ** d] + [block] * levels)
    assert np.unique(t.cells).size == t.points.shape[0]      # no point is unused


@pytest.mark.parametrize(("d", "hollow", "edge", "levels"), CASES)
def test_the_mesh_conforms_and_is_right_handed(d, hollow, edge, levels):
    t = cubed_topology(edge, levels, dimension=d, hollow=hollow)
    counts = shared_faces(t.cells)
    assert set(counts.tolist()) <= {1, 2}
    boundary = 2 * d * edge ** (d - 1) * (2 if hollow else 1)
    assert int((counts == 1).sum()) == boundary
    assert corner_jacobians(t.points, t.cells).min() > 0.0
    assert np.unique(t.points, axis=0).shape == t.points.shape   # no point twice


def outwardness(points, faces):
    """Positive for a face pointing away from the origin: its normal on
    its centroid, or in the plane the turn of a side about the origin."""
    v = points[faces]
    centre = v.mean(axis=1)
    if faces.shape[1] == 2:
        side = v[:, 1] - v[:, 0]
        return centre[:, 0] * side[:, 1] - centre[:, 1] * side[:, 0]
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 3] - v[:, 0])
    return np.einsum("ij,ij->i", normal, centre)


@pytest.mark.parametrize(("d", "hollow", "edge", "levels"), CASES)
def test_the_faces_of_a_level(d, hollow, edge, levels):
    t = cubed_topology(edge, levels, dimension=d, hollow=hollow)
    cell_faces = {}
    for cell, level in zip(t.cells, t.cell_levels):
        for face in cell[_FACES[d]]:
            cell_faces.setdefault(tuple(sorted(face.tolist())), []).append(int(level))
    for level in range(levels + 1):
        faces = t.faces(level)
        assert np.all(outwardness(t.points, faces) > 0.0)
        assert np.all(outwardness(t.points, t.faces(level, inward=True)) < 0.0)
        # every point of the surface lies on it
        s = np.abs(t.points[t.nodes_at(level)]).max(axis=1)
        np.testing.assert_allclose(s, 1.0 + level, rtol=0, atol=1e-13)
        below = [] if (level == 0 and hollow) else [level]
        above = [level + 1] if level < levels else []
        for face in faces:
            assert sorted(cell_faces[tuple(sorted(face.tolist()))]) == below + above


@pytest.mark.parametrize("d", [3, 2])
def test_the_last_direction_of_a_block_is_radial(d):
    t = cubed_topology(3, 2, dimension=d)
    s = np.abs(t.points).max(axis=1)[t.cells]
    half = 2 ** (d - 1)
    block = t.cell_levels > 0
    low = s[block][:, [0, 1]] if d == 2 else s[block][:, :half]
    high = s[block][:, [3, 2]] if d == 2 else s[block][:, half:]
    np.testing.assert_allclose(high - low, 1.0, atol=1e-13)


def test_what_is_refused():
    with pytest.raises(ValueError, match="edge must be at least 1"):
        cubed_topology(0, 1)
    with pytest.raises(ValueError, match="at least 0 levels"):
        cubed_topology(2, -1)
    with pytest.raises(ValueError, match="at least 1 levels"):
        cubed_topology(2, 0, hollow=True)
    with pytest.raises(ValueError, match="3 dimensions or 2"):
        cubed_topology(2, 1, dimension=4)
    with pytest.raises(ValueError, match="levels 0..1"):
        cubed_topology(2, 1).nodes_at(2)
