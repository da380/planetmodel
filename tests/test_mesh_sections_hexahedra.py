"""A mesh of hexahedra cut by a plane: the pieces tile the section.

The oracle is the tetrahedral cut: a lattice of hexahedra filling the
unit cube, its inner points jittered so that its faces are not planes,
leaves pieces of the same total area as the cube's six Kuhn tetrahedra
whatever the plane, and its boundary quadrilaterals segments of the
same total length.  A cubed sphere cut through its centre leaves each
layer's annulus.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib",
                                 reason="needs the planetmodel[plot] extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from planetmodel.mesh3d._cubed import cubed_map, cubed_topology  # noqa: E402
from planetmodel.plotting import in_plane, mesh_section, plane_cut  # noqa: E402
from test_mesh_sections import CORNERS, area, kuhn, length  # noqa: E402

HEX = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
QUAD = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def lattice(n, *, jitter=0.0, seed=0):
    """The unit cube in n^3 hexahedra, its inner points moved at random
    by up to `jitter` of a cell: the points, the cells and the
    quadrilaterals of its boundary."""
    g = np.arange(n + 1) / n
    index = np.stack(np.meshgrid(*[np.arange(n + 1)] * 3, indexing="ij"),
                     axis=-1).reshape(-1, 3)
    points = g[index]
    inner = np.all((index > 0) & (index < n), axis=1)
    rng = np.random.default_rng(seed)
    points[inner] += jitter / n * rng.uniform(-1.0, 1.0, (int(inner.sum()), 3))
    number = lambda ijk: (ijk[..., 0] * (n + 1) + ijk[..., 1]) * (n + 1) + ijk[..., 2]
    base = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"),
                    axis=-1).reshape(-1, 3)
    cells = number(base[:, None, :] + HEX[None])
    faces = []
    square = np.stack(np.meshgrid(*[np.arange(n)] * 2, indexing="ij"),
                      axis=-1).reshape(-1, 2)
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for side in (0, n):
            ijk = np.zeros((square.shape[0], 4, 3), dtype=int)
            ijk[..., axis] = side
            ijk[..., others] = square[:, None, :] + QUAD[None]
            faces.append(number(ijk))
    return points, cells, np.concatenate(faces)


PLANES = [([0.5, 0.5, 0.5], [[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]]),
          ([0.0, 0.0, 0.3], [[1, 0, 0], [0, 1, 0.2]]),
          ([0.2, 0.7, 0.4], [[1.0, 0.3, -0.2], [0.1, 1.0, 0.7]]),
          ([0.5, 0.5, 0.5], [[0, 1, 0], [0, 0, 1]]),
          ([0.9, 0.1, 0.8], [[1.0, 2.0, 0.5], [-0.3, 0.1, 1.0]])]


@pytest.mark.parametrize("jitter", [0.0, 0.3])
@pytest.mark.parametrize(("origin", "axes"), PLANES)
def test_hexahedra_tile_the_section_as_tetrahedra_do(origin, axes, jitter):
    tets, triangles = kuhn()
    want = area(plane_cut(CORNERS, tets, origin, axes).pieces).sum()
    rim = length(plane_cut(CORNERS, triangles, origin, axes).pieces).sum()
    points, cells, faces = lattice(3, jitter=jitter)
    cut = plane_cut(points, cells, origin, axes)
    assert cut.pieces.ndim == 3 and 4 <= cut.pieces.shape[1] <= 6
    assert np.isclose(area(cut.pieces).sum(), want, rtol=1e-12)
    assert np.all(area(cut.pieces) > 0.0)
    assert cut.index.size == np.unique(cut.index).size      # one piece a cell
    edge = plane_cut(points, faces, origin, axes, tensor=True)
    assert edge.pieces.shape[1:] == (2, 2)
    assert np.isclose(length(edge.pieces).sum(), rim, rtol=1e-12)


def test_a_cube_cut_across_its_diagonal_is_a_hexagon():
    cut = plane_cut(HEX.astype(float), [np.arange(8)], np.full(3, 0.5),
                    [[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]])
    assert cut.pieces.shape == (1, 6, 2)
    assert np.isclose(area(cut.pieces)[0], 3 * np.sqrt(3) / 4, rtol=1e-14)
    sides = np.linalg.norm(np.roll(cut.pieces[0], -1, axis=0) - cut.pieces[0], axis=1)
    assert np.allclose(sides, np.sqrt(0.5))
    # a corner cut off is a triangle, its last vertex repeated
    corner = plane_cut(HEX.astype(float), [np.arange(8)], [0.2, 0.0, 0.0],
                       [[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]])
    assert corner.pieces.shape == (1, 4, 2)
    assert np.array_equal(corner.pieces[0, 3], corner.pieces[0, 2])
    assert np.isclose(area(corner.pieces)[0], np.sqrt(3) / 2 * 0.2 ** 2, rtol=1e-12)


def test_a_face_in_the_plane_is_left_once_by_the_cell_above_it():
    points, cells, _ = lattice(2)
    for height, pieces in ((0.0, 4), (0.5, 4), (1.0, 0)):
        cut = plane_cut(points, cells, [0.0, 0.0, height], [[1, 0, 0], [0, 1, 0]])
        assert cut.pieces.shape[0] == pieces
        assert np.isclose(area(cut.pieces).sum(), 1.0 if pieces else 0.0, rtol=1e-15)
    # the cells above the middle plane are the ones that leave its faces
    cut = plane_cut(points, cells, [0.0, 0.0, 0.5], [[1, 0, 0], [0, 1, 0]])
    assert np.all(points[cells[cut.index]][..., 2].max(axis=1) == 1.0)


def test_a_mesh_of_quadrilaterals_in_a_plane_is_its_own_section():
    t = cubed_topology(3, 2, dimension=2)
    x = np.zeros((t.points.shape[0], 3))
    x[:, :2] = cubed_map(t.points, spheres=[0.6, 1.0], blend=[0.0, 1.0],
                         half_width=0.25)
    flat = in_plane(x, t.cells, np.zeros(3), np.eye(3)[:2])
    assert flat.pieces.shape == (t.cells.shape[0], 4, 2)
    assert np.array_equal(flat.index, np.arange(t.cells.shape[0]))
    assert np.array_equal(flat.pieces, x[t.cells][..., :2])
    # the straight-sided cells fill the polygon through the outer nodes
    rim = x[t.faces(2)]
    polygon = 0.5 * np.abs(np.sum(rim[:, 0, 0] * rim[:, 1, 1]
                                  - rim[:, 1, 0] * rim[:, 0, 1]))
    assert np.isclose(area(flat.pieces).sum(), polygon, rtol=1e-13)
    fig, ax = plt.subplots()
    shades = np.array(["#c3c2b7", "#e1e0d9", "#f4f3f0"])
    drawn = mesh_section(ax, flat, colours=shades[t.cell_levels])
    assert isinstance(drawn, PolyCollection)
    assert len(drawn.get_paths()) == t.cells.shape[0]
    circles = mesh_section(ax, in_plane(x, t.faces(2), np.zeros(3), np.eye(3)[:2]),
                           colours="k")
    assert isinstance(circles, LineCollection)
    plt.close(fig)
    with pytest.raises(ValueError, match="not in it"):
        in_plane(x + [0.0, 0.0, 1.0], t.cells, np.zeros(3), np.eye(3)[:2])


@pytest.mark.parametrize("axes", [np.eye(3)[:2], [[1.0, 0.2, 0.1], [0.0, 1.0, 0.7]]])
def test_a_cubed_sphere_cut_through_its_centre(axes):
    """Each layer leaves its annulus, to the chords of its cells (a
    hundredth, at twelve cells an edge), and an interface's faces the
    circle between two."""
    spheres = np.array([0.5, 0.8, 1.0])
    blend = np.array([0.0, 0.5, 1.0])
    t = cubed_topology(12, spheres.size - 1 + blend.size - 1)
    x = cubed_map(t.points, spheres=spheres, blend=blend, half_width=0.2)
    cut = plane_cut(x, t.cells, np.zeros(3), axes)
    assert cut.index.size == np.unique(cut.index).size
    areas = area(cut.pieces)
    assert np.isclose(areas.sum(), np.pi, rtol=1e-2)
    levels = t.cell_levels[cut.index]
    inner = areas[levels <= blend.size - 1].sum()
    assert np.isclose(inner, np.pi * spheres[0] ** 2, rtol=1e-2)
    for k in (1, 2):
        ring = areas[levels == blend.size - 1 + k].sum()
        want = np.pi * (spheres[k] ** 2 - spheres[k - 1] ** 2)
        assert np.isclose(ring, want, rtol=1e-2)
        circle = plane_cut(x, t.faces(blend.size - 1 + k), np.zeros(3), axes,
                           tensor=True)
        assert np.isclose(length(circle.pieces).sum(), 2 * np.pi * spheres[k],
                          rtol=1e-2)
        # the circle runs along the edges of the cells' pieces
        ends = circle.pieces.reshape(-1, 2)
        assert np.allclose(np.linalg.norm(ends, axis=1), spheres[k], rtol=1e-2)
    fig, ax = plt.subplots()
    colours = np.array(["#c3c2b7", "#e1e0d9", "#f4f3f0", "#a3a29b", "#52514e"])
    drawn = mesh_section(ax, cut, colours=colours[t.cell_levels], edgecolor="k")
    assert len(drawn.get_paths()) == cut.index.size
    plt.close(fig)


def test_what_is_refused():
    points, cells, faces = lattice(1)
    with pytest.raises(ValueError, match="crinkle is a section of tetrahedra"):
        plane_cut(points, cells, np.full(3, 0.5), np.eye(3)[:2], kind="crinkle")
    with pytest.raises(ValueError, match="crinkle is a section of tetrahedra"):
        plane_cut(points, faces, np.full(3, 0.5), np.eye(3)[:2], kind="crinkle",
                  tensor=True)
    with pytest.raises(ValueError, match="no tensor elements"):
        plane_cut(points, faces[:, :3], np.full(3, 0.5), np.eye(3)[:2], tensor=True)
    with pytest.raises(ValueError, match="or of 8 for hexahedra"):
        plane_cut(points, cells[:, :6], np.full(3, 0.5), np.eye(3)[:2])
    with pytest.raises(ValueError, match="2, 3 or 4"):
        in_plane(points, cells, np.zeros(3), np.eye(3)[:2])
