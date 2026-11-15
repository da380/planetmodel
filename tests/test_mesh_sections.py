"""A simplex mesh cut by a plane: the pieces tile the section exactly.

The oracle is the unit cube in its six Kuhn tetrahedra and its twelve
boundary triangles: the polygons cut from the cells have the area of
the cube's section by the plane, whatever the plane, and the segments
cut from the faces its perimeter.  A block of it drawn in three
dimensions shows the walls away from the viewer, each tiled by its
cells' sections; a fault's curves are drawn by kind, in space or
projected onto a plane.
"""
from itertools import permutations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib",
                                 reason="needs the planetmodel[plot] extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from planetmodel.plotting import (BLOCK_VIEW, FAULT_CURVE_STYLES,  # noqa: E402
                                  PlaneCut, block_planes, clip_polygons,
                                  fault_curves, mesh_block, mesh_section, plane_cut)

CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                   dtype=float)


def _vertex(p):
    return int(np.flatnonzero(np.all(CORNERS == p, axis=1))[0])


def kuhn():
    """The unit cube's six tetrahedra about its diagonal, and its twelve
    boundary triangles."""
    e = np.eye(3)
    cells = []
    for order in permutations(range(3)):
        path = np.cumsum([np.zeros(3)] + [e[a] for a in order], axis=0)
        cells.append([_vertex(p) for p in path])
    faces = []
    for axis in range(3):
        for side in (0.0, 1.0):
            on = [v for v in range(8) if CORNERS[v, axis] == side]
            faces += [c for c in cells if sum(v in on for v in c) == 3
                      for c in [[v for v in c if v in on]]]
    return np.array(cells), np.array(faces)


def area(polygons):
    s, t = polygons[..., 0], polygons[..., 1]
    return 0.5 * np.abs(np.sum(s * np.roll(t, -1, axis=1)
                               - np.roll(s, -1, axis=1) * t, axis=1))


def length(segments):
    return np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)


def test_the_cube_s_cells_and_faces_tile_its_section():
    cells, faces = kuhn()
    assert len(faces) == 12
    centre = np.full(3, 0.5)
    # the plane through the centre normal to the diagonal: a regular
    # hexagon of side sqrt(2)/2
    axes = [[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]]
    cut = plane_cut(CORNERS, cells, centre, axes)
    assert cut.pieces.shape == (6, 4, 2) and sorted(cut.index) == list(range(6))
    assert np.isclose(area(cut.pieces).sum(), 3 * np.sqrt(3) / 4, rtol=1e-14)
    assert np.allclose(cut.normal, np.ones(3) / np.sqrt(3), rtol=0, atol=1e-15)
    rim = plane_cut(CORNERS, faces, centre, axes)
    assert rim.pieces.shape == (12, 2, 2)
    assert np.isclose(length(rim.pieces).sum(), 3 * np.sqrt(2), rtol=1e-14)
    # an oblique plane cutting off a corner region and crossing the cube
    for z in (0.3, 0.71):
        tilted = plane_cut(CORNERS, cells, [0.0, 0.0, z], [[1, 0, 0], [0, 1, 0.2]])
        # the section of the cube by a plane z = z0 + 0.2 y over the square
        assert np.isclose(area(tilted.pieces).sum(), np.hypot(1.0, 0.2), rtol=1e-14)


def test_a_face_in_the_plane_is_left_once_by_the_cell_above_it():
    cells, faces = kuhn()
    bottom = plane_cut(CORNERS, cells, [0.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0]])
    assert np.isclose(area(bottom.pieces).sum(), 1.0, rtol=1e-15)
    # the two cells over the face leave it; two touching it along an edge
    # leave nothing
    assert bottom.pieces.shape[0] == 2
    # the top face has no cell on the normal's side: nothing is left
    top = plane_cut(CORNERS, cells, [0.0, 0.0, 1.0], [[1, 0, 0], [0, 1, 0]])
    assert top.pieces.shape[0] == 0
    # a plane through the diagonal: the cells meeting it along an edge
    # leave nothing, two leave the rectangle 1 by sqrt(2)
    diagonal = plane_cut(CORNERS, cells, [0.0, 0.0, 0.0], [[1, 1, 0], [0, 0, 1]])
    assert diagonal.pieces.shape[0] == 2
    assert np.isclose(area(diagonal.pieces).sum(), np.sqrt(2.0), rtol=1e-14)
    # a hair off it, the cells around the diagonal edge leave nothing more
    hair = plane_cut(CORNERS, cells, [1e-15, 0.0, 0.0], [[1, 1, 0], [0, 0, 1]])
    assert np.isclose(area(hair.pieces).sum(), np.sqrt(2.0), rtol=1e-12)
    assert np.all(area(hair.pieces) > 0.1)
    # a plane meeting the cube at a corner alone leaves nothing, the cube
    # on either side of it
    for origin in ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]):
        corner = plane_cut(CORNERS, cells, origin, [[1, -1, 0], [1, 1, -2]])
        assert corner.pieces.shape[0] == 0


def test_coordinates_are_the_projections_onto_the_plane():
    cells, _ = kuhn()
    cut = plane_cut(CORNERS, cells, [0.0, 0.0, 0.5], [[2, 0, 0], [1, 0, 3]])
    assert np.allclose(cut.axes, [[1, 0, 0], [0, 0, 1]], rtol=0, atol=1e-15)
    assert np.allclose(cut.coordinates([[0.3, 7.0, 0.9]]), [[0.3, 0.4]],
                       rtol=0, atol=1e-15)
    # the plane is y = 0: the pieces in space lie in it, and project back
    space = cut.in_space()
    assert space.shape == cut.pieces.shape[:2] + (3,)
    assert np.all(space[..., 1] == 0.0)
    assert np.allclose(cut.coordinates(space), cut.pieces, rtol=0, atol=1e-15)


def test_a_polygon_is_clipped_to_one_side_of_a_plane():
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    triangle = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float)
    segment = np.array([[0, 0, 0], [0, 0, 2]], dtype=float)
    beyond = triangle + [5.0, 0.0, 0.0]
    parts, index = clip_polygons([square, beyond, triangle, segment],
                                 [1.0, 0.0, 0.0], 0.25)
    assert index.tolist() == [0, 2, 3]
    s, t = parts[0][:, 0], parts[0][:, 1]
    assert np.isclose(0.5 * abs(np.sum(s * np.roll(t, -1) - np.roll(s, -1) * t)),
                      0.25, rtol=1e-15)
    assert np.all(parts[0][:, 0] <= 0.25)
    assert np.all(parts[2] == segment)
    # several planes bound the region between them; a segment crossing one
    # is cut where it crosses
    (strip,), _ = clip_polygons([square], [[1, 0, 0], [-1, 0, 0]], [0.75, -0.25])
    assert np.allclose(np.unique(strip[:, 0]), [0.25, 0.75], rtol=0, atol=1e-15)
    (short,), _ = clip_polygons([segment], [0, 0, 1], 0.5)
    assert np.allclose(short, [[0, 0, 0], [0, 0, 0.5]], rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="vertices"):
        clip_polygons([square[:, :2]], [1.0, 0.0, 0.0], 0.0)
    with pytest.raises(ValueError, match="offsets"):
        clip_polygons([square], [[1.0, 0.0, 0.0]], [0.0, 1.0])


def test_a_section_is_drawn_as_polygons_or_segments_in_each_simplex_s_colour():
    cells, faces = kuhn()
    cut = plane_cut(CORNERS, cells, np.full(3, 0.5), [[1, -1, 0], [1, 1, -2]])
    colours = ["#2a78d6", "#eb6834"] * 3
    fig, ax = plt.subplots()
    polygons = mesh_section(ax, cut, colours=colours, edgecolor="k", linewidth=0.3)
    assert isinstance(polygons, PolyCollection)
    assert len(polygons.get_paths()) == 6
    expected = matplotlib.colors.to_rgba_array(colours)[cut.index]
    assert np.allclose(polygons.get_facecolor(), expected)
    assert ax.get_aspect() == 1.0
    rim = plane_cut(CORNERS, faces, np.full(3, 0.5), [[1, -1, 0], [1, 1, -2]])
    lines = mesh_section(ax, rim, colours="#1a1a19", linewidth=2.0)
    assert isinstance(lines, LineCollection) and len(lines.get_segments()) == 12
    lo, hi = ax.get_xlim()
    assert lo <= rim.pieces[..., 0].min() and hi >= rim.pieces[..., 0].max()
    plt.close(fig)


def test_refusals():
    cells, _ = kuhn()
    with pytest.raises(ValueError, match="parallel"):
        plane_cut(CORNERS, cells, np.zeros(3), [[1, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="3 or 4"):
        plane_cut(CORNERS, cells[:, :2], np.zeros(3), np.eye(3)[:2])
    with pytest.raises(ValueError, match="outside"):
        plane_cut(CORNERS, cells + 8, np.zeros(3), np.eye(3)[:2])
    cut = plane_cut(CORNERS, cells, np.full(3, 0.5), np.eye(3)[:2])
    assert isinstance(cut, PlaneCut)
    fig = plt.figure()
    polar = fig.add_subplot(1, 2, 1, projection="polar")
    with pytest.raises(ValueError, match="plain axes"):
        mesh_section(polar, cut)
    ax = fig.add_subplot(1, 2, 2)
    with pytest.raises(ValueError, match="one colour"):
        mesh_section(ax, cut, colours=["r", "g"])
    plt.close(fig)
    with pytest.raises(ValueError, match="'cut' or 'crinkle'"):
        plane_cut(CORNERS, cells, np.full(3, 0.5), np.eye(3)[:2], kind="slice")
    with pytest.raises(ValueError, match="crinkle is a section of tetrahedra"):
        plane_cut(CORNERS, kuhn()[1], np.full(3, 0.5), np.eye(3)[:2], kind="crinkle")


# ------------------------------------------------------------ the crinkle

def lattice(n):
    """The unit cube in n^3 cubes, each in its six Kuhn tetrahedra: the
    lattice's points and the cells."""
    g = np.arange(n + 1) / n
    points = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    number = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k
    local, _ = kuhn()
    cells = [[number(i + a, j + b, k + c) for a, b, c in CORNERS[cell].astype(int)]
             for i in range(n) for j in range(n) for k in range(n) for cell in local]
    return points, np.array(cells)


@pytest.mark.parametrize(("axes", "height"), [
    ([[1, 0, 0], [0, 1, 0]], 0.5), ([[1, 0, 0], [0, 1, 0]], 0.45),
    ([[0, 1, 0], [1, 0, 0]], 0.5), ([[0, 1, 0], [1, 0, 0]], 0.55)])
def test_a_crinkle_by_a_lattice_plane_is_its_faces(axes, height):
    """Two cubes deep, a plane on the lattice's middle plane z = 1/2 or
    just behind it, the normal up or down: the cells kept are the layer
    on the side opposite the normal, and the crinkle is the eight
    triangles of the middle plane, whole, tiling the section; each the
    face of its cell."""
    points, cells = lattice(2)
    cut = plane_cut(points, cells, [0.0, 0.0, height], axes, kind="crinkle")
    assert cut.pieces.shape == (8, 4, 2)
    assert np.allclose(area(cut.pieces), 1 / 8, rtol=1e-14)
    below = points[cells].mean(axis=1)[:, 2] < 0.5
    up = np.cross(*np.asarray(axes, dtype=float))[2] > 0
    assert np.all(below[cut.index] == up)
    for piece, cell in zip(cut.pieces, cut.index):
        corners = points[cells[cell]]
        on = corners[np.isclose(corners[:, 2], 0.5)]
        assert len(on) == 3
        assert np.allclose(np.sort(on @ np.asarray(axes, float).T, axis=0),
                           np.sort(piece[:3], axis=0), rtol=0, atol=1e-15)


def test_a_crinkle_takes_whole_faces_turned_to_the_viewer_farthest_first():
    """A plane at a slant through a lattice of 3^3 cubes: every piece is a
    face of its cell projected, the cell kept (a corner on the negative
    side) and the cell across the face wholly on the positive side, or
    the face on the cube's boundary and its cell passed through by the
    plane; the face turned towards the positive side, the pieces from
    the farthest; the faces cover the cut's section."""
    points, cells = lattice(3)
    origin, axes = np.full(3, 0.5), np.array([[1.0, 0.3, -0.2], [0.1, 1.0, 0.4]])
    cut = plane_cut(points, cells, origin, axes, kind="crinkle")
    n = cut.normal
    d = (points[cells] - origin) @ n
    assert np.all(d[cut.index].min(axis=1) < 0.0)
    rim, depths = 0, []
    for piece, cell in zip(cut.pieces, cut.index):
        corners = points[cells[cell]]
        projected = cut.coordinates(corners)
        match = [int(np.flatnonzero(np.all(np.isclose(projected, v, rtol=0,
                                                      atol=1e-14), axis=1))[0])
                 for v in piece[:3]]
        face = corners[match]
        apex = corners[np.setdiff1d(range(4), match)[0]]
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        outward = normal if (apex - face[0]) @ normal < 0 else -normal
        assert outward @ n > 0.0
        shared = np.flatnonzero(np.isin(cells, cells[cell][match]).sum(axis=1) == 3)
        other = shared[shared != cell]
        if len(other):
            assert d[other[0]].min() > 0.0
        else:
            rim += 1
            assert np.any(np.isclose(face, 0.0) | np.isclose(face, 1.0), axis=0).any()
            assert d[cell].max() > 0.0
        depths.append(((face.mean(axis=0) - origin) @ n))
    assert 0 < rim < len(cut.pieces)
    assert np.all(np.diff(depths) >= 0.0)
    rng = np.random.default_rng(1)
    section = plane_cut(points, cells, origin, axes).pieces
    inside = np.concatenate([rng.dirichlet(np.ones(4), 4) @ p for p in section])
    paths = [matplotlib.path.Path(p[:3]) for p in cut.pieces]
    assert all(any(q.contains_point(x, radius=1e-12) for q in paths) for x in inside)


def test_a_crinkle_is_drawn_in_its_cells_colours():
    points, cells = lattice(2)
    cut = plane_cut(points, cells, [0.0, 0.0, 0.5], [[1, 0, 0], [0, 1, 0]],
                    kind="crinkle")
    colours = np.where(np.arange(len(cells)) % 2 == 0, "#2a78d6", "#eb6834")
    fig, ax = plt.subplots()
    polygons = mesh_section(ax, cut, colours=list(colours), edgecolor="k")
    assert len(polygons.get_paths()) == 8
    assert np.allclose(polygons.get_facecolor(),
                       matplotlib.colors.to_rgba_array(list(colours))[cut.index])
    plt.close(fig)


def area3(polygons):
    """The areas of planar polygons in space, each (k, 3)."""
    out = []
    for p in polygons:
        n = sum(np.cross(p[i], p[(i + 1) % len(p)]) for i in range(len(p)))
        out.append(0.5 * np.linalg.norm(n))
    return np.array(out)


def test_a_block_s_planes_hold_its_inside():
    normals, offsets = block_planes([[0.0, 1.0], [-1.0, 2.0], [-3.0, 0.0]])
    inside, outside = np.array([0.5, 0.0, -1.0]), np.array([0.5, 2.5, -1.0])
    assert np.all(normals @ inside <= offsets)
    assert not np.all(normals @ outside <= offsets)
    assert normals.shape == (6, 3) and offsets.tolist() == [0, 1, 3, 1, 2, 0]
    with pytest.raises(ValueError, match="each lower below its upper"):
        block_planes([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])


def test_a_block_shows_the_walls_away_from_the_viewer():
    """Seen from the default view (above the corner of least y and
    greatest x), the walls drawn are x = 0, y = 1 and z = 0, each tiled
    by the cells' sections, each piece its cell's colour; the boundary
    triangles leave the walls' edges on them."""
    cells, faces = kuhn()
    colours = ["#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#52514e", "#c3c2b7"]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    walls = mesh_block(ax, CORNERS, cells, [[0, 1]] * 3, colours=colours,
                       lines=faces, edgecolor="k", linewidth=0.3)
    fig.canvas.draw()
    pieces = walls.get_paths()
    assert len(ax.collections) == 2 and len(ax.collections[1].get_segments())
    assert ax.computed_zorder is False
    assert (ax.elev, ax.azim) == BLOCK_VIEW
    assert ax.get_xlim() == (0.0, 1.0) and ax.get_zlim() == (0.0, 1.0)
    plt.close(fig)
    # the same pieces as the cells' sections by the three walls
    normals, offsets = block_planes([[0, 1]] * 3)
    total, count = 0.0, 0
    for origin, axes in (([0, 0, 0], [[0, 1, 0], [0, 0, 1]]),
                         ([0, 1, 0], [[1, 0, 0], [0, 0, 1]]),
                         ([0, 0, 0], [[1, 0, 0], [0, 1, 0]])):
        cut = plane_cut(CORNERS, cells, origin, axes)
        parts, _ = clip_polygons(cut.in_space(), normals, offsets)
        total += area3(parts).sum()
        count += len(parts)
    assert np.isclose(total, 3.0, rtol=1e-12) and count == len(pieces)


@pytest.mark.parametrize("view", [(25.0, -60.0), (25.0, 120.0), (-30.0, 30.0)])
def test_a_block_on_the_mesh_s_own_faces_shows_three_walls_from_any_view(view):
    """Bounds that are the cube's own faces: each wall drawn is a face of
    the cube, tiled by the cells inside it, area 1, from any view."""
    cells, _ = kuhn()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    walls = mesh_block(ax, CORNERS, cells, [[0, 1]] * 3, view=view)
    fig.canvas.draw()
    assert len(walls.get_paths()) == 6
    plt.close(fig)


def test_a_block_s_refusals():
    cells, _ = kuhn()
    fig = plt.figure()
    plain = fig.add_subplot(1, 2, 1)
    with pytest.raises(ValueError, match="three-dimensional axes"):
        mesh_block(plain, CORNERS, cells, [[0, 1]] * 3)
    space = fig.add_subplot(1, 2, 2, projection="3d")
    with pytest.raises(ValueError, match="one for each cell"):
        mesh_block(space, CORNERS, cells, [[0, 1]] * 3, colours=["r", "g"])
    plt.close(fig)


def test_a_fault_s_curves_are_drawn_by_kind():
    """Each kind in its own style, labelled, its edges joined end to end;
    in space as they are, on plain axes projected onto the plane given."""
    edges = {"tip": np.array([[0, 1], [1, 3]]), "free_edge": np.array([[4, 5]]),
             "junction": None, "face_edge": np.empty((0, 2), dtype=np.int64)}
    fig = plt.figure()
    space = fig.add_subplot(1, 2, 1, projection="3d")
    drawn = fault_curves(space, CORNERS, edges, color="#2a78d6")
    assert set(drawn) == {"tip", "free_edge"}
    assert drawn["tip"].get_label() == "tip"
    assert drawn["free_edge"].get_label() == "free edge"
    assert np.isclose(drawn["free_edge"].get_linewidth()[0],
                      FAULT_CURVE_STYLES["free_edge"]["linewidth"])
    plain = fig.add_subplot(1, 2, 2)
    drawn = fault_curves(plain, CORNERS, edges,
                         plane=([0.0, 0.0, 0.0], [[0, 1, 0], [0, 0, 1]]))
    # the tip's two edges joined at node 1 into one polyline, 0 - 1 - 3
    (line,) = drawn["tip"].get_segments()
    assert np.allclose(line, CORNERS[[0, 1, 3]][:, 1:])
    with pytest.raises(ValueError, match="give plane"):
        fault_curves(plain, CORNERS, edges)
    with pytest.raises(ValueError, match="the kinds of a fault's curves"):
        fault_curves(space, CORNERS, {"edge": edges["tip"]})
    with pytest.raises(ValueError, match="rows of node indices"):
        fault_curves(space, CORNERS, {"tip": edges["tip"] * 1.0})
    plt.close(fig)


def test_edges_are_joined_into_polylines_broken_at_branches():
    """A path, a branch where three edges meet, and a loop: runs from the
    ends and the branch, the loop closed on its first node."""
    from planetmodel.plotting.meshes import _polylines
    lines = _polylines(np.array([[0, 1], [1, 2], [2, 3], [2, 4],
                                 [5, 6], [6, 7], [7, 5]]))
    runs = sorted(tuple(r.tolist()) for r in lines)
    assert runs == [(0, 1, 2), (2, 3), (2, 4), (5, 6, 7, 5)]
