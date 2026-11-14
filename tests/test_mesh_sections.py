"""A simplex mesh cut by a plane: the pieces tile the section exactly.

The oracle is the unit cube in its six Kuhn tetrahedra and its twelve
boundary triangles: the polygons cut from the cells have the area of
the cube's section by the plane, whatever the plane, and the segments
cut from the faces its perimeter.
"""
from itertools import permutations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib",
                                 reason="needs the planetmodel[plot] extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from planetmodel.plotting import (PlaneCut, clip_polygons, mesh_section,  # noqa: E402
                                  plane_cut)

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
