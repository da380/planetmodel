"""What a flat mesh needs without gmsh: `planetmodel.cartesian` and a
spec of a box import light, a flat manifest (schema 7) is assembled,
written and read with gmsh unimportable, its rules (the layered
coordinate, the mesh's axes, a point pulled into a layer) are the box's,
and a mesh lying in a plane is cut whole for drawing."""
import json
import subprocess
import sys

import numpy as np
import pytest

from planetmodel import Skeleton, cartesian
from planetmodel.mesh3d import MeshSpec, UniformInterfaces, manifest as sc
from planetmodel.plotting import in_plane


def flat_card(*, coordinates=("x", "z")):
    """A flat manifest assembled by hand: two layers of a rectangle over
    x in [-2, 3], or of a box over that and y in [0, 1]."""
    g = cartesian.Geometry(Skeleton([0.0, 0.4, 1.5]),
                           cartesian.Rectangle((-2.0, 3.0)) if len(coordinates) == 2
                           else cartesian.Box([(-2.0, 3.0), (0.0, 1.0)]))
    b = g.skeleton.boundaries
    layers = [sc.FlatLayerEntry.from_layer(lay, attribute=i + 1, depth_top=b[i],
                                           depth_bottom=b[i + 1])
              for i, lay in enumerate(g.layers)]
    interfaces = [sc.FlatInterfaceEntry.from_interface(f, attribute=k + 1,
                                                       depth=f.radius)
                  for k, f in enumerate(g.interfaces)]
    values = {"base": -1.5, "x_min": -2.0, "x_max": 3.0, "y_min": 0.0, "y_max": 1.0}
    faces = [sc.FaceEntry(attribute=3 + k, name=name, axis=axis, value=values[name],
                          outward=outward)
             for k, (name, axis, outward) in enumerate(
                 row for row in sc.FACES if row[1] in coordinates)]
    return sc.MeshManifest.from_build(
        mesh=sc.mesh_block("m.mesh", format="mfem", nodes="reference"),
        layers=layers, interfaces=interfaces, faces=faces, coordinates=coordinates)


def test_a_flat_manifest_is_assembled_written_and_read_without_gmsh(tmp_path):
    card = flat_card()
    assert card.schema == sc.SCHEMA_FLAT and card.is_flat
    path = sc.write(tmp_path / "m", card)
    data = json.loads(path.read_text())
    assert data["coordinates"] == ["x", "z"]
    assert [f["name"] for f in data["faces"]] == ["base", "x_min", "x_max"]
    code = f"""
import sys
sys.modules["gmsh"] = None
import planetmodel.cartesian
from planetmodel import Skeleton, cartesian
from planetmodel.mesh3d import MeshSpec, UniformInterfaces, manifest
card = manifest.read({str(path)!r})
assert card.face_attribute("x_max") == 5 and card.surface_attribute == 1
g = cartesian.Geometry(Skeleton([0.0, 1.0]), cartesian.Box([(0, 1), (0, 1)]))
assert MeshSpec(g, UniformInterfaces(0.2, 0.3, 0.3)).dimension == 3
try:
    from planetmodel.mesh3d import build_layered_mesh
except ImportError as error:
    assert "planetmodel[meshing]" in str(error), error
else:
    raise AssertionError("the builder needed no gmsh")
heavy = [m for m in ("mfem", "matplotlib", "netCDF4", "ducc0") if m in sys.modules]
assert not heavy, heavy
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=tmp_path)
    assert r.returncode == 0, r.stderr


def test_the_rules_of_a_flat_manifest_are_the_box_s():
    card = flat_card()
    assert card.coordinate == "depth"
    assert np.array_equal(card.boundaries, [0.0, 0.4, 1.5])
    assert card.layer_interval(card.layers[1]) == (0.4, 1.5)
    assert card.interface_level(card.interfaces[1]) == 0.4
    assert card.scale == 3.0
    assert card.mesh_axes(2) == (0, 2)
    with pytest.raises(ValueError, match="coordinates are"):
        card.mesh_axes(3)
    assert flat_card(coordinates=("x", "y", "z")).mesh_axes(3) == (0, 1, 2)
    X = np.array([[0.5, 7.0, 0.1], [1.0, 0.0, -0.2], [-1.0, 0.3, -2.0]])
    assert np.array_equal(card.layered_coordinate(X), [-0.1, 0.2, 2.0])
    # pulled vertically into [0.4, 1.5], the lateral place kept
    pulled = card.pulled_into(X, (0.4, 1.5))
    assert np.array_equal(pulled[:, :2], X[:, :2])
    assert np.array_equal(pulled[:, 2], [-0.4, -0.4, -1.5])
    with pytest.raises(ValueError, match="no inner radius"):
        card.inner_radius


def test_a_mesh_in_a_plane_is_cut_whole():
    """Two triangles and their rim in the plane z = 0: each kept whole, the
    polygons tiling the square, the segments its perimeter; a corner off
    the plane is refused."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    cut = in_plane(points, triangles, [0.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0]])
    assert cut.pieces.shape == (2, 4, 2) and cut.index.tolist() == [0, 1]
    s, t = cut.pieces[..., 0], cut.pieces[..., 1]
    area = 0.5 * np.abs(np.sum(s * np.roll(t, -1, axis=1) - np.roll(s, -1, axis=1) * t,
                               axis=1))
    assert np.isclose(area.sum(), 1.0, rtol=1e-15)
    rim = in_plane(points, [[0, 1], [1, 2], [2, 3], [3, 0]], [0, 0, 0],
                   [[1, 0, 0], [0, 1, 0]])
    assert rim.pieces.shape == (4, 2, 2)
    assert np.isclose(np.linalg.norm(rim.pieces[:, 1] - rim.pieces[:, 0],
                                     axis=1).sum(), 4.0, rtol=1e-15)
    # the plane's own coordinates: the (x, z) plane of a rectangle's mesh
    side = in_plane(points[:, [0, 2, 1]], triangles, [0, 0, 0], [[1, 0, 0], [0, 0, 1]])
    assert np.array_equal(side.pieces, cut.pieces)
    bent = points.copy()
    bent[2, 2] = 1e-3
    with pytest.raises(ValueError, match="off the plane"):
        in_plane(bent, triangles, [0, 0, 0], [[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="2 or 3 corner"):
        in_plane(points, np.array([[0, 1, 2, 3]]), [0, 0, 0], [[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="outside"):
        in_plane(points, triangles + 4, [0, 0, 0], [[1, 0, 0], [0, 1, 0]])


def test_a_flat_spec_reads_its_dimension_from_the_domain():
    sizing = UniformInterfaces(0.2, 0.3, 0.3)
    box = cartesian.Geometry(Skeleton([0.0, 1.0]), cartesian.Box([(0, 1), (0, 1)]))
    rectangle = cartesian.Geometry(Skeleton([0.0, 1.0]), cartesian.Rectangle((0, 1)))
    assert MeshSpec(box, sizing).dimension == 3
    assert MeshSpec(rectangle, sizing).dimension == 2
    assert MeshSpec(rectangle, sizing).domain is rectangle
    assert MeshSpec(box, sizing).outer_radius == 1.0
