"""A written mesh read back as plain arrays (`read_mesh`): every node,
cell and boundary element the build counted, the attributes the
manifest numbers, the faces' cells, the groups' names and the cells'
shape by the measure the build checks; and each
cell's layer by position, a cell with its corners on a sphere taking its
neighbours'."""
import numpy as np
import pytest

from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, build_layered_mesh, cell_quality, manifest,
                                read_mesh)
from planetmodel.mesh3d.arrays import MeshArrays

from conftest import COARSE, full_geometry

pytestmark = pytest.mark.gmsh


@pytest.mark.parametrize("dimension", [2, 3])
def test_a_mesh_reads_back_as_arrays(dimension, tmp_path):
    res = build_layered_mesh(MeshSpec(full_geometry(), COARSE, dimension=dimension,
                                      order=2), tmp_path / "ball")
    card = manifest.read(res.manifest_path)
    mesh = read_mesh(res.msh_path)
    d = dimension
    assert mesh.dimension == d
    assert mesh.points.shape == (res.counts["nodes"], 3)
    assert np.all(np.diff(mesh.tags) > 0)
    assert len(mesh.cells) == res.counts[f"dim{d}"]
    assert len(mesh.faces) == res.counts[f"dim{d - 1}"]
    # quadratic simplices, corners first
    assert mesh.cells.shape[1] == {2: 6, 3: 10}[d]
    assert mesh.faces.shape[1] == {2: 3, 3: 6}[d]
    assert sorted(set(mesh.cell_attributes.tolist())) == [1, 2, 3]
    assert sorted(set(mesh.face_attributes.tolist())) == [1, 2, 3]
    assert mesh.curves == {} or d == 3
    for e in card.layers:
        assert mesh.names[(d, e["attribute"])] == e["name"]
    for e in card.interfaces:
        assert mesh.names[(d - 1, e["attribute"])] == e["name"]
    # every node inside the unit ball, an interface's nodes on its sphere
    r = np.linalg.norm(mesh.points, axis=1)
    assert r.max() <= 1.0 + 1e-12
    for e in card.interfaces:
        on = np.unique(mesh.faces_with(e["attribute"])[:, :d])
        assert np.allclose(r[on], e["radius"], rtol=0, atol=1e-12)
    # a face between two layers has a cell in each, one on the surface one
    for e in card.interfaces:
        cells = mesh.face_cells(mesh.faces_with(e["attribute"]))
        below, above = e["between_layers"]
        if above < 0:
            assert np.all(cells[:, 1] == -1)
            assert np.all(mesh.cell_attributes[cells[:, 0]] == below + 1)
        else:
            assert np.all(cells[:, 0] < cells[:, 1])
            got = np.sort(mesh.cell_attributes[cells], axis=1)
            assert np.all(got == [below + 1, above + 1])
    # every cell's shape in the same order, by the measure the build checks
    quality = cell_quality(res.msh_path)
    assert quality.shape == (len(mesh.cells),)
    assert quality.min() == pytest.approx(res.validation.min_sicn, rel=1e-12)
    assert 0.0 < quality.min() and quality.max() <= 1.0 + 1e-12
    # corners that are no face of any cell, and a malformed request
    assert mesh.face_cells(np.array([[0, 0, 0][:d]])).tolist() == [[-1, -1]]
    with pytest.raises(ValueError, match="node indices"):
        mesh.face_cells(np.zeros((2, d), dtype=float))


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no mesh file"):
        read_mesh(tmp_path / "nowhere.msh")
    with pytest.raises(FileNotFoundError, match="no mesh file"):
        cell_quality(tmp_path / "nowhere.msh")


def _two_layers_card():
    """The manifest of a unit ball of two layers, the interface at 0.6,
    assembled by hand."""
    g = Geometry(Skeleton([0.0, 0.6, 1.0]), layer_names=["core", "mantle"],
                 interface_names=["cmb", "surface"])
    b = g.skeleton.boundaries
    layers = [manifest.LayerEntry.from_layer(lay, attribute=i + 1, r_inner=b[i],
                                             r_outer=b[i + 1], in_geometry=True)
              for i, lay in enumerate(g.layers)]
    faces = [manifest.InterfaceEntry.from_interface(f, attribute=k + 1,
                                                    radius=f.radius)
             for k, f in enumerate(g.interfaces)]
    return manifest.MeshManifest.from_build(
        mesh=manifest.mesh_block("m.msh", format="msh", nodes="reference"),
        layers=layers, interfaces=faces)


@pytest.mark.parametrize("fold", [True, False])
def test_a_cell_with_its_corners_on_a_sphere_takes_its_neighbours_layer(fold):
    """Four corners A, B, C, D on the sphere r = 0.6, A and D nearer the
    pole: the interface triangulated across BC folds inward, and the cell
    ABCD fills the fold from the mantle (its sides ABD and ACD shared with
    mantle cells over them); triangulated across AD it is convex, and ABCD
    lies in the core under it (its sides ABC and BCD shared with core
    cells).  Either way its corners lie on the sphere, and its layer is
    its neighbours' across the sides that are no interface's."""
    def on(theta, phi, r=0.6):
        return r * np.array([np.sin(theta) * np.cos(phi),
                             np.sin(theta) * np.sin(phi), np.cos(theta)])
    A, D = on(0.1, 0.0), on(0.1, np.pi)
    B, C = on(0.15, 0.5 * np.pi), on(0.15, 1.5 * np.pi)
    O, P = np.array([0.0, 0.0, 0.3]), np.array([0.0, 0.0, 0.8])
    points = np.array([A, B, C, D, O, P])
    a, b, c, d, o, p = range(6)
    cells = [[o, a, b, c], [o, b, c, d], [a, b, c, d], [p, a, b, d], [p, a, c, d]]
    layer, interface = (([1, 1, 2, 2, 2], [[a, b, c], [b, c, d]]) if fold
                        else ([1, 1, 1, 2, 2], [[a, b, d], [a, c, d]]))
    mesh = MeshArrays(dimension=3, points=points, tags=np.arange(1, 7),
                      cells=np.array(cells), cell_attributes=np.array(layer),
                      faces=np.array(interface), face_attributes=np.array([1, 1]))
    assert np.allclose(np.linalg.norm(points[[a, b, c, d]], axis=1), 0.6)
    assert mesh.layers_by_position(_two_layers_card()).tolist() == layer
    # alone, with no neighbour to say, it is refused
    alone = MeshArrays(dimension=3, points=points, tags=np.arange(1, 7),
                       cells=np.array([[a, b, c, d]]), cell_attributes=np.array([2]),
                       faces=np.array(interface), face_attributes=np.array([1, 1]))
    with pytest.raises(ValueError, match="1 cells have their corners on an "
                                         "interface and no neighbour"):
        alone.layers_by_position(_two_layers_card())
