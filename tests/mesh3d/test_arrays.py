"""A written mesh read back as plain arrays (`read_mesh`): every node,
cell and boundary element the build counted, the attributes the
manifest numbers, the faces' cells, and the groups' names."""
import numpy as np
import pytest

from planetmodel.mesh3d import MeshSpec, build_layered_mesh, manifest, read_mesh

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
    # corners that are no face of any cell, and a malformed request
    assert mesh.face_cells(np.array([[0, 0, 0][:d]])).tolist() == [[-1, -1]]
    with pytest.raises(ValueError, match="node indices"):
        mesh.face_cells(np.zeros((2, d), dtype=float))


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no mesh file"):
        read_mesh(tmp_path / "nowhere.msh")
