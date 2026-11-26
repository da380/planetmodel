"""A mesh in MFEM's format read as plain arrays: a cubed sphere's cells,
faces and attributes, the shape of its cells, and what is stated for
simplices alone."""
import numpy as np
import pytest

from cubed_cases import COARSE, ball, shell
from planetmodel.mesh3d import (CubedSphereSpec, Shell, build_cubed_sphere,
                                cell_quality, export_mfem_mesh, manifest as sc,
                                read_mesh)
from planetmodel.mesh3d._cubed import cubed_map, cubed_topology

pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = pytest.mark.mfem


@pytest.mark.parametrize("dimension", [3, 2])
@pytest.mark.parametrize("make", [ball, shell])
def test_a_cubed_sphere_as_arrays(make, dimension, tmp_path):
    spec = CubedSphereSpec(make(), 3, sizing=COARSE, dimension=dimension, order=2,
                           shells=(Shell(ratio=0.2),))
    res = build_cubed_sphere(spec, tmp_path / "m")
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    d = dimension
    assert mesh.dimension == d and mesh.tensor
    assert mesh.points.shape == (res.counts["nodes"], 3)
    assert mesh.cells.shape == (spec.cells, 2 ** d)
    assert mesh.faces.shape == (len(card.interfaces) * 2 * d * 3 ** (d - 1),
                                2 ** (d - 1))
    assert not mesh.curves and not mesh.names
    if d == 2:
        assert np.all(mesh.points[:, 2] == 0.0)
    # the cells are the topology's, corner for corner, at the mapped points
    t = cubed_topology(spec.edge, spec.levels, dimension=d, hollow=spec.hollow)
    x = cubed_map(t.points, spheres=spec.spheres, blend=spec.blend,
                  half_width=spec.half_width or 0.0)
    np.testing.assert_array_equal(mesh.cells, t.cells)
    np.testing.assert_allclose(mesh.points[:t.points.shape[0], :d], x, atol=1e-15)
    np.testing.assert_array_equal(mesh.layers(card), mesh.cell_attributes)
    for entry in card.interfaces:
        rows = mesh.faces_with(entry["attribute"])
        np.testing.assert_allclose(np.linalg.norm(mesh.points[rows], axis=-1),
                                   entry["radius"], rtol=1e-14)
    assert set(mesh.cell_attributes.tolist()) == {e["attribute"]
                                                  for e in card.layers}
    quality = cell_quality(res.mesh_path)
    assert quality.shape == (spec.cells,)
    assert quality.min() == pytest.approx(res.validation.min_sicn, rel=1e-12)
    assert 0.0 < quality.min() and quality.max() <= 1.0 + 1e-12


def test_a_delivery_reads_the_same(tmp_path):
    res = build_cubed_sphere(CubedSphereSpec(ball(), 2, sizing=COARSE), tmp_path / "m")
    export = export_mfem_mesh(res, tmp_path / "out")
    built, delivered = read_mesh(res.mesh_path), read_mesh(export.mesh_path)
    np.testing.assert_array_equal(built.cells, delivered.cells)
    np.testing.assert_array_equal(built.faces, delivered.faces)
    np.testing.assert_allclose(built.points, delivered.points, atol=1e-15)


def test_the_rules_for_simplices_refuse_hexahedra(tmp_path):
    res = build_cubed_sphere(CubedSphereSpec(ball(), 2, sizing=COARSE), tmp_path / "m")
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    with pytest.raises(ValueError, match="stated for a mesh of simplices"):
        mesh.layers_by_position(card)
    with pytest.raises(ValueError, match="stated for a mesh of simplices"):
        mesh.face_cells(mesh.faces)
    with pytest.raises(ValueError, match="stated for a mesh of simplices"):
        mesh.walls(card)
    with pytest.raises(FileNotFoundError):
        read_mesh(tmp_path / "missing.mesh")


@pytest.mark.gmsh
def test_tetrahedra_in_mfem_s_format(tmp_path):
    """A delivery of a mesh gmsh built is read by its corners too, and the
    shape of its cells is the MSH file's to give."""
    pytest.importorskip("gmsh", reason="needs the planetmodel[meshing] extra")
    from planetmodel.mesh3d import MeshSpec, UniformInterfaces, build_layered_mesh
    res = build_layered_mesh(MeshSpec(ball(), UniformInterfaces(0.3, 0.3, 0.3)),
                             tmp_path / "tets")
    export = export_mfem_mesh(res, tmp_path / "out")
    msh, mesh = read_mesh(res.mesh_path), read_mesh(export.mesh_path)
    assert not mesh.tensor and mesh.cells.shape == (msh.cells.shape[0], 4)
    assert sorted(np.bincount(mesh.cell_attributes).tolist()) == sorted(
        np.bincount(msh.cell_attributes).tolist())
    np.testing.assert_array_equal(mesh.layers_by_position(sc.read(res.manifest_path)),
                                  mesh.cell_attributes)
    with pytest.raises(ValueError, match="measured by gmsh"):
        cell_quality(export.mesh_path)
