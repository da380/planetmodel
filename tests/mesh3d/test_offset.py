"""The offset two-body meshes."""
import numpy as np
import pytest

import gmsh

from planetmodel.mesh3d import build_offset_mesh, manifest as sc
from planetmodel.mesh3d._session import session

from conftest import COARSE

pytestmark = pytest.mark.gmsh


@pytest.fixture(scope="module")
def offset3(tmp_path_factory):
    return build_offset_mesh(tmp_path_factory.mktemp("offset") / "ball",
                             inner_radius=0.4, outer_radius=1.0, offset=0.3,
                             sizing=COARSE, order=2)


def test_an_offset_ball_builds_and_validates(offset3):
    res = offset3
    assert res.validation.ok, res.validation.failures
    assert res.counts["layers"] == 2 and res.counts["interfaces"] == 2
    assert res.geometry is None and res.spec is None and res.mapping is None
    assert res.msh_path.exists() and res.manifest_path.exists()


def test_the_offset_manifest_says_what_it_knows(offset3):
    card = sc.read(offset3.manifest_path)
    sc.validate_against(card, layer_count=2, interface_count=2)
    assert card.delivery == "physical"
    assert card.geometry["kind"] == "two_sphere"
    assert card.geometry["inclusion_radius"] == pytest.approx(0.4)
    assert card.geometry["offset"] == pytest.approx(0.3)
    assert card.geometry["outer_radius"] == 1.0
    assert card.geometry["n_layers"] == 2
    assert [lay["name"] for lay in card.layers] == ["inclusion", "matrix"]
    assert all(lay["in_geometry"] for lay in card.layers)
    assert [f["name"] for f in card.interfaces] == ["inclusion_boundary", "surface"]
    assert [f["between_layers"] for f in card.interfaces] == [[0, 1], [1, -1]]
    assert card.mapping["kind"] == "IdentityMapping"
    assert card.mapping["applied_to_nodes"] is False
    # the inclusion's node-average radius is about the origin, so it
    # exceeds its own radius by the offset
    inner = card.interfaces[0]["mean_radius"]
    assert 0.4 < inner < 0.4 + 0.3
    assert card.interfaces[1]["mean_radius"] == pytest.approx(1.0, abs=1e-3)


def test_the_inclusion_sits_where_it_was_put(offset3):
    with session(name="nodes"):
        gmsh.merge(str(offset3.msh_path))
        (face,) = gmsh.model.getEntitiesForPhysicalGroup(2, 1)
        _, coords, _ = gmsh.model.mesh.getNodes(2, face, includeBoundary=True)
    xyz = coords.reshape(-1, 3)
    assert np.allclose(np.linalg.norm(xyz - [0.0, 0.0, 0.3], axis=1), 0.4,
                       atol=2e-3)


@pytest.mark.parametrize("offset", [0.0, 0.3])
def test_a_disc_builds_in_two_dimensions(offset, tmp_path):
    res = build_offset_mesh(tmp_path / "disc", inner_radius=0.4, outer_radius=1.0,
                            offset=offset, sizing=COARSE, dimension=2)
    assert res.validation.ok
    card = sc.read(res.manifest_path)
    assert card.geometry["kind"] == "two_disc"
    assert card.mesh["dimension"] == 2
    with session(name="nodes"):
        gmsh.merge(str(res.msh_path))
        (curve,) = gmsh.model.getEntitiesForPhysicalGroup(1, 1)
        _, coords, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
    xyz = coords.reshape(-1, 3)
    assert np.allclose(np.linalg.norm(xyz - [offset, 0.0, 0.0], axis=1), 0.4,
                       atol=2e-3)


def test_an_offset_mesh_keeps_the_numbers_it_was_given(tmp_path):
    res = build_offset_mesh(tmp_path / "five", inner_radius=2.0, outer_radius=5.0,
                            offset=1.5, sizing=COARSE.__class__(0.75, 1.5, 1.5),
                            dimension=2, order=1)
    assert res.validation.ok
    card = sc.read(res.manifest_path)
    assert card.geometry["inclusion_radius"] == 2.0
    assert card.geometry["offset"] == 1.5
    assert card.geometry["outer_radius"] == 5.0
    assert card.layers[1]["r_outer"] == 5.0
    assert card.sizing["per_interface"][0]["size"] == 0.75
    with session(name="nodes"):
        gmsh.merge(str(res.msh_path))
        _, coords, _ = gmsh.model.mesh.getNodes()
    r = np.linalg.norm(coords.reshape(-1, 3), axis=1)
    assert r.max() == pytest.approx(5.0, abs=1e-9)


def test_offset_refusals(tmp_path):
    kw = dict(sizing=COARSE, dimension=2)
    with pytest.raises(ValueError, match="0 < inner_radius < outer_radius"):
        build_offset_mesh(tmp_path / "a", inner_radius=1.0, outer_radius=0.5, **kw)
    with pytest.raises(ValueError, match="strictly enclosed"):
        build_offset_mesh(tmp_path / "b", inner_radius=0.4, outer_radius=1.0,
                          offset=0.7, **kw)
    with pytest.raises(ValueError, match="dimension must be"):
        build_offset_mesh(tmp_path / "c", inner_radius=0.4, outer_radius=1.0,
                          sizing=COARSE, dimension=1)
    with pytest.raises(ValueError, match="element order"):
        build_offset_mesh(tmp_path / "d", inner_radius=0.4, outer_radius=1.0,
                          order=5, **kw)
    with pytest.raises(ValueError, match="no sizing given"):
        build_offset_mesh(tmp_path / "e", inner_radius=0.4, outer_radius=1.0)
    with pytest.raises(TypeError, match="divisor"):
        build_offset_mesh(tmp_path / "f", inner_radius=0.4, outer_radius=1.0,
                          divisor=2.0, **kw)
    assert not list(tmp_path.iterdir())
