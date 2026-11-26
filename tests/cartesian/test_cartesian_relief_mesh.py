"""A unit box meshed under topography from a series and from a grid.

The reference box is meshed a few elements across and the mapping moves
it at the export: under a Fourier relief on the surface and a gridded root
on the Moho, mapped, the surface's nodes lie on the series and the Moho's
on the gridded relief to rounding, the base's stay where they were and the
sides' on their planes; a rectangle under a gridded relief likewise.
"""
import numpy as np
import pytest

pytest.importorskip("gmsh", reason="needs the planetmodel[meshing] extra")

from planetmodel import Skeleton  # noqa: E402
from planetmodel import cartesian as flat  # noqa: E402
from planetmodel.mesh3d import (MeshSpec, UniformInterfaces,  # noqa: E402
                                build_layered_mesh, manifest as sc, read_mesh)

pytestmark = pytest.mark.gmsh

DEPTHS = [0.0, 0.4, 1.0]
UNIT = flat.Box([(0.0, 1.0), (-0.5, 0.5)])
SIZING = UniformInterfaces(0.15, 0.3, 0.3)


def reliefs(domain):
    """A range from a short series and a root sampled on a grid: heights,
    positive up, of 0.08 at most."""
    g = flat.Grid(domain, 8)
    t = g.transform
    c = np.zeros(t.size)
    along = [0] * (g.dimension - 1)
    c[0] = 0.02 * np.sqrt(g.area)
    c[t.component_of([1] + along)] = 0.04 * np.sqrt(g.area / 2)
    surface = flat.FourierFunction(g, c)
    fine = flat.Grid(domain, 32)
    root = flat.GriddedFunction.sampled(lambda x, y: -0.5 * surface(x, y), fine)
    return surface, root


@pytest.mark.parametrize("domain", [UNIT, flat.Rectangle((0.0, 1.0))],
                         ids=["box", "rectangle"])
def test_a_mesh_under_a_series_and_a_grid_lies_on_both(tmp_path, domain):
    sk = Skeleton(DEPTHS)
    surface, root = reliefs(domain)
    g = flat.Geometry(sk, domain, layer_names=["crust", "mantle"],
                      interface_names=["surface", "moho"])
    g = g.stretched(flat.layer_linear(sk, [surface, root, None]))
    res = build_layered_mesh(MeshSpec(g, SIZING, order=2), tmp_path / "box")
    assert res.validation.ok and res.mapping is g.mapping
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    # a rectangle's mesh is in (x, z), at y = 0
    axes = card.mesh_axes(len(domain.axes) + 1)
    points = np.zeros((len(mesh.points), 3))
    points[:, axes] = mesh.points[:, :len(axes)]
    for k, relief in enumerate((surface, root)):
        nodes = np.unique(mesh.faces_with(card.interfaces[k]["attribute"]))
        ref = points[nodes]
        assert np.abs(ref[:, 2] + DEPTHS[k]).max() < 1e-12
        x = res.mapping(ref)
        assert np.abs(x[:, 2] - (relief(x[:, 0], x[:, 1]) - DEPTHS[k])).max() < 1e-12
    on_top = np.unique(mesh.faces_with(card.interfaces[0]["attribute"]))
    top = res.mapping(points[on_top])
    assert top[:, 2].max() > 0.05 and top[:, 2].min() < -0.015
    base = [f for f in card.faces if f["name"] == "base"][0]
    nodes = np.unique(mesh.faces_with(base["attribute"]))
    assert np.allclose(res.mapping(points[nodes])[:, 2], -1.0, rtol=0, atol=1e-15)
