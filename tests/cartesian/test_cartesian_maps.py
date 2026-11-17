"""A flat model drawn: a plane slice as a map on a box's grid and a vertical
section, depth downward."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib",
                                 reason="needs the planetmodel[plot] extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from planetmodel import cartesian as flat  # noqa: E402
from planetmodel.cartesian import slices  # noqa: E402


def test_a_map_and_a_section_of_a_flat_model(tmp_path):
    """A map on a box's grid fills each node's cell, the period tiled; a
    section runs depth downward, layer by layer; a rectangle's grid makes no
    map."""
    from planetmodel.plotting import grid_cell_edges, plane_map, vertical_section
    box = flat.Box([(0.0, 2.0), (-1.0, 1.0)])
    grid = flat.Grid(box, (8, 4))
    xe, ye = grid_cell_edges(grid)
    assert np.allclose(xe, -0.125 + 0.25 * np.arange(9))
    assert np.isclose(ye[-1] - ye[0], 2.0)
    m = flat.LayeredIsotropicElastic([0.0, 0.3, 1.0], box, rho=[1.0, 2.0],
                                     vp=[2.0, 3.0], vs=[1.0, 1.5])
    m = m.with_field(1, "vs", flat.AnalyticField(
        (0.3, 1.0), lambda d, x, y: 1.5 + 0.1 * np.sin(np.pi * x) * np.cos(np.pi * y)
        + 0.0 * d, name="vs"), replace=True)
    fig, (a, b) = plt.subplots(1, 2)
    mesh = plane_map(a, slices.plane(m, "vs", depth=0.5, grid=grid,
                                     relative_to="mean"), colorbar=True)
    assert mesh.get_array().size == 32 and a.get_aspect() == 1.0
    lo, hi = mesh.get_clim()
    assert lo == -hi
    sec = slices.section(m, "rho", (0.0, 0.0), end=(2.0, 0.0), n=4, m=21)
    meshes = vertical_section(b, sec, colorbar=True)
    assert len(meshes) == 2 and b.yaxis_inverted()
    path = tmp_path / "flat-map-section.png"
    fig.savefig(path)
    assert path.stat().st_size > 0
    plt.close(fig)
    rect = flat.LayeredIsotropicElastic([0.0, 1.0], flat.Rectangle((0.0, 1.0)),
                                        rho=[1.0], vp=[2.0], vs=[1.0])
    with pytest.raises(ValueError, match="rectangle's plane is a line"):
        plane_map(a, slices.plane(rect, "rho", depth=0.5,
                                  grid=flat.Grid(rect.geometry.domain, 4)))
