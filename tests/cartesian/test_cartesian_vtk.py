"""VTK files of a flat model: parsed back, their points the box's nodes
closed by the far ends of its periods, z up and at their physical
positions by default, their arrays the model's fields and departures."""
import base64
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest

from planetmodel import VECTOR
from planetmodel import cartesian as flat
from planetmodel.cartesian import FLAT
from planetmodel.plotting import write_flat_vtk

BOX = flat.Box([(0.0, 2.0), (-1.0, 1.0)])


def decode(text, dtype):
    """VTK's compressed binary: a UInt64 header, then the zlib blocks."""
    text = text.strip()
    blocks = int(np.frombuffer(base64.b64decode(text[:32]), dtype="<u8")[0])
    head_chars = 4 * ((8 * (3 + blocks) + 2) // 3)
    header = np.frombuffer(base64.b64decode(text[:head_chars]), dtype="<u8")
    data = base64.b64decode(text[head_chars:])
    out, at = [], 0
    for size in header[3:]:
        out.append(zlib.decompress(data[at:at + int(size)]))
        at += int(size)
    return np.frombuffer(b"".join(out), dtype=dtype)


def read_block(path):
    grid = ET.parse(path).getroot().find("StructuredGrid")
    nx, ny, nz = (int(e) + 1 for e in grid.get("WholeExtent").split()[1::2])
    points = decode(grid.find("Piece/Points/DataArray").text, "<f8")
    arrays = {}
    for da in grid.findall("Piece/PointData/DataArray"):
        dtype = "<f8" if da.get("type") == "Float64" else "<i4"
        k = int(da.get("NumberOfComponents"))
        arrays[da.get("Name")] = decode(da.text, dtype).reshape(nz, ny, nx, k)
    return points.reshape(nz, ny, nx, 3), arrays


def lateral(d, x, y):
    return 4.0 + 0.3 * d * np.cos(np.pi * x) + 0.2 * d * np.sin(np.pi * y)


@pytest.fixture
def model():
    m = flat.LayeredIsotropicElastic([0.0, 0.4, 1.0], BOX, rho=[2.0, 3.0],
                                     vp=[5.0, 7.0], vs=[3.0, 4.0],
                                     layer_names=["upper", "lower"])
    m = m.with_field(1, "vs", flat.AnalyticField((0.4, 1.0), lateral, name="vs"),
                     replace=True)
    return m.with_field(1, "flow", flat.AnalyticField(
        (0.4, 1.0), lambda d, x, y: np.stack(np.broadcast_arrays(y, -x, -0.5 * d),
                                             axis=-1),
        character=VECTOR, frame="cartesian", name="flow"))


def test_a_flat_model_is_written_layer_by_layer_on_its_closed_grid(model, tmp_path):
    grid = flat.Grid(BOX, (4, 3))
    out = write_flat_vtk(tmp_path / "box", model, grid=grid, n=3,
                         departures={"vs": "mean"})
    sets = ET.parse(out).getroot().findall("vtkMultiBlockDataSet/DataSet")
    assert [s.get("name") for s in sets] == ["upper", "lower"]
    points, arrays = read_block(tmp_path / "box" / "01_lower.vts")
    x = np.append(grid.x, 2.0)
    y = np.append(grid.y, 1.0)
    D, X, Y = np.meshgrid(np.linspace(0.4, 1.0, 3), x, y, indexing="ij")
    assert np.allclose(points, FLAT.points(D, X, Y))
    assert np.allclose(arrays["vs"][..., 0], lateral(D, X, Y))
    assert np.allclose(arrays["flow"], np.stack([Y, -X, -0.5 * D], axis=-1))
    assert np.allclose(arrays["vs_percent_from_mean"][..., 0],
                       100.0 * (lateral(D, X, Y) / 4.0 - 1.0))
    assert np.all(arrays["layer"] == 1)
    _, upper = read_block(tmp_path / "box" / "00_upper.vts")
    assert "vs_percent_from_mean" in upper and "flow" not in upper


def test_the_points_are_physical_by_default_and_a_section_is_a_sheet(model, tmp_path):
    g = flat.Grid(BOX, (8, 6))
    xg, yg = g.nodes
    top = flat.FourierFunction(g, g.transform.analyse_grid(
        np.broadcast_to(0.05 * (1.0 + np.cos(np.pi * xg)) + 0.0 * yg, g.shape)))
    raised = model.stretched(flat.layer_linear(model.skeleton, [top, None, None]))
    write_flat_vtk(tmp_path / "raised", raised, grid=g, n=3)
    points, _ = read_block(tmp_path / "raised" / "00_upper.vts")
    assert np.allclose(points[0, ..., 2], top(points[0, ..., 0], points[0, ..., 1]),
                       atol=1e-14)
    rect = flat.LayeredIsotropicElastic([0.0, 1.0], flat.Rectangle((0.0, 2.0)),
                                        rho=[1.0], vp=[2.0], vs=[1.0])
    write_flat_vtk(tmp_path / "sheet", rect, n=2)
    points, arrays = read_block(tmp_path / "sheet" / "00_layer.vts")
    assert points.shape == (2, flat.Grid(rect.geometry.domain, 48).shape[0] + 1, 1, 3)
    assert np.allclose(points[..., 1], 0.0) and np.allclose(arrays["rho"], 1.0)
    from planetmodel import LayeredIsotropicElastic
    with pytest.raises(TypeError, match="write_vtk"):
        write_flat_vtk(tmp_path / "x", LayeredIsotropicElastic([0.0, 1.0], rho=[1.0],
                                                               vp=[1.0], vs=[1.0]))
