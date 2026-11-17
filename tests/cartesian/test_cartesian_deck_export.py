"""A model read from a deck of depths, meshed and exported for MFEM as a
consumer takes it: a unit box a few elements across, built from the
model's own geometry, every field written by layer, each dof holding the
model's value at its depth, the viscosity zero where a layer has none,
and the interfaces' one-sided values those of the deck."""
import numpy as np
import pytest

pytest.importorskip("gmsh", reason="needs the planetmodel[meshing] extra")

from planetmodel import Deck, cartesian, write_deck  # noqa: E402
from planetmodel.mesh3d import (MeshSpec, UniformInterfaces,  # noqa: E402
                                build_layered_mesh, export_mfem, manifest as sc)
from planetmodel.mesh3d.export import MESH_READ_OPTIONS, _node_array  # noqa: E402

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]

#: Coarse sizing: a few elements across a unit box.
SIZING = UniformInterfaces(0.25, 0.4, 0.3)


def unit_deck() -> Deck:
    """Three layers to a depth of 1, the density rising in each, the
    deepest a Maxwell body."""
    depth = np.array([0.0, 0.15, 0.3, 0.3, 0.45, 0.6, 0.6, 0.8, 1.0])
    rho = np.array([2.6, 2.65, 2.7, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45]) * 1e3
    vp = np.array([6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.1, 8.1, 8.1]) * 1e3
    vs = vp / 1.75
    eta = np.array([np.nan] * 6 + [1e20] * 3)
    columns = {"rho": rho, "vpv": vp, "vsv": vs, "vph": vp, "vsh": vs,
               "eta": np.ones(depth.size), "viscosity": eta}
    return Deck(depth, columns, header={"name": "unit", "layers": ("a", "b", "c"),
                                        "interfaces": ("surface", "ab", "bc")})


def test_a_deck_read_flat_model_is_exported_by_layer(tmp_path):
    path = write_deck(tmp_path / "unit.deck", unit_deck(), cartesian.MAXWELL)
    model = cartesian.MaxwellDeckModel(path, cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                                       kind="linear")
    res = build_layered_mesh(MeshSpec(model.geometry, SIZING), tmp_path / "unit")
    export = export_mfem(res, tmp_path / "out", model=model,
                         fields=["rho", "N", "viscosity"])
    o = MESH_READ_OPTIONS
    mesh = mfem.Mesh(str(export.mesh_path), o["generate_edges"], o["refine"],
                     o["fix_orientation"])
    card = sc.read(export.manifest_path)
    assert [lay["name"] for lay in card.layers] == ["a", "b", "c"]
    gf = {name: mfem.GridFunction(mesh, str(p)) for name, p in
          export.field_paths.items()}
    fes = gf["rho"].FESpace()
    X = np.zeros((fes.GetNDofs(), 3))
    attribute = np.zeros(fes.GetNDofs(), dtype=int)
    for e in range(mesh.GetNE()):
        pts = mfem.DenseMatrix()
        mesh.GetElementTransformation(e).Transform(fes.GetFE(e).GetNodes(), pts)
        dofs = np.asarray(fes.GetElementDofs(e), dtype=int)
        X[dofs] = pts.GetDataArray().T
        attribute[dofs] = mesh.GetAttribute(e)
    depth = -X[:, 2]
    values = {name: np.array(_node_array(g), dtype=float)[:, 0]
              for name, g in gf.items()}
    for k in (1, 2, 3):
        on = attribute == k
        layer = model.layer(k - 1)
        lo, hi = layer.interval
        d = np.clip(depth[on], lo, hi)
        for name in ("rho", "N"):
            assert np.allclose(values[name][on], layer[name](d), rtol=1e-12, atol=0)
        want = layer["viscosity"](d) if "viscosity" in layer else 0.0
        assert np.allclose(values["viscosity"][on], want, rtol=1e-12, atol=0)
    # the one-sided values at each interface are the deck's, the deeper first
    assert [f["values"]["rho"] for f in card.interfaces] == \
        [[2600.0, None], [3200.0, 2700.0], [3350.0, 3300.0]]
