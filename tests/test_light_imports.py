"""`import planetmodel` pulls in numpy and scipy only: no extra, and not
ducc0, a dependency imported where a transform is made."""
import subprocess
import sys


def test_import_is_light(extra_modules):
    code = (
        "import sys, planetmodel\n"
        f"heavy = [m for m in {extra_modules + ('ducc0',)!r} if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_a_manifest_and_its_rules_need_no_gmsh(tmp_path):
    """With gmsh unimportable, a mesh manifest is read and the rules of a
    fault's faces applied, and what builds or reads a mesh is refused,
    naming the `meshing` extra."""
    import json

    from planetmodel.mesh3d import manifest

    card = manifest.MeshManifest.from_build(
        mesh=manifest.mesh_block("m.mesh", format="mfem", nodes="physical"),
        layers=[manifest.LayerEntry(attribute=1, name="ball", r_inner=0.0,
                                    r_outer=1.0, in_geometry=True)],
        interfaces=[manifest.InterfaceEntry(attribute=1, name="surface", radius=1.0,
                                            between_layers=[0, -1])],
        faults=[manifest.FaultEntry(name="f", attribute=2, layers=[1], tip=1)])
    path = manifest.write(tmp_path / "m", card)
    assert json.loads(path.read_text())["faults"][0]["name"] == "f"
    code = f"""
import sys
sys.modules["gmsh"] = None
import numpy as np
from planetmodel.mesh3d import manifest, MeshSpec, export_mfem_mesh, read_mesh
card = manifest.read({str(path)!r})
assert card.faults[0]["tip"] == 1 and card.surface_attribute == 1
points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1], [0.2, 0.2, -1]],
                  dtype=float)
assert manifest.fault_side(points, [0, 1, 2], [0, 1, 2, 3]) == 1
assert manifest.plus_cell(points, [[0, 1, 2]], [[[4, 0, 1, 2], [0, 1, 2, 3]]])[0] == 1
(edges,) = manifest.fault_curves([[[0, 1, 2]]], [[0, 1, 9]])
assert edges.free_edge.tolist() == [[0, 1]]
for attempt in ("from planetmodel.mesh3d import build_layered_mesh",
                "from planetmodel.mesh3d import build_offset_mesh",
                "read_mesh('m.msh')"):
    try:
        exec(attempt)
    except ImportError as error:
        assert "planetmodel[meshing]" in str(error), error
    else:
        raise AssertionError(attempt + " needed no gmsh")
assert sys.modules["gmsh"] is None
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=tmp_path)
    assert r.returncode == 0, r.stderr
