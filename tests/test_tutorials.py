"""Every tutorial runs headless to completion."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

TUTORIALS = Path(__file__).resolve().parent.parent / "examples" / "tutorials"


def run(script: str, *args: str) -> None:
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, str(TUTORIALS / script), *args], cwd=TUTORIALS,
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.parametrize("script", ["01_skeleton_and_geometry.py",
                                    "02_an_analytic_mapping.py",
                                    "03_a_radial_mesh.py"])
def test_core_tutorials(script):
    run(script)


@pytest.mark.gmsh
@pytest.mark.mfem
def test_mesh_tutorial():
    pytest.importorskip("gmsh")
    pytest.importorskip("mfem")
    run("04_a_mesh_for_mfem.py", "--temp")
