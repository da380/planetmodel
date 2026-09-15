"""Every tutorial runs headless to completion."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

TUTORIALS = Path(__file__).resolve().parent.parent / "examples" / "tutorials"


def run(script: str, *args: str, strict: bool = True) -> None:
    """Run a tutorial headless; `strict` turns every warning into an error,
    which the MFEM tutorials cannot have since PyMFEM's own import warns."""
    env = dict(os.environ, MPLBACKEND="Agg")
    flags = ["-W", "error"] if strict else []
    r = subprocess.run([sys.executable, *flags, str(TUTORIALS / script), *args],
                       cwd=TUTORIALS, capture_output=True, text=True, env=env,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.parametrize("script", ["01_skeleton_and_geometry.py",
                                    "02_an_analytic_mapping.py",
                                    "03_a_radial_mesh.py",
                                    "05_fields.py",
                                    "06_a_model_of_your_own.py",
                                    "07_prem.py",
                                    "08_simple_models.py",
                                    "10_love_numbers.py",
                                    "11_random_fields.py",
                                    "12_models_from_decks.py"])
def test_core_tutorials(script):
    run(script)


@pytest.mark.gmsh
@pytest.mark.mfem
@pytest.mark.parametrize("script", ["04_a_mesh_for_mfem.py", "09_fields_for_mfem.py"])
def test_mesh_tutorials(script):
    pytest.importorskip("gmsh")
    pytest.importorskip("mfem")
    run(script, "--temp", strict=False)
