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
                                    "10_random_fields.py",
                                    "11_models_from_decks.py"])
def test_core_tutorials(script):
    run(script)


@pytest.mark.netcdf
@pytest.mark.maps
def test_the_model_file_tutorial():
    """Not strict: netCDF4 1.7.4 warns under numpy 2.5 when it writes."""
    pytest.importorskip("netCDF4")
    pytest.importorskip("cartopy")
    run("12_a_laterally_varying_model.py", strict=False)


@pytest.mark.maps
def test_the_pictures_tutorial():
    pytest.importorskip("cartopy")
    run("13_pictures.py")


@pytest.mark.netcdf
@pytest.mark.maps
def test_the_kernels_tutorial():
    """Writes its kernel with the Python and draws it; not strict, since
    netCDF4 1.7.4 warns under numpy 2.5 when it writes."""
    pytest.importorskip("netCDF4")
    pytest.importorskip("cartopy")
    run("15_kernels.py", strict=False)


@pytest.mark.network
@pytest.mark.netcdf
@pytest.mark.maps
def test_the_published_models_tutorial():
    """Fetches its data sets, so it runs with the registry check (`-m
    network`); not strict, since netCDF4 1.7.4 warns under numpy 2.5."""
    pytest.importorskip("netCDF4")
    run("14_published_models.py", strict=False)


@pytest.mark.gmsh
@pytest.mark.mfem
@pytest.mark.parametrize("script", ["04_a_mesh_for_mfem.py", "09_fields_for_mfem.py"])
def test_mesh_tutorials(script):
    pytest.importorskip("gmsh")
    pytest.importorskip("mfem")
    run(script, "--temp", strict=False)


# -- the notebooks ----------------------------------------------------------------

def _make_notebooks() -> object:
    """The notebook generator, loaded from its script."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_notebooks",
                                                  TUTORIALS / "make_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_tutorial_is_listed_in_the_readme():
    readme = (TUTORIALS.parent / "README.md").read_text(encoding="utf-8")
    missing = [s.name for s in _make_notebooks().scripts() if s.name not in readme]
    assert not missing, f"not in examples/README.md: {missing}"


def test_the_notebooks_are_current():
    """Each tutorial has a notebook generated from it, up to date: the
    notebooks are what Colab opens, so a stale one teaches an older
    tutorial than the script the suite ran."""
    changes = _make_notebooks().stale()
    assert not changes, ("the notebooks are stale; run "
                         "examples/tutorials/make_notebooks.py: " + ", ".join(changes))


def test_the_badges_open_the_generated_notebooks():
    """Each Colab badge opens its own tutorial's notebook on the branch the
    notebooks install the library from."""
    mk = _make_notebooks()
    readme = (TUTORIALS.parent / "README.md").read_text(encoding="utf-8")
    missing = [s.name for s in mk.scripts() if mk.badge(s) not in readme]
    assert not missing, f"no badge for {missing}"


def test_the_notebooks_show_their_figures_rather_than_saving_them():
    """Every figure a script saves is one its notebook shows: the three
    lines that save and announce it are left out, and are the only way a
    tutorial writes a figure."""
    import json
    mk = _make_notebooks()
    for script in mk.scripts():
        text = script.read_text(encoding="utf-8")
        saves = [line for line in text.splitlines() if "savefig" in line]
        assert all(mk.SAVING.match(line) for line in saves), (script.name, saves)
        nb = json.loads((TUTORIALS / "notebooks" / f"{script.stem}.ipynb").read_text())
        code = "".join("".join(c["source"]) for c in nb["cells"]
                       if c["cell_type"] == "code")
        assert "savefig" not in code and "figure written" not in code, script.name
        assert text.count("fig.savefig") == text.count('print("figure written to"')


def test_a_notebook_runs_as_jupyter_would():
    """The code cells of a notebook, run in order from the notebooks
    directory as a kernel runs them, without `__file__` or the script's
    arguments: the generated first cell supplies the script's place."""
    import json
    nb = json.loads((TUTORIALS / "notebooks" / "07_prem.ipynb").read_text())
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"]
                        if c["cell_type"] == "code")
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, "-c", code, "-f", "kernel.json"],
                       cwd=TUTORIALS / "notebooks", capture_output=True, text=True,
                       env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "figure written" not in r.stdout and "PREM" in r.stdout
