"""Generates the tutorial notebooks from the tutorial scripts.

The numbered scripts in this directory are the source of the tutorials.
They are written in cells, the ``# %%`` convention that editors such as
VS Code and Spyder run cell by cell, and the test suite runs them
headless.  Each is converted here into a notebook under ``notebooks/``,
for working in Jupyter or trying the library online in Google Colab
without installing anything.  A notebook gains one cell, before its
first code cell, that installs the library when it runs on Colab and,
anywhere, gives the notebook the script's place in the repository, so
that the figures and data a tutorial reaches through ``__file__`` are
found as the script finds them; on Colab the data files a tutorial reads
are fetched from GitHub.

The conversion is mechanical and deterministic.  A ``# %% [markdown]``
cell becomes a markdown cell with its comment markers removed, any other
``# %%`` cell a code cell with the three lines that save a figure and
announce its file left out (``out = FIGURES / "..."``, ``fig.savefig(out,
dpi=...)``, ``print("figure written to", out)``), so that the figure,
left open at the end of its cell, is shown in the notebook instead; the
notebook is written in the layout of
nbformat and nbstripout (cell ids numbered, no outputs), so an unchanged
script regenerates its notebook byte for byte.  A notebook whose script
has gone is deleted.  Run it by hand,

    poetry run python examples/tutorials/make_notebooks.py

after editing a tutorial.  It exits with status 1 when it has changed
anything, and the test suite checks that the notebooks are current, so
a commit with stale notebooks fails a test rather than shipping them.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE / "notebooks"

MARKER = "# %%"

#: The lines of a script that save a figure and say so, left out of its
#: notebook, where the figure is shown instead.
SAVING = re.compile(r'^\s*(out = FIGURES / "[^"]+"|fig\.savefig\(out(, dpi=\d+)?\)'
                    r'|print\("figure written to", out\))$')

#: The branch of the repository the Colab cell installs from and fetches
#: data from, and the README's badges open the notebooks on.
BRANCH = "main"

#: The repository on GitHub.
REPOSITORY = "da380/planetmodel"

#: The extras each tutorial installs on Colab; `plot` unless listed.
EXTRAS = {
    "04_a_mesh_for_mfem": "meshing,mfem",
    "09_fields_for_mfem": "meshing,mfem",
    "12_a_laterally_varying_model": "netcdf,maps",
    "13_pictures": "maps",
    "14_published_models": "netcdf,maps",
    "15_kernels": "netcdf,maps",
}
DEFAULT_EXTRAS = "plot"

#: What a Colab install may not manage, said in the cell.
NOTES = {
    "04_a_mesh_for_mfem": "\n# PyMFEM may have no wheel for Colab's Python and build.",
    "09_fields_for_mfem": "\n# PyMFEM may have no wheel for Colab's Python and build.",
}

#: The files under `examples/data` each tutorial reads, fetched on Colab.
DATA = {
    "11_models_from_decks": ("prem.200", "prem.nocrust"),
}

INSTALL = """\
# Generated from examples/tutorials/{name}.
# Edit the script, not this notebook.  On Google Colab the library is
# installed first, from GitHub, with the extras this tutorial needs.{note}
# Everywhere, the notebook is given the script's place, so that the files
# it writes and the data it reads are found as the script finds them; its
# figures are shown here rather than saved.
import subprocess
import sys
from pathlib import Path
{imports}
if "google.colab" in sys.modules:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "planetmodel[{extras}] @ "
         "git+https://github.com/{repository}@{branch}"],
        check=True)
here = Path.cwd()
tutorials = here.parent if here.name == "notebooks" else here / "examples" / "tutorials"
__file__ = str(tutorials / "{name}")
sys.argv = [__file__]
(tutorials.parent / "figures").mkdir(parents=True, exist_ok=True)
if "get_ipython" in globals():          # figures shown in the notebook, sharp
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")"""

FETCH = """

for name in {files!r}:
    target = tutorials.parent / "data" / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/{repository}/{branch}/examples/data/"
            + name, target)"""

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def scripts() -> list[Path]:
    """The numbered tutorial scripts, in reading order."""
    return sorted(HERE.glob("[0-9][0-9]_*.py"))


def _cells(script: Path) -> list[tuple[str, str]]:
    """The script's cells as (kind, source) pairs, kind 'markdown' or
    'code', with empty cells dropped."""
    blocks: list[tuple[str, list[str]]] = []
    for number, line in enumerate(script.read_text(encoding="utf-8").splitlines(),
                                  start=1):
        if line.startswith(MARKER):
            tag = line[len(MARKER):].strip()
            if tag not in ("", "[markdown]"):
                raise ValueError(f"{script.name}:{number}: unknown cell tag {tag!r}")
            blocks.append(("markdown" if tag else "code", []))
        elif blocks:
            blocks[-1][1].append(line)
        elif line.strip():
            raise ValueError(f"{script.name}:{number}: text before the first cell")
    cells = []
    for kind, lines in blocks:
        if kind == "code":
            lines = [line for line in lines if not SAVING.match(line)]
        if kind == "markdown":
            text = []
            for line in lines:
                if line.strip() and line != "#" and not line.startswith("# "):
                    raise ValueError(f"{script.name}: markdown line not commented: "
                                     f"{line!r}")
                text.append(line[2:] if line.startswith("# ") else "")
            lines = text
        source = "\n".join(lines).strip("\n")
        if source.strip():
            cells.append((kind, source))
    return cells


def _cell(kind: str, source: str, number: int) -> dict[str, object]:
    """One cell as nbformat 4.5 lays it out."""
    cell: dict[str, object] = {"cell_type": kind, "id": str(number), "metadata": {},
                               "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def notebook(script: Path) -> str:
    """The text of the notebook generated from a script."""
    cells = _cells(script)
    first_code = next((i for i, (kind, _) in enumerate(cells) if kind == "code"), 0)
    install = INSTALL.format(name=script.name,
                             extras=EXTRAS.get(script.stem, DEFAULT_EXTRAS),
                             note=NOTES.get(script.stem, ""), branch=BRANCH,
                             repository=REPOSITORY,
                             imports=("import urllib.request\n" if script.stem in DATA
                                      else ""))
    if script.stem in DATA:
        install += FETCH.format(files=DATA[script.stem], branch=BRANCH,
                                repository=REPOSITORY)
    cells.insert(first_code, ("code", install))
    nb = {"cells": [_cell(kind, source, i) for i, (kind, source) in enumerate(cells)],
          "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False,
                      separators=(",", ": ")) + "\n"


def stale() -> list[str]:
    """What `main` would change: notebooks to write or delete."""
    wanted = set()
    changes = []
    for script in scripts():
        target = NOTEBOOKS / (script.stem + ".ipynb")
        wanted.add(target)
        if (not target.exists()
                or target.read_text(encoding="utf-8") != notebook(script)):
            changes.append(f"write {target.name}")
    for orphan in sorted(set(NOTEBOOKS.glob("*.ipynb")) - wanted):
        changes.append(f"delete {orphan.name}")
    return changes


def badge(script: Path) -> str:
    """The Markdown of the Colab badge that opens a script's notebook."""
    return ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPOSITORY}/blob/{BRANCH}/"
            f"examples/tutorials/notebooks/{script.stem}.ipynb)")


def main() -> int:
    NOTEBOOKS.mkdir(exist_ok=True)
    changed = []
    wanted = set()
    for script in scripts():
        target = NOTEBOOKS / (script.stem + ".ipynb")
        wanted.add(target)
        text = notebook(script)
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
            changed.append(f"wrote {target.relative_to(HERE.parent.parent)}")
    for orphan in sorted(set(NOTEBOOKS.glob("*.ipynb")) - wanted):
        orphan.unlink()
        changed.append(f"deleted {orphan.relative_to(HERE.parent.parent)}")
    for line in changed:
        print(line)
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
