"""A mesh recipe: one TOML file, one command, a mesh and its manifest.

A recipe names a deck and its reader, the surgery to perform on the
body (interfaces to drop, where to cut, shells to insert, a vacuum
buffer), the mesh dimension and order, the sizing policy, optional
surfaces with relief and the mapping rule, and where to write.  `python
-m planetmodel.mesh3d recipe.toml` builds it; `--check` reads the
recipe and reports the body without meshing.  Everything in the file is
also a `MeshSpec` in Python, which is what the recipe reader produces.

This script writes a tiny deck and recipe to a temporary directory,
checks it, builds it and reads the manifest back.  It needs the
`meshing` extra (gmsh); the mesh is a few hundred elements.
"""
import tempfile
from pathlib import Path

import numpy as np

try:
    import gmsh  # noqa: F401
except ImportError:
    print("gmsh is not installed (pip install 'planetmodel[meshing]'); skipping")
    raise SystemExit(0)

from planetmodel.io import manifest as mf
from planetmodel.io import recipe as rp
from planetmodel.mesh3d import __main__ as cli

RECIPE = """
[model]
source = "planet.deck"
reader = "isotropic_deck"
rref_m = 1.0e6

[geometry]
insert = ["floor"]
buffer = { ratio = 0.2, name = "buffer" }

  [geometry.floor]
  radius_km = 400
  role = "control"

[mesh]
dimension = 3
order = 1

[sizing]
policy = "uniform_interfaces"
h_min_km = 150
h_max_km = 300
decay_width_km = 300

[output]
path = "mesh/small"
"""


def write_deck(path):
    """A five-knot isotropic deck: three shells out to 960 km, all solid."""
    rows = []
    for lo, hi in ((0.0, 0.2e6), (0.2e6, 0.6e6), (0.6e6, 0.96e6)):
        for r in np.linspace(lo, hi, 5):
            rows.append(f"{r:12.1f} {5000.0:9.1f} {8000.0:9.1f} "
                        f"{4500.0:9.1f} {1000.0:9.1f} {600.0:9.1f}")
    path.write_text("small planet\n 1 1.0 1\n"
                    f" {len(rows)} 5 10\n" + "\n".join(rows) + "\n")


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_deck(root / "planet.deck")
    (root / "recipe.toml").write_text(RECIPE)

    # -- reading: the recipe becomes a MeshSpec ---------------------------------
    card = rp.read(root / "recipe.toml")
    spec = card.spec
    assert spec.order == 1 and spec.delivery == "physical"
    assert spec.body.skeleton.nlayers == 3                       # the deck's shells
    print("body from the recipe:", [lay.state for lay in spec.body.layers],
          "outer radius", spec.body.skeleton.boundaries[-1])

    # -- --check reads and reports without meshing -----------------------------
    assert cli.main([str(root / "recipe.toml"), "--check"]) == 0

    # -- building ---------------------------------------------------------------
    result = rp.build(root / "recipe.toml")
    assert result.msh_path.exists() and result.manifest_path.exists()
    card = mf.read(result.manifest_path)
    mf.validate_structure(card)
    assert card.provenance["recipe"] == "recipe.toml"
    assert card.provenance["command"] == "python -m planetmodel.mesh3d recipe.toml"
    assert [lay["name"] for lay in card.layers][-1] == "buffer"
    assert "floor" in [f["name"] for f in card.interfaces]      # the inserted control
    assert card.vacuum_attributes == (len(card.layers),)
    assert card.validation["negative_jacobians"] == 0
    print("mesh:", card.mesh["n_elements"], "elements;",
          "layers:", [lay["attribute"] for lay in card.layers])

    # The command line does the same thing.
    assert cli.main([str(root / "recipe.toml")]) == 0

print("ok: a recipe reads to a MeshSpec, builds, and its manifest says which recipe")
