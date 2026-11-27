"""The migration script, `scripts/migrate_deck.py`: every deck of
`tests/data/legacy_decks`, written by planetmodel 2.2's writers, becomes
a model file whose model is the one 2.2 read from the deck.

Beside each deck is what 2.2's reader made of it (`<deck>.json`): the
model's name, its behaviours, its boundaries, the names of its layers
and interfaces, and every field of every layer at three points of it,
none a knot.  A Maxwell, a prony and a burgers deck of radii, a
PREM-like Earth with a viscosity graded through its lower mantle; the
same three of depths, and the elastic deck of depths with quality
factors.  Bare tables and what is refused, too.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from planetmodel import Viscoelastic, cartesian
from planetmodel.testing import check_model, check_model_file

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

from planetmodel.cartesian.modelfile import (  # noqa: E402
    read_model_file as read_flat_model_file)
from planetmodel.modelfile import read_model, read_model_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECKS = ROOT / "tests" / "data" / "legacy_decks"
NAMES = ("maxwell", "prony", "burgers", "flat_maxwell", "flat_prony", "flat_burgers",
         "flat_elastic")


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "migrate_deck", ROOT / "scripts" / "migrate_deck.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def agrees(model, oracle, *, rtol, sampled=1e-9):
    """The model is the oracle's: its names, its boundaries and every field
    at the stored points, to `rtol`, or to `sampled` for the branches of a
    Burgers layer, which a file holds as sampled on the deck's knots."""
    assert model.name == oracle["name"]
    assert list(model.behaviours) == oracle["behaviours"]
    assert np.array_equal(model.skeleton.boundaries, oracle["boundaries"])
    assert [f.name for f in model.geometry.interfaces] == oracle["interfaces"]
    burgers = oracle["kind"] == "burgers"
    for layer, want in zip(model.layers, oracle["layers"], strict=True):
        assert layer.name == want["name"]
        assert set(layer.names) == set(want["fields"]), layer.index
        at = np.array(want["at"])
        for name, values in want["fields"].items():
            tol = sampled if burgers and name.startswith("relaxation_") else rtol
            assert np.allclose(layer[name](at), values, rtol=tol, atol=0), (
                layer.index, name)


@pytest.mark.parametrize("name", NAMES)
def test_a_legacy_deck_becomes_the_model_file_of_the_model_it_was(name, script,
                                                                 tmp_path):
    oracle = json.loads((DECKS / f"{name}.json").read_text())
    domain = cartesian.Box(oracle["box"]) if "box" in oracle else None
    out = tmp_path / f"{name}.nc"
    model = script.migrate(DECKS / f"{name}.deck", out, domain=domain)
    agrees(model, oracle, rtol=1e-13, sampled=1e-13)
    check_model(model)
    check_model_file(out, model)
    data = (read_flat_model_file if "box" in oracle else read_model_file)(out)
    assert data.meta["deck"] == f"{name}.deck"
    assert list(data.behaviours) == oracle["behaviours"]
    back = read_model(out)
    assert type(back) is type(model) and isinstance(back, Viscoelastic)
    assert isinstance(back, cartesian.Model) == ("box" in oracle)
    agrees(back, oracle, rtol=1e-12)
    # the columns joined in their logarithm state that rule in the file
    rules = {var: v.attributes["radial_interpolation"]
             for part in data.parts for var, v in part.variables.items()}
    logged = {"maxwell": ["viscosity"], "prony": ["relaxation_time_1"]}
    for var in logged.get(oracle["kind"], []):
        assert rules[var] == "log-polynomial", var
    assert rules["rho"] == "polynomial"
    # its elements are the spans between the deck's knots
    knots = np.unique(np.loadtxt((DECKS / f"{name}.deck").read_text().splitlines()[2:],
                                 usecols=0))
    mesh = data.parts[0].mesh
    assert mesh.ngll == script.NGLL
    assert np.allclose(np.append(mesh.left, mesh.right[-1]), knots, rtol=1e-15)


def test_the_command_line_writes_the_file_and_says_what_it_holds(script, tmp_path,
                                                                 capsys):
    out = tmp_path / "flat.nc"
    script.main([str(DECKS / "flat_burgers.deck"), str(out),
                 "--domain=0,400e3,-200e3,200e3"])
    said = capsys.readouterr().out
    assert "3 layers, behaviours Elastic Viscoelastic" in said
    assert "asthenosphere" in said and "relaxation_time_2" in said
    agrees(read_model(out), json.loads((DECKS / "flat_burgers.json").read_text()),
           rtol=1e-12)
    line = tmp_path / "line.nc"
    script.main([str(DECKS / "flat_maxwell.deck"), str(line), "--domain", "0,400e3"])
    assert read_model(line).geometry.domain == cartesian.Rectangle((0.0, 400e3))


def test_a_bare_table_is_read_by_naming_its_columns(script, tmp_path):
    """The rows of the Maxwell deck under two lines that name nothing: the
    same fields, the names a deck without naming lines has, and the name
    given."""
    oracle = json.loads((DECKS / "maxwell.json").read_text())
    lines = (DECKS / "maxwell.deck").read_text().splitlines()
    columns = lines[1].split()[1:]
    bare = tmp_path / "bare.deck"
    bare.write_text("\n".join(["a table", "in SI (see the paper)"] + lines[4:]) + "\n")
    with pytest.raises(ValueError, match="the second line names the columns.*bare "
                                         "table is read with --columns"):
        script.model_of(bare)
    model = script.migrate(bare, tmp_path / "bare.nc", columns=columns,
                           header_lines=2, name="PREM (no ocean)")
    n = len(oracle["layers"])
    named = {**oracle, "interfaces": [None] * (n - 1) + ["surface"],
             "layers": [{**layer, "name": None} for layer in oracle["layers"]]}
    agrees(model, named, rtol=1e-13)
    agrees(read_model(tmp_path / "bare.nc"), named, rtol=1e-12)
    # a table of the density alone is a model that gravitates and no more
    rho = tmp_path / "rho.deck"
    rho.write_text("\n".join(" ".join(ln.split()[:2]) for ln in lines[4:]) + "\n")
    assert script.model_of(rho, columns=["rho"]).behaviours == ("SelfGravitating",)


def test_what_is_no_such_deck_is_refused_saying_why(script, tmp_path):
    lines = (DECKS / "burgers.deck").read_text().splitlines()
    with pytest.raises(ValueError, match="a table of depths holds no lateral extent"):
        script.model_of(DECKS / "flat_maxwell.deck")
    with pytest.raises(ValueError, match="takes no lateral domain"):
        script.model_of(DECKS / "maxwell.deck", domain=cartesian.Rectangle((0.0, 1.0)))
    said = tmp_path / "said.deck"
    said.write_text("\n".join([lines[0], lines[1] + " (SI), as written"]
                              + lines[2:]) + "\n")
    with pytest.raises(ValueError, match="read with --columns and --header-lines"):
        script.model_of(said)
    twice = tmp_path / "twice.deck"
    twice.write_text("\n".join(lines[:3] + [lines[2]] + lines[3:]) + "\n")
    with pytest.raises(ValueError, match="one `# layers` line; the deck has two"):
        script.model_of(twice)
    short = tmp_path / "short.deck"
    short.write_text("\n".join(lines[:2] + ["# layers a b"] + lines[3:]) + "\n")
    with pytest.raises(ValueError, match="layer"):
        script.model_of(short)
    half = tmp_path / "half.deck"
    rows = [" ".join(ln.split()[:-1] + ["nan"]) for ln in lines[4:]]
    half.write_text("\n".join(lines[:4] + rows) + "\n")
    with pytest.raises(ValueError, match=r"holds \['kelvin_viscosity'\] without the "
                                         "rest of a Burgers body"):
        script.model_of(half)
    with pytest.raises(ValueError, match="--domain takes"):
        script.domain_of("0,1,2")
