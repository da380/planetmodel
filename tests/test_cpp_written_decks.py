"""The decks the C++ writer wrote, read by the Python: each fixture the C++
read and wrote back is, character for character, what the Python's own
`write_deck` makes of it, and the Maxwell, prony and burgers decks the C++
built are the Python's writing of the same decks and read with their names
and values, the prony deck's layer of one branch completed alike.

The C++ tests leave the decks in their build tree; PLANETMODEL_CPP_WRITTEN
names that directory (`build/cpp/tests/written` for the build of
CLAUDE.md), and without it these tests are skipped.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from planetmodel import (BURGERS, MAXWELL, MINEOS, ND, PRONY, TVEL, Deck, DeckFormat,
                         read_deck, write_deck)
from planetmodel.deck import maxwell_names, prony_branches

WRITTEN = os.environ.get("PLANETMODEL_CPP_WRITTEN")
FIXTURES = Path(__file__).resolve().parents[1] / "cpp" / "tests" / "fixtures"
pytestmark = pytest.mark.skipif(
    WRITTEN is None,
    reason="PLANETMODEL_CPP_WRITTEN names no directory of C++-written files")

#: Each fixture the C++ writes back, and its format.
CASES = {"prem200": MINEOS, "nocrust": MINEOS, "maxwell": MAXWELL, "prony": PRONY,
         "burgers": BURGERS, "nd": ND, "tvel": TVEL,
         "knots": DeckFormat(("rho", "vp"), header_lines=1)}


@pytest.mark.parametrize("label", sorted(CASES))
def test_a_fixture_written_back_by_the_cpp_is_the_pythons_writing(label, tmp_path):
    written = Path(WRITTEN) / f"deck_{label}.txt"
    assert written.exists(), f"the C++ tests wrote no {written.name}"
    fmt = CASES[label]
    ours = write_deck(tmp_path / written.name,
                      read_deck(FIXTURES / f"deck_{label}.txt", fmt), fmt)
    assert written.read_text() == ours.read_text()


def test_the_maxwell_deck_the_cpp_built_reads_with_its_names(tmp_path):
    written = Path(WRITTEN) / "deck_maxwell_built.txt"
    assert written.exists(), f"the C++ tests wrote no {written.name}"
    nan = np.nan
    deck = Deck([0.0, 0.5e6, 1.0e6, 1.0e6, 1.5e6, 2.0e6],
                {"rho": [1.1e4, 1.05e4, 1.0e4, 4.5e3, 4.0e3, 3.5e3],
                 "vpv": [9e3, 8.9e3, 8.8e3, 8e3, 7.5e3, 7e3],
                 "vsv": [0.0, 0.0, 0.0, 4.5e3, 4.2e3, 4e3],
                 "vph": [9e3, 8.9e3, 8.8e3, 8.1e3, 7.6e3, 7.1e3],
                 "vsh": [0.0, 0.0, 0.0, 4.6e3, 4.3e3, 4.1e3],
                 "eta": [1.0, 1.0, 1.0, 0.95, 0.97, 1.0],
                 "viscosity": [nan, nan, nan, 1e23, 1e22, 1e21]},
                header={"name": "a body built in C++", "layers": ("core", None),
                        "interfaces": ("cmb", "surface")})
    ours = write_deck(tmp_path / written.name, deck, MAXWELL)
    assert written.read_text() == ours.read_text()
    back = read_deck(written, MAXWELL)
    assert back.header["name"] == "a body built in C++"
    assert maxwell_names(back) == (["core", None], ["cmb", "surface"])
    for name in deck.names:
        assert np.allclose(back[name], deck[name], rtol=1e-12, equal_nan=True)


#: The elastic columns of the two-layer-and-more bodies the C++ builds.
_ELASTIC = {"rho": [1.1e4, 1.0e4, 4.5e3, 4.2e3, 4.2e3, 4.0e3, 3.8e3, 3.3e3, 3.2e3],
            "vpv": [9e3, 8.8e3, 8e3, 7.8e3, 7.8e3, 7.5e3, 7e3, 6.5e3, 6.4e3],
            "vsv": [0.0, 0.0, 4.5e3, 4.4e3, 4.4e3, 4.2e3, 4e3, 3.7e3, 3.6e3],
            "vph": [9e3, 8.8e3, 8.1e3, 7.9e3, 7.9e3, 7.6e3, 7.1e3, 6.6e3, 6.5e3],
            "vsh": [0.0, 0.0, 4.6e3, 4.5e3, 4.5e3, 4.3e3, 4.1e3, 3.8e3, 3.7e3],
            "eta": [1.0, 1.0, 0.95, 0.96, 0.96, 0.97, 1.0, 1.0, 1.0]}


def test_the_prony_deck_the_cpp_built_is_the_pythons_and_completes_alike(tmp_path):
    written = Path(WRITTEN) / "deck_prony_built.txt"
    assert written.exists(), f"the C++ tests wrote no {written.name}"
    nan = np.nan
    deck = Deck([0.0, 1.0e6, 1.0e6, 1.5e6, 1.5e6, 1.8e6, 2.0e6, 2.0e6, 2.1e6],
                {**_ELASTIC,
                 "relaxation_strength_1": [nan, nan, 1.0, 1.0, 0.4, 0.45, 0.5, nan,
                                           nan],
                 "relaxation_time_1": [nan, nan, 3e11, 1e11, 1e9, 2e9, 4e9, nan, nan],
                 "relaxation_strength_2": [nan, nan, nan, nan, 0.5, 0.45, 0.3, nan,
                                           nan],
                 "relaxation_time_2": [nan, nan, nan, nan, 1e12, 3e12, 1e13, nan, nan]},
                header={"name": "a body built in C++",
                        "layers": ("core", "lower_mantle", None, "lid"),
                        "interfaces": ("cmb", None, None, "surface")})
    ours = write_deck(tmp_path / written.name, deck, PRONY)
    assert written.read_text() == ours.read_text()
    back = read_deck(written, PRONY)
    assert prony_branches(back) == 2
    assert maxwell_names(back) == (["core", "lower_mantle", None, "lid"],
                                   ["cmb", None, None, "surface"])
    assert list(back["relaxation_strength_2"][2:4]) == [0.0, 0.0]
    assert list(back["relaxation_time_2"][2:4]) == [3e11, 1e11]
    for name in deck.names:
        held = ~np.isnan(deck[name])
        assert np.allclose(back[name][held], deck[name][held], rtol=1e-12), name


def test_the_burgers_deck_the_cpp_built_is_the_pythons(tmp_path):
    written = Path(WRITTEN) / "deck_burgers_built.txt"
    assert written.exists(), f"the C++ tests wrote no {written.name}"
    nan = np.nan
    elastic = {name: values[:5] + values[6:] for name, values in _ELASTIC.items()}
    deck = Deck([0.0, 1.0e6, 1.0e6, 1.5e6, 1.5e6, 2.0e6, 2.0e6, 2.1e6],
                {**elastic,
                 "viscosity": [nan, nan, 1e22, 3e21, 1e21, 1e20, nan, nan],
                 "kelvin_viscosity": [nan, nan, 1e20, 5e19, nan, nan, nan, nan],
                 "kelvin_modulus_fraction": [nan, nan, 0.5, 1.5, nan, nan, nan, nan]},
                header={"name": "a Burgers body built in C++",
                        "layers": ("core", None, "asthenosphere", "lid")})
    ours = write_deck(tmp_path / written.name, deck, BURGERS)
    assert written.read_text() == ours.read_text()
    back = read_deck(written, BURGERS)
    assert maxwell_names(back)[0] == ["core", None, "asthenosphere", "lid"]
    for name in deck.names:
        assert np.allclose(back[name], deck[name], rtol=1e-12, equal_nan=True), name
