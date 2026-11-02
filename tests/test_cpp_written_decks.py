"""The decks the C++ writer wrote, read by the Python: each fixture the C++
read and wrote back is, character for character, what the Python's own
`write_deck` makes of it, and the Maxwell deck the C++ built reads with its
names and values.

The C++ tests leave the decks in their build tree; PLANETMODEL_CPP_WRITTEN
names that directory (`build/cpp/tests/written` for the build of
CLAUDE.md), and without it these tests are skipped.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from planetmodel import (MAXWELL, MINEOS, ND, TVEL, Deck, DeckFormat, read_deck,
                         write_deck)
from planetmodel.deck import maxwell_names

WRITTEN = os.environ.get("PLANETMODEL_CPP_WRITTEN")
FIXTURES = Path(__file__).resolve().parents[1] / "cpp" / "tests" / "fixtures"
pytestmark = pytest.mark.skipif(
    WRITTEN is None,
    reason="PLANETMODEL_CPP_WRITTEN names no directory of C++-written files")

#: Each fixture the C++ writes back, and its format.
CASES = {"prem200": MINEOS, "nocrust": MINEOS, "maxwell": MAXWELL, "nd": ND,
         "tvel": TVEL, "knots": DeckFormat(("rho", "vp"), header_lines=1)}


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
