"""Decks: structure from repeated radii, fields by interpolation, the mineos
format on PREM's own tabulation, custom formats, and the round trip."""
from pathlib import Path

import numpy as np
import pytest

from planetmodel import (DENSITY, SCALAR, Deck, DeckFormat, Elastic, FieldSpec,
                         Geometry, MINEOS, MineosModel, Model, PREM, SelfGravitating,
                         Tabulated, Viscoelastic, deck_layers, read_deck, testing,
                         write_deck)
from planetmodel.deck import KINDS, deck_knots, mineos_names
from planetmodel.layerfunction import PolynomialLayer
from planetmodel.units import VISCOSITY

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
PREM_DECK = DATA / "prem.200"
ISO_DECK = DATA / "prem.nocrust"


@pytest.fixture(scope="module")
def deck():
    return read_deck(PREM_DECK, MINEOS)


@pytest.fixture(scope="module")
def mineos(deck):
    return MineosModel(deck)


# -- structure -----------------------------------------------------------------

def test_a_repeated_radius_is_a_boundary(deck):
    assert deck.nknots == 220 and deck.nlayers == 13
    assert deck.names == ("rho", "vpv", "vsv", "qkappa", "qmu", "vph", "vsh", "eta")
    b = deck.boundaries
    assert b[0] == 0.0 and b[-1] == 6371e3 and 1221.5e3 in b and 3480e3 in b
    assert np.allclose(b, PREM().skeleton.boundaries)
    assert deck.header["name"] == "prem.200" and deck.header["tref"] == 1.0
    assert deck.header["nic"] == 40 and deck.header["noc"] == 112
    assert not deck.radius.flags.writeable and not deck["rho"].flags.writeable


def test_structure_refusals():
    with pytest.raises(ValueError, match="non-decreasing"):
        Deck([0.0, 2.0, 1.0], {"rho": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="more than twice"):
        Deck([0.0, 1.0, 1.0, 1.0, 2.0], {"rho": np.ones(5)})
    with pytest.raises(ValueError, match="one knot"):
        Deck([0.0, 1.0, 1.0], {"rho": np.ones(3)})
    with pytest.raises(ValueError, match="shape"):
        Deck([0.0, 1.0], {"rho": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError, match="no column"):
        Deck([0.0, 1.0], {"rho": [1.0, 2.0]})["vp"]


def test_the_format_names_the_columns():
    lines = ["title", "0 1.0 1", "3 0 0", "0 1 2", "1 2 3", "2 3 4"]
    with pytest.raises(ValueError, match="columns after"):
        read_deck(lines, MINEOS)
    with pytest.raises(ValueError, match="columns after"):
        read_deck(lines, ("rho",), header_lines=3)
    d = read_deck(lines, ("rho", "vp"), header_lines=3)
    assert d.header == {"lines": ("title", "0 1.0 1", "3 0 0")} and d.nlayers == 1
    with pytest.raises(ValueError, match="set by the DeckFormat"):
        read_deck(lines, MINEOS, header_lines=3)
    with pytest.raises(ValueError, match="fewer than"):
        read_deck(["title"], MINEOS)
    with pytest.raises(ValueError, match="no knots"):
        read_deck(lines[:3], MINEOS)


# -- fields by interpolation ---------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_interpolants_pass_through_the_knots_and_are_polynomial(deck, kind):
    sk, layers = deck_layers(deck, kind=kind)
    assert sk.nlayers == 13 and len(layers) == 13
    for s, fields in zip(deck.layers(), layers):
        r = deck.radius[s]
        for name, f in fields.items():
            assert isinstance(f.function, PolynomialLayer) and f.name == name
            assert np.allclose(f(r), deck[name][s], rtol=1e-12, atol=1e-9)
        assert fields["rho"].character == DENSITY and fields["vpv"].character == SCALAR


def test_two_knots_give_a_line_and_unknown_kinds_are_refused():
    d = Deck([0.0, 1.0, 1.0, 3.0], {"rho": [1.0, 3.0, 5.0, 1.0]})
    for kind in KINDS:
        _, layers = deck_layers(d, kind=kind)
        assert layers[0]["rho"].function.degree == 1
        assert np.isclose(layers[0]["rho"](0.5), 2.0)
        assert np.isclose(layers[1]["rho"](2.0), 3.0)
    with pytest.raises(ValueError, match="kind"):
        deck_layers(d, kind="quintic")


def test_nan_throughout_a_layer_means_absent_and_partial_nan_is_refused():
    d = Deck([0.0, 1.0, 1.0, 2.0], {"rho": [1.0, 1.0, 2.0, 2.0],
                                    "viscosity": [np.nan, np.nan, 1e21, 1e21]})
    _, layers = deck_layers(d)
    assert "viscosity" not in layers[0] and "viscosity" in layers[1]
    bad = Deck([0.0, 1.0, 2.0], {"rho": [1.0, np.nan, 2.0]})
    with pytest.raises(ValueError, match="NaN on part"):
        deck_layers(bad)


def test_specs_give_a_custom_column_its_character():
    d = Deck([0.0, 1.0], {"rho": [1.0, 1.0], "damping": [2.0, 3.0]})
    _, layers = deck_layers(d, specs={"damping": FieldSpec(DENSITY, VISCOSITY)})
    assert layers[0]["damping"].character == DENSITY
    _, plain = deck_layers(d)
    assert plain[0]["damping"].character == SCALAR


# -- the mineos model ----------------------------------------------------------

def test_mineos_model_is_prem_to_the_tabulation(mineos):
    prem = PREM()
    assert mineos.nlayers == 13 and mineos.name == "prem.200"
    assert mineos.layer(0).name == "inner_core" and mineos.layer(1).name == "outer_core"
    faces = [f.name for f in mineos.geometry.interfaces]
    assert faces[0] == "icb" and faces[1] == "cmb" and faces[-1] == "surface"
    rng = np.random.default_rng(0)
    for i in range(prem.nlayers):
        lo, hi = prem.skeleton.interval(i)
        r = lo + (hi - lo) * rng.uniform(0.02, 0.98, 6)
        for name in ("rho", "vpv", "vsh", "A", "L"):
            got, want = mineos.layer(i)[name](r), prem.layer(i)[name](r)
            assert np.allclose(got, want, rtol=3e-5), (i, name)
    assert np.isclose(mineos.mass(), prem.mass(), rtol=1e-6)
    assert mineos.is_fluid("outer_core") and not mineos.is_fluid(2)
    assert mineos.elastic_moduli("outer_core").symmetry.name == "ISOTROPIC"
    assert "qmu" in mineos.layer("outer_core")          # kept as read, zero
    assert mineos.layer("outer_core")["qmu"](2000e3) == 0.0
    assert np.isclose(mineos.reference_omega(), 2 * np.pi)
    assert "omega_ref" in mineos.constants
    testing.check_model(mineos)


def test_moduli_of_a_deck_are_exact_products_of_the_interpolants(mineos):
    rng = np.random.default_rng(1)
    for layer in mineos.layers:
        lo, hi = layer.interval
        r = lo + (hi - lo) * rng.uniform(0.0, 1.0, 20)
        assert np.allclose(layer["A"](r), layer["rho"](r) * layer["vph"](r) ** 2,
                           rtol=1e-14)
        assert np.allclose(layer["F"](r),
                           layer["eta"](r) * (layer["A"](r) - 2 * layer["L"](r)),
                           rtol=1e-14)
        assert isinstance(layer["A"].function, PolynomialLayer)


def test_the_isotropic_deck_warns_about_its_header_and_reads_isotropic():
    with pytest.warns(UserWarning, match="220 knots"):
        m = MineosModel(ISO_DECK)
    assert m.nlayers == 10 and m.layer(0).name == "inner_core"
    assert m.layer(0).names == ("rho", "vp", "vs", "qkappa", "qmu", "A", "C", "F", "L",
                                "N")
    assert m.is_fluid("outer_core") and m.elastic_moduli(3).symmetry.name == "ISOTROPIC"
    assert m.layer(0)["A"](500e3) == m.layer(0)["C"](500e3)
    testing.check_model(m)


def test_mineos_names_from_an_inconsistent_header():
    d = Deck([0.0, 1.0, 1.0, 2.0], {"rho": [1.0, 1.0, 2.0, 2.0]},
             header={"nic": 3, "noc": 3})
    with pytest.warns(UserWarning, match="does not sit"):
        layers, faces = mineos_names(d)
    assert layers == [None, None] and faces == [None, "surface"]


# -- the reference frequency ----------------------------------------------------

def test_the_header_period_is_the_reference_frequency(deck):
    lines = Path(PREM_DECK).read_text().splitlines()
    lines[1] = "1 100.0 1"
    slow = MineosModel(read_deck(lines, MINEOS))
    assert np.isclose(slow.reference_omega(), 2 * np.pi / 100.0)
    r = 5000e3
    static = slow.layer(3)["L"](r)
    at_ref = slow.moduli_at(3, slow.reference_omega())["L"](r)
    assert np.isclose(at_ref.real, static) and at_ref.imag > 0.0
    frozen = slow.frozen(slow.reference_omega())
    assert np.isclose(frozen.layer(3)["L"](r).real, static)
    nd = slow.nondimensionalised()
    assert np.isclose(nd.reference_omega() / nd.scales.time, 2 * np.pi / 100.0)


# -- round trips ----------------------------------------------------------------

def test_to_deck_and_write_deck_round_trip(mineos, deck, tmp_path):
    back = mineos.to_deck()
    for name in deck.names:
        assert np.allclose(back[name], deck[name], rtol=1e-12, atol=1e-9)
    assert "A" in back and np.isnan(back["A"]).sum() == 0
    assert back.header["name"] == "prem.200"
    path = write_deck(tmp_path / "prem.out", back, MINEOS)
    text = path.read_text().splitlines()
    assert text[0] == "prem.200" and text[1].split()[:2] == ["1", "1"]
    assert text[2].split() == ["220", "40", "112"]
    again = read_deck(path, MINEOS)
    assert again.names == deck.names
    for name in deck.names:
        assert np.allclose(again[name], deck[name], rtol=1e-7)
    assert np.allclose(again.radius, deck.radius)
    with pytest.raises(ValueError, match="no longer matches"):
        mineos.refined([2000e3]).to_deck()
    with pytest.raises(ValueError, match="needs columns"):
        write_deck(tmp_path / "bad", Deck([0.0, 1.0], {"rho": [1.0, 1.0]}), MINEOS)


def test_a_custom_format_and_a_model_type_of_ones_own(tmp_path):
    """PREM's elastic part with a viscosity in the solid regions, written
    in a format of its own and read back as a GIA-style model type."""
    prem = PREM(ocean=False)
    radii, cols = [], {n: [] for n in ("rho", "vpv", "vsv", "vph", "vsh", "eta",
                                       "viscosity")}
    eta = {"lower_mantle": 1e22, "upper_lower_mantle": 1e22}
    for layer in prem.layers:
        lo, hi = layer.interval
        r = np.linspace(lo, hi, 8)
        radii.append(r)
        for n in cols:
            if n == "viscosity":
                v = (np.full(8, np.nan) if prem.is_fluid(layer.index)
                     else np.full(8, eta.get(layer.name, 1e21)))
            else:
                v = layer[n](r)
            cols[n].append(v)
    deck = Deck(np.concatenate(radii), {n: np.concatenate(v) for n, v in cols.items()},
                header={"title": "PREM with a Maxwell mantle"})
    assert np.isnan(deck["viscosity"]).sum() == 8

    GIA = DeckFormat(("rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity"),
                     name="gia", header_lines=2,
                     parse_header=lambda lines: {"title": lines[0].strip(),
                                                 "columns": tuple(lines[1].split())},
                     write_header=lambda d: [d.header["title"],
                                             "r rho vpv vsv vph vsh eta viscosity"])
    path = write_deck(tmp_path / "gia.deck", deck, GIA)
    assert path.read_text().splitlines()[1].startswith("r rho")

    class GIAModel(Elastic, SelfGravitating, Viscoelastic, Tabulated, Model):
        def __init__(self, source, *, kind="cubic"):
            d = read_deck(source, GIA)
            sk, layers = deck_layers(d, kind=kind)
            self.knots, self.header = deck_knots(d), d.header
            super().__init__(Geometry(sk), layers)

    m = GIAModel(path)
    assert m.nlayers == prem.nlayers
    assert m.header["title"] == "PREM with a Maxwell mantle"
    assert "viscosity" not in m.layer(1) and m.layer(3)["viscosity"](5000e3) == 1e22
    assert m.is_viscoelastic(3) and not m.is_viscoelastic(1)
    assert np.isclose(m.mass(), prem.mass(), rtol=1e-4)
    omega = 2 * np.pi / (1000 * 3.15576e7)
    fz = m.frozen(omega)
    assert type(fz) is GIAModel and fz.layer(3)["L"].dtype == np.complex128
    L = fz.layer(3)["L"](5000e3)
    assert abs(L) < m.layer(3)["L"](5000e3)        # relaxed at a thousand years
    testing.check_model(m)
    back = m.to_deck(columns=["rho", "viscosity"])
    assert np.isnan(back["viscosity"]).sum() == 8
    assert np.allclose(back["rho"], deck["rho"])
