"""Decks: structure from repeated radii, fields by interpolation, the mineos
format on PREM's own tabulation, custom formats, and the round trip."""
from pathlib import Path

import numpy as np
import pytest

from planetmodel import (DENSITY, SCALAR, Deck, DeckFormat, Elastic, FieldSpec,
                         Geometry, MINEOS, MineosModel, Model, PREM, SelfGravitating,
                         Tabulated, Viscoelastic, deck_layers, read_deck, testing,
                         write_deck)
from planetmodel.deck import KINDS, deck_knots, mineos_names, nd_names
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


def test_the_cubic_kinds_are_scipys_splines():
    """`cubic` is not-a-knot, the parabola through three knots; `natural`
    has no curvature at a layer's ends."""
    from scipy.interpolate import CubicSpline
    d = Deck([0.0, 0.3, 0.5, 1.0, 1.0, 1.4, 2.0],
             {"rho": [5.0, 4.0, 4.5, 3.0, 2.0, 2.5, 1.0]})
    r = np.linspace(0.0, 1.0, 11)
    for kind, bc in (("cubic", "not-a-knot"), ("natural", "natural")):
        _, layers = deck_layers(d, kind=kind)
        want = CubicSpline([0.0, 0.3, 0.5, 1.0], [5.0, 4.0, 4.5, 3.0], bc_type=bc)
        assert np.allclose(layers[0]["rho"](r), want(r), rtol=1e-14, atol=0)
    _, cubic = deck_layers(d, kind="cubic")
    parabola = np.polyfit([1.0, 1.4, 2.0], [2.0, 2.5, 1.0], 2)
    s = np.linspace(1.0, 2.0, 7)
    assert np.allclose(cubic[1]["rho"](s), np.polyval(parabola, s), rtol=1e-13)
    _, natural = deck_layers(d, kind="natural")
    assert np.isclose(natural[1]["rho"].derivative(nu=2)(1.0), 0.0, atol=1e-12)
    assert np.isclose(natural[1]["rho"].derivative(nu=2)(2.0), 0.0, atol=1e-12)


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
    assert back.names == deck.names                 # the columns it was read with
    full = mineos.to_deck(columns=[*deck.names, "A"])
    assert np.isnan(full["A"]).sum() == 0
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
    # a format naming a subset of the deck's columns writes that subset
    rho_only = DeckFormat(("rho",), name="rho")
    path = write_deck(tmp_path / "rho.deck", back, rho_only)
    assert read_deck(path, rho_only).names == ("rho",)


def test_a_custom_format_and_a_model_type_of_ones_own(tmp_path):
    """PREM's elastic part with a viscosity in the solid regions, written
    in a format of its own and read back as a model type of its own."""
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

    own = DeckFormat(("rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity"),
                     name="own", header_lines=2,
                     parse_header=lambda lines: {"title": lines[0].strip(),
                                                 "columns": tuple(lines[1].split())},
                     write_header=lambda d: [d.header["title"],
                                             "r rho vpv vsv vph vsh eta viscosity"])
    path = write_deck(tmp_path / "own.deck", deck, own)
    assert path.read_text().splitlines()[1].startswith("r rho")

    class OwnModel(Elastic, SelfGravitating, Viscoelastic, Tabulated, Model):
        def __init__(self, source, *, kind="cubic"):
            d = read_deck(source, own)
            sk, layers = deck_layers(d, kind=kind)
            self.knots, self.header = deck_knots(d), d.header
            super().__init__(Geometry(sk), layers)

    m = OwnModel(path)
    assert m.nlayers == prem.nlayers
    assert m.header["title"] == "PREM with a Maxwell mantle"
    assert "viscosity" not in m.layer(1) and m.layer(3)["viscosity"](5000e3) == 1e22
    assert m.is_viscoelastic(3) and not m.is_viscoelastic(1)
    assert np.isclose(m.mass(), prem.mass(), rtol=1e-4)
    omega = 2 * np.pi / (1000 * 3.15576e7)
    fz = m.frozen(omega)
    assert type(fz) is OwnModel and fz.layer(3)["L"].dtype == np.complex128
    L = fz.layer(3)["L"](5000e3)
    assert abs(L) < m.layer(3)["L"](5000e3)        # relaxed at a thousand years
    testing.check_model(m)
    back = m.to_deck(columns=["rho", "viscosity"])
    assert np.isnan(back["viscosity"]).sum() == 8
    assert np.allclose(back["rho"], deck["rho"])


def test_the_maxwell_format_round_trips(tmp_path):
    from planetmodel import MAXWELL, PREM
    from planetmodel.deck import MAXWELL_COLUMNS
    elastic = PREM().elastic().truncated(6368e3, name="surface")
    radii, columns = [], {name: [] for name in MAXWELL_COLUMNS}
    for layer in elastic.layers:
        lo, hi = layer.interval
        r = np.linspace(lo, hi, 5)
        radii.append(r)
        for name in MAXWELL_COLUMNS:
            if name == "viscosity":
                eta = np.nan if elastic.is_fluid(layer.index) else 1e21
                columns[name].append(np.full(r.size, eta))
            else:
                columns[name].append(layer[name](r))
    deck = Deck(np.concatenate(radii),
                {n: np.concatenate(v) for n, v in columns.items()},
                header={"name": "PREM with a viscosity"})
    path = write_deck(tmp_path / "maxwell.deck", deck, MAXWELL)
    lines = path.read_text().splitlines()
    assert lines[0] == "PREM with a viscosity"
    assert lines[1].split() == ["r", *MAXWELL_COLUMNS]
    back = read_deck(path, MAXWELL)
    assert back.header["name"] == "PREM with a viscosity"
    assert np.allclose(back.radius, deck.radius)
    for name in MAXWELL_COLUMNS:
        assert np.allclose(back.columns[name], deck.columns[name], equal_nan=True)
    _, layers = deck_layers(back)
    assert "viscosity" not in layers[1] and "viscosity" in layers[2]
    bad = tmp_path / "bad.deck"
    bad.write_text("title\nr rho vp vs\n" + "\n".join(lines[2:]) + "\n")
    with pytest.raises(ValueError, match="names its columns"):
        read_deck(bad, MAXWELL)


def test_a_converted_deck_model_writes_its_deck_in_si(mineos, deck):
    """The knots follow the model's length unit, and a deck is in SI."""
    from planetmodel.units import Scales

    nd = mineos.nondimensionalised()
    assert np.isclose(nd.knots[-1][-1], 1.0)
    for model in (nd, mineos.converted(Scales(mass=1000.0))):
        back = model.to_deck()
        for name in deck.names:
            assert np.allclose(back[name], deck[name], rtol=1e-9, atol=1e-6)
        assert np.allclose(back.radius, deck.radius, rtol=1e-12)


def test_to_deck_holds_the_layering_to_a_relative_tolerance(mineos):
    with pytest.raises(ValueError, match="no longer matches"):
        mineos.truncated(6371e3 - 50.0).to_deck()


def test_the_mineos_header_round_trips_its_trailing_tokens(deck, tmp_path):
    path = write_deck(tmp_path / "again.200", deck, MINEOS)
    lines = path.read_text().splitlines()
    original = PREM_DECK.read_text().splitlines()
    for ours, theirs in zip(lines[1:3], original[1:3]):
        assert [float(x) for x in ours.split()] == [float(x) for x in theirs.split()]


def test_a_header_without_an_inner_core_is_not_warned_about():
    import warnings

    radius = [3480e3, 4000e3, 4000e3, 6371e3]
    from planetmodel.deck import MINEOS_COLUMNS

    columns = {name: np.ones(4) for name in MINEOS_COLUMNS}
    d = Deck(radius, columns, header={"nic": 0, "noc": 0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layers, faces = mineos_names(d)
    assert layers == [None, None] and faces[-1] == "surface"


# ------------------------------------------------- TauP's named discontinuities

def nd_file(width=6):
    """A small two-layer-cored body in TauP's form: a label before the first
    row, labels between repeated depths (TauP's own and one of words), an
    unlabelled repeat, comments of the three kinds, `width` numbers a row."""
    rows = [("crust", None),
            (0.0, [5.8, 3.4, 2.7, 1400.0, 600.0]),
            (10.0, [5.8, 3.4, 2.7, 1400.0, 600.0]),
            ("mantle", None),
            (10.0, [7.8, 4.4, 3.3, 1000.0, 400.0]),
            (40.0, [7.9, 4.5, 3.4, 1000.0, 400.0]),
            (40.0, [8.0, 4.6, 3.45, 900.0, 350.0]),         # a repeat, no label
            (60.0, [8.1, 4.7, 3.5, 900.0, 350.0]),
            ("lower mantle", None),
            (60.0, [8.2, 4.8, 3.6, 800.0, 300.0]),
            (80.0, [8.3, 4.9, 3.7, 800.0, 300.0]),
            ("outer-core", None),
            (80.0, [6.0, 0.0, 9.0, 50000.0, 0.0]),
            (95.0, [6.5, 0.0, 9.5, 50000.0, 0.0]),
            ("inner-core", None),
            (95.0, [7.0, 3.0, 11.0, 500.0, 100.0]),
            (100.0, [7.1, 3.1, 11.1, 500.0, 100.0])]
    lines = ["# a test body, 100 km in radius", "/* a block", "   comment */"]
    for depth, values in rows:
        if values is None:
            lines.append(depth)
        else:
            row = [depth] + values[:width - 1]
            lines.append("  ".join(f"{v:g}" for v in row) + "   // a row")
    return lines


def test_an_nd_table_reads_to_an_si_deck_radius_first():
    from planetmodel import ND
    d = read_deck(nd_file(), ND)
    assert d.names == ("vp", "vs", "rho", "qp", "qmu")
    assert d.radius[0] == 0.0 and d.radius[-1] == 100e3
    assert list(d.boundaries) == [0.0, 5e3, 20e3, 40e3, 60e3, 90e3, 100e3]
    assert d["vp"][-1] == 5.8e3 and d["rho"][0] == 11.1e3 and d["qmu"][-1] == 600.0
    labels = dict((r, lab) for r, lab in d.header["labels"])
    assert labels == {100e3: "crust", 90e3: "mantle", 40e3: "lower-mantle",
                      20e3: "outer-core", 5e3: "inner-core"}
    layers, faces = nd_names(d)
    assert layers == ["inner_core", "outer_core", "lower_mantle", None, "mantle",
                      "crust"]
    assert faces == ["icb", "cmb", None, None, "moho", "surface"]


@pytest.mark.parametrize("width", [3, 4, 5, 6])
def test_every_width_of_row_names_its_columns(width):
    from planetmodel import ND
    d = read_deck(nd_file(width), ND)
    assert d.names == ("vp", "vs", "rho", "qp", "qmu")[:width - 1]


def test_an_nd_table_writes_back(tmp_path):
    from planetmodel import ND
    d = read_deck(nd_file(), ND)
    path = write_deck(tmp_path / "body.nd", d, ND)
    back = read_deck(path, ND)
    assert np.array_equal(back.radius, d.radius)
    for name in d.names:
        assert np.allclose(back[name], d[name], rtol=1e-12)
    assert back.header["labels"] == d.header["labels"]
    absent = Deck(d.radius, {**d.columns, "vs": np.where(d.radius < 20e3, np.nan,
                                                          d["vs"])})
    with pytest.raises(ValueError, match="every column on every row"):
        write_deck(tmp_path / "bad.nd", absent, ND)


def test_a_malformed_nd_table_is_refused_by_name():
    from planetmodel import ND
    lines = nd_file()
    with pytest.raises(ValueError, match="the same on every row"):
        read_deck(lines[:6] + ["10 7.8 4.4"] + lines[6:], ND)
    with pytest.raises(ValueError, match="three to six numbers"):
        read_deck(["0 1 2 3 4 5 6", "10 1 2 3 4 5 6"], ND)
    with pytest.raises(ValueError, match="sits at no boundary"):
        read_deck(["0 5.8 3.4", "label", "10 5.8 3.4"], ND)
    with pytest.raises(ValueError, match="must not decrease"):
        read_deck(["10 5.8 3.4", "0 5.8 3.4"], ND)
    with pytest.raises(ValueError, match="a row is numbers"):
        read_deck(["0 5.8 x", "10 5.8 3.4"], ND)


def test_a_tvel_table_keeps_its_two_header_lines(tmp_path):
    from planetmodel import TVEL
    lines = ["a P model", "an S model"] + [ln for ln in nd_file(4)
                                            if ln[0].isdigit()]
    d = read_deck(lines, TVEL)
    assert d.header["lines"] == ("a P model", "an S model")
    assert d.names == ("vp", "vs", "rho")
    back = read_deck(write_deck(tmp_path / "body.tvel", d, TVEL), TVEL)
    assert np.array_equal(back.radius, d.radius)
    assert back.header["lines"] == d.header["lines"]


# -- the Maxwell format: names in the header -------------------------------------

def _maxwell_lines(*names: str,
                   column_line: str = "r rho vpv vsv vph vsh eta viscosity"):
    """A two-layer Maxwell deck, a fluid core under a viscous mantle, with
    the given naming lines after the column line."""
    rows = ["0 10 9 0 9 0 1 nan", "1 10 9 0 9 0 1 nan",
            "1 4 8 4 8 4 1 1e21", "2 3 7 3 7 3 1 1e22"]
    return ["a Maxwell body", column_line, *names, *rows]


def test_the_naming_lines_name_the_layers_and_interfaces():
    from planetmodel import MAXWELL
    from planetmodel.deck import maxwell_names
    d = read_deck(_maxwell_lines("# layers core -", "# interfaces cmb top"), MAXWELL)
    assert d.header["name"] == "a Maxwell body"
    assert d.header["layers"] == ("core", None)
    assert maxwell_names(d) == (["core", None], ["cmb", "top"])
    plain = read_deck(_maxwell_lines(), MAXWELL)
    assert maxwell_names(plain) == ([None, None], [None, "surface"])
    only = read_deck(_maxwell_lines("# layers core mantle"), MAXWELL)
    assert maxwell_names(only) == (["core", "mantle"], [None, "surface"])
    # a comment after the header's lines is a comment, wherever it sits
    commented = read_deck(_maxwell_lines("# a note", "# layers core mantle"), MAXWELL)
    assert "layers" not in commented.header and commented.nknots == 4


def test_naming_lines_of_the_wrong_count_or_twice_are_refused():
    from planetmodel import MAXWELL
    from planetmodel.deck import maxwell_names
    three = read_deck(_maxwell_lines("# layers a b c"), MAXWELL)
    with pytest.raises(ValueError, match="names 3 layers; its table has 2"):
        maxwell_names(three)
    one = read_deck(_maxwell_lines("# interfaces top"), MAXWELL)
    with pytest.raises(ValueError, match="names 1 interfaces; its table has 2"):
        maxwell_names(one)
    with pytest.raises(ValueError, match="one `# layers` line"):
        read_deck(_maxwell_lines("# layers a b", "# layers c d"), MAXWELL)


def test_the_column_line_may_end_in_a_comment():
    from planetmodel import MAXWELL
    d = read_deck(_maxwell_lines(
        column_line="r rho vpv vsv vph vsh eta viscosity # (SI); an example"), MAXWELL)
    assert d.names == MAXWELL.columns


def test_names_are_written_where_the_header_has_any(tmp_path):
    from planetmodel import MAXWELL
    d = read_deck(_maxwell_lines("# layers core -", "# interfaces - surface"), MAXWELL)
    lines = write_deck(tmp_path / "named.deck", d, MAXWELL).read_text().splitlines()
    assert lines[2:4] == ["# layers core -", "# interfaces - surface"]
    unnamed = Deck(d.radius, d.columns, header={"name": "x", "layers": (None, None)})
    path = write_deck(tmp_path / "plain.deck", unnamed, MAXWELL)
    lines = path.read_text().splitlines()
    assert lines[1].startswith("r rho") and lines[2][0].isspace()
    for bad in ("two words", "-", "a#b", ""):
        named = Deck(d.radius, d.columns, header={"layers": ("core", bad)})
        with pytest.raises(ValueError, match="one word"):
            write_deck(tmp_path / "bad.deck", named, MAXWELL)
    wrong = Deck(d.radius, d.columns, header={"layers": ("core",)})
    with pytest.raises(ValueError, match="names 1 layers"):
        write_deck(tmp_path / "bad.deck", wrong, MAXWELL)


def test_a_header_writer_may_only_add_its_own_comment_lines(tmp_path):
    d = Deck([0.0, 1.0], {"a": [1.0, 2.0]}, header={"lines": ("t", "# extra")})
    with pytest.raises(ValueError, match="1 header lines; 2 were produced"):
        write_deck(tmp_path / "x", d, DeckFormat(("a",), header_lines=1))
    keyed = DeckFormat(("a",), header_lines=1, header_comments=("extra",))
    path = write_deck(tmp_path / "y", d, keyed)
    assert read_deck(path, keyed).header["lines"] == ("t", "# extra")
    with pytest.raises(ValueError, match="up to 1 `#` lines"):
        write_deck(tmp_path / "z", Deck([0.0, 1.0], {"a": [1.0, 2.0]},
                                        header={"lines": ("t", "# other")}), keyed)


# -- writing: the widest column set, and fill values -----------------------------

def test_a_mineos_deck_is_written_with_its_widest_columns_and_no_loss_as_zero(tmp_path):
    from planetmodel import MAXWELL
    from planetmodel.deck import MINEOS_COLUMNS, MINEOS_ISOTROPIC_COLUMNS
    names = set(MINEOS_COLUMNS) | set(MINEOS_ISOTROPIC_COLUMNS)
    columns = {n: np.full(4, 2.0) for n in names}
    columns["qmu"] = np.array([np.nan, np.nan, 100.0, 100.0])
    d = Deck([0.0, 1.0, 1.0, 2.0], columns, header={"name": "both"})
    back = read_deck(write_deck(tmp_path / "both.deck", d, MINEOS), MINEOS)
    assert back.names == MINEOS_COLUMNS
    assert list(back["qmu"]) == [0.0, 0.0, 100.0, 100.0]
    viscous = Deck(d.radius, {n: d[n] for n in MAXWELL.columns if n in d}
                   | {"viscosity": columns["qmu"]})
    text = write_deck(tmp_path / "v.deck", viscous, MAXWELL).read_text()
    assert "nan" in text                     # a Maxwell deck fills nothing


# -- per-column kinds ---------------------------------------------------------------

def test_a_column_may_be_joined_by_a_kind_of_its_own():
    d = Deck([0.0, 0.3, 0.5, 1.0], {"rho": [5.0, 4.0, 4.5, 3.0],
                                    "viscosity": [1e20, 1e23, 1e21, 1e22]})
    _, cubic = deck_layers(d)
    _, mixed = deck_layers(d, kinds={"viscosity": "linear"})
    r = np.linspace(0.0, 1.0, 41)
    assert np.array_equal(mixed[0]["rho"](r), cubic[0]["rho"](r))
    assert np.min(cubic[0]["viscosity"](r)) < 0.0          # a cubic overshoots
    assert np.min(mixed[0]["viscosity"](r)) >= 1e20
    assert mixed[0]["viscosity"].function.degree == 1
    with pytest.raises(ValueError, match="lacks"):
        deck_layers(d, kinds={"qmu": "linear"})
    with pytest.raises(ValueError, match="kind must be one of"):
        deck_layers(d, kinds={"viscosity": "quintic"})


# -- any model as a deck ------------------------------------------------------------

def _maxwell_prem():
    """PREM without its ocean, a viscosity on every mantle layer below the
    low-velocity zone: 1e22 Pa s below 670 km, 1e21 above."""
    from planetmodel.fields import constant_field
    model = PREM(ocean=False)
    for i, layer in enumerate(model.layers):
        if 2 <= i <= 7:
            eta = 1e22 if i <= 4 else 1e21
            field = constant_field(layer.interval, eta, name="viscosity")
            model = model.with_field(i, "viscosity", field)
    return model


def test_a_sampled_deck_holds_the_model_exactly_on_cubic_layers():
    """PREM's layers are each one polynomial of degree three at most, so a
    not-a-knot cubic through four or more knots a piece gives them back."""
    from planetmodel import sample_deck
    from planetmodel.deck import MAXWELL_COLUMNS, SAMPLE_SPACING
    model = _maxwell_prem()
    d = sample_deck(model, MAXWELL_COLUMNS)
    assert d.nlayers == model.nlayers
    assert np.allclose(d.boundaries, model.skeleton.boundaries, rtol=0, atol=0)
    gaps = np.diff(d.radius)
    assert np.max(gaps) <= SAMPLE_SPACING * 6368e3 * (1 + 1e-12)
    assert min(s.stop - s.start for s in d.layers()) >= 4
    assert d.header["layers"][0] == "inner_core"
    assert d.header["interfaces"][-1] == "surface"
    assert d.header["name"] == "PREM (no ocean)"
    _, layers = deck_layers(d, kinds={"viscosity": "linear"})
    for i, layer in enumerate(model.layers):
        lo, hi = layer.interval
        r = np.linspace(lo, hi, 29)
        for name in ("rho", "vpv", "vsv", "vph", "vsh", "eta"):
            assert np.allclose(layers[i][name](r), layer[name](r), rtol=1e-12,
                               atol=1e-9), (i, name)
        assert ("viscosity" in layers[i]) == ("viscosity" in layer)


def test_a_velocity_a_layer_lacks_comes_from_its_moduli():
    """pyslfp's Maxwell PREM is made isotropic first, so it holds vp and vs;
    a Maxwell deck asks for the transversely isotropic five."""
    from planetmodel import sample_deck
    from planetmodel.deck import MAXWELL_COLUMNS
    iso = _maxwell_prem().isotropic().without_field("qmu").without_field("qkappa")
    assert "vpv" not in iso.layer("lid")
    d = sample_deck(iso, MAXWELL_COLUMNS)
    s = d.layers()[iso.layer("lid").index]
    r = d.radius[s]
    assert np.allclose(d["vpv"][s], iso.layer("lid")["vp"](r), rtol=1e-14)
    assert np.allclose(d["vsh"][s], iso.layer("lid")["vs"](r), rtol=1e-14)
    assert np.allclose(d["eta"][s], 1.0, rtol=1e-14)
    with pytest.raises(ValueError, match="transversely isotropic"):
        sample_deck(PREM(), ["rho", "vp"])          # PREM's lid is not isotropic
    blank = sample_deck(iso, ["rho", "qmu"])
    assert np.isnan(blank["qmu"]).all()


def test_sampling_refuses_what_is_not_a_column_or_a_layering():
    from planetmodel import sample_deck
    prem = PREM()
    with pytest.raises(ValueError, match="once"):
        sample_deck(prem, ["rho", "rho"])
    with pytest.raises(ValueError, match="positive"):
        sample_deck(prem, ["rho"], spacing=0.0)
    with pytest.raises(ValueError, match="no longer matches"):
        sample_deck(prem, ["rho"], knots=[np.array([0.0, 1.0])])
    frozen = prem.frozen(2 * np.pi / 100.0)
    with pytest.raises(ValueError, match="not a real radial field"):
        sample_deck(frozen, ["L"])
    frozen_iso = prem.isotropic().frozen(2 * np.pi / 100.0)
    with pytest.raises(ValueError, match="moduli of layer 0 are not real"):
        sample_deck(frozen_iso, ["vpv"])


def test_a_model_in_other_scales_is_sampled_in_si():
    from planetmodel import sample_deck
    model = _maxwell_prem()
    nd = model.nondimensionalised()
    d = sample_deck(nd, ["rho", "viscosity"])
    si = sample_deck(model, ["rho", "viscosity"])
    assert np.allclose(d.radius, si.radius, rtol=1e-13)
    assert np.allclose(d["rho"], si["rho"], rtol=1e-12)
    assert np.allclose(d["viscosity"], si["viscosity"], rtol=1e-12, equal_nan=True)
    k = [np.linspace(*L.interval, 5) for L in nd.layers]
    given = sample_deck(nd, ["rho"], knots=k)
    assert np.isclose(given.radius[-1], 6368e3, rtol=1e-13)


def test_prem_writes_as_a_mineos_deck(tmp_path):
    from planetmodel import sample_deck
    prem = PREM()
    path = write_deck(tmp_path / "prem.deck", sample_deck(prem, MINEOS.columns[8]),
                      MINEOS)
    m = MineosModel(path)
    assert m.nlayers == prem.nlayers and m.name == "PREM"
    assert m.layer(0).name == "inner_core" and m.layer(1).name == "outer_core"
    assert m.layer("outer_core")["qmu"](2000e3) == 0.0          # no loss, as zero
    assert np.isclose(m.reference_omega(), 2 * np.pi)
    assert np.isclose(m.mass(), prem.mass(), rtol=1e-7)
    # a model's declared reference period is written as the mineos tref
    assert np.isclose(sample_deck(m, ["rho"]).header["tref"], 1.0)


# -- the Maxwell deck model ---------------------------------------------------------

def test_a_maxwell_deck_model_reads_back_what_was_written(tmp_path):
    from planetmodel import MAXWELL, MaxwellDeckModel, sample_deck
    from planetmodel.deck import MAXWELL_COLUMNS
    model = _maxwell_prem()
    path = write_deck(tmp_path / "maxwell.deck", sample_deck(model, MAXWELL_COLUMNS),
                      MAXWELL)
    m = MaxwellDeckModel(path)
    assert m.name == "PREM (no ocean)" and m.nlayers == model.nlayers
    assert [L.name for L in m.layers] == [L.name for L in model.layers]
    assert ([f.name for f in m.geometry.interfaces]
            == [f.name for f in model.geometry.interfaces])
    for i, layer in enumerate(model.layers):
        assert ("viscosity" in m.layer(i)) == ("viscosity" in layer)
        lo, hi = layer.interval
        r = np.linspace(lo, hi, 13)
        assert np.allclose(m.layer(i)["rho"](r), layer["rho"](r), rtol=1e-7)
    assert m.layer("lower_mantle")["viscosity"].function.degree == 1
    assert m.columns == MAXWELL_COLUMNS
    testing.check_model(m)
    again = MaxwellDeckModel(write_deck(tmp_path / "again.deck", m.to_deck(), MAXWELL))
    assert [L.name for L in again.layers] == [L.name for L in m.layers]


def test_a_maxwell_deck_model_relaxes_its_mantle_at_glacial_periods(tmp_path):
    from planetmodel import MAXWELL, MaxwellDeckModel, sample_deck
    from planetmodel.deck import MAXWELL_COLUMNS
    path = write_deck(tmp_path / "maxwell.deck",
                      sample_deck(_maxwell_prem(), MAXWELL_COLUMNS), MAXWELL)
    m = MaxwellDeckModel(path)
    assert m.is_viscoelastic("lower_mantle") and not m.is_viscoelastic("lid")
    omega = 2 * np.pi / (1000 * 3.15576e7)
    fz = m.frozen(omega)
    assert type(fz) is MaxwellDeckModel
    assert fz.layer("lower_mantle")["L"].dtype == np.complex128
    r = 5000e3
    assert abs(fz.layer("lower_mantle")["L"](r)) < m.layer("lower_mantle")["L"](r)
    assert np.isclose(fz.layer("lid")["L"](6300e3), m.layer("lid")["L"](6300e3))
    nd = m.nondimensionalised()
    assert type(nd) is MaxwellDeckModel and np.isclose(nd.knots[-1][-1], 1.0)
