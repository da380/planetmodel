"""Decks: structure from repeated radii, fields by interpolation, a table
brought in as a model (PREM's own tabulation, a table of named columns,
the names and the constants the caller's to give), and a model written as
a table under its format's own header, lines given, or none."""
import warnings
from pathlib import Path

import numpy as np
import pytest

from planetmodel import (DENSITY, SCALAR, Deck, DeckFormat, Elastic, FieldSpec,
                         LayeredIsotropicElastic, MINEOS, ND, PREM, TVEL, Model,
                         Scales, Viscoelastic, deck_layers, model_from_deck, read_deck,
                         sample_deck, testing, write_deck)
from planetmodel.behaviours import BEHAVIOURS
from planetmodel.deck import (KINDS, LOG_KINDS, MINEOS_COLUMNS,
                              MINEOS_ISOTROPIC_COLUMNS, SAMPLE_SPACING, deck_knots,
                              mineos_constants, mineos_header, mineos_names, nd_names)
from planetmodel.fields import constant_field
from planetmodel.layerfunction import ExponentialLayer, PolynomialLayer
from planetmodel.units import VISCOSITY

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
PREM_DECK = DATA / "prem.200"
ISO_DECK = DATA / "prem.nocrust"

#: The elastic columns of a transversely isotropic body beside a viscosity.
VISCOUS = ("rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity")


def imported(deck, **kw):
    """The model of a mineos deck as its header describes it: every
    behaviour, the core named by the header's counts, the reference
    frequency its period, the name its title."""
    layers, faces = mineos_names(deck)
    return model_from_deck(deck, behaviours=BEHAVIOURS, layer_names=layers,
                           interface_names=faces, constants=mineos_constants(deck),
                           name=mineos_header(deck)["name"], **kw)


@pytest.fixture(scope="module")
def deck():
    return read_deck(PREM_DECK, MINEOS)


@pytest.fixture(scope="module")
def mineos(deck):
    return imported(deck)


# -- the surface -----------------------------------------------------------------

def test_the_surface_is_a_table_its_import_and_its_export():
    """The module's names, each needed to read a table, make a model of it
    or write a model as one; the top level has the ten a caller starts
    from, and the model types of the catalogue are none of a deck's."""
    import planetmodel
    from planetmodel import catalogue, deck as module
    assert set(module.__all__) == {
        "Deck", "DeckFormat", "DeckCoordinate", "RADIUS", "HeaderNames", "Table",
        "MINEOS", "MINEOS_COLUMNS", "MINEOS_ISOTROPIC_COLUMNS", "ND", "TVEL",
        "ND_COLUMNS", "ND_LABELS", "KINDS", "LOG_KINDS", "read_deck", "deck_layers",
        "model_from_deck", "mineos_header", "mineos_names", "mineos_constants",
        "nd_names", "sample_deck", "SAMPLE_SPACING", "deck_knots", "write_deck"}
    assert set(module.__all__) & set(planetmodel.__all__) == {
        "Deck", "DeckFormat", "MINEOS", "ND", "TVEL", "read_deck", "deck_layers",
        "model_from_deck", "sample_deck", "write_deck"}
    assert set(catalogue.__all__) == {"PREM", "LayeredIsotropicElastic", "PREM_RADIUS",
                                      "isotropic_layers", "LayerValue"}


# -- structure -----------------------------------------------------------------

def test_a_repeated_radius_is_a_boundary(deck):
    assert deck.nknots == 220 and deck.nlayers == 13
    assert deck.names == ("rho", "vpv", "vsv", "qkappa", "qmu", "vph", "vsh", "eta")
    b = deck.boundaries
    assert b[0] == 0.0 and b[-1] == 6371e3 and 1221.5e3 in b and 3480e3 in b
    assert np.allclose(b, PREM().skeleton.boundaries)
    assert not deck.radius.flags.writeable and not deck["rho"].flags.writeable


def test_a_deck_read_keeps_its_header_lines_and_reads_no_meaning_from_them(deck):
    """The lines as they stand; what they say is read only by asking."""
    assert deck.header == {"lines": tuple(PREM_DECK.read_text().splitlines()[:3])}
    h = mineos_header(deck)
    assert h == {"name": "prem.200", "ifanis": 1, "tref": 1.0, "ifdeck": 1,
                 "nknot": 220, "nic": 40, "noc": 112}
    assert np.isclose(mineos_constants(deck)["omega_ref"].value_si, 2.0 * np.pi)
    rule = "a mineos deck's header is a title line"
    with pytest.raises(ValueError, match=rule + ".*the deck holds 0 header lines"):
        mineos_header(Deck([0.0, 1.0], {"rho": [1.0, 1.0]}))
    junk = Deck([0.0, 1.0], {"rho": [1.0, 1.0]},
                header={"lines": ("title", "not numbers", "1 2")})
    with pytest.raises(ValueError, match=rule + ".*got 'not numbers' and '1 2'"):
        mineos_header(junk)


def test_a_deck_is_itself_and_no_other(deck):
    """A deck holds arrays: it is compared and hashed by identity."""
    other = read_deck(PREM_DECK, MINEOS)
    assert deck == deck and deck != other
    assert hash(deck) != hash(other) and len({deck, deck, other}) == 2


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
    with pytest.raises(ValueError, match=r"a mineos deck has \[5, 8\] columns after "
                                         "the knot, not 2"):
        read_deck(lines, MINEOS)
    with pytest.raises(ValueError, match="1 columns after the knot.*the table has 2"):
        read_deck(lines, ("rho",), header_lines=3)
    with pytest.raises(TypeError, match="got the string 'rho'"):
        read_deck(lines, "rho", header_lines=3)
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


# -- a table brought in as a model ----------------------------------------------

def test_the_model_of_prems_deck_is_prem_to_the_tabulation(mineos):
    prem = PREM()
    assert mineos.nlayers == 13 and mineos.name == "prem.200"
    assert mineos.behaviours == ("Elastic", "ConstantQ", "SelfGravitating",
                                 "Viscoelastic") == prem.behaviours
    assert isinstance(mineos, Model)
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


def test_the_isotropic_deck_s_header_is_warned_about_where_it_is_asked():
    """`prem.nocrust` declares 220 knots over a table of 214: reading it
    says nothing, and the names its header implies come with a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = read_deck(ISO_DECK, MINEOS)
        plain = model_from_deck(d, behaviours=[Elastic])
    assert plain.nlayers == 10 and plain.layer(0).name is None and plain.name is None
    with pytest.warns(UserWarning, match="220 knots"):
        m = imported(d)
    assert m.nlayers == 10 and m.layer(0).name == "inner_core"
    assert m.layer(0).names == ("rho", "vp", "vs", "qkappa", "qmu", "A", "C", "F", "L",
                                "N")
    assert m.is_fluid("outer_core") and m.elastic_moduli(3).symmetry.name == "ISOTROPIC"
    assert m.layer(0)["A"](500e3) == m.layer(0)["C"](500e3)
    testing.check_model(m)


def test_mineos_names_from_an_inconsistent_header():
    d = Deck([0.0, 1.0, 1.0, 2.0], {"rho": [1.0, 1.0, 2.0, 2.0]},
             header={"lines": ("t", "0 1 1", "4 3 3")})
    with pytest.warns(UserWarning, match="does not sit"):
        layers, faces = mineos_names(d)
    assert layers == [None, None] and faces == [None, "surface"]


def test_a_header_without_an_inner_core_is_not_warned_about():
    radius = [3480e3, 4000e3, 4000e3, 6371e3]
    columns = {name: np.ones(4) for name in MINEOS_COLUMNS}
    d = Deck(radius, columns, header={"lines": ("t", "1 0 1", "4 0 0")})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layers, faces = mineos_names(d)
        assert mineos_constants(d) == {}                 # a tref of 0: none
    assert layers == [None, None] and faces == [None, None, "surface"]


def test_the_header_period_is_the_reference_frequency_where_it_is_passed_on():
    lines = Path(PREM_DECK).read_text().splitlines()
    lines[1] = "1 100.0 1"
    hundred = read_deck(lines, MINEOS)
    slow = imported(hundred)
    assert np.isclose(slow.reference_omega(), 2 * np.pi / 100.0)
    r = 5000e3
    static = slow.layer(3)["L"](r)
    at_ref = slow.moduli_at(3, slow.reference_omega())["L"](r)
    assert np.isclose(at_ref.real, static) and at_ref.imag > 0.0
    frozen = slow.frozen(slow.reference_omega())
    assert np.isclose(frozen.layer(3)["L"](r).real, static)
    nd = slow.nondimensionalised()
    assert np.isclose(nd.reference_omega() / nd.scales.time, 2 * np.pi / 100.0)
    # not passed on, the model declares none: a period of one second
    plain = model_from_deck(hundred, behaviours=BEHAVIOURS)
    assert "omega_ref" not in plain.constants
    assert np.isclose(plain.reference_omega(), 2 * np.pi)


def test_a_table_under_a_header_of_nonsense_and_under_none_are_the_same_model():
    """Nothing is read from a header: the rows of PREM's deck under three
    lines that mean nothing, and under no line at all, as a format of the
    column names, give the same fields, no names and no constants."""
    rows = PREM_DECK.read_text().splitlines()[3:]
    bare = read_deck(rows, MINEOS_COLUMNS)
    nonsense = ["%% not a header", "the quick brown fox", "- - -"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        under = read_deck(nonsense + rows, MINEOS)
        a = model_from_deck(bare, behaviours=BEHAVIOURS)
        b = model_from_deck(under, behaviours=BEHAVIOURS)
    assert under.header["lines"] == tuple(nonsense) and bare.header["lines"] == ()
    assert type(a) is type(b) and a.name is None and b.name is None
    assert a.constants.keys() == b.constants.keys() and "omega_ref" not in a.constants
    assert [f.name for f in b.geometry.interfaces] == [None] * 13
    for la, lb in zip(a.layers, b.layers):
        assert la.names == lb.names and la.name is None and lb.name is None
        r = np.linspace(*la.interval, 7)
        for name in la.names:
            assert np.array_equal(la[name](r), lb[name](r)), name
    with pytest.raises(ValueError, match="a mineos deck's header is a title line"):
        mineos_names(under)


#: A table the library has no format for: a fluid core under a mantle
#: whose viscosity falls by two decades, and a column of the caller's own.
OWN_COLUMNS = ("rho", "vp", "vs", "viscosity", "damping")
OWN_TABLE = ["# a table of my own: r rho vp vs viscosity damping",
             "0     10e3 9e3 0   nan  1",
             "1e6   10e3 9e3 0   nan  2",
             "1e6   4e3  8e3 4e3 1e22 3",
             "1.5e6 4e3  8e3 4e3 1e21 4",
             "2e6   3e3  7e3 3e3 1e20 5"]


def test_a_table_of_named_columns_is_imported_by_its_column_names():
    d = read_deck(OWN_TABLE, OWN_COLUMNS, header_lines=1)
    assert d.names == OWN_COLUMNS and d.header["lines"] == (OWN_TABLE[0],)
    spec = FieldSpec(SCALAR, VISCOSITY)
    m = model_from_deck(d, behaviours=[Elastic, Viscoelastic],
                        kinds={"viscosity": "log-linear"}, specs={"damping": spec},
                        layer_names=["core", "mantle"],
                        interface_names=["cmb", "surface"], name="my own")
    assert m.behaviours == ("Elastic", "Viscoelastic") and m.name == "my own"
    assert [lay.name for lay in m.layers] == ["core", "mantle"]
    assert [f.name for f in m.geometry.interfaces] == ["cmb", "surface"]
    core, mantle = m.layers
    assert "viscosity" not in core and m.is_fluid("core")
    eta = mantle["viscosity"]
    assert isinstance(eta.function, ExponentialLayer)
    assert np.isclose(eta(1.25e6), np.sqrt(1e22 * 1e21), rtol=1e-13, atol=0)
    assert m.specs["damping"] == spec and mantle["damping"](1.5e6) == 4.0
    assert mantle["L"](1e6) == 4e3 * 4e3 ** 2            # Elastic completed the moduli
    year = 3.15576e7
    relaxed = m.frozen(2 * np.pi / (1e5 * year)).layer("mantle")["L"](1.5e6)
    assert abs(relaxed) < mantle["L"](1.5e6)
    testing.check_model(m)
    # with no behaviours it is a plain Model of the columns alone
    plain = model_from_deck(d)
    assert type(plain) is Model and plain.layer(1).names == OWN_COLUMNS
    assert plain.layer(0).names == ("rho", "vp", "vs", "damping")
    with pytest.raises(ValueError, match="layer"):
        model_from_deck(d, layer_names=["core"])


def test_the_scales_are_the_units_of_the_table_s_numbers():
    """A table in km, km/s and g/cm^3 is a model in those scales."""
    d = Deck([0.0, 1000.0, 1000.0, 2000.0], {"rho": [10.0, 10.0, 4.0, 3.0],
                                              "vp": [9.0, 9.0, 8.0, 7.0],
                                              "vs": [0.0, 0.0, 4.0, 3.0]})
    km = Scales(length=1e3, mass=1e12)
    m = model_from_deck(d, behaviours=[Elastic], scales=km)
    assert m.scales == km and m.skeleton.boundaries[-1] == 2000.0
    si = m.in_si()
    assert np.isclose(si.skeleton.boundaries[-1], 2e6)
    assert np.isclose(si.layer(1)["rho"](2e6), 3e3)
    assert np.isclose(si.layer(1)["vs"](2e6), 3e3)
    assert np.isclose(si.layer(1)["L"](2e6), 3e3 * 3e3 ** 2)
    testing.check_model(m)


def test_a_table_of_radii_takes_no_lateral_domain():
    d = Deck([0.0, 1.0], {"rho": [1.0, 1.0]})
    with pytest.raises(ValueError, match="a table of radii is a model of the whole "
                                         "sphere and takes no lateral domain"):
        model_from_deck(d, domain=((0.0, 1.0),))


# -- a model written as a table -------------------------------------------------

def test_a_model_imported_and_exported_gives_the_table_s_numbers_back(
        mineos, deck, tmp_path):
    back = sample_deck(mineos, deck.names, knots=deck_knots(deck))
    assert back.names == deck.names and np.array_equal(back.radius, deck.radius)
    for name in deck.names:
        assert np.allclose(back[name], deck[name], rtol=1e-12, atol=1e-9)
    assert back.header["name"] == "prem.200" and np.isclose(back.header["tref"], 1.0)
    assert back.header["layers"][:2] == ("inner_core", "outer_core")
    path = write_deck(tmp_path / "prem.out", back, MINEOS)
    again = read_deck(path, MINEOS)
    assert again.names == deck.names
    for name in deck.names:
        assert np.allclose(again[name], deck[name], rtol=1e-7)
    assert np.allclose(again.radius, deck.radius)
    # the header written is the table's own: it reads as the one it came under
    assert mineos_header(again) == mineos_header(deck)
    assert mineos_names(again) == mineos_names(deck)
    with pytest.raises(ValueError, match="no longer matches"):
        sample_deck(mineos.refined([2000e3]), deck.names, knots=deck_knots(deck))
    with pytest.raises(ValueError, match="no longer matches"):
        sample_deck(mineos.truncated(6371e3 - 50.0), deck.names,
                    knots=deck_knots(deck))
    with pytest.raises(ValueError, match="needs columns"):
        write_deck(tmp_path / "bad", Deck([0.0, 1.0], {"rho": [1.0, 1.0]}), MINEOS)


def test_a_deck_is_written_under_its_format_s_header_lines_given_or_none(tmp_path):
    d = sample_deck(PREM(), MINEOS_COLUMNS, spacing=0.1)
    own = write_deck(tmp_path / "own", d, MINEOS).read_text().splitlines()
    assert own[:2] == ["PREM", "1 1 1"] and own[2].split()[0] == str(d.nknots)
    assert len(own) == 3 + d.nknots
    asked = write_deck(tmp_path / "asked", d, MINEOS, header=True)
    assert asked.read_text().splitlines() == own
    none = write_deck(tmp_path / "none", d, MINEOS, header=False)
    assert none.read_text().splitlines() == own[3:]
    mine = ["a title of my own", "# and a note, as it is"]
    given = write_deck(tmp_path / "given", d, MINEOS, header=mine)
    assert given.read_text().splitlines() == mine + own[3:]
    empty = write_deck(tmp_path / "empty", d, MINEOS, header=[])
    assert empty.read_text().splitlines() == own[3:]
    with pytest.raises(TypeError, match="got the string 'a title'"):
        write_deck(tmp_path / "bad", d, MINEOS, header="a title")
    # the rows alone are a table of named columns, read back by naming them
    bare = read_deck(none, MINEOS_COLUMNS)
    assert np.array_equal(bare["vph"], read_deck(tmp_path / "own", MINEOS)["vph"])


def test_a_plain_table_of_named_columns_has_no_header_of_its_own(tmp_path):
    d = sample_deck(PREM(), MINEOS_COLUMNS, spacing=0.1)
    for format in (("rho", "vsv"), DeckFormat(("rho", "vsv"), name="two")):
        path = write_deck(tmp_path / "two.txt", d, format)
        lines = path.read_text().splitlines()
        assert len(lines) == d.nknots and len(lines[0].split()) == 3
        back = read_deck(path, ("rho", "vsv"))
        assert np.allclose(back["vsv"], d["vsv"], rtol=1e-7)
        assert np.allclose(back.radius, d.radius, rtol=0, atol=1e-3)
    named = write_deck(tmp_path / "named.txt", d, ("rho", "vsv"),
                       header=["# r rho vsv"])
    assert named.read_text().splitlines()[0] == "# r rho vsv"
    assert read_deck(named, ("rho", "vsv")).nknots == d.nknots   # a comment, skipped
    with pytest.raises(TypeError, match="got the string 'rho'"):
        write_deck(tmp_path / "bad", d, "rho")
    with pytest.raises(ValueError, match="needs columns"):
        write_deck(tmp_path / "bad", d, ("rho", "viscosity"))


def test_prem_s_generated_mineos_header_carries_the_counts_of_its_tabulation(
        deck, tmp_path):
    """On the knots of `prem.200`, the counts of its header: 220 knots, 40
    in the inner core and 112 to the top of the outer core, the innermost
    layer whose S velocity is zero throughout (the ocean is fluid too).
    The title is the model's name, and the reference period the 1 s of
    the `omega_ref` that `PREM` declares, as the file's own is: read
    back, the model declares the same constant."""
    d = sample_deck(PREM(), MINEOS_COLUMNS, knots=deck_knots(deck))
    lines = write_deck(tmp_path / "prem.deck", d, MINEOS).read_text().splitlines()
    theirs = PREM_DECK.read_text().splitlines()
    assert lines[0] == "PREM"
    assert lines[2].split() == theirs[2].split() == ["220", "40", "112"]
    assert lines[1] == "1 1 1"
    assert [float(x) for x in lines[1].split()] == [
        float(x) for x in theirs[1].split()[:3]] == [1.0, 1.0, 1.0]
    for variant in (PREM(ocean=False), PREM().nondimensionalised()):
        assert sample_deck(variant, MINEOS_COLUMNS).header["tref"] == (
            pytest.approx(1.0, rel=1e-14))
    m = imported(read_deck(tmp_path / "prem.deck", MINEOS))
    assert m.nlayers == 13 and m.name == "PREM"
    assert m.layer(0).name == "inner_core" and m.layer(1).name == "outer_core"
    assert m.layer("outer_core")["qmu"](2000e3) == 0.0          # no loss, as zero
    assert m.constants["omega_ref"] == PREM().constants["omega_ref"]
    assert m.constants["omega_ref"] == mineos_constants(deck)["omega_ref"]
    assert np.isclose(m.mass(), PREM().mass(), rtol=1e-7)


def test_the_generated_counts_are_where_the_fluid_is_in_the_table(tmp_path):
    """A body with no fluid layer is written `0 0`; a fluid layer at the
    centre has no inner core beneath it; a model that declares no
    reference frequency is written with a `tref` of 0; an isotropic table
    with `ifanis` 0."""
    def header(vs):
        model = LayeredIsotropicElastic([0.0, 1e6, 2e6, 3e6], rho=[9e3, 8e3, 4e3],
                                        vp=[9e3, 8e3, 7e3], vs=vs, name="a body")
        d = sample_deck(model, MINEOS_ISOTROPIC_COLUMNS, spacing=0.1)
        knots = [s.stop - s.start for s in d.layers()]
        path = write_deck(tmp_path / "body.deck", d, MINEOS)
        return path, knots, path.read_text().splitlines()[:3]

    path, knots, lines = header([4e3, 3e3, 3e3])
    assert lines == ["a body", "0 0 1", f"{sum(knots)} 0 0"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_deck(path, MINEOS)
        assert mineos_names(back) == ([None] * 3, [None, None, "surface"])
        assert mineos_constants(back) == {}
    assert np.all(back["qmu"] == 0.0)                 # no quality factor: no loss
    _, knots, lines = header([4e3, 0.0, 3e3])
    assert lines[2] == f"{sum(knots)} {knots[0]} {knots[0] + knots[1]}"
    path, knots, lines = header([0.0, 3e3, 3e3])
    assert lines[2] == f"{sum(knots)} 0 {knots[0]}"
    assert mineos_names(read_deck(path, MINEOS)) == (
        ["outer_core", None, None], ["cmb", None, "surface"])
    _, knots, lines = header([0.0, 3e3, 0.0])         # the innermost fluid layer
    assert lines[2] == f"{sum(knots)} 0 {knots[0]}"


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
    d = read_deck(nd_file(width), ND)
    assert d.names == ("vp", "vs", "rho", "qp", "qmu")[:width - 1]


def test_an_nd_table_writes_back(tmp_path):
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


@pytest.mark.parametrize("inner", [0.0, 0.5])
def test_a_label_at_the_innermost_boundary_is_refused_by_name(inner):
    """A label names the layer below its boundary, and none lies below the
    innermost, of a solid body or a hollow one."""
    radius = np.array([inner, 1.0, 1.0, 2.0])
    columns = {"vp": np.ones(4), "vs": np.ones(4)}
    below = Deck(radius, columns, header={"labels": ((inner, "inner-core"),)})
    with pytest.raises(ValueError, match="the label 'inner-core' sits at the "
                                         "innermost boundary, where no layer lies "
                                         "below it"):
        nd_names(below)
    # at any other boundary it names the layer below, and its interface
    held = Deck(radius, columns, header={"labels": ((1.0, "inner-core"),
                                                    (2.0, "mantle"))})
    layers, faces = nd_names(held)
    assert layers == ["inner_core", "mantle"]
    assert faces == [None] * (inner > 0) + ["icb", "surface"]


def test_an_nd_table_of_a_model_is_labelled_from_its_interface_names(tmp_path):
    """TauP knows the Moho and the core by `mantle`, `outer-core` and
    `inner-core`: a model's table holds those at its interfaces `moho`,
    `cmb` and `icb`, each between the two rows of its depth, and no other
    label; read back, the names they imply are the model's."""
    d = read_deck(nd_file(), ND)
    layers, faces = nd_names(d)
    model = model_from_deck(d, kind="linear", layer_names=layers,
                            interface_names=faces, name="a body")
    testing.check_model(model)
    sampled = sample_deck(model, d.names, knots=deck_knots(d))
    assert "labels" not in sampled.header
    lines = write_deck(tmp_path / "model.nd", sampled, ND).read_text().splitlines()
    words = [(i, ln) for i, ln in enumerate(lines) if not ln[0].isdigit()]
    assert [w for _, w in words] == ["mantle", "outer-core", "inner-core"]
    for (i, _), depth in zip(words, ("10", "80", "95")):
        assert lines[i - 1].split()[0] == lines[i + 1].split()[0] == depth
    back = read_deck(tmp_path / "model.nd", ND)
    assert nd_names(back)[1] == faces
    for name in d.names:
        assert np.allclose(back[name], d[name], rtol=1e-9)
    # a hollow body's inner boundary is an interface too, and none of TauP's
    hollow = model.hollowed(5e3)
    assert [f.name for f in hollow.geometry.interfaces][:2] == ["icb", "cmb"]
    lines = write_deck(tmp_path / "hollow.nd", sample_deck(hollow, d.names),
                       ND).read_text().splitlines()
    assert [ln for ln in lines if not ln[0].isdigit()] == ["mantle", "outer-core"]
    # header lines given come first, and the labels are the table's all the same
    noted = write_deck(tmp_path / "noted.nd", sampled, ND, header=["# a body"])
    assert noted.read_text().splitlines()[0] == "# a body"
    assert nd_names(read_deck(noted, ND))[1] == faces
    wrong = Deck(sampled.radius, sampled.columns, header={"interfaces": ("moho",)})
    with pytest.raises(ValueError, match="names 1 interfaces; its table has 6 layers"):
        write_deck(tmp_path / "wrong.nd", wrong, ND)


def test_a_tvel_table_is_written_under_two_title_lines_and_holds_no_labels(tmp_path):
    lines = ["a P model", "an S model"] + [ln for ln in nd_file(4)
                                            if ln[0].isdigit()]
    d = read_deck(lines, TVEL)
    assert d.header["lines"] == ("a P model", "an S model")
    assert d.names == ("vp", "vs", "rho")
    kept = read_deck(write_deck(tmp_path / "body.tvel", d, TVEL,
                                header=d.header["lines"]), TVEL)
    assert np.array_equal(kept.radius, d.radius)
    assert kept.header["lines"] == d.header["lines"]
    layers, faces = nd_names(read_deck(nd_file(4), ND))
    model = model_from_deck(d, kind="linear", layer_names=layers,
                            interface_names=faces, name="a body")
    path = write_deck(tmp_path / "model.tvel", sample_deck(model, d.names), TVEL)
    text = path.read_text().splitlines()
    assert text[:2] == ["a body - P", "a body - S"]
    assert all(ln[0].isdigit() for ln in text[2:])
    assert read_deck(path, TVEL).nlayers == d.nlayers


# -- writing: the widest column set, and fill values -----------------------------

def test_a_mineos_deck_is_written_with_its_widest_columns_and_no_loss_as_zero(tmp_path):
    names = set(MINEOS_COLUMNS) | set(MINEOS_ISOTROPIC_COLUMNS)
    columns = {n: np.full(4, 2.0) for n in names}
    columns["qmu"] = np.array([np.nan, np.nan, 100.0, 100.0])
    d = Deck([0.0, 1.0, 1.0, 2.0], columns, header={"name": "both"})
    back = read_deck(write_deck(tmp_path / "both.deck", d, MINEOS), MINEOS)
    assert back.names == MINEOS_COLUMNS
    assert list(back["qmu"]) == [0.0, 0.0, 100.0, 100.0]
    assert mineos_header(back)["ifanis"] == 1
    viscous = Deck(d.radius, {n: d[n] for n in VISCOUS if n in d}
                   | {"viscosity": columns["qmu"]})
    text = write_deck(tmp_path / "v.deck", viscous, VISCOUS).read_text()
    assert "nan" in text                     # a table of named columns fills nothing
    iso = Deck(d.radius, {n: d[n] for n in MINEOS_ISOTROPIC_COLUMNS})
    written = read_deck(write_deck(tmp_path / "iso.deck", iso, MINEOS), MINEOS)
    assert mineos_header(written)["ifanis"] == 0


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
    with pytest.raises(ValueError, match="kind must be one of"):
        deck_layers(d, kind="log-quintic")


# -- kinds in the logarithm -----------------------------------------------------------

#: Knots of one layer, and values over five decades on them, up and down.
_KNOTS = np.array([0.0, 0.13, 0.31, 0.52, 0.74, 1.0])
_SPAN = 10.0 ** np.array([19.0, 21.5, 20.0, 23.0, 22.0, 24.0])


def test_log_linear_is_geometric_between_knots():
    """y_i (y_{i+1} / y_i)^t on every piece, to the rounding of the
    logarithms of the ratios, and the first knot to the bit; two knots give
    it under every kind in the logarithm."""
    _, layers = deck_layers(Deck(_KNOTS, {"viscosity": _SPAN}), kind="log-linear")
    f = layers[0]["viscosity"]
    assert isinstance(f.function, ExponentialLayer) and f.function.exponent.degree == 1
    assert f(0.0) == _SPAN[0]
    t = np.linspace(0.0, 1.0, 11)
    for i in range(_KNOTS.size - 1):
        x = _KNOTS[i] + t * (_KNOTS[i + 1] - _KNOTS[i])
        want = _SPAN[i] * (_SPAN[i + 1] / _SPAN[i]) ** t
        assert np.allclose(f(x), want, rtol=5e-15, atol=0), i
    two = Deck([0.0, 1.0], {"viscosity": [1e19, 1e23]})
    for kind in LOG_KINDS:
        _, layers = deck_layers(two, kind=kind)
        eta = layers[0]["viscosity"]([0.25, 0.5, 0.75])
        assert np.allclose(eta, [1e20, 1e21, 1e22], rtol=2e-15, atol=0), kind


@pytest.mark.parametrize("kind", LOG_KINDS)
def test_a_column_joined_in_its_logarithm_is_the_exponential_of_its_kind(kind):
    """The logarithm of the column over its first value is the kind's
    polynomial through the logarithms; the column passes through the knots;
    its first and second derivatives agree with central differences between
    them, and its integral with quadrature piece by piece."""
    from scipy.integrate import quad
    _, layers = deck_layers(Deck(_KNOTS, {"viscosity": _SPAN}),
                            kinds={"viscosity": kind})
    f = layers[0]["viscosity"]
    assert isinstance(f.function, ExponentialLayer)
    assert np.allclose(f(_KNOTS), _SPAN, rtol=5e-15, atol=0)
    plain = kind.removeprefix("log-")
    _, logs = deck_layers(Deck(_KNOTS, {"log": np.log(_SPAN / _SPAN[0])}), kind=plain)
    x = np.sort(np.concatenate([0.5 * (_KNOTS[1:] + _KNOTS[:-1]), _KNOTS[:-1] + 0.03]))
    assert np.allclose(np.log(f(x) / _SPAN[0]), logs[0]["log"](x), rtol=0, atol=1e-13)
    h = 1e-6
    for nu in (1, 2):
        before, d = f.derivative(nu=nu - 1), f.derivative(nu=nu)
        fd = (before(x + h) - before(x - h)) / (2.0 * h)
        assert np.allclose(d(x), fd, rtol=1e-6, atol=0), nu
    whole = sum(quad(lambda s: float(f(s)), a, b, epsrel=1e-12, epsabs=0, limit=200)[0]
                for a, b in zip(_KNOTS[:-1], _KNOTS[1:]))
    assert f.integrate(0.0, 1.0) == pytest.approx(whole, rel=1e-10, abs=0)


@pytest.mark.parametrize("values, kind, says", [
    ([np.nan, np.nan, 1e21, -1.0, 1e22], "log-linear",
     "column 'viscosity' is -1 at r = 2 on layer 1; a column joined by log-linear, its "
     "logarithm interpolated, is positive and finite at every knot of a layer holding "
     "it"),
    ([np.nan, np.nan, 0.0, 1e21, 1e22], "log-cubic",
     "column 'viscosity' is 0 at r = 1 on layer 1; a column joined by log-cubic"),
    ([1.0, 2.0, 1e21, np.inf, 1e22], "log-pchip",
     "column 'viscosity' is inf at r = 2 on layer 1"),
])
def test_a_column_joined_in_its_logarithm_is_positive_at_every_knot(values, kind, says):
    d = Deck([0.0, 1.0, 1.0, 2.0, 3.0], {"rho": [4.0, 4.0, 3.0, 3.0, 3.0],
                                         "viscosity": values})
    with pytest.raises(ValueError) as caught:
        deck_layers(d, kinds={"viscosity": kind})
    assert says in str(caught.value)
    with pytest.raises(ValueError, match="spans more than the range of a float"):
        deck_layers(Deck([0.0, 1.0], {"v": [1e-200, 1e200]}), kind="log-linear")


#: Kinds for the families of Prony branches, by their stems: the strengths
#: by straight lines, the times in their logarithm.
BRANCH_KINDS = {"relaxation_strength": "linear", "relaxation_time": "log-linear"}


def test_a_family_is_joined_by_the_kind_its_stem_is_given():
    d = Deck([0.0, 0.3, 0.5, 1.0], {"rho": [5.0, 4.0, 4.5, 3.0],
                                    "relaxation_strength_1": [0.1, 0.9, 0.2, 0.8],
                                    "relaxation_time_1": [1e8, 1e12, 1e9, 1e11]})
    _, cubic = deck_layers(d)
    _, joined = deck_layers(d, kinds=BRANCH_KINDS)
    r = np.linspace(0.0, 1.0, 41)
    assert np.array_equal(joined[0]["rho"](r), cubic[0]["rho"](r))
    assert np.min(cubic[0]["relaxation_time_1"](r)) < 0.0      # a cubic overshoots
    assert np.min(joined[0]["relaxation_time_1"](r)) >= 1e8
    w = joined[0]["relaxation_strength_1"](r)
    assert np.all((w >= 0.1) & (w <= 0.9))
    strength, time = (joined[0][n].function for n in ("relaxation_strength_1",
                                                      "relaxation_time_1"))
    assert isinstance(strength, PolynomialLayer) and strength.degree == 1
    assert isinstance(time, ExponentialLayer) and time.exponent.degree == 1
    assert np.isclose(time(0.15), 1e10, rtol=1e-14, atol=0)    # geometric between knots
    _, own = deck_layers(d, kinds={"relaxation_time": "linear",
                                   "relaxation_time_1": "cubic"})
    assert np.array_equal(own[0]["relaxation_time_1"](r),
                          cubic[0]["relaxation_time_1"](r))
    with pytest.raises(ValueError, match="lacks"):
        deck_layers(Deck([0.0, 1.0], {"rho": [1.0, 2.0]}), kinds=BRANCH_KINDS)


def _graded_viscous_deck():
    """`_maxwell_prem` sampled, with the lower mantle's viscosity falling
    from 1e23 Pa s at its base to 1e21 at its top, exponentially in
    radius."""
    model = _maxwell_prem()
    d = sample_deck(model, VISCOUS)
    eta = np.array(d["viscosity"])
    s = d.layers()[model.layer("lower_mantle").index]
    x = (d.radius[s] - d.radius[s][0]) / np.ptp(d.radius[s])
    eta[s] = 10.0 ** (23.0 - 2.0 * x)
    return Deck(d.radius, {**d.columns, "viscosity": eta}, header=d.header)


def _viscous_model(deck, **kw):
    """The model of a deck of `VISCOUS` sampled from a model, with that
    model's names."""
    return model_from_deck(deck, behaviours=[b for b in BEHAVIOURS
                                             if b.__name__ != "ConstantQ"],
                           layer_names=deck.header["layers"],
                           interface_names=deck.header["interfaces"],
                           name=deck.header["name"], **kw)


def test_a_viscosity_constant_through_a_layer_is_that_constant_under_every_kind():
    """Joined in its logarithm or by straight lines, a viscosity constant
    through each layer is the same field, with the same derivative and the
    same frozen moduli, to the bit."""
    d = sample_deck(_maxwell_prem(), VISCOUS)
    log = _viscous_model(d, kinds={"viscosity": "log-linear"})
    lines = _viscous_model(d, kinds={"viscosity": "linear"})
    omega = 2.0 * np.pi / (1000 * 3.15576e7)
    flog, flines = log.frozen(omega), lines.frozen(omega)
    viscous = [lay.index for lay in log.layers if "viscosity" in lay]
    assert viscous == [2, 3, 4, 5, 6, 7]
    for i in viscous:
        r = np.linspace(*log.layer(i).interval, 101)
        a, b = log.layer(i)["viscosity"], lines.layer(i)["viscosity"]
        assert a.function.degree == 0 and np.all(a(r) == a(r[0]))
        assert np.array_equal(a(r), b(r)) and np.array_equal(a.derivative()(r),
                                                            b.derivative()(r))
        assert np.array_equal(flog.layer(i)["N"](r), flines.layer(i)["N"](r))


def test_a_graded_viscosity_is_geometric_in_its_logarithm_and_above_it_by_lines(
        tmp_path):
    """A viscosity varying through a layer, written and read back: with
    `log-linear` geometric between knots, which holds a viscosity
    exponential in radius to the digits written; with `linear` the straight
    lines through the values, above it between knots by their arithmetic
    over geometric mean; both through the knots."""
    d = _graded_viscous_deck()
    path = write_deck(tmp_path / "graded.deck", d, VISCOUS)
    read = read_deck(path, VISCOUS)
    read = Deck(read.radius, read.columns, header=d.header)
    log = _viscous_model(read, kinds={"viscosity": "log-linear"})
    lines = _viscous_model(read, kinds={"viscosity": "linear"})
    assert [lay.name for lay in log.layers] == [lay.name for lay in
                                               _maxwell_prem().layers]
    i = log.layer("lower_mantle").index
    a, b = log.layer(i)["viscosity"], lines.layer(i)["viscosity"]
    assert isinstance(a.function, ExponentialLayer)
    assert isinstance(b.function, PolynomialLayer) and b.function.degree == 1
    r = deck_knots(read)[i]
    y = read["viscosity"][read.layers()[i]]
    assert np.allclose(a(r), y, rtol=1e-14, atol=0) and np.allclose(b(r), y, rtol=1e-15)
    middle = 0.5 * (r[1:] + r[:-1])
    assert np.allclose(a(middle), np.sqrt(y[1:] * y[:-1]), rtol=1e-14, atol=0)
    assert np.allclose(b(middle), 0.5 * (y[1:] + y[:-1]), rtol=1e-14, atol=0)
    x = (middle - r[0]) / (r[-1] - r[0])
    assert np.allclose(a(middle), 10.0 ** (23.0 - 2.0 * x), rtol=5e-8, atol=0)
    above = b(middle) / a(middle) - 1.0
    assert np.all(above > 0.0) and 6e-4 < np.max(above) < 8e-4
    # exported on its knots, the table comes back character for character
    again = write_deck(tmp_path / "again.deck",
                       sample_deck(log, VISCOUS, knots=deck_knots(read)), VISCOUS)
    assert again.read_text() == path.read_text()
    testing.check_model(log)


def test_a_viscous_model_relaxes_its_mantle_at_glacial_periods():
    m = _viscous_model(sample_deck(_maxwell_prem(), VISCOUS),
                       kinds={"viscosity": "log-linear"})
    assert m.behaviours == ("Elastic", "SelfGravitating", "Viscoelastic")
    assert m.is_viscoelastic("lower_mantle") and not m.is_viscoelastic("lid")
    omega = 2 * np.pi / (1000 * 3.15576e7)
    fz = m.frozen(omega)
    assert type(fz) is type(m)
    assert fz.layer("lower_mantle")["L"].dtype == np.complex128
    r = 5000e3
    assert abs(fz.layer("lower_mantle")["L"](r)) < m.layer("lower_mantle")["L"](r)
    assert np.isclose(fz.layer("lid")["L"](6300e3), m.layer("lid")["L"](6300e3))
    assert type(m.nondimensionalised()) is type(m)


# -- any model as a deck ------------------------------------------------------------

def _maxwell_prem():
    """PREM without its ocean, a viscosity on every mantle layer below the
    low-velocity zone: 1e22 Pa s below 670 km, 1e21 above."""
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
    model = _maxwell_prem()
    d = sample_deck(model, VISCOUS)
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
    """A model made isotropic holds vp and vs; asked for the transversely
    isotropic five, it gives them from its moduli."""
    iso = _maxwell_prem().isotropic().without_field("qmu").without_field("qkappa")
    assert "vpv" not in iso.layer("lid")
    d = sample_deck(iso, VISCOUS)
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
    # a declared reference period is in the header in seconds, for a mineos tref
    assert given.header["tref"] == pytest.approx(1.0, rel=1e-14)
    assert given.header["name"] == "PREM (no ocean)"
    bare = nd.replaced(constants={})
    assert "tref" not in sample_deck(bare, ["rho"], knots=k).header
