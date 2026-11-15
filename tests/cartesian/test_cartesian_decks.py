"""Decks of depths: each format written, read and written again character
for character; the deck-read models held to the model contract; the
formats sharing their definitions with the radius-first ones; the base
padded below the last knot exactly; the names a header gives from the
surface down; the knots named by their depth in the column line and in
every message; and what is refused."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel import (BURGERS, MAXWELL, PRONY, Deck, deck_layers, read_deck,
                         sample_deck, write_deck)
from planetmodel import cartesian as flat
from planetmodel.conversion import prony
from planetmodel.deck import (MAXWELL_COLUMNS, MAXWELL_KINDS, MINEOS_COLUMNS,
                              MINEOS_ISOTROPIC_COLUMNS, RADIUS, deck_from,
                              prony_columns)
from planetmodel.layerfunction import ExponentialLayer
from planetmodel.testing import check_model

BOX = flat.Box([(0.0, 400e3), (-200e3, 200e3)])
YEAR = 3.15576e7


def crust(**kw):
    """The reference flat model, its asthenosphere a Maxwell body."""
    return flat.CrustOverMantle(BOX, viscosity=1e20, **kw)


def round_trip(tmp_path, deck, format, model_type, stem):
    """The deck written in `format`, read as `model_type`, and written back
    from the model: the model and both texts."""
    first = write_deck(tmp_path / f"{stem}.deck", deck, format)
    model = model_type(first, BOX)
    again = write_deck(tmp_path / f"{stem}.again", model.to_deck(), format)
    return model, first.read_text(), again.read_text()


# -- one definition each ----------------------------------------------------------

def test_the_formats_of_depths_are_the_radius_first_ones_stated_for_depth():
    for depth, radius in ((flat.MAXWELL, MAXWELL), (flat.PRONY, PRONY),
                          (flat.BURGERS, BURGERS)):
        assert depth.columns is radius.columns and depth.name == radius.name
        assert depth.families == radius.families
        assert depth.header_comments == radius.header_comments
        assert depth.complete is radius.complete
        assert depth.coordinate is flat.DEPTH and radius.coordinate is RADIUS
    assert flat.ELASTIC.names(8) == MINEOS_COLUMNS
    assert flat.ELASTIC.names(5) == MINEOS_ISOTROPIC_COLUMNS
    assert flat.DEPTH.field_type is flat.DepthField and flat.DEPTH.word == "depth"
    assert flat.DEPTH.coordinates is flat.FLAT and flat.DEPTH.label == "depth"


# -- round trips, character for character -----------------------------------------

def test_a_maxwell_deck_of_depths_round_trips(tmp_path):
    model, first, again = round_trip(tmp_path, sample_deck(crust(), MAXWELL_COLUMNS),
                                     flat.MAXWELL, flat.MaxwellDeckModel, "maxwell")
    assert again == first
    lines = first.splitlines()
    assert lines[:4] == ["crust over mantle (Maxwell asthenosphere)",
                         "depth rho vpv vsv vph vsh eta viscosity",
                         "# layers crust lithospheric_mantle asthenosphere",
                         "# interfaces surface moho lab"]
    assert float(lines[4].split()[0]) == 0.0           # the surface, first
    assert [lay.name for lay in model.layers] == ["crust", "lithospheric_mantle",
                                                  "asthenosphere"]
    assert [f.name for f in model.geometry.interfaces] == ["surface", "moho", "lab"]
    assert model.geometry.interface("moho").between == (1, 0)
    assert model.geometry.domain == BOX and model.name == crust().name
    for layer in model.layers:
        assert all(type(f) is flat.DepthField for f in layer.fields.values())
    # a column constant through a layer is that constant
    assert model.layer(-1)["viscosity"].radial_degree == 0
    assert model.layer(-1)["viscosity"](150e3) == 1e20
    check_model(model)


def test_an_elastic_deck_of_depths_round_trips_isotropic_and_not(tmp_path):
    iso = crust().elastic()
    model, first, again = round_trip(tmp_path,
                                     sample_deck(iso, MINEOS_ISOTROPIC_COLUMNS),
                                     flat.ELASTIC, flat.ElasticDeckModel, "iso")
    assert again == first
    assert first.splitlines()[1] == "depth rho vp vs qkappa qmu"
    assert "qmu" not in model.layer(0)                    # NaN throughout: absent
    check_model(model)
    # a transversely isotropic deck with quality factors, the shear one
    # absent from a fluid layer on top
    layered = flat.LayeredIsotropicElastic(
        [0.0, 4e3, 30e3, 100e3], BOX, rho=[1020.0, 2800.0, 3300.0],
        vp=[1450.0, 6400.0, [8000.0, 1e-3]], vs=[0.0, 3700.0, 4500.0],
        interface_names=["surface", "seafloor", "moho"])
    deck = sample_deck(layered, MINEOS_COLUMNS)
    q = {"qkappa": np.full(deck.nknots, 57823.0), "qmu": np.full(deck.nknots, 600.0)}
    q["qmu"][deck.layers()[0]] = np.nan
    deck = Deck(deck.radius, {**deck.columns, **q}, header=deck.header)
    vti, first, again = round_trip(tmp_path, deck, flat.ELASTIC, flat.ElasticDeckModel,
                                   "vti")
    assert again == first
    assert first.splitlines()[1] == "depth rho vpv vsv qkappa qmu vph vsh eta"
    assert vti.is_fluid(0) and "qmu" not in vti.layer(0) and "qmu" in vti.layer(1)
    omega = 2.0 * np.pi / 20.0
    assert np.iscomplexobj(vti.moduli_at(1, omega)["L"](10e3))
    check_model(vti)


def test_a_prony_deck_of_depths_round_trips(tmp_path):
    series = prony.PronySeries((0.3, 0.2), (0.5, 5.0), name="two")
    model = prony.with_branches(crust(), series)
    deck = prony.prony_deck(model, format=flat.PRONY)
    read, first, again = round_trip(tmp_path, deck, flat.PRONY, flat.PronyDeckModel,
                                    "prony")
    assert again == first
    assert first.splitlines()[1] == "depth " + " ".join(prony_columns(2))
    assert len(pm.rheology.branches(read.layer(-1))) == 2
    assert not pm.rheology.branches(read.layer(0))
    check_model(read)
    omega = 2.0 * np.pi / (100.0 * YEAR)
    d = read.knots[-1]
    assert np.allclose(read.frozen(omega).layer(-1)["N"](d),
                       model.frozen(omega).layer(-1)["N"](d), rtol=1e-7, atol=0)


def test_a_burgers_deck_of_depths_round_trips(tmp_path):
    model = prony.with_burgers(crust(), ["asthenosphere"], kelvin_viscosity=1e19,
                               kelvin_modulus_fraction=0.5)
    deck = prony.burgers_deck(model, format=flat.BURGERS)
    read, first, again = round_trip(tmp_path, deck, flat.BURGERS,
                                    flat.BurgersDeckModel, "burgers")
    assert again == first
    rows = [line.split() for line in first.splitlines()[4:]]
    assert {(r[-2], r[-1]) for r in rows if r[-1] != "nan"} == {("1e+19", "0.5")}
    assert "viscosity" not in read.layer(-1)
    assert len(read.layer(-1).names) > len(read.layer(0).names)
    check_model(read)


def test_a_written_deck_of_depths_reads_and_writes_back_as_it_stands(tmp_path):
    text = ("two layers\n"
            "depth rho vpv vsv vph vsh eta viscosity # SI\n"
            "# layers - mantle\n"
            "# interfaces surface -\n"
            "         0.000            2700            6000            3500"
            "            6000            3500               1             nan\n"
            "     20000.000            2700            6000            3500"
            "            6000            3500               1             nan\n"
            "     20000.000            3300            8000            4500"
            "            8000            4500               1           1e+21\n"
            "     60000.000            3350            8100            4550"
            "            8100            4550               1           1e+20\n"
            "    100000.000            3400            8200            4600"
            "            8200            4600               1           1e+19\n")
    path = tmp_path / "two.deck"
    path.write_text(text)
    model = flat.MaxwellDeckModel(path, flat.Rectangle((0.0, 1e6)))
    assert [lay.name for lay in model.layers] == [None, "mantle"]
    assert [f.name for f in model.geometry.interfaces] == ["surface", None]
    # the viscosity is joined in its logarithm, as a deck of radii's is:
    # geometric between knots
    eta = model.layer("mantle")["viscosity"]
    assert isinstance(eta.function, ExponentialLayer)
    assert np.isclose(eta(40e3), np.sqrt(1e21 * 1e20), rtol=1e-12)
    again = write_deck(tmp_path / "again.deck", model.to_deck(), flat.MAXWELL)
    # the comment on the column line is not kept; every other character is
    assert again.read_text() == text.replace(" # SI", "")
    check_model(model)


# -- the base, and padding below it -------------------------------------------------

def test_padding_below_the_last_knot_holds_its_values_exactly(tmp_path):
    path = write_deck(tmp_path / "m.deck", sample_deck(crust(), MAXWELL_COLUMNS),
                      flat.MAXWELL)
    model = flat.MaxwellDeckModel(path, BOX)
    deck = read_deck(path, flat.MAXWELL)
    base = float(deck.radius[-1])
    assert model.skeleton.boundaries[-1] == base
    padded = model.extended([base + 300e3, base + 800e3], fields="hold",
                            names=["pad", "far"])
    assert [f.radius for f in padded.geometry.interfaces][-2:] == [base, base + 300e3]
    depths = np.linspace(base, base + 800e3, 41)
    for name in deck.names:
        for layer in padded.layers[-2:]:
            held = layer[name]
            assert type(held) is flat.DepthField
            lo, hi = layer.interval
            at = depths[(depths >= lo) & (depths <= hi)]
            assert np.all(held(at) == deck[name][-1]), name
    assert np.all(padded.layer("far")["N"](depths[-3:]) == model.layer(-1)["N"](base))
    check_model(padded)


# -- names, columns and messages ----------------------------------------------------

def _rows(*rows):
    return [" ".join(str(v) for v in row) for row in rows]


TWO = _rows((0, 2700, 6000, 3500, 6000, 3500, 1, "nan"),
            (20e3, 2700, 6000, 3500, 6000, 3500, 1, "nan"),
            (20e3, 3300, 8000, 4500, 8000, 4500, 1, 1e21),
            (60e3, 3300, 8000, 4500, 8000, 4500, 1, 1e20))
LINE = "depth rho vpv vsv vph vsh eta viscosity"


def test_the_names_of_a_header_of_depths_run_from_the_surface_down():
    named = read_deck(["two", LINE, "# layers crust mantle",
                       "# interfaces top moho", *TWO], flat.MAXWELL)
    assert flat.depth_names(named) == (["crust", "mantle"], ["top", "moho"])
    plain = read_deck(["two", LINE, *TWO], flat.MAXWELL)
    assert flat.depth_names(plain) == ([None, None], ["surface", None])
    # a body whose top lies below the free surface has no surface by default
    deep = Deck(plain.radius + 5e3, plain.columns)
    assert flat.depth_names(deep) == ([None, None], [None, None])
    three = read_deck(["two", LINE, "# interfaces a b c", *TWO], flat.MAXWELL)
    with pytest.raises(ValueError, match="names 3 interfaces; its table has 2"):
        flat.depth_names(three)


def test_the_knots_are_named_by_their_depth_in_the_column_line_and_messages():
    with pytest.raises(ValueError, match="a Maxwell deck's second line names its "
                                         "columns, depth rho vpv"):
        read_deck(["two", "r rho vpv vsv vph vsh eta viscosity", *TWO], flat.MAXWELL)
    with pytest.raises(ValueError, match="names its columns, r rho"):
        read_deck(["two", LINE, *TWO], MAXWELL)
    with pytest.raises(ValueError, match="columns after the depth"):
        read_deck(["two", LINE, *[row + " 1" for row in TWO]], flat.MAXWELL)
    with pytest.raises(ValueError, match="is 0 at depth = 20000 on layer 1"):
        deck = read_deck(["two", LINE, *TWO], flat.MAXWELL)
        bad = np.array(deck["viscosity"])
        bad[2] = 0.0
        deck_layers(Deck(deck.radius, {**deck.columns, "viscosity": bad}),
                    kinds=MAXWELL_KINDS, coordinate=flat.DEPTH)
    lines = ["p", "depth rho vpv vsv vph vsh eta relaxation_strength_1 "
             "relaxation_time_1",
             *_rows((0, 2700, 6000, 3500, 6000, 3500, 1, 1.5, 1e9),
                    (1e3, 2700, 6000, 3500, 6000, 3500, 1, 0.5, 1e9))]
    with pytest.raises(ValueError, match="relaxation_strength_1 is 1.5 at depth = 0"):
        read_deck(lines, flat.PRONY)
    lines = ["b", "depth " + " ".join(MAXWELL_COLUMNS) + " kelvin_viscosity "
             "kelvin_modulus_fraction",
             *_rows((0, 2700, 6000, 3500, 6000, 3500, 1, 1e21, 1e19, 0.5),
                    (1e3, 2700, 6000, 3500, 6000, 3500, 1, 1e21, 1e19, -2))]
    with pytest.raises(ValueError, match="is -2 at depth = 1000; it is positive"):
        read_deck(lines, flat.BURGERS)
    iso = ["e", "depth rho vp vs qkappa qmu",
           *_rows((0, 2700, 6000, 3500, 1, 1, 1, 1, 1), (1e3, 2700, 6000, 3500,
                                                         1, 1, 1, 1, 1))]
    with pytest.raises(ValueError, match="names 5 columns after the depth; the rows "
                                         "have 8"):
        read_deck(iso, flat.ELASTIC)
    with pytest.raises(ValueError, match="an elastic deck's second line names its "
                                         "columns, depth rho vpv .* or depth rho vp"):
        read_deck(["e", "depth rho vp vs", *TWO], flat.ELASTIC)


def test_a_deck_read_model_needs_its_box(tmp_path):
    path = tmp_path / "two.deck"
    path.write_text("\n".join(["two", LINE, *TWO]) + "\n")
    with pytest.raises(TypeError, match="domain"):
        flat.MaxwellDeckModel(path)
    with pytest.raises(TypeError, match="expected a Box or a Rectangle"):
        flat.MaxwellDeckModel(path, ((0.0, 1.0), (0.0, 1.0)))
    model = flat.MaxwellDeckModel(read_deck(path, flat.MAXWELL), BOX)
    assert model.geometry.domain == BOX


def test_a_deck_given_is_held_to_its_format_as_one_read():
    lines = ["p", "depth rho vpv vsv vph vsh eta " + " ".join(prony_columns(2)[6:]),
             *_rows((0, 2700, 6000, 3500, 6000, 3500, 1, 0.5, 1e9, "nan", "nan"),
                    (1e3, 2700, 6000, 3500, 6000, 3500, 1, 0.5, 1e9, "nan", "nan"))]
    raw = np.loadtxt(lines[2:], ndmin=2)
    deck = Deck(raw[:, 0], {n: raw[:, j + 1] for j, n in enumerate(prony_columns(2))})
    held = deck_from(deck, flat.PRONY)
    assert np.all(held["relaxation_strength_2"] == 0.0)            # completed
    assert np.all(held["relaxation_time_2"] == held["relaxation_time_1"])
    model = flat.PronyDeckModel(deck, BOX)
    assert "relaxation_strength_2" not in model.layer(0)          # the writer's
    assert "relaxation_strength_1" in model.layer(0)


# -- the step of a deck sampled from a model -----------------------------------------

def test_the_default_knots_are_spaced_by_a_fraction_of_the_span():
    model = crust().hollowed(50e3)
    deck = sample_deck(model, ["rho"], spacing=1.0 / 50.0)
    gaps = np.diff(deck.radius)
    assert gaps[gaps > 0].max() <= (200e3 - 50e3) / 50.0 * (1 + 1e-12)
    assert deck.radius[0] == 50e3 and deck.radius[-1] == 200e3


def test_the_example_of_the_module_docstring_reads():
    lines = ["a crust over a mantle", "depth rho vp vs qkappa qmu",
             "# layers crust mantle", "# interfaces surface moho",
             "0       2800  6400  3700  nan  nan", "30000   2800  6400  3700  nan  nan",
             "30000   3300  8000  4500  nan  nan", "200000  3300  8000  4500  nan  nan"]
    model = flat.ElasticDeckModel(read_deck(lines, flat.ELASTIC), BOX)
    assert [lay.name for lay in model.layers] == ["crust", "mantle"]
    assert model.layer("mantle")["L"](100e3) == 3300.0 * 4500.0 ** 2
    assert "qmu" not in model.layer("crust") and model.name == "a crust over a mantle"
    check_model(model)
