"""Tables of depths: a table brought in as a flat model over a box, held
to the model contract, and written back character for character on its
own knots; the names given from the surface down; the base padded below
the last knot exactly; the knots named by their depth in every message;
and what is refused."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel import (Deck, Elastic, Viscoelastic, deck_layers, model_from_deck,
                         read_deck, sample_deck, write_deck)
from planetmodel import cartesian as flat
from planetmodel.conversion import prony
from planetmodel.deck import (MINEOS_COLUMNS, MINEOS_ISOTROPIC_COLUMNS, RADIUS,
                              deck_knots)
from planetmodel.layerfunction import ExponentialLayer
from planetmodel.testing import check_model

BOX = flat.Box([(0.0, 400e3), (-200e3, 200e3)])
YEAR = 3.15576e7

#: The elastic columns of a transversely isotropic body beside a viscosity.
VISCOUS = ("rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity")

#: A viscosity joined in its logarithm.
LOG = {"viscosity": "log-linear"}


def crust(**kw):
    """The reference flat model, its asthenosphere a Maxwell body."""
    return flat.CrustOverMantle(BOX, viscosity=1e20, **kw)


def imported(deck, **kw):
    """The flat model of a deck sampled from a model, over `BOX`, with that
    model's names."""
    kw.setdefault("behaviours", [Elastic, Viscoelastic])
    return model_from_deck(deck, coordinate=flat.DEPTH, domain=BOX,
                           layer_names=deck.header["layers"],
                           interface_names=deck.header["interfaces"],
                           name=deck.header["name"], **kw)


def round_trip(tmp_path, deck, columns, stem, **kw):
    """The deck written as a table of `columns`, read and made a model with
    the deck's names, and written back from the model on the table's knots:
    the model and both texts."""
    first = write_deck(tmp_path / f"{stem}.deck", deck, columns)
    read = read_deck(first, columns)
    model = imported(Deck(read.radius, read.columns, header=deck.header), **kw)
    again = write_deck(tmp_path / f"{stem}.again",
                       sample_deck(model, columns, knots=deck_knots(read)), columns)
    return model, first.read_text(), again.read_text()


# -- the coordinate ---------------------------------------------------------------

def test_depth_is_the_coordinate_of_a_flat_model():
    assert flat.DEPTH.word == "depth" and RADIUS.word == "r"
    assert flat.DEPTH.base is flat.Model and RADIUS.base is pm.Model
    assert flat.DEPTH.field_type is flat.DepthField
    assert RADIUS.field_type is pm.RadialField


# -- round trips, character for character -----------------------------------------

def test_a_viscous_table_of_depths_round_trips(tmp_path):
    model, first, again = round_trip(tmp_path, sample_deck(crust(), VISCOUS), VISCOUS,
                                     "viscous", kinds=LOG)
    assert again == first
    assert float(first.splitlines()[0].split()[0]) == 0.0      # the surface, first
    assert isinstance(model, flat.Model)
    assert model.behaviours == ("Elastic", "Viscoelastic")
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
    omega = 2.0 * np.pi / (100.0 * YEAR)
    d = np.linspace(100e3, 200e3, 5)
    assert np.allclose(model.frozen(omega).layer(-1)["N"](d),
                       crust().frozen(omega).layer(-1)["N"](d), rtol=1e-7, atol=0)


def test_an_elastic_table_of_depths_round_trips_isotropic_and_not(tmp_path):
    iso = crust().elastic()
    model, first, again = round_trip(tmp_path,
                                     sample_deck(iso, MINEOS_ISOTROPIC_COLUMNS),
                                     MINEOS_ISOTROPIC_COLUMNS, "iso",
                                     behaviours=[Elastic, pm.ConstantQ])
    assert again == first
    assert "qmu" not in model.layer(0)                    # NaN throughout: absent
    check_model(model)
    # a transversely isotropic table with quality factors, the shear one
    # absent from a fluid layer on top
    layered = flat.LayeredIsotropicElastic(
        [0.0, 4e3, 30e3, 100e3], BOX, rho=[1020.0, 2800.0, 3300.0],
        vp=[1450.0, 6400.0, [8000.0, 1e-3]], vs=[0.0, 3700.0, 4500.0],
        interface_names=["surface", "seafloor", "moho"], name="an ocean")
    deck = sample_deck(layered, MINEOS_COLUMNS)
    q = {"qkappa": np.full(deck.nknots, 57823.0), "qmu": np.full(deck.nknots, 600.0)}
    q["qmu"][deck.layers()[0]] = np.nan
    deck = Deck(deck.radius, {**deck.columns, **q}, header=deck.header)
    vti, first, again = round_trip(tmp_path, deck, MINEOS_COLUMNS, "vti",
                                   behaviours=[Elastic, pm.ConstantQ])
    assert again == first
    assert vti.is_fluid(0) and "qmu" not in vti.layer(0) and "qmu" in vti.layer(1)
    omega = 2.0 * np.pi / 20.0
    assert np.iscomplexobj(vti.moduli_at(1, omega)["L"](10e3))
    check_model(vti)


def test_a_table_of_branches_round_trips(tmp_path):
    """Prony branches are columns like any other: the strengths joined by
    straight lines and the times in their logarithm, by their stems."""
    series = prony.PronySeries((0.3, 0.2), (0.5, 5.0), name="two")
    model = prony.with_branches(crust(), series)
    assert "viscosity" not in model.layer(-1)
    columns = VISCOUS[:-1] + ("relaxation_strength_1", "relaxation_time_1",
                              "relaxation_strength_2", "relaxation_time_2")
    read, first, again = round_trip(
        tmp_path, sample_deck(model, columns), columns, "prony",
        kinds={"relaxation_strength": "linear", "relaxation_time": "log-linear"})
    assert again == first
    assert len(pm.rheology.branches(read.layer(-1))) == 2
    assert not pm.rheology.branches(read.layer(0))
    check_model(read)
    omega = 2.0 * np.pi / (100.0 * YEAR)
    d = np.linspace(100e3, 200e3, 5)
    assert np.allclose(read.frozen(omega).layer(-1)["N"](d),
                       model.frozen(omega).layer(-1)["N"](d), rtol=1e-7, atol=0)


def test_a_written_table_of_depths_reads_and_writes_back_as_it_stands(tmp_path):
    text = ("# two layers: depth rho vpv vsv vph vsh eta viscosity (SI)\n"
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
    deck = read_deck(path, VISCOUS, header_lines=1)
    model = model_from_deck(deck, behaviours=[Elastic, Viscoelastic], kinds=LOG,
                            coordinate=flat.DEPTH, domain=flat.Rectangle((0.0, 1e6)),
                            layer_names=[None, "mantle"],
                            interface_names=["surface", None])
    assert [lay.name for lay in model.layers] == [None, "mantle"]
    assert [f.name for f in model.geometry.interfaces] == ["surface", None]
    # the viscosity is joined in its logarithm: geometric between knots
    eta = model.layer("mantle")["viscosity"]
    assert isinstance(eta.function, ExponentialLayer)
    assert np.isclose(eta(40e3), np.sqrt(1e21 * 1e20), rtol=1e-12)
    again = write_deck(tmp_path / "again.deck",
                       sample_deck(model, VISCOUS, knots=deck_knots(deck)), VISCOUS,
                       header=deck.header["lines"])
    assert again.read_text() == text
    check_model(model)


# -- the base, and padding below it -------------------------------------------------

def test_padding_below_the_last_knot_holds_its_values_exactly():
    deck = sample_deck(crust(), VISCOUS)
    model = imported(deck, kinds=LOG)
    base = float(deck.radius[-1])
    assert model.skeleton.boundaries[-1] == base
    padded = model.extended([base + 300e3, base + 800e3], fields="hold",
                            layer_names=["pad", "far"])
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


# -- names, the box and messages ----------------------------------------------------

def _rows(*rows):
    return [" ".join(str(v) for v in row) for row in rows]


TWO = _rows((0, 2700, 6000, 3500, 6000, 3500, 1, "nan"),
            (20e3, 2700, 6000, 3500, 6000, 3500, 1, "nan"),
            (20e3, 3300, 8000, 4500, 8000, 4500, 1, 1e21),
            (60e3, 3300, 8000, 4500, 8000, 4500, 1, 1e20))


def test_the_names_given_run_from_the_surface_down_one_interface_a_layer():
    deck = read_deck(TWO, VISCOUS)
    named = model_from_deck(deck, coordinate=flat.DEPTH, domain=BOX,
                            layer_names=["crust", "mantle"],
                            interface_names=["top", "moho"])
    assert [lay.name for lay in named.layers] == ["crust", "mantle"]
    assert named.geometry.interface("top").radius == 0.0
    assert named.geometry.interface("moho").radius == 20e3
    plain = model_from_deck(deck, coordinate=flat.DEPTH, domain=BOX)
    assert type(plain) is flat.Model
    assert [f.name for f in plain.geometry.interfaces] == [None, None]
    with pytest.raises(ValueError, match="interface"):
        model_from_deck(deck, coordinate=flat.DEPTH, domain=BOX,
                        interface_names=["a", "b", "c"])


def test_a_table_of_depths_needs_its_box():
    deck = read_deck(TWO, VISCOUS)
    with pytest.raises(ValueError, match="a table of depths holds no lateral extent"):
        model_from_deck(deck, coordinate=flat.DEPTH)
    with pytest.raises(TypeError, match="expected a Box or a Rectangle"):
        model_from_deck(deck, coordinate=flat.DEPTH,
                        domain=((0.0, 1.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="a table of radii .* takes no lateral domain"):
        model_from_deck(deck, domain=BOX)
    model = model_from_deck(deck, coordinate=flat.DEPTH, domain=BOX)
    assert model.geometry.domain == BOX


def test_the_knots_are_named_by_their_depth_in_messages():
    deck = read_deck(TWO, VISCOUS)
    bad = np.array(deck["viscosity"])
    bad[2] = 0.0
    zero = Deck(deck.radius, {**deck.columns, "viscosity": bad})
    with pytest.raises(ValueError, match="is 0 at depth = 20000 on layer 1"):
        deck_layers(zero, kinds=LOG, coordinate=flat.DEPTH)
    with pytest.raises(ValueError, match="is 0 at depth = 20000 on layer 1"):
        model_from_deck(zero, kinds=LOG, coordinate=flat.DEPTH, domain=BOX)
    with pytest.raises(ValueError, match="is 0 at r = 20000 on layer 1"):
        deck_layers(zero, kinds=LOG)


# -- the step of a deck sampled from a model -----------------------------------------

def test_the_default_knots_are_spaced_by_a_fraction_of_the_span():
    model = crust().hollowed(50e3)
    deck = sample_deck(model, ["rho"], spacing=1.0 / 50.0)
    gaps = np.diff(deck.radius)
    assert gaps[gaps > 0].max() <= (200e3 - 50e3) / 50.0 * (1 + 1e-12)
    assert deck.radius[0] == 50e3 and deck.radius[-1] == 200e3


def test_the_example_of_the_module_docstring_reads():
    lines = ["depth rho vp vs",
             "0       2800  6400  3700", "30000   2800  6400  3700",
             "30000   3300  8000  4500", "200000  3300  8000  4500"]
    deck = read_deck(lines, ["rho", "vp", "vs"], header_lines=1)
    model = model_from_deck(deck, coordinate=flat.DEPTH, domain=BOX,
                            behaviours=[Elastic], layer_names=["crust", "mantle"],
                            interface_names=["surface", "moho"])
    assert [lay.name for lay in model.layers] == ["crust", "mantle"]
    assert model.layer("mantle")["L"](100e3) == 3300.0 * 4500.0 ** 2
    assert model.geometry.interface("moho").radius == 30e3
    check_model(model)
