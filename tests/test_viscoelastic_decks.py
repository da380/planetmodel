"""The prony and burgers decks: the formats written and read back, a
Maxwell deck as a one-branch prony deck of the same body, a Burgers body
as two branches and as the Kelvin element it was made from, three
branches with a relaxed part beside elastic layers and a layer of fewer
branches, a burgers deck's branches against their closed form, the
model types read from both, and every refusal by its message."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel import (BURGERS, MAXWELL, PRONY, BurgersDeckModel, Deck,
                         MaxwellDeckModel, PronyDeckModel, deck_layers, read_deck,
                         sample_deck, write_deck)
from planetmodel.conversion import prony
from planetmodel.deck import (BURGERS_COLUMNS, BURGERS_KINDS, MAXWELL_COLUMNS,
                              MAXWELL_KINDS, PRONY_COLUMNS, PRONY_KINDS,
                              STRENGTH_ROUNDING, prony_branches, prony_columns)
from planetmodel.fields import constant_field
from planetmodel.layerfunction import ExponentialLayer, PolynomialLayer

YEAR = 3.15576e7

#: Laplace variables, as multiples of a layer's Maxwell time's inverse.
DECADES = np.logspace(-4.0, 4.0, 17)


def _maxwell_prem():
    """PREM without its ocean, a viscosity on every mantle layer below the
    low-velocity zone: 1e22 Pa s below 670 km, 1e21 above."""
    model = pm.PREM(ocean=False)
    for i, layer in enumerate(model.layers):
        if 2 <= i <= 7:
            eta = 1e22 if i <= 4 else 1e21
            field = constant_field(layer.interval, eta, name="viscosity")
            model = model.with_field(i, "viscosity", field)
    return model


def _modulus(layer, r, s):
    """The layer's shear modulus in the Laplace variable, M(s) at the
    radii `r` (rows) and the values `s` (columns): mu s tau / (1 + s tau)
    of a viscosity, mu (1 - sum_k w_k / (1 + s tau_k)) of Prony branches."""
    mu = pm.kappa_mu(layer)[1](r)[:, None]
    s = np.asarray(s)[None, :]
    if "viscosity" in layer:
        tau = (layer["viscosity"](r) / pm.kappa_mu(layer)[1](r))[:, None]
        return mu * s * tau / (1.0 + s * tau)
    out = np.ones(np.broadcast_shapes(mu.shape, s.shape))
    for w, tau in pm.rheology.branches(layer):
        out = out - w(r)[:, None] / (1.0 + s * tau(r)[:, None])
    return mu * out


def _viscous(model):
    return [lay.index for lay in model.layers
            if "viscosity" in lay or pm.rheology.branches(lay)]


# -- a Maxwell deck as one branch -----------------------------------------------

def test_a_maxwell_deck_is_a_one_branch_prony_deck_of_the_same_body(tmp_path):
    """At every knot the one branch is the Maxwell body to the eight digits
    of the time written, M(s) over eight decades of s about the Maxwell
    time and the moduli frozen at glacial and seismic periods.  Between
    knots the time is geometric in the knots' eta / mu, the viscosity being
    constant through each layer, where the Maxwell deck's is eta over mu's
    cubic: they differ by the curvature of log mu between the knots, at
    second order in their spacing, 8.2e-5 at most on these knots, about
    32 km apart."""
    path = write_deck(tmp_path / "m.deck",
                      sample_deck(_maxwell_prem(), MAXWELL_COLUMNS), MAXWELL)
    maxwell = MaxwellDeckModel(path)
    deck = prony.prony_deck(maxwell)
    assert deck.names == prony_columns(1) and prony_branches(deck) == 1
    assert np.array_equal(deck.radius, read_deck(path, MAXWELL).radius)
    one = PronyDeckModel(write_deck(tmp_path / "p.deck", deck, PRONY))
    assert one.columns == prony_columns(1)
    assert [lay.name for lay in one.layers] == [lay.name for lay in maxwell.layers]
    viscous = _viscous(maxwell)
    assert viscous == _viscous(one) == [2, 3, 4, 5, 6, 7]
    worst_knots = worst_between = 0.0
    for i in viscous:
        r = maxwell.knots[i]
        assert np.array_equal(one.layer(i)["relaxation_strength_1"](r), np.ones(r.size))
        tau = maxwell.layer(i)["viscosity"](r) / pm.kappa_mu(maxwell.layer(i))[1](r)
        s = DECADES / np.median(tau)
        a, b = _modulus(maxwell.layer(i), r, s), _modulus(one.layer(i), r, s)
        worst_knots = max(worst_knots, float(np.max(np.abs(b / a - 1.0))))
        middle = 0.5 * (r[1:] + r[:-1])
        time = one.layer(i)["relaxation_time_1"]
        written = one.deck["relaxation_time_1"][one.deck.layers()[i]]
        geometric = np.sqrt(written[1:] * written[:-1])
        assert np.allclose(time(middle), geometric, rtol=1e-14, atol=0)
        between = (time(middle) / (maxwell.layer(i)["viscosity"](middle)
                                   / pm.kappa_mu(maxwell.layer(i))[1](middle)))
        worst_between = max(worst_between, float(np.max(np.abs(between - 1.0))))
    assert worst_knots < 1e-7                     # eight digits of tau
    assert 1e-5 < worst_between < 1e-4            # the curvature of log mu
    for omega in (2.0 * np.pi / (1000.0 * YEAR), 2.0 * np.pi / 100.0):
        fa, fb = maxwell.frozen(omega), one.frozen(omega)
        for i in viscous:
            r = maxwell.knots[i]
            assert np.allclose(fb.layer(i)["N"](r), fa.layer(i)["N"](r), rtol=1e-7,
                               atol=0), (omega, i)
    pm.testing.check_model(one)
    again = write_deck(tmp_path / "again.deck", one.to_deck(), PRONY)
    assert again.read_text() == (tmp_path / "p.deck").read_text()


# -- a Burgers body as two branches, and as its Kelvin element --------------------

def _burgers_prem():
    """`_maxwell_prem` with its lower mantle and middle transition zone
    Burgers bodies, a Kelvin element of 2e20 Pa s and half the shear
    modulus; the other viscous layers Maxwell bodies."""
    return prony.with_burgers(_maxwell_prem(),
                              ["lower_mantle", "transition_zone_middle"],
                              kelvin_viscosity=2e20, kelvin_modulus_fraction=0.5)


def test_a_burgers_model_is_a_two_branch_prony_deck(tmp_path):
    """The Burgers layers' two branches as the model holds them; a Maxwell
    layer's one branch completed with a strength of 0 and its own time;
    read back, each layer holds the branches it was written from, and the
    body frozen at a period is the model's to the eight digits written."""
    model = _burgers_prem()
    deck = prony.prony_deck(model)
    assert prony_branches(deck) == 2 and deck.names == prony_columns(2)
    burgers = [model.layer(n).index for n in ("lower_mantle", "transition_zone_middle")]
    for i, s in enumerate(deck.layers()):
        r = deck.radius[s]
        if i in burgers:
            for k in (1, 2):
                for stem in ("relaxation_strength", "relaxation_time"):
                    name = f"{stem}_{k}"
                    assert np.allclose(deck[name][s], model.layer(i)[name](r),
                                       rtol=1e-15, atol=0)
        elif "viscosity" in model.layer(i):
            assert np.array_equal(deck["relaxation_strength_1"][s], np.ones(r.size))
            assert np.array_equal(deck["relaxation_strength_2"][s], np.zeros(r.size))
            assert np.array_equal(deck["relaxation_time_2"][s],
                                  deck["relaxation_time_1"][s])
        else:
            assert np.isnan(deck["relaxation_time_2"][s]).all()
    path = write_deck(tmp_path / "b.prony", deck, PRONY)
    back = PronyDeckModel(path)
    for i in _viscous(model):
        held = len(pm.rheology.branches(back.layer(i)))
        assert held == (2 if i in burgers else 1), i
    omega = 2.0 * np.pi / (30.0 * YEAR)
    fm, fb = model.frozen(omega), back.frozen(omega)
    for i in _viscous(model):
        r = back.knots[i]
        assert np.allclose(fb.layer(i)["N"](r), fm.layer(i)["N"](r), rtol=1e-7, atol=0)
    again = write_deck(tmp_path / "again.prony", back.to_deck(), PRONY)
    assert again.read_text() == path.read_text()


def test_a_burgers_deck_round_trips_the_kelvin_element(tmp_path):
    """Written as a burgers deck the Burgers layers give back the Kelvin
    element they were made from, digit for digit, and the steady viscosity;
    read back, the model holds their two branches, `burgers_terms` of the
    interpolated columns everywhere, the body's own modulus at the knots,
    and writes the same deck; a Maxwell layer holds its viscosity."""
    model = _burgers_prem()
    deck = prony.burgers_deck(model)
    assert deck.names == BURGERS_COLUMNS
    burgers = [model.layer(n).index for n in ("lower_mantle", "transition_zone_middle")]
    plain = _maxwell_prem()
    for i, s in enumerate(deck.layers()):
        r = deck.radius[s]
        if i in burgers:
            assert np.allclose(deck["kelvin_viscosity"][s], 2e20, rtol=1e-12, atol=0)
            assert np.allclose(deck["kelvin_modulus_fraction"][s], 0.5, rtol=1e-12,
                               atol=0)
            assert np.allclose(deck["viscosity"][s], plain.layer(i)["viscosity"](r),
                               rtol=1e-12, atol=0)
        else:
            assert np.isnan(deck["kelvin_viscosity"][s]).all()
            assert (np.isnan(deck["viscosity"][s]).all()
                    == ("viscosity" not in model.layer(i)))
    path = write_deck(tmp_path / "b.burgers", deck, BURGERS)
    text = path.read_text().splitlines()
    assert text[1].split() == ["r", *BURGERS_COLUMNS]
    rows = [line.split() for line in text[2:] if not line.startswith("#")]
    kelvin = {(row[-2], row[-1]) for row in rows if row[-1] != "nan"}
    assert kelvin == {("2e+20", "0.5")}
    read = BurgersDeckModel(path)
    _, raw = deck_layers(read.deck, kinds=BURGERS_KINDS)
    for i in burgers:
        lay = read.layer(i)
        assert "viscosity" not in lay and len(pm.rheology.branches(lay)) == 2
        lo, hi = lay.interval
        r = np.concatenate([read.knots[i], np.linspace(lo, hi, 23)])
        eta, eta_k, f = (raw[i][n](r) for n in BURGERS_COLUMNS[-3:])
        mu = pm.kappa_mu(lay)[1](r)
        w, tau = prony.burgers_terms(mu, eta, kelvin_viscosity=eta_k,
                                     kelvin_modulus_fraction=f)
        for k in (1, 2):
            assert np.allclose(lay[f"relaxation_strength_{k}"](r), w[k - 1], rtol=1e-14,
                               atol=0)
            assert np.allclose(lay[f"relaxation_time_{k}"](r), tau[k - 1], rtol=1e-14,
                               atol=0)
        omega = 2.0 * np.pi / (30.0 * YEAR)
        knots = read.knots[i]
        mu = pm.kappa_mu(lay)[1](knots)
        s = 1j * omega
        want = 1.0 / (1.0 / mu + 1.0 / (eta[:knots.size] * s)
                      + 1.0 / (f[:knots.size] * mu + eta_k[:knots.size] * s))
        got = read.frozen(omega).layer(i)["N"](knots) - lay["N"](knots) + mu
        assert np.allclose(got, want, rtol=1e-12, atol=0)
    assert "viscosity" in read.layer("transition_zone_upper")
    pm.testing.check_model(read)
    again = write_deck(tmp_path / "again.burgers", read.to_deck(), BURGERS)
    assert again.read_text() == path.read_text()


def test_a_burgers_deck_of_varying_columns_is_a_prony_deck_of_the_same_body(tmp_path):
    """A burgers deck whose three columns vary by orders of magnitude across
    its layers, read and written back unchanged, and written as a prony
    deck of two branches: the same body at every knot to the eight digits
    written, over eight decades of s."""
    base = sample_deck(pm.PREM(ocean=False), PRONY_COLUMNS)
    columns = {n: np.full(base.nknots, np.nan) for n in BURGERS_COLUMNS[-3:]}
    for i, s in enumerate(base.layers()):
        x = (base.radius[s] - base.radius[s][0]) / np.ptp(base.radius[s])
        if 2 <= i <= 4:
            columns["viscosity"][s] = 10.0 ** (21.0 + 2.0 * x ** 2)
        elif 5 <= i <= 8:
            columns["viscosity"][s] = 10.0 ** (20.0 + x)
            columns["kelvin_viscosity"][s] = 10.0 ** (19.0 + x)
            columns["kelvin_modulus_fraction"][s] = 0.5 + 0.5 * x
    deck = Deck(base.radius, {**base.columns, **columns},
                header={"name": "varying", **{k: base.header[k] for k in
                                              ("layers", "interfaces")}})
    path = write_deck(tmp_path / "v.burgers", deck, BURGERS)
    model = BurgersDeckModel(path)
    assert write_deck(tmp_path / "w.burgers", model.to_deck(),
                      BURGERS).read_text() == path.read_text()
    branches = PronyDeckModel(write_deck(tmp_path / "v.prony", prony.prony_deck(model),
                                         PRONY))
    assert branches.columns == prony_columns(2)
    for i in range(2, 9):
        r = model.knots[i]
        s = DECADES / float(np.median(columns["viscosity"][base.layers()[i]]) / 7e10)
        a, b = _modulus(model.layer(i), r, s), _modulus(branches.layer(i), r, s)
        assert np.allclose(b, a, rtol=1e-7, atol=0), i
        assert len(pm.rheology.branches(branches.layer(i))) == (1 if i <= 4 else 2)


# -- three branches, a relaxed part, elastic layers, fewer branches ----------------

def test_three_branches_with_a_relaxed_part_beside_fewer_and_none(tmp_path):
    """Three branches summing to 0.6 on the lower layers, so 0.4 of the
    shear modulus stays at long times; one, a viscosity, on the layers
    above, written with strengths of 0 and the time of its branch; elastic
    and fluid layers NaN.  Read back, every field is the model's at the
    knots to the eight digits written, the layer of one branch holds one,
    and the long-time modulus is the relaxed part."""
    series = prony.PronySeries((0.3, 0.2, 0.1), (0.1, 1.0, 10.0), name="three")
    model = prony.with_branches(_maxwell_prem(), series, layers=[2, 3, 4])
    deck = prony.prony_deck(model)
    assert prony_branches(deck) == 3
    path = write_deck(tmp_path / "three.prony", deck, PRONY)
    back = PronyDeckModel(path)
    assert back.columns == prony_columns(3)
    for i, lay in enumerate(model.layers):
        held = pm.rheology.branches(lay)
        n = len(held) if held else (1 if "viscosity" in lay else 0)
        assert len(pm.rheology.branches(back.layer(i))) == n, i
        r = back.knots[i]
        for k in range(1, n + 1):
            for stem in ("relaxation_strength", "relaxation_time"):
                name = f"{stem}_{k}"
                want = deck[name][deck.layers()[i]]
                assert np.allclose(back.layer(i)[name](r), want, rtol=1e-7, atol=0)
        if n == 1:              # the writer's completion, in the text
            s = deck.layers()[i]
            assert np.array_equal(deck["relaxation_strength_3"][s], np.zeros(r.size))
            line = path.read_text().splitlines()[4 + s.start]
            values = line.split()
            assert values[9] == values[11] == "0"
            assert values[10] == values[12] == values[8]
    for i in (2, 3, 4):
        r = back.knots[i]
        total = sum(back.layer(i)[f"relaxation_strength_{k}"](r) for k in (1, 2, 3))
        assert np.allclose(total, 0.6, rtol=0, atol=1e-15)
        slow = back.frozen(1e-20).layer(i)
        mu = pm.kappa_mu(back.layer(i))[1](r)
        assert np.allclose(slow["N"](r) - back.layer(i)["N"](r) + mu, 0.4 * mu,
                           rtol=1e-6, atol=0)
    assert not pm.rheology.branches(back.layer("lid"))
    assert not pm.rheology.branches(back.layer("outer_core"))
    again = write_deck(tmp_path / "again.prony", back.to_deck(), PRONY)
    assert again.read_text() == path.read_text()


# -- the formats' rules ----------------------------------------------------------

_TWO = ("r rho vpv vsv vph vsh eta relaxation_strength_1 relaxation_time_1 "
        "relaxation_strength_2 relaxation_time_2")
_GOOD = ("0 4 8 4 8 4 1 0.5 1e9 0.25 1e11", "1 4 8 4 8 4 1 0.5 1e9 0.25 1e11")


def _prony(line=_TWO, rows=_GOOD):
    return read_deck(["a body", line, *rows], PRONY)


def _burgers(*rows, line=" ".join(("r",) + BURGERS_COLUMNS)):
    return read_deck(["a body", line, *rows], BURGERS)


@pytest.mark.parametrize("line, rows, says", [
    ("r rho vp vs relaxation_strength_1 relaxation_time_1", _GOOD,
     "a prony deck's second line names its columns, r rho vpv vsv vph vsh eta then "
     "relaxation_strength_k relaxation_time_k for each branch k = 1 to N; got "
     "'r rho vp"),
    ("r rho vpv vsv vph vsh eta viscosity", _GOOD,
     "'viscosity' stands where 'relaxation_strength_1' belongs"),
    ("r rho vpv vsv vph vsh eta", ("0 4 8 4 8 4 1", "1 4 8 4 8 4 1"),
     "a prony deck has at least one branch; its column line names none after eta"),
    ("r rho vpv vsv vph vsh eta relaxation_time_1 relaxation_strength_1", _GOOD,
     "a prony deck's branch columns come in pairs, relaxation_strength_k "
     "relaxation_time_k for k = 1, 2, ... in order; 'relaxation_time_1' stands where "
     "'relaxation_strength_1' belongs"),
    ("r rho vpv vsv vph vsh eta relaxation_strength_1 relaxation_time_1 "
     "relaxation_strength_3 relaxation_time_3", _GOOD,
     "'relaxation_strength_3' stands where 'relaxation_strength_2' belongs"),
    ("r rho vpv vsv vph vsh eta relaxation_strength_1 relaxation_time_1 "
     "relaxation_strength_2", _GOOD, "branch 2 has no 'relaxation_time_2'"),
    ("r rho vpv vsv vph vsh eta relaxation_strength_1 relaxation_time_1", _GOOD,
     "the column line names 8 columns after the radius; the rows have 10"),
    (_TWO, ("0 4 8 4 8 4 1 0.5 1e9 0.25", "1 4 8 4 8 4 1 0.5 1e9 0.25"),
     "then 2 for each index k = 1 to N"),
])
def test_a_prony_column_line_is_refused_saying_what_is_wrong(line, rows, says):
    with pytest.raises(ValueError) as caught:
        _prony(line, rows)
    assert says in str(caught.value)


@pytest.mark.parametrize("rows, says", [
    (("0 4 8 4 8 4 1 0.5 1e9 nan nan", "1 4 8 4 8 4 1 0.5 1e9 nan nan"),
     "the deck: layer 0: holds nan in some of its branch columns and numbers in "
     "others"),
    (("0 4 8 4 8 4 1 0.5 1e9 0 1e11", "1 4 8 4 8 4 1 nan 1e9 0 1e11"),
     "holds nan in some of its branch columns"),
    (("0 4 8 4 8 4 1 1.5 1e9 0 1e11", "1 4 8 4 8 4 1 0.5 1e9 0 1e11"),
     "layer 0: relaxation_strength_1 is 1.5 at r = 0; a strength lies in [0, 1]"),
    (("0 4 8 4 8 4 1 0.5 1e9 0 1e11", "1 4 8 4 8 4 1 0.5 1e9 -0.1 1e11"),
     "relaxation_strength_2 is -0.1 at r = 1; a strength lies in [0, 1]"),
    (("0 4 8 4 8 4 1 0.5 1e9 0 0", "1 4 8 4 8 4 1 0.5 1e9 0 1e11"),
     "relaxation_time_2 is 0 at r = 0; a time is positive and finite"),
    (("0 4 8 4 8 4 1 0.5 inf 0 1", "1 4 8 4 8 4 1 0.5 1e9 0 1e11"),
     "relaxation_time_1 is inf at r = 0"),
    (("0 4 8 4 8 4 1 0.6 1e9 0.5 1", "1 4 8 4 8 4 1 0.5 1e9 0 1e11"),
     "layer 0: the strengths sum to 1.1 at r = 0, more than one"),
])
def test_a_prony_deck_breaking_its_rules_is_refused_where_it_is(rows, says):
    with pytest.raises(ValueError) as caught:
        _prony(rows=rows)
    assert says in str(caught.value)


def test_a_prony_deck_within_its_rules_is_read():
    d = _prony()
    assert prony_branches(d) == 2 and d.names == prony_columns(2)
    # strengths of 0 and of 1, and a sum past one within the rounding of eight
    # digits a branch
    _prony(rows=("0 4 8 4 8 4 1 1 1e9 0 1", "1 4 8 4 8 4 1 0.50000001 1e9 0.5 1e11"))
    elastic = _prony(rows=("0 4 8 4 8 4 1 nan nan nan nan",
                           "1 4 8 4 8 4 1 nan nan nan nan"))
    assert not deck_layers(elastic)[1][0].keys() & set(prony_columns(2)[6:])
    with pytest.raises(ValueError, match="more than one"):
        _prony(rows=("0 4 8 4 8 4 1 1 1e9 0 1",
                     f"1 4 8 4 8 4 1 0.5 1e9 {0.5 + 3 * STRENGTH_ROUNDING!r} 1e11"))


@pytest.mark.parametrize("rows, says", [
    (("0 4 8 4 8 4 1 nan 1e20 0.5", "1 4 8 4 8 4 1 nan 1e20 0.5"),
     "layer 0: holds ['kelvin_viscosity', 'kelvin_modulus_fraction']; a layer of a "
     "burgers deck holds a viscosity alone (a Maxwell body), a viscosity and both "
     "Kelvin columns (a Burgers body), or none of the three (elastic)"),
    (("0 4 8 4 8 4 1 1e21 1e20 nan", "1 4 8 4 8 4 1 1e21 1e20 nan"),
     "holds ['viscosity', 'kelvin_viscosity']"),
    (("0 4 8 4 8 4 1 1e21 1e20 0.5", "1 4 8 4 8 4 1 1e21 nan 0.5"),
     "kelvin_viscosity is nan on part of the layer"),
    (("0 4 8 4 8 4 1 1e21 1e20 0.5", "1 4 8 4 8 4 1 1e21 1e20 -2"),
     "kelvin_modulus_fraction is -2 at r = 1; it is positive and finite"),
    (("0 4 8 4 8 4 1 0 nan nan", "1 4 8 4 8 4 1 1e21 nan nan"),
     "viscosity is 0 at r = 0; it is positive and finite"),
])
def test_a_burgers_deck_breaking_its_rules_is_refused(rows, says):
    with pytest.raises(ValueError) as caught:
        _burgers(*rows)
    assert says in str(caught.value)
    _burgers("0 4 8 4 8 4 1 1e21 1e20 0.5", "1 4 8 4 8 4 1 1e21 1e20 0.5")
    _burgers("0 4 8 4 8 4 1 1e21 nan nan", "1 4 8 4 8 4 1 1e21 nan nan")
    _burgers("0 4 8 4 8 4 1 nan nan nan", "1 4 8 4 8 4 1 nan nan nan")
    with pytest.raises(ValueError,
                       match="a burgers deck's second line names its columns"):
        _burgers("0 1 1 1 1 1 1 1", "1 1 1 1 1 1 1 1",
                 line="r rho vpv vsv vph vsh eta viscosity")


#: A wrong column line of each viscoelastic deck and the whole refusal, which
#: the C++ says character for character (cpp/tests/test_deck.cpp).
COLUMN_LINE_REFUSALS = (
    (MAXWELL, "r rho vp vs viscosity # a note",
     "a Maxwell deck's second line names its columns, r rho vpv vsv vph vsh eta "
     "viscosity; got 'r rho vp vs viscosity # a note'"),
    (PRONY, "r rho vp vs relaxation_strength_1 relaxation_time_1",
     "a prony deck's second line names its columns, r rho vpv vsv vph vsh eta then "
     "relaxation_strength_k relaxation_time_k for each branch k = 1 to N; got "
     "'r rho vp vs relaxation_strength_1 relaxation_time_1'"),
    (BURGERS, "r rho vpv vsv vph vsh eta viscosity",
     "a burgers deck's second line names its columns, r rho vpv vsv vph vsh eta "
     "viscosity kelvin_viscosity kelvin_modulus_fraction; got "
     "'r rho vpv vsv vph vsh eta viscosity'"),
)


@pytest.mark.parametrize("format_, line, says", COLUMN_LINE_REFUSALS)
def test_a_wrong_column_line_is_refused_quoting_it(format_, line, says):
    with pytest.raises(ValueError) as caught:
        read_deck(["a body", line, "0 4 8 4 8 4 1 1e21", "1 4 8 4 8 4 1 1e21"], format_)
    assert str(caught.value) == says


def test_writing_holds_a_deck_to_its_format(tmp_path):
    """A layer of branches that do not run from 1, a value out of bounds,
    a branch without its partner, none at all: refused on writing as on
    reading; a layer of branches 1 to m completed."""
    nan = np.nan
    elastic = {n: [4.0, 4.0] for n in PRONY_COLUMNS}

    def written(**branches):
        deck = Deck([0.0, 1.0], {**elastic, **branches})
        return write_deck(tmp_path / "x", deck, PRONY)

    with pytest.raises(ValueError, match="holds nan in some of its branch columns"):
        written(relaxation_strength_1=[nan, nan], relaxation_time_1=[nan, nan],
                relaxation_strength_2=[0.5, 0.5], relaxation_time_2=[1.0, 1.0])
    with pytest.raises(ValueError, match=r"relaxation_strength_1 is 2 at r = 0"):
        written(relaxation_strength_1=[2.0, 0.5], relaxation_time_1=[1.0, 1.0])
    with pytest.raises(ValueError) as caught:
        written(relaxation_strength_1=[0.5, 0.5], relaxation_time_1=[1.0, 1.0],
                relaxation_strength_2=[0.5, 0.5])
    assert ("a prony deck holds relaxation_strength_k, relaxation_time_k for k = 1 to "
            "N, N at least one, each of them; it has ['relaxation_strength_1', "
            "'relaxation_time_1', 'relaxation_strength_2']") in str(caught.value)
    with pytest.raises(ValueError, match="each of them; it has none"):
        written()
    path = written(relaxation_strength_1=[0.5, 0.5], relaxation_time_1=[3.0, 4.0],
                   relaxation_strength_2=[nan, nan], relaxation_time_2=[nan, nan])
    back = read_deck(path, PRONY)
    assert list(back["relaxation_strength_2"]) == [0.0, 0.0]
    assert list(back["relaxation_time_2"]) == [3.0, 4.0]
    with pytest.raises(ValueError, match="positive and finite"):
        write_deck(tmp_path / "y", Deck([0.0, 1.0], {
            **{n: [4.0, 4.0] for n in MAXWELL_COLUMNS[:-1]}, "viscosity": [-1.0, 1.0],
            "kelvin_viscosity": [nan, nan], "kelvin_modulus_fraction": [nan, nan]}),
            BURGERS)
    with pytest.raises(ValueError, match="at least one"):
        prony_columns(0)
    with pytest.raises(ValueError, match="relaxation_strength_1 is 2"):
        PronyDeckModel(Deck([0.0, 1.0], {**elastic, "relaxation_strength_1": [2.0, 0.5],
                                          "relaxation_time_1": [1.0, 1.0]}))


def test_the_conversions_refuse_what_is_no_such_body():
    model = _maxwell_prem()
    with pytest.raises(ValueError, match="a prony deck has at least one branch"):
        prony.prony_deck(pm.PREM())
    lower = model.layer("lower_mantle")
    both = model.with_field(lower.index, "relaxation_strength_1", constant_field(
        lower.interval, 1.0, name="relaxation_strength_1")).with_field(
        lower.index, "relaxation_time_1", constant_field(lower.interval, 1e10,
                                                         name="relaxation_time_1"))
    for convert in (prony.prony_deck, prony.burgers_deck):
        with pytest.raises(ValueError, match="holds a viscosity and Prony branches"):
            convert(both)
    three = prony.with_branches(model, prony.PronySeries(
        (0.3, 0.3, 0.4), (0.1, 1.0, 10.0), name="three"), layers=["lower_mantle"])
    with pytest.raises(ValueError,
                       match="holds 3 Prony branches; a Burgers body has two"):
        prony.burgers_deck(three)
    relaxed = prony.with_branches(model, prony.PronySeries(
        (0.3, 0.3), (0.1, 1.0), name="relaxed"), layers=["lower_mantle"])
    with pytest.raises(ValueError, match="sum to 0.6 at r = .*, not one; a body with a "
                                         "relaxed part is no Maxwell or Burgers body"):
        prony.burgers_deck(relaxed)
    solid = prony.with_branches(model, prony.PronySeries((0.9,), (1.0,), name="solid"),
                                layers=["lower_mantle"])
    with pytest.raises(ValueError, match="sum to 0.9"):
        prony.burgers_deck(solid)
    # one branch of strength one is a Maxwell body, written with its viscosity
    one = prony.with_branches(model, prony.maxwell_series(), layers=["lower_mantle"])
    d = prony.burgers_deck(one)
    s = d.layers()[lower.index]
    assert np.allclose(d["viscosity"][s], 1e22, rtol=1e-13, atol=0)
    assert np.isnan(d["kelvin_viscosity"][s]).all()


def test_the_viscoelastic_decks_join_times_and_viscosities_in_their_logarithm():
    assert dict(MAXWELL_KINDS) == {"viscosity": "log-linear"}
    assert dict(PRONY_KINDS) == {"relaxation_strength": "linear",
                                 "relaxation_time": "log-linear"}
    assert dict(BURGERS_KINDS) == {"viscosity": "log-linear",
                                   "kelvin_viscosity": "log-linear",
                                   "kelvin_modulus_fraction": "linear"}


def test_a_family_is_joined_by_the_kind_its_stem_is_given():
    d = Deck([0.0, 0.3, 0.5, 1.0], {"rho": [5.0, 4.0, 4.5, 3.0],
                                    "relaxation_strength_1": [0.1, 0.9, 0.2, 0.8],
                                    "relaxation_time_1": [1e8, 1e12, 1e9, 1e11]})
    _, cubic = deck_layers(d)
    _, joined = deck_layers(d, kinds=PRONY_KINDS)
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
        deck_layers(Deck([0.0, 1.0], {"rho": [1.0, 2.0]}), kinds=PRONY_KINDS)
