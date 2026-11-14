"""Prony branches: the vocabulary's families, freezing a generalised
Maxwell body, the Maxwell, Burgers and fitted Andrade series against
their closed forms, the branches on a model, its refusals, its units and
the model file."""
import math

import numpy as np
import pytest

import planetmodel as pm
from planetmodel import rheology, vocabulary
from planetmodel.conversion import prony
from planetmodel.units import DIMENSIONLESS, TIME


def viscous_prem():
    prem = pm.PREM()
    for lay in prem.layers:
        lo, hi = lay.interval
        if 3480e3 <= lo and hi <= 6291e3:
            prem = prem.with_field(lay.index, "viscosity", pm.constant_field(
                (lo, hi), 2e21 if hi <= 5701e3 else 5e20, name="viscosity"))
    return prem


def test_the_vocabulary_s_families():
    assert vocabulary.member("relaxation_time_3") == ("relaxation_time", 3)
    assert vocabulary.member("relaxation_strength_12") == ("relaxation_strength", 12)
    for name in ("relaxation_time_0", "relaxation_time_03", "relaxation_time_",
                 "relaxation_time", "relaxation_time_x", "viscosity_1", "vs"):
        assert vocabulary.member(name) is None, name
    assert vocabulary.shipped("relaxation_time_7").dimensions == TIME
    assert vocabulary.spec("relaxation_strength_2").dimensions == DIMENSIONLESS
    assert vocabulary.shipped("vs") is vocabulary.VOCABULARY["vs"]
    with pytest.raises(KeyError, match="families"):
        vocabulary.spec("relaxation_time_0")
    assert rheology.is_rheology_name("relaxation_time_2")
    assert rheology.is_rheology_name("viscosity")
    assert not rheology.is_rheology_name("vsv")


def test_one_branch_freezes_as_the_viscosity_does():
    m = viscous_prem()
    b = prony.with_branches(m, prony.maxwell_series())
    i = m.layer("lower_mantle").index
    assert "viscosity" not in b.layer(i) and "relaxation_time_1" in b.layer(i)
    r = np.linspace(*m.layer(i).interval, 7)
    for omega in (2 * np.pi / (1000 * 3.15e7), 2 * np.pi / 3.15e9):
        a = m.frozen(omega).layer(i)
        c = b.frozen(omega).layer(i)
        for k in ("A", "C", "F", "L", "N"):
            assert np.allclose(c[k](r), a[k](r), rtol=1e-13, atol=0), k
    pm.testing.check_model(b)


def test_burgers_is_its_closed_form():
    strength, ratio = 5.0, 5.0
    s = prony.burgers_series(strength=strength, viscosity_ratio=ratio)
    assert len(s.strengths) == 2 and sum(s.strengths) == pytest.approx(1.0, abs=1e-15)
    w = np.logspace(-4, 4, 41)
    # J / J_M with mu_M = eta_M = 1, mu_K = 1 / strength, eta_K = 1 / ratio
    j = 1.0 + 1.0 / (1j * w) + 1.0 / (1.0 / strength + 1j * w / ratio)
    assert np.allclose(s.modulus(w), 1.0 / j, rtol=1e-13, atol=1e-15)
    # on a model: mu(omega) = mu_U / J(i omega) with the layer's own Maxwell time
    m = viscous_prem()
    b = prony.with_branches(m, s, layers=["lower_mantle"])
    i = m.layer("lower_mantle").index
    r = np.array([5000e3])
    omega = 2 * np.pi / (100 * 3.15e7)
    mu = pm.kappa_mu(m.layer(i))[1](r)
    tau = 2e21 / mu
    want = mu / (1.0 + 1.0 / (1j * omega * tau)
                 + 1.0 / (1.0 / strength + 1j * omega * tau / ratio))
    frozen = b.frozen(omega).layer(i)
    assert np.allclose(frozen["N"](r) - m.layer(i)["N"](r) + mu, want, rtol=1e-12)
    with pytest.raises(ValueError, match="positive"):
        prony.burgers_series(strength=0.0, viscosity_ratio=1.0)


def test_an_andrade_fit_holds_its_misfit_and_flows():
    s = prony.andrade_series(1.0 / 3.0)
    assert s.misfit < 0.02
    assert sum(s.strengths) == pytest.approx(1.0, abs=1e-6)
    assert sum(w * c for w, c in zip(s.strengths, s.times)) == pytest.approx(1.0,
                                                                            abs=1e-6)
    w = np.logspace(-3, 3, 61)
    g = 1.0 + math.gamma(4.0 / 3.0) * (1j * w) ** (-1.0 / 3.0) + 1.0 / (1j * w)
    assert np.max(np.abs(s.modulus(w) * g - 1.0)) <= s.misfit * 1.01
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        prony.andrade_series(1.0)
    with pytest.raises(ValueError, match="band"):
        prony.fitted_series(lambda x: 1.0 + 0 * x, band=(1.0, 0.1))


def test_the_series_refuses_what_is_not_a_body():
    with pytest.raises(ValueError, match="more than the whole"):
        prony.PronySeries((0.7, 0.5), (1.0, 2.0), name="x")
    with pytest.raises(ValueError, match="positive"):
        prony.PronySeries((0.5,), (0.0,), name="x")
    with pytest.raises(ValueError, match="as many"):
        prony.PronySeries((0.5,), (1.0, 2.0), name="x")


def test_freezing_refuses_a_muddled_layer():
    m = viscous_prem()
    i = m.layer("lower_mantle").index
    lay = m.layer(i)
    one = pm.constant_field(lay.interval, 0.5, name="relaxation_strength_1")
    time = pm.constant_field(lay.interval, 1e10, name="relaxation_time_1")
    both = m.with_field(i, "relaxation_strength_1", one).with_field(
        i, "relaxation_time_1", time)
    with pytest.raises(ValueError, match="viscosity and Prony branches"):
        both.frozen(1e-9)
    alone = both.without_field("viscosity", layers=[i]).without_field(
        "relaxation_time_1", layers=[i])
    with pytest.raises(ValueError, match="branch 1 holds"):
        alone.frozen(1e-9)
    gap = both.without_field("viscosity", layers=[i]).with_field(
        i, "relaxation_strength_3", one).with_field(i, "relaxation_time_3", time)
    with pytest.raises(ValueError, match=r"numbered \[1, 3\]"):
        gap.frozen(1e-9)


def test_branches_are_rheology_and_have_units():
    m = prony.with_branches(viscous_prem(), prony.burgers_series(strength=2.0,
                                                                 viscosity_ratio=10.0))
    i = m.layer("lower_mantle").index
    assert "relaxation_time_2" not in m.elastic().layer(i)
    assert m.layer(i).fields.keys() >= {"relaxation_strength_1", "relaxation_time_2"}
    nd = m.nondimensionalised()
    r = np.array([5000e3])
    k = nd.scales.factor(TIME)
    assert np.allclose(k * nd.layer(i)["relaxation_time_2"](r / nd.scales.length),
                       m.layer(i)["relaxation_time_2"](r), rtol=1e-12)
    with pytest.raises(ValueError, match="frozen"):
        m.frozen(1e-9).with_field(i, "relaxation_time_1", pm.constant_field(
            m.layer(i).interval, 1.0, name="relaxation_time_1"), replace=True)
    with pytest.raises(ValueError, match="no solid layer"):
        prony.with_branches(pm.PREM(), prony.maxwell_series())


def test_a_laterally_varying_time_is_a_log_perturbation():
    from planetmodel.conversion import building, parameters
    m = viscous_prem()
    f = pm.AnalyticField((3480e3, 5701e3), lambda r, t, p: 0.02 * np.cos(t) + 0 * r * p)
    for n in ("vsv", "vsh"):
        m = pm.perturbed(m, n, f, kind="relative")
    dt = building.temperature_anomaly(building.velocity_anomaly(m),
                                      scaling=parameters.SL3D_DENSITY_SCALING,
                                      expansivity=parameters.SL3D_EXPANSIVITY)
    m = building.with_viscosity(m, dt, parameters.MILNE_2018_VISCOSITY)
    b = prony.with_branches(m, prony.burgers_series(strength=5.0, viscosity_ratio=5.0))
    i = m.layer("lower_mantle").index
    tau = b.layer(i)["relaxation_time_2"]
    assert isinstance(tau, pm.PerturbedField) and tau.kind == "log"
    r, t, p = np.array([5000e3, 4000e3]), np.array([0.3, 2.0]), np.array([1.0, 4.0])
    c = prony.burgers_series(strength=5.0, viscosity_ratio=5.0).times[1]
    want = c * m.layer(i)["viscosity"](r, t, p) / pm.kappa_mu(m.layer(i))[1](r, t, p)
    assert np.allclose(tau(r, t, p), want, rtol=1e-12)
    flat = prony.with_branches(viscous_prem(), prony.maxwell_series())
    assert not isinstance(flat.layer(i)["relaxation_time_1"], pm.PerturbedField)


@pytest.mark.netcdf
def test_branches_write_and_read_back(tmp_path):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.modelfile import read_model, write_model
    m = prony.with_branches(viscous_prem(), prony.burgers_series(strength=5.0,
                                                                 viscosity_ratio=5.0))
    path = write_model(tmp_path / "prony.nc", m, grid=pm.gauss_legendre(2))
    pm.testing.check_model_file(path, m)
    back = read_model(path)
    i = m.layer("lower_mantle").index
    assert back.spec("relaxation_time_1").dimensions == TIME
    assert len(rheology.branches(back.layer(i))) == 2


def _sl3dpp_burgers(mu, eta, kelvin_modulus, kelvin_viscosity):
    """sl3dpp's BurgersPronyTerms, transcribed line for line from its C++
    (its `Rheology.cpp`): the (modulus, rate) of the slower term, then of
    the faster."""
    a, b, c = mu / eta, mu / kelvin_viscosity, kelvin_modulus / kelvin_viscosity
    gap = math.sqrt((a - c) * (a - c) + b * b + 2.0 * b * (a + c))
    fast = 0.5 * (a + b + c + gap)
    slow = a * c / fast

    def plus(s, product):
        return gap + s if s >= 0.0 else 4.0 * product / (gap - s)

    above_slow = c * 0.5 * plus(b + c - a, a * b) / fast
    below_fast = 0.5 * plus(a + b - c, b * c)
    return (mu * above_slow / gap, slow), (mu * below_fast / gap, fast)


def test_burgers_terms_are_the_burgers_body():
    """The two terms against the body's own modulus in the Laplace
    variable, 1 / M(s) = 1 / mu + 1 / (eta s) + 1 / (f mu + eta_K s), over
    ten decades of s and Kelvin elements far faster and far slower than
    the Maxwell time; the unrelaxed modulus and the steady viscosity
    held; and the ratios' series, `burgers_series`, the same terms."""
    rng = np.random.default_rng(5)
    n = 200
    mu = 10 ** rng.uniform(9.0, 11.5, n)
    eta = 10 ** rng.uniform(17.0, 23.0, n)
    eta_k = eta * 10 ** rng.uniform(-4.0, 2.0, n)
    f = 10 ** rng.uniform(-2.0, 1.0, n)
    w, tau = prony.burgers_terms(mu, eta, kelvin_viscosity=eta_k,
                                 kelvin_modulus_fraction=f)
    assert w.shape == tau.shape == (2, n)
    assert np.all(w > 0.0) and np.all(tau[0] < tau[1])
    assert np.allclose(w.sum(axis=0), 1.0, rtol=0, atol=2e-15)
    assert np.allclose(w[0] * tau[0] + w[1] * tau[1], eta / mu, rtol=1e-13, atol=0)
    tau_m = eta / mu
    for x in np.logspace(-5, 5, 21):
        s = x / tau_m
        want = 1.0 / (1.0 / mu + 1.0 / (eta * s) + 1.0 / (f * mu + eta_k * s))
        got = mu * (w[0] * s / (s + 1.0 / tau[0]) + w[1] * s / (s + 1.0 / tau[1]))
        assert np.allclose(got, want, rtol=1e-12, atol=0), x
    # a Kelvin element a fixed fraction of the Maxwell one: the ratios' series
    series = prony.burgers_series(strength=1.0 / f[0],
                                  viscosity_ratio=eta[0] / eta_k[0])
    assert np.allclose(w[:, 0], series.strengths, rtol=1e-13, atol=0)
    assert np.allclose(tau[:, 0], np.array(series.times) * tau_m[0], rtol=1e-13, atol=0)
    for bad in ({"kelvin_viscosity": 0.0, "kelvin_modulus_fraction": 1.0},
                {"kelvin_viscosity": 1e18, "kelvin_modulus_fraction": -1.0},
                {"kelvin_viscosity": np.inf, "kelvin_modulus_fraction": 1.0}):
        with pytest.raises(ValueError, match="positive and finite"):
            prony.burgers_terms(3e10, 1e19, **bad)


def test_burgers_parameters_invert_burgers_terms():
    """The Kelvin element and the viscosity of two branches against the
    Burgers body they came from, over the same ranges as above (Kelvin
    elements far faster and far slower than the Maxwell time), in either
    order; any two branches without a relaxed part, against the modulus
    of the body found, 1 / M(s) = 1 / mu + 1 / (eta s) + 1 / (f mu +
    eta_K s), over ten decades of s; and the refusals."""
    rng = np.random.default_rng(7)
    n = 200
    mu = 10 ** rng.uniform(9.0, 11.5, n)
    eta = 10 ** rng.uniform(17.0, 23.0, n)
    eta_k = eta * 10 ** rng.uniform(-4.0, 2.0, n)
    f = 10 ** rng.uniform(-2.0, 1.0, n)
    w, tau = prony.burgers_terms(mu, eta, kelvin_viscosity=eta_k,
                                 kelvin_modulus_fraction=f)
    for order in ([0, 1], [1, 0]):
        back = prony.burgers_parameters(mu, w[order], tau[order])
        for got, want in zip(back, (eta, eta_k, f)):
            assert np.allclose(got, want, rtol=1e-13, atol=0)
    # any two branches whose strengths sum to one are a Burgers body
    w1 = rng.uniform(0.01, 0.99, n)
    t = np.sort(10 ** rng.uniform(6.0, 12.0, (2, n)), axis=0)
    t[1] = t[1] * 1.001
    strengths = np.stack([w1, 1.0 - w1])
    v, vk, fk = prony.burgers_parameters(mu, strengths, t)
    assert np.all(v > 0) and np.all(vk > 0) and np.all(fk > 0)
    for x in np.logspace(-5, 5, 21):
        s = x / (v / mu)
        want = 1.0 / (1.0 / mu + 1.0 / (v * s) + 1.0 / (fk * mu + vk * s))
        got = mu * (strengths[0] * s / (s + 1.0 / t[0])
                    + strengths[1] * s / (s + 1.0 / t[1]))
        assert np.allclose(got, want, rtol=1e-9, atol=0), x
    # the ratios' series: Delta = 1 / f, eta_M / eta_K
    series = prony.burgers_series(strength=5.0, viscosity_ratio=10.0)
    v, vk, fk = prony.burgers_parameters(1.0, series.strengths, series.times)
    assert np.allclose([v, vk, fk], [1.0, 0.1, 0.2], rtol=1e-13, atol=0)
    with pytest.raises(ValueError, match="positive and finite"):
        prony.burgers_parameters(1.0, [0.5, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="positive and finite"):
        prony.burgers_parameters(np.inf, [0.5, 0.5], [1.0, 2.0])
    with pytest.raises(ValueError, match="same time"):
        prony.burgers_parameters(1.0, [0.5, 0.5], [2.0, 2.0])
    with pytest.raises(ValueError, match=r"shape \(2, ...\)"):
        prony.burgers_parameters(1.0, [0.2, 0.3, 0.5], [1.0, 2.0, 3.0])


def test_burgers_terms_reproduce_the_post_seismic_example():
    """sl3dpp's post-seismic example: a Burgers asthenosphere of steady
    viscosity 1e19 Pa s, its Kelvin element of the layer's shear modulus
    and a tenth of its viscosity, over the asthenosphere of PREM made
    isotropic (Voigt; its low-velocity zone, between 220 and 80 km
    depth), as sl3dpp's BurgersPronyTerms forms the terms."""
    prem = pm.PREM(ocean=False).isotropic()
    r = np.linspace(6151e3, 6291e3, 9)
    mu = np.asarray(prem.layer(prem.skeleton.locate(6200e3).layer)["mu"](r))
    w, tau = prony.burgers_terms(mu, 1e19, kelvin_viscosity=1e18,
                                 kelvin_modulus_fraction=1.0)
    for i, m in enumerate(mu):
        (mu_slow, rate_slow), (mu_fast, rate_fast) = _sl3dpp_burgers(
            float(m), 1e19, float(m), 1e18)
        assert np.allclose([w[0, i] * m, 1.0 / tau[0, i], w[1, i] * m, 1.0 / tau[1, i]],
                           [mu_fast, rate_fast, mu_slow, rate_slow], rtol=1e-14, atol=0)
    year = 3.15576e7
    # the Maxwell time eta / mu is 4.7 to 5 years there; the transient's
    # relaxation, months
    assert 4.5 < float(np.min(1e19 / mu)) / year < float(np.max(1e19 / mu)) / year < 5.5
    assert 0.1 < float(np.max(tau[0])) / year < 0.5


def test_with_burgers_makes_named_layers_burgers_bodies():
    m = viscous_prem()
    b = prony.with_burgers(m, ["lower_mantle", "transition_zone_middle"],
                           kelvin_viscosity=2e20, kelvin_modulus_fraction=0.5)
    for name in ("lower_mantle", "transition_zone_middle"):
        i = m.layer(name).index
        lay = b.layer(i)
        assert "viscosity" not in lay and len(rheology.branches(lay)) == 2
        r = np.linspace(*lay.interval, 5)
        eta = m.layer(i)["viscosity"](r)
        mu = pm.kappa_mu(m.layer(i))[1](r)
        w, tau = prony.burgers_terms(mu, eta, kelvin_viscosity=2e20,
                                     kelvin_modulus_fraction=0.5)
        for k in range(2):
            assert np.array_equal(lay[f"relaxation_strength_{k + 1}"](r), w[k])
            assert np.array_equal(lay[f"relaxation_time_{k + 1}"](r), tau[k])
        # the frozen modulus is the body's own, mu / (mu J(i omega))
        omega = 2 * np.pi / (30 * 3.15e7)
        frozen = b.frozen(omega).layer(i)
        s = 1j * omega
        want = 1.0 / (1.0 / mu + 1.0 / (eta * s) + 1.0 / (0.5 * mu + 2e20 * s))
        assert np.allclose(frozen["N"](r) - m.layer(i)["N"](r) + mu, want, rtol=1e-12)
    # the other mantle layers keep their Maxwell viscosity
    upper = m.layer("transition_zone_upper").index
    assert "viscosity" in b.layer(upper)
    # a Kelvin element a fixed fraction of a constant viscosity: the series
    lower = m.layer("lower_mantle").index
    c = prony.with_branches(m, prony.burgers_series(strength=2.0, viscosity_ratio=10.0),
                            layers=["lower_mantle"])
    r = np.linspace(*m.layer(lower).interval, 5)
    for key in ("relaxation_strength_1", "relaxation_time_1", "relaxation_strength_2",
                "relaxation_time_2"):
        assert np.allclose(b.layer(lower)[key](r), c.layer(lower)[key](r), rtol=1e-13,
                           atol=0), key
    pm.testing.check_model(b)
    with pytest.raises(KeyError, match="holds no 'viscosity'"):
        prony.with_burgers(m, ["upper_crust"], kelvin_viscosity=1e18,
                           kelvin_modulus_fraction=1.0)
    with pytest.raises(ValueError, match="fluid"):
        prony.with_burgers(m.with_field(m.layer("outer_core").index, "viscosity",
                                        pm.constant_field(m.layer("outer_core").interval,
                                                          1.0, name="viscosity")),
                           ["outer_core"], kelvin_viscosity=1e18,
                           kelvin_modulus_fraction=1.0)
    for bad in ({"kelvin_viscosity": 0.0, "kelvin_modulus_fraction": 1.0},
                {"kelvin_viscosity": 1e18, "kelvin_modulus_fraction": np.nan}):
        with pytest.raises(ValueError, match="positive number"):
            prony.with_burgers(m, ["lower_mantle"], **bad)
    with pytest.raises(TypeError):
        prony.with_burgers(m, "lower_mantle", kelvin_viscosity=1e18,
                           kelvin_modulus_fraction=1.0)
    with pytest.raises(TypeError):
        prony.with_burgers(m, ["lower_mantle"], kelvin_viscosity=1e18)


def test_a_burgers_model_tabulates_and_reads_back_as_a_deck(tmp_path):
    """A deck of the Maxwell columns and the two branches: the Burgers
    layers hold the branches and no viscosity, the Maxwell layers the
    viscosity and no branches, and the deck read back is the same body."""
    from planetmodel.deck import (MAXWELL_COLUMNS, DeckFormat, deck_layers, read_deck,
                                  sample_deck, write_deck)
    m = prony.with_burgers(viscous_prem(), ["lower_mantle"], kelvin_viscosity=2e20,
                           kelvin_modulus_fraction=0.5)
    branches = tuple(f"{stem}_{k}" for k in (1, 2) for stem in rheology.BRANCH_STEMS)
    burgers = DeckFormat(MAXWELL_COLUMNS + branches, name="burgers")
    path = write_deck(tmp_path / "burgers.deck", sample_deck(m, burgers.columns),
                      burgers)
    deck = read_deck(path, burgers)
    skeleton, layers = deck_layers(deck, kinds={n: "linear" for n in
                                                ("viscosity",) + branches})
    i = m.layer("lower_mantle").index
    assert set(branches) <= set(layers[i]) and "viscosity" not in layers[i]
    upper = m.layer("transition_zone_upper").index
    assert "viscosity" in layers[upper] and not set(branches) & set(layers[upper])
    r = deck.radius[deck.layers()[i]]          # its knots, eight digits written
    for name in branches:
        assert np.allclose(layers[i][name](r), m.layer(i)[name](r), rtol=1e-7), name
    assert len(rheology.branches(layers[i])) == 2
