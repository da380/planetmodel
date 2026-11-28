"""Rheology from a layer's fields, and a model frozen at a frequency."""
import numpy as np
import pytest

from planetmodel import (constant_field, frozen, frozen_moduli, LayeredIsotropicElastic,
                         is_viscoelastic, kappa_mu, moduli, PREM, testing)
from planetmodel.units import FREQUENCY


def test_is_viscoelastic():
    model = PREM(ocean=False)
    assert all(is_viscoelastic(layer) for layer in model.layers)   # qkappa everywhere
    plain = LayeredIsotropicElastic.homogeneous(1.0, rho=1.0, vp=2.0, vs=1.0)
    assert not is_viscoelastic(plain.layer(0))
    assert frozen_moduli(plain.layer(0), 1.0, reference_omega=1.0)["A"].dtype == float


def test_maxwell_layer_limits():
    rho, mu, a, eta = 5500.0, 1.0e11, 6371e3, 1e21
    model = LayeredIsotropicElastic.homogeneous(a, rho=rho, vp=8000.0,
                                                vs=np.sqrt(mu / rho))
    model = model.with_field(0, "viscosity",
                             constant_field((0.0, a), eta, name="viscosity"))
    layer = model.layer(0)
    base = moduli(layer)
    kappa, mu_f = kappa_mu(layer)
    r = np.array([1e6, 4e6])
    tau = eta / mu
    fast = frozen_moduli(layer, 1e8 / tau, reference_omega=1.0)
    assert fast["L"].dtype == complex
    assert np.allclose(fast["L"](r), base["L"](r), rtol=1e-7)
    assert np.allclose(fast["A"](r), base["A"](r), rtol=1e-7)
    slow = frozen_moduli(layer, 1e-8 / tau, reference_omega=1.0)
    assert np.allclose(slow["L"](r), 0.0, atol=1e-7 * mu)
    assert np.allclose(slow["A"](r), kappa(r), rtol=1e-7)
    mid = frozen_moduli(layer, 1.0 / tau, reference_omega=1.0)
    z = 1j
    want = mu * z / (1.0 + z)
    assert np.allclose(mid["N"](r), want)
    assert np.allclose(mid["F"](r), kappa(r) - 2.0 * want / 3.0)
    for f in mid.values():
        testing.check_field(f)


def test_constant_q_band_on_prem():
    model = PREM(ocean=False)
    omega0 = 2.0 * np.pi
    layer = model.layer("lower_mantle")
    kappa, mu = kappa_mu(layer)
    base = moduli(layer)
    r = np.array([4.0e6, 5.0e6])
    at_ref = frozen_moduli(layer, omega0, reference_omega=omega0)
    qmu, qkappa = layer["qmu"](r), layer["qkappa"](r)
    assert np.allclose(at_ref["L"](r).real, base["L"](r))
    assert np.allclose(at_ref["L"](r).imag, mu(r) / qmu)
    assert np.allclose(at_ref["A"](r).imag,
                       kappa(r) / qkappa + 4.0 * mu(r) / (3.0 * qmu))
    assert np.allclose(at_ref["F"](r).imag,
                       kappa(r) / qkappa - 2.0 * mu(r) / (3.0 * qmu))
    assert np.allclose(at_ref["C"](r).real - at_ref["A"](r).real,
                       base["C"](r) - base["A"](r))               # anisotropy kept real
    lower = frozen_moduli(layer, omega0 / np.e, reference_omega=omega0)
    assert np.allclose(lower["L"](r).real, base["L"](r) - (2.0 / np.pi) * mu(r) / qmu)
    core = model.layer("outer_core")
    fc = frozen_moduli(core, omega0, reference_omega=omega0)
    assert np.allclose(fc["L"](r[:1] * 0.6), 0.0) and fc["A"].dtype == complex


def test_frozen_model():
    model = PREM(ocean=False)
    omega = 2.0 * np.pi / 43200.0
    cold = frozen(model, omega)
    testing.check_model(cold)
    assert cold.constant("omega") == omega and cold.G == model.G
    for layer in cold.layers:
        assert all(n in layer for n in ("A", "C", "F", "L", "N"))
        assert layer["A"].dtype == complex
        assert "vpv" in layer
    # dispersion from the 1 s reference to 12 h softens the mantle by a per cent
    warm = moduli(model.layer("lower_mantle"))["A"](5e6)
    cold_A = moduli(cold.layer("lower_mantle"))["A"](5e6)
    assert 0.97 < cold_A.real / warm < 0.995 and cold_A.imag > 0.0
    nd = frozen(model.nondimensionalised(), omega / 3.0)
    assert nd.constant("omega") == omega / 3.0
    assert np.isclose(nd.in_si().constant("omega"),
                      omega / 3.0 * nd.scales.factor(FREQUENCY))
    plain = LayeredIsotropicElastic.homogeneous(1.0, rho=1.0, vp=2.0, vs=1.0)
    same = frozen(plain, 3.0)
    assert same.constant("omega") == 3.0 and same.layer(0)["vp"].dtype == float
    assert same.layer(0)["A"].dtype == float
    with pytest.raises(ValueError):
        frozen(model, 0.0)
    with pytest.raises(ValueError):
        frozen(model, 1.0, reference_omega=-1.0)


def test_a_frozen_model_is_not_frozen_again():
    cold = frozen(PREM(ocean=False), 1e-3)
    with pytest.raises(ValueError, match="already frozen"):
        frozen(cold, 1e-3)


def test_a_frozen_model_refuses_what_would_leave_its_moduli_stale():
    """Replacing rho, a velocity or a Q field would rebuild or outdate the
    complex moduli, and coarsening would refit them from the real
    velocities; a field the moduli do not depend on is still attached."""
    from planetmodel.rheology import is_frozen

    warm = PREM(ocean=False)
    cold = frozen(warm, 2 * np.pi / 3600.0)
    assert is_frozen(cold) and not is_frozen(warm)
    lay = cold.layer("lower_mantle")
    iv = lay.interval
    for name in ("rho", "vsv", "L", "qmu"):
        with pytest.raises(ValueError, match=f"frozen at omega.*derived from '{name}'"):
            cold.with_field("lower_mantle", name, lay[name], replace=True)
    with pytest.raises(ValueError, match="frozen at omega.*viscosity"):
        cold.with_field("lower_mantle", "viscosity",
                        constant_field(iv, 1e21, name="viscosity"))
    with pytest.raises(ValueError, match="frozen at omega.*coarsening"):
        cold.coarsened(drop=["d771"])
    tagged = cold.with_field("lower_mantle", "tracer", constant_field(iv, 1.0))
    assert tagged.layer("lower_mantle")["L"] is lay["L"]
    # the way out the refusal names: change the elastic model, freeze again
    vsv = warm.layer("lower_mantle")["vsv"]
    slow = warm.with_field("lower_mantle", "vsv", 0.9 * vsv, replace=True)
    assert frozen(slow, 2 * np.pi / 3600.0).layer("lower_mantle")["L"].dtype == complex


def test_a_frozen_model_takes_the_names_of_power_law_creep():
    """Freezing reads the linear rheology alone, so the names of power-law
    creep are attached to a frozen model as any other field is, and the
    names a linear rheology is read from are the ones refused."""
    from planetmodel.character import DENSITY
    from planetmodel.rheology import (POWER_LAW_NAMES, is_linear_rheology_name,
                                      is_rheology_name)

    warm = PREM(ocean=False)
    omega = 2 * np.pi / 3600.0
    cold = frozen(warm, omega)
    lay = cold.layer("lower_mantle")
    iv = lay.interval
    creep = {"stress_exponent": constant_field(iv, 3.5, name="stress_exponent"),
             "transition_stress": constant_field(iv, 1e6, character=DENSITY,
                                                 name="transition_stress")}
    assert tuple(creep) == POWER_LAW_NAMES
    held, creeping = warm, cold
    for name, field in creep.items():
        assert is_rheology_name(name) and not is_linear_rheology_name(name)
        creeping = creeping.with_field("lower_mantle", name, field)
        held = held.with_field("lower_mantle", name, field)
    assert all(name in creeping.layer("lower_mantle") for name in creep)
    assert creeping.layer("lower_mantle")["L"] is lay["L"]
    testing.check_model(creeping)
    # freezing a model that holds them gives the moduli of the one that does not
    r = np.linspace(*iv, 7)
    again = frozen(held, omega).layer("lower_mantle")
    for n in ("A", "C", "F", "L", "N"):
        assert again[n].dtype == complex and np.array_equal(again[n](r), lay[n](r))
    # a branch's names are a linear rheology's, and refused like its others
    for name in ("viscosity", "qmu", "qkappa", "relaxation_strength_1",
                 "relaxation_time_2"):
        assert is_rheology_name(name) and is_linear_rheology_name(name)
    with pytest.raises(ValueError, match="derived from 'relaxation_time_1'"):
        cold.with_field("lower_mantle", "relaxation_time_1",
                        constant_field(iv, 1e10, name="relaxation_time_1"))


def test_moduli_conversions_keep_complex_numbers():
    from planetmodel.materials import kappa_mu_from_moduli, moduli_from_velocities

    cold = frozen(PREM(ocean=False), 1e-3).layer("lower_mantle")
    r = 5e6
    five = [cold[n](r) for n in ("A", "C", "F", "L", "N")]
    kappa, mu = kappa_mu_from_moduli(*five)
    k_field, mu_field = kappa_mu(cold)
    assert np.iscomplexobj(mu) and np.isclose(mu, mu_field(r), rtol=1e-12)
    assert np.isclose(kappa, k_field(r), rtol=1e-12)
    out = moduli_from_velocities(1.0, 2.0 + 0.1j, 1.0)
    assert np.isclose(out["C"], (2.0 + 0.1j) ** 2)
