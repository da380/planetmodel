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
                             constant_field(eta, (0.0, a), name="viscosity"))
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
