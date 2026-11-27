"""The base of the lithosphere found in a model's fields: at an isotherm,
a viscosity and a Maxwell time against fields whose base is known in
closed form, at its edges, as geometry through `moved`, and as a lid."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel.conversion import lithosphere
from planetmodel.datasets import moved

TOP = 6368e3
GRID = pm.equiangular(18, 36)


def gradient(t):
    """dT/d(depth) in K/m, 16 K/km at the north pole, 8 at the south."""
    return 0.012 + 0.004 * np.cos(t)


def warm(model):
    """The model with T = 300 + G(theta) (TOP - r) on every layer above
    5701 km."""
    for lay in model.layers:
        lo, hi = lay.interval
        if lo >= 5701e3:
            model = model.with_field(lay.index, "temperature", pm.AnalyticField(
                (lo, hi), lambda r, t, p: 300.0 + gradient(t) * (TOP - r) + 0 * p,
                name="temperature"))
    return model


def nodes():
    t = np.concatenate([np.repeat(GRID.colatitudes, GRID.nphi), [0.0, np.pi]])
    p = np.concatenate([np.tile(GRID.longitudes, GRID.ntheta), [0.0, 0.0]])
    return t, p


def test_an_isotherm_is_where_the_closed_form_puts_it():
    m = warm(pm.PREM(ocean=False))
    base = lithosphere.isotherm(m, 1573.0, grid=GRID)
    t, p = nodes()
    want = TOP - 1273.0 / gradient(t)
    assert np.allclose(base.radius(t, p), want, rtol=0, atol=1e-2)
    assert base.measures == "radius" and "1573 K isotherm" in base.name
    # hot at once below the top, and never hot within the layers asked
    assert np.allclose(lithosphere.isotherm(m, 250.0, grid=GRID).radius(t, p), TOP)
    deep = lithosphere.isotherm(m, 1e5, grid=GRID)
    assert np.allclose(deep.radius(t, p), 5701e3)
    with pytest.raises(ValueError, match="no layer holds"):
        lithosphere.isotherm(pm.PREM(), 1573.0, grid=GRID)
    with pytest.raises(ValueError, match="not a temperature"):
        lithosphere.isotherm(m, 1573.0, grid=GRID, name="rho")


def viscous(model, scale=30e3):
    """eta = 1e25 exp(-(TOP - r) / H(theta)) on every solid layer above
    5701 km, H = scale (1 + 0.5 cos theta)."""
    for lay in model.layers:
        lo, hi = lay.interval
        if lo >= 5701e3:
            model = model.with_field(lay.index, "viscosity", pm.AnalyticField(
                (lo, hi), lambda r, t, p: 1e25 * np.exp(
                    -(TOP - r) / (scale * (1 + 0.5 * np.cos(t)))) + 0 * p,
                name="viscosity"))
    return model


def test_a_viscosity_threshold_and_a_maxwell_time():
    m = viscous(pm.PREM(ocean=False))
    t, p = nodes()
    h = 30e3 * (1 + 0.5 * np.cos(t))
    base = lithosphere.viscosity_threshold(m, 1e23, grid=GRID)
    assert np.allclose(base.radius(t, p), TOP - h * np.log(100.0), rtol=0, atol=1e-2)
    # eta / mu = 1e4 years: eta = 1e4 yr mu, mu read from the layer it lies in
    year = 365.25 * 86400.0
    mx = lithosphere.maxwell_threshold(m, 1e4 * year, grid=GRID)
    r = mx.radius(t, p)
    lay = [m.layer(m.skeleton.locate(x).layer) for x in r[:3]]
    for x, tt, pp, L in zip(r[:3], t[:3], p[:3], lay):
        mu = pm.kappa_mu(L)[1](np.array([x]), np.array([tt]), np.array([pp]))[0]
        eta = L["viscosity"](np.array([x]), np.array([tt]), np.array([pp]))[0]
        assert eta / mu == pytest.approx(1e4 * year, rel=1e-6)


def test_the_base_as_geometry_and_as_a_lid():
    m = viscous(pm.PREM(ocean=False), scale=25e3)
    base = lithosphere.viscosity_threshold(m, 1e23, grid=GRID)
    plain, _ = pm.PREM(ocean=False).coarsened(drop=["d80", "d220", "moho", "d15"])
    plain = plain.refined([6271e3], interface_names=["lab"])
    g = moved(plain, {"lab": base})
    pm.testing.check_geometry(g.geometry)
    t, p = np.meshgrid(GRID.colatitudes, GRID.longitudes, indexing="ij")
    lab = g.geometry.interface("lab").radius
    x = g.geometry.mapping(pm.frames.cartesian_points(np.full(t.shape, lab), t, p))
    assert np.allclose(np.linalg.norm(x, axis=-1), base.radius(t, p), atol=1e-3)
    # as a lid: 1e30 Pa s above the base, the field below it
    lid = lithosphere.with_lid(m, base, viscosity=1e30)
    i = m.skeleton.locate(6200e3).layer             # the base crosses it
    eta = lid.layer(i)["viscosity"]
    assert not isinstance(eta, pm.PerturbedField)          # fully 3D before
    r = np.array([6280e3, 6160e3])
    tt, pp = np.array([1.0, 1.0]), np.array([2.0, 2.0])
    shallow, deep = base.radius(tt, pp) < r, base.radius(tt, pp) >= r
    got = eta(r, tt, pp)
    assert shallow.any() and deep.any()
    assert np.allclose(got[shallow], 1e30, rtol=1e-12)
    assert np.allclose(got[deep], m.layer(i)["viscosity"](r, tt, pp)[deep], rtol=1e-12)
    low = m.layer(m.skeleton.locate(5800e3).layer)
    assert lid.layer(low.index)["viscosity"] is low["viscosity"]
    # a radial viscosity stays a log perturbation of itself
    flat = pm.PREM(ocean=False)
    for lay in flat.layers:
        if lay.interval[0] >= 5701e3:
            flat = flat.with_field(lay.index, "viscosity", pm.constant_field(
                lay.interval, 1e21, name="viscosity"))
    lidded = lithosphere.with_lid(flat, base, viscosity=1e30)
    f = lidded.layer(i)["viscosity"]
    assert isinstance(f, pm.PerturbedField) and f.kind == "log"
    assert np.allclose(f(r, tt, pp), np.where(shallow, 1e30, 1e21), rtol=1e-12)
    assert lidded.citation == flat.citation
