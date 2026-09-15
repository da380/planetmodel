"""Gravity through the fields, against a Gauss-panel oracle and known masses."""
import numpy as np
import pytest
from scipy.integrate import fixed_quad, quad

from planetmodel import Geometry, Skeleton
from planetmodel.catalogue import LayeredIsotropicElastic, PREM
from planetmodel.character import DENSITY
from planetmodel.fields import RadialField
from planetmodel.mesh1d.gravity import gravity, mass
from planetmodel.model import Model
from planetmodel.units import GRAVITY, MASS, G_SI


def gauss_gravity(model, radii, *, n=8, panels=4):
    """The oracle: M(r) by n-point Gauss-Legendre on `panels` per layer,
    exact for polynomial densities of degree <= 2n - 3."""
    r = np.asarray(radii, dtype=float).reshape(-1)

    def span(f, lo, hi):
        edges = np.linspace(lo, hi, panels + 1)
        return sum(fixed_quad(lambda s: f(s) * s * s, a, c, n=n)[0]
                   for a, c in zip(edges[:-1], edges[1:]))

    out = np.empty(r.shape)
    for j, x in enumerate(r):
        total = 0.0
        for i in range(model.nlayers):
            lo, hi = model.skeleton.interval(i)
            if x <= lo:
                break
            total += span(model.layer(i)["rho"], lo, min(x, hi))
        out[j] = 4.0 * np.pi * model.G * total / x ** 2 if x > 0.0 else 0.0
    return out.reshape(np.shape(radii))


def test_prem_gravity_matches_the_gauss_oracle_to_machine_precision():
    m = PREM()
    r = np.concatenate(([0.0], np.linspace(1e5, 6371e3, 40), m.skeleton.boundaries))
    got = gravity(m, r)
    assert np.allclose(got, gauss_gravity(m, r), rtol=1e-13, atol=1e-13 * 10.0)
    assert got[0] == 0.0


def test_prem_mass_and_surface_gravity():
    m = PREM()
    M = mass(m)
    assert np.isclose(M, 5.974e24, rtol=3e-4)
    assert np.isclose(gravity(m, 6371e3), G_SI * M / 6371e3 ** 2, rtol=1e-14)
    assert np.isclose(mass(m, radius=3480e3), 1.94e24, rtol=5e-3)
    assert gravity(m, 3480e3) > gravity(m, 6371e3)


def test_a_homogeneous_sphere_is_linear_in_r():
    m = LayeredIsotropicElastic.homogeneous(2.0, rho=3.0, vp=1.0, vs=0.5)
    r = np.linspace(0.0, 2.0, 9)
    assert np.allclose(gravity(m, r), 4.0 * np.pi * G_SI * 3.0 * r / 3.0, rtol=1e-14)
    assert np.isclose(mass(m), 4.0 * np.pi * 3.0 * 8.0 / 3.0, rtol=1e-14)


def test_a_hollow_model_has_no_mass_inside():
    m = LayeredIsotropicElastic([0.5, 1.0], rho=[2.0], vp=[1.0], vs=[0.0])
    assert gravity(m, 0.5) == 0.0
    assert np.isclose(mass(m), 4.0 * np.pi * 2.0 * (1.0 - 0.125) / 3.0, rtol=1e-14)


def test_an_analytic_density_integrates_by_quadrature():
    sk = Skeleton([0.0, 1.0])
    rho = RadialField((0.0, 1.0), lambda r: np.exp(-r), character=DENSITY)
    m = Model(Geometry(sk), [{"rho": rho}])
    want, _ = quad(lambda s: np.exp(-s) * s * s, 0.0, 0.7)
    assert np.isclose(gravity(m, 0.7), 4.0 * np.pi * G_SI * want / 0.49, rtol=1e-9)


def test_non_dimensional_gravity_redimensionalises_exactly():
    m = PREM()
    nd = m.nondimensionalised()
    r = np.linspace(0.1, 1.0, 7)
    back = gravity(nd, r) * nd.scales.factor(GRAVITY)
    assert np.allclose(back, gravity(m, r * 6371e3), rtol=1e-13)
    assert np.isclose(mass(nd) * nd.scales.factor(MASS), mass(m), rtol=1e-13)
    assert np.isclose(nd.G, 1.0)


def test_refusals():
    m = PREM()
    with pytest.raises(ValueError, match="lie in the model"):
        gravity(m, 7e6)
    with pytest.raises(ValueError, match="outside the model"):
        mass(m, radius=7e6)
    with pytest.raises(KeyError, match="needs 'rho'"):
        gravity(m.without_field("rho", layers=[3]), 1e6)


def test_a_density_that_depends_on_direction_is_refused():
    from planetmodel import DENSITY, AnalyticField, PREM
    m = PREM(ocean=False)
    lid = m.layer("lid")
    shaped = AnalyticField(lid.interval, lambda r, t, p: 3.3e3 * (1 + 0.01 * np.cos(t)),
                           character=DENSITY, name="rho")
    bumpy = m.with_field("lid", "rho", shaped, replace=True)
    with pytest.raises(ValueError, match="depends on direction"):
        bumpy.gravity(6300e3)
    with pytest.raises(ValueError, match="depends on direction"):
        bumpy.mass()


def test_gravity_fields_agree_with_gravity_and_are_exact():
    from planetmodel import gravity_fields, testing
    m = PREM()
    fields = gravity_fields(m)
    assert len(fields) == m.nlayers and all(f.name == "g" for f in fields)
    for layer, f in zip(m.layers, fields):
        lo, hi = layer.interval
        r = np.linspace(lo, hi, 9)
        assert np.allclose(f(r), m.gravity(r), rtol=1e-13, atol=0.0)
        h = 1e-3 * (hi - lo)
        inner = r[1:-1]
        fd = (m.gravity(inner + h) - m.gravity(inner - h)) / (2 * h)
        assert np.allclose(f.derivative()(inner), fd, rtol=1e-5)
        testing.check_field(f)
    # continuous across the CMB, zero at the centre, the surface value
    cmb = m.geometry.interface("cmb").radius
    assert np.isclose(fields[1](cmb), fields[2](cmb), rtol=1e-13)
    assert fields[0](0.0) == 0.0
    assert np.isclose(fields[-1](6371e3), m.gravity(6371e3))
    # with_gravity attaches them under the vocabulary name; the copy keeps
    # its class and survives conversion
    held = m.with_gravity()
    assert type(held) is type(m)
    assert all("g" in layer for layer in held.layers)
    assert np.allclose(held.layer(3)["g"](5000e3), fields[3](5000e3))
    with pytest.raises(ValueError, match="already holds"):
        held.with_gravity()
    assert "g" in held.with_gravity(replace=True).layer(0)
    nd = held.nondimensionalised()
    assert np.isclose(nd.layer(-1)["g"](1.0), nd.gravity(1.0), rtol=1e-12)
    testing.check_model(held.truncated(6000e3))


def test_gravity_fields_on_a_numeric_density():
    from planetmodel import DENSITY, Geometry, Model, RadialField, Skeleton, gravity
    from planetmodel import gravity_fields
    sk = Skeleton([0.0, 0.5, 1.0])
    rho = [RadialField(sk.interval(i), lambda r: 2.0 - np.exp(-r), character=DENSITY,
                       name="rho") for i in range(2)]
    m = Model(Geometry(sk), [{"rho": rho[0]}, {"rho": rho[1]}])
    fields = gravity_fields(m)
    r = np.array([0.1, 0.4, 0.5])
    assert np.allclose(fields[0](r), gravity(m, r), rtol=1e-10)
    assert np.allclose(fields[1](np.array([0.5, 0.9])), gravity(m, [0.5, 0.9]),
                       rtol=1e-10)
