"""Gravity through the fields, against a Gauss-panel oracle and known masses."""
import numpy as np
import pytest
from scipy.integrate import fixed_quad, quad

from planetmodel import Geometry, Skeleton
from planetmodel.catalogue import LayeredIsotropicElastic, PREM
from planetmodel.character import DENSITY
from planetmodel.fields import RadialField
from planetmodel.layerfunction import polynomial_layer
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


def test_a_radial_density_that_is_not_a_radial_field_integrates_by_quadrature():
    """A density that is radial without being a `RadialField` (a
    perturbed, a composed or a held one) is integrated by quadrature,
    and gives what the same density gives written as a polynomial."""
    from planetmodel import ComposedField, RadialMesh, perturbed
    from planetmodel.fields import HeldField
    from planetmodel.mesh1d.gravity import layer_masses, pressure
    prem = PREM()
    lower = prem.layer("lower_mantle")
    i, iv, rho = lower.index, lower.interval, lower["rho"]
    exact = prem.with_field(i, "rho", 1.01 * rho, replace=True)
    composed = ComposedField(lambda x: 1.01 * x, (rho,), character=DENSITY,
                             name="rho")
    plain = Model(prem.geometry, [{"rho": layer["rho"]} for layer in prem.layers])
    r = np.array([0.0, 3000e3, iv[0], 4800e3, iv[1], 6000e3, 6371e3])
    for model in (perturbed(prem, "rho", RadialField(iv, 0.01), kind="relative"),
                  plain.with_field(i, "rho", composed, replace=True)):
        field = model.layer(i)["rho"]
        assert field.is_radial and not isinstance(field, RadialField)
        assert np.allclose(gravity(model, r), gravity(exact, r), rtol=1e-9)
        assert np.allclose(gravity(model, r), gauss_gravity(model, r), rtol=1e-9)
        assert np.allclose(layer_masses(model), layer_masses(exact), rtol=1e-9)
        assert np.isclose(mass(model), mass(exact), rtol=1e-9)
        assert np.isclose(mass(model, radius=4800e3), mass(exact, radius=4800e3),
                          rtol=1e-9)
        assert np.allclose(pressure(model, r), pressure(exact, r), rtol=1e-9,
                           atol=1e-9 * float(pressure(exact, 0.0)))
    model = perturbed(prem, "rho", RadialField(iv, 0.01), kind="relative")
    inside = np.array([iv[0], 4800e3, iv[1]])
    assert np.allclose(model.with_gravity().layer(i)["g"](inside),
                       gravity(exact, inside), rtol=1e-9)
    mesh = RadialMesh(model, ngll=3, drmax=1500e3)
    assert np.allclose(mesh.nodal_gravity(model), gravity(exact, mesh.r), rtol=1e-9)
    # a radial density held beyond the surface, a `HeldField`
    a = float(prem.skeleton.boundaries[-1])
    top = ComposedField(lambda x: x, (prem.layer(-1)["rho"],), character=DENSITY,
                        name="rho")
    shell = plain.with_field(-1, "rho", top, replace=True).extended(
        [a + 100e3], fields="hold")
    assert isinstance(shell.layer(-1)["rho"], HeldField)
    want = mass(prem) + 4.0 * np.pi * float(top(a)) * ((a + 100e3) ** 3 - a ** 3) / 3.0
    assert np.isclose(mass(shell), want, rtol=1e-12)


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
    with pytest.raises(ValueError, match="lie in the model"):
        mass(m, radius=7e6)
    with pytest.raises(ValueError, match="lie in the model"):
        gravity(m, np.nan)
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


def test_with_stratification_matches_the_direct_ratio():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[(2.0, -1.0), 1.0],
                                vp=[1.0, 2.0], vs=[0.0, 1.0])
    held = m.with_stratification()
    assert type(held) is type(m)
    core = held.layer(0)
    assert "drho_dphi0" in core and "drho_dphi0" not in held.layer(1)
    r = np.linspace(0.05, 0.5, 8)
    assert np.allclose(core["drho_dphi0"](r), -1.0 / gravity(m, r), rtol=1e-12)
    assert core["drho_dphi0"](0.0) == 0.0
    with pytest.raises(ValueError, match="already holds"):
        held.with_stratification()
    assert "drho_dphi0" in held.with_stratification(replace=True).layer(0)
    solid = LayeredIsotropicElastic([0.0, 1.0], rho=[1.0], vp=[2.0], vs=[1.0])
    with pytest.raises(ValueError, match="no fluid layer"):
        solid.with_stratification()


def test_radii_are_held_to_the_geometry_tolerance():
    """A radius an ulp beyond the surface is the surface, as the fields
    take it; the antiderivative agrees with a quadrature point by point."""
    nd = PREM().nondimensionalised()
    beyond = np.nextafter(1.0, 2.0)
    assert gravity(nd, beyond) == gravity(nd, 1.0)
    assert mass(nd, radius=beyond) == mass(nd)
    r = np.linspace(0.0, 1.0, 37)
    each = np.array([4 * np.pi * sum(
        float(f.integrate(f.interval[0], min(x, f.interval[1])))
        for f in ((lay["rho"] * RadialField(lay.interval, polynomial_layer(
            lay.interval, [0.0, 0.0, 1.0]))) for lay in nd.layers)
        if f.interval[0] < x) for x in r])
    g = gravity(nd, r)
    safe = np.where(r > 0, r, 1.0)
    assert np.allclose(g, np.where(r > 0, nd.G * each / safe ** 2, 0.0), rtol=1e-12,
                       atol=0.0)


# -- pressure ------------------------------------------------------------------------

def test_prem_pressure_matches_adaptive_quadrature_and_its_table():
    from planetmodel.mesh1d.gravity import pressure
    prem = PREM()
    b = prem.skeleton.boundaries

    def rho_g(s):
        i = int(min(np.searchsorted(b, s, side="right") - 1, prem.nlayers - 1))
        return float(prem.layer(i)["rho"](np.array(s))) * float(gravity(prem, s))

    for r in (0.0, 1000e3, 3480e3, 4000e3, 6000e3, 6360e3):
        cuts = [r] + [x for x in b if r < x < b[-1]] + [b[-1]]
        want = sum(quad(rho_g, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)[0]
                   for lo, hi in zip(cuts[:-1], cuts[1:]))
        assert float(pressure(prem, r)) == pytest.approx(want, rel=1e-13)
    # PREM's own table, about 364 GPa at the centre and 135.8 at the CMB
    centre, cmb = pressure(prem, [0.0, 3480e3])
    assert centre == pytest.approx(363.9e9, rel=2e-3)
    assert cmb == pytest.approx(135.8e9, rel=2e-3)
    assert float(pressure(prem, b[-1])) == 0.0


def test_a_uniform_sphere_s_pressure_is_the_closed_form():
    from planetmodel.mesh1d.gravity import pressure
    a, rho = 6.0e6, 5000.0
    ball = LayeredIsotropicElastic.homogeneous(a, rho=rho, vp=8e3, vs=4e3)
    r = np.linspace(0.0, a, 13)
    want = 2.0 * np.pi / 3.0 * G_SI * rho ** 2 * (a ** 2 - r ** 2)
    assert np.allclose(pressure(ball, r), want, rtol=1e-14, atol=1e-6)


def test_pressure_fields_are_continuous_with_the_derivative_minus_rho_g():
    from planetmodel.mesh1d.gravity import pressure, pressure_fields
    prem = PREM()
    fields = pressure_fields(prem)
    for below, above in zip(fields[:-1], fields[1:]):
        r = below.interval[1]
        assert float(below(np.array(r))) == pytest.approx(float(above(np.array(r))),
                                                          rel=1e-14)
    assert float(fields[-1](np.array(prem.skeleton.boundaries[-1]))) == 0.0
    for i, f in enumerate(fields):
        lo, hi = f.interval
        r = np.linspace(lo, hi, 7)[1:-1]
        rho = prem.layer(i)["rho"](r)
        assert np.allclose(f.function.derivative()(r), -rho * gravity(prem, r),
                           rtol=1e-14)
        h = 1e-6 * (hi - lo)
        fd = (f(r + h) - f(r - h)) / (2.0 * h)
        assert np.allclose(fd, -rho * gravity(prem, r), rtol=1e-6)
        assert f.character == DENSITY and f.name == "pressure"
    assert np.allclose(pressure(prem, [1221.5e3, 3480e3]),
                       [float(fields[0](np.array(1221.5e3))),
                        float(fields[1](np.array(3480e3)))], rtol=1e-15)


def test_with_pressure_attaches_it_in_the_model_s_units():
    prem = PREM()
    m = prem.with_pressure()
    assert m.spec("pressure").dimensions == m.spec("kappa").dimensions
    assert m.layer("lower_mantle")["pressure"](np.array(5000e3)) == pytest.approx(
        float(prem.pressure(5000e3)), rel=1e-15)
    with pytest.raises(ValueError, match="already holds 'pressure'"):
        m.with_pressure()
    assert m.with_pressure(replace=True).nlayers == m.nlayers
    nd = prem.nondimensionalised()
    k = nd.scales.factor(nd.spec("pressure").dimensions)
    r = np.array([0.0, 2e6, 5e6]) / nd.scales.length
    assert np.allclose(k * nd.pressure(r), prem.pressure(r * nd.scales.length),
                       rtol=1e-13)
