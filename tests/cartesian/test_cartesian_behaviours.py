"""Behaviours and conversions on flat models: what is pointwise (the
elastic description, constant-Q dispersion, freezing a Maxwell or Prony
body, Prony and Burgers branches, the laws applied to an anomaly, power-
law creep) gives on a flat layer what it gives on a spherical one at the
same material values; what is the sphere's by nature (gravity from the
mass inside a radius, a mean over the sphere, the lithosphere on radial
lines, sl3d's Earth chain) refuses a flat object by name."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel import LayeredIsotropicElastic as SphericalLayered
from planetmodel import cartesian as flat
from planetmodel.behaviours import SelfGravitating
from planetmodel.conversion import building, lithosphere, parameters, prony
from planetmodel.fields import ComposedField
from planetmodel.means import quadrature_mean, spherical_mean, spherically_averaged
from planetmodel.mesh1d import gravity as gravity_module
from planetmodel.perturbation import PerturbedField
from planetmodel.rheology import dispersive_moduli
from planetmodel.sampling import gauss_legendre

BOX = flat.Box([(0.0, 400e3), (-200e3, 200e3)])
YEAR = 3.15576e7
EDGES = [0.0, 30e3, 100e3, 200e3]
VALUES = {"rho": [2800.0, 3300.0, 3300.0], "vp": [6400.0, 8000.0, 8000.0],
          "vs": [3700.0, 4500.0, 4500.0]}
FIVE = ("A", "C", "F", "L", "N")


def pair(*, viscosity=None):
    """The reference flat model and a spherical one of the same material
    values on the same intervals (a ball of radius 200 km), both with the
    asthenosphere's viscosity where given."""
    box = flat.CrustOverMantle(BOX, viscosity=viscosity)
    sphere = SphericalLayered(EDGES, **VALUES)
    if viscosity is not None:
        iv = sphere.layer(-1).interval
        sphere = sphere.with_field(-1, "viscosity",
                                   pm.constant_field(iv, viscosity, name="viscosity"))
    return box, sphere


def same(a, b, d, *, rtol=1e-14):
    """Whether two fields agree to rounding at the values `d` of the
    layered coordinate, laterally where they need it."""
    x = np.linspace(0.1, 0.9, d.size)
    va = np.asarray(a.evaluate(d, x, x))
    vb = np.asarray(b.evaluate(d, x, x))
    return np.allclose(va, vb, rtol=rtol, atol=0.0)


def test_the_elastic_description_is_pointwise():
    box, sphere = pair()
    for i in range(3):
        lo, hi = box.layer(i).interval
        d = np.linspace(lo, hi, 9)
        for name in FIVE:
            assert same(box.layer(i)[name], sphere.layer(i)[name], d), (i, name)
        kb, mb = box.kappa_mu(i)
        ks, ms = sphere.kappa_mu(i)
        assert same(kb, ks, d) and same(mb, ms, d)
        assert box.is_fluid(i) == sphere.is_fluid(i) is False
        rho, vp, vs = (VALUES[k][i] for k in ("rho", "vp", "vs"))
        assert np.allclose(mb(d), rho * vs ** 2, rtol=1e-15, atol=0)
        assert np.allclose(kb(d), rho * (vp ** 2 - 4.0 * vs ** 2 / 3.0), rtol=1e-14,
                           atol=0)
    iso = box.isotropic()
    assert type(iso) is flat.CrustOverMantle and "kappa" in iso.layer(0)
    assert type(iso.layer(0)["mu"]) is flat.DepthField
    swapped = box.with_field(0, "vs", flat.constant_field((0.0, 30e3), 3500.0,
                                                          name="vs"), replace=True)
    assert np.isclose(swapped.layer(0)["L"](10e3), 2800.0 * 3500.0 ** 2, rtol=1e-15)


def test_constant_q_disperses_a_flat_layer_as_a_spherical_one():
    box, sphere = pair()
    omega, reference = 2.0 * np.pi / 50.0, 2.0 * np.pi
    qbox = flat.constant_field((30e3, 100e3), 80.0, name="qmu")
    qsph = pm.constant_field((30e3, 100e3), 80.0, name="qmu")
    box = box.with_field(1, "qmu", qbox)
    sphere = sphere.with_field(1, "qmu", qsph)
    got = box.moduli_at(1, omega, reference_omega=reference)
    want = dispersive_moduli(sphere.layer(1), omega, reference_omega=reference)
    d = np.linspace(30e3, 100e3, 7)
    for name in FIVE:
        assert np.iscomplexobj(got[name](d)) == (name in ("L", "N", "A", "C", "F"))
        assert same(got[name], want[name], d), name
    assert box.reference_omega() == 2.0 * np.pi


def test_freezing_a_maxwell_body_is_pointwise():
    box, sphere = pair(viscosity=1e20)
    for omega in (2.0 * np.pi / (1000.0 * YEAR), 2.0 * np.pi / 100.0):
        fb, fs = box.frozen(omega), pm.rheology.frozen(sphere, omega)
        assert type(fb) is flat.CrustOverMantle and pm.rheology.is_frozen(fb)
        d = np.linspace(100e3, 200e3, 7)
        for name in FIVE:
            assert same(fb.layer(-1)[name], fs.layer(-1)[name], d), (omega, name)
        mu, eta = 3300.0 * 4500.0 ** 2, 1e20
        s = 1j * omega * eta / mu
        assert np.allclose(fb.layer(-1)["N"](d), mu * s / (1.0 + s), rtol=1e-12, atol=0)
        assert fb.layer(-1)["N"].coordinates is flat.FLAT
    plain = sphere.without_field("viscosity")
    assert box.elastic().field_names() == plain.field_names()


def test_prony_and_burgers_branches_are_pointwise():
    box, sphere = pair(viscosity=1e20)
    series = prony.PronySeries((0.5, 0.3), (0.2, 3.0), name="two")
    omega = 2.0 * np.pi / (30.0 * YEAR)
    d = np.linspace(100e3, 200e3, 7)
    for make in (lambda m: prony.with_branches(m, series),
                 lambda m: prony.with_burgers(m, ["asthenosphere"] if m is box else [2],
                                              kelvin_viscosity=1e19,
                                              kelvin_modulus_fraction=0.5)):
        b, s = make(box), make(sphere)
        assert type(b) is flat.CrustOverMantle
        held = b.layer(-1)
        assert all(f.coordinates is flat.FLAT for f in held.fields.values())
        assert len(pm.rheology.branches(held)) == 2
        for name in FIVE:
            want = pm.rheology.frozen(s, omega).layer(-1)[name]
            assert same(b.frozen(omega).layer(-1)[name], want, d, rtol=1e-13), name
    w = prony.with_branches(box, series).layer(-1)["relaxation_strength_1"]
    assert type(w) is flat.DepthField and w(150e3) == 0.5


def test_the_laws_apply_to_a_flat_anomaly_point_by_point():
    box, _ = pair(viscosity=1e20)
    vs = box.layer(-1)["vs"]
    dlnvs = flat.AnalyticField(vs.interval,
                               lambda d, x, y: 0.01 * np.sin(2 * np.pi * x / 400e3)
                               * (d / 200e3))
    model = box.with_field(-1, "vs", PerturbedField(vs, dlnvs, kind="relative",
                                                    name="vs"), replace=True)
    anomaly = building.velocity_anomaly(model, layers=["asthenosphere"])
    dt = building.temperature_anomaly(
        anomaly, sensitivity=parameters.Sensitivity(-1e-4, name="flat test"))
    law = parameters.ExponentialViscosity(0.02, name="flat test")
    out = building.with_viscosity(model, dt, law)
    eta = out.layer(-1)["viscosity"]
    d = np.array([120e3, 150e3, 190e3])
    x = np.array([100e3, 50e3, 300e3])
    y = np.zeros(3)
    rel = 0.01 * np.sin(2 * np.pi * x / 400e3) * d / 200e3
    want = 1e20 * np.exp(-0.02 * rel / -1e-4)
    assert np.allclose(eta.evaluate(d, x, y), want, rtol=1e-12, atol=0)
    assert eta.coordinates is flat.FLAT and type(out) is flat.CrustOverMantle
    creeping = building.with_power_law(box, exponent=3.0, transition_stress=1e6)
    assert type(creeping.layer(-1)["stress_exponent"]) is flat.DepthField
    assert creeping.layer(-1)["transition_stress"](150e3) == 1e6


# -- what is the sphere's refuses a flat object by name -------------------------------

SPHERE_ONLY = "is defined in the coordinates 'spherical' alone; this {} is in 'flat'"


def test_gravity_refuses_a_flat_model_by_name():
    box, _ = pair()
    says = SPHERE_ONLY.format("CrustOverMantle")
    for call in (lambda: gravity_module.gravity(box, [10e3]),
                 lambda: gravity_module.mass(box),
                 lambda: gravity_module.pressure(box, [10e3]),
                 lambda: gravity_module.gravity_fields(box),
                 lambda: gravity_module.pressure_fields(box)):
        with pytest.raises(ValueError, match="gravity from the mass inside a radius "
                                             + says):
            call()

    class Slab(SelfGravitating, flat.Model):
        """A flat model type mixing in what computes gravity."""

    slab = Slab(box.geometry, [dict(lay.fields) for lay in box.layers])
    for call in (lambda: slab.gravity([1e3]), lambda: slab.with_gravity(),
                 lambda: slab.with_pressure(), lambda: slab.mass()):
        with pytest.raises(ValueError, match="this Slab is in 'flat'"):
            call()


def test_a_mean_over_the_sphere_refuses_a_flat_field_or_model():
    box, _ = pair()
    rho = box.layer(0)["rho"]
    lateral = flat.AnalyticField((0.0, 30e3), lambda d, x, y: 2800.0 + 0.0 * x)
    perturbed = PerturbedField(rho, ComposedField(lambda v: 0.0 * v, [lateral],
                                                  character=pm.SCALAR), kind="relative")
    grid = gauss_legendre(4)
    for call, who in ((lambda: spherical_mean(rho), "DepthField"),
                      (lambda: rho.spherical_mean(), "DepthField"),
                      (lambda: spherical_mean(lateral, grid=grid), "AnalyticField"),
                      (lambda: quadrature_mean(lateral, grid), "AnalyticField"),
                      (lambda: perturbed.spherical_mean(grid=grid), "PerturbedField"),
                      (lambda: spherically_averaged(box), "CrustOverMantle")):
        with pytest.raises(ValueError, match="a spherical mean "
                                             + SPHERE_ONLY.format(who)):
            call()
    # a fully 3D speed's reference is its spherical mean: refused too
    model = box.with_field(-1, "vs", flat.AnalyticField(
        (100e3, 200e3), lambda d, x, y: 4500.0 + 10.0 * np.sin(x / 1e5)), replace=True)
    with pytest.raises(ValueError, match="a spherical mean " + SPHERE_ONLY.format(
            "AnalyticField")):
        building.velocity_anomaly(model, layers=[2], grid=grid)


def test_what_reads_radial_lines_or_the_earth_refuses_a_flat_model():
    box, _ = pair(viscosity=1e20)
    grid = gauss_legendre(4)
    with pytest.raises(ValueError, match="the base of the lithosphere on radial lines "
                                         + SPHERE_ONLY.format("CrustOverMantle")):
        lithosphere.viscosity_threshold(box, 1e21, grid=grid)
    with pytest.raises(ValueError, match="the base of the lithosphere on radial lines"):
        lithosphere.maxwell_threshold(box, 1e3 * YEAR, grid=grid)
    with pytest.raises(ValueError, match="a lid below a surface measuring a radius"):
        lithosphere.with_lid(box, None, viscosity=1e23)
    with pytest.raises(ValueError, match="sl3d's chain " + SPHERE_ONLY.format(
            "CrustOverMantle")):
        building.sl3d_viscosity(box)
