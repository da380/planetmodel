"""Processing a data set's fields onto a model of one's own, each step held
to its formula on fields built here."""
import numpy as np
import pytest
from scipy.interpolate import PPoly

import planetmodel as pm
from planetmodel.datasets import (GridData, Perturbation, RegionalGrid, filtered,
                                  merged, over_reference, raised_cosine, rereferenced,
                                  spread, standalone)
from planetmodel.materials import voigt_vs

CMB, MOHO = 3480e3, 6346.6e3
R = 6371e3


def harmonic(lo=CMB, hi=MOHO, lmax=6, seed=0, scale=0.01):
    """A harmonic field, linear in radius in each coefficient."""
    rng = np.random.default_rng(seed)
    c = scale * rng.standard_normal((2, 2, lmax + 1, lmax + 1))
    for l in range(lmax + 1):
        c[:, :, l, l + 1:] = 0.0
    c[:, 1, :, 0] = 0.0
    return pm.HarmonicField((lo, hi), PPoly(c[:, None] / np.array([hi - lo, 1.0])
                                            .reshape(2, 1, 1, 1, 1), [lo, hi]))


def points(rng, lo, hi, n=100):
    return (rng.uniform(lo, hi, n), np.arccos(rng.uniform(-1, 1, n)),
            rng.uniform(0, 2 * np.pi, n))


def simple(vs, *, moho=MOHO):
    """An isotropic model: core, mantle to `moho`, crust."""
    return pm.LayeredIsotropicElastic([0.0, 1221.5e3, CMB, moho, R],
                                      rho=[1.3e4, 1.1e4, 4.5e3, 2.8e3],
                                      vp=[1.1e4, 9e3, 1.2e4, 6.5e3],
                                      vs=[3.5e3, 0.0, vs, 3.6e3],
                                      layer_names=["ic", "oc", "mantle", "crust"],
                                      interface_names=["icb", "cmb", "moho", "surface"])


def test_the_raised_cosine():
    assert raised_cosine([-1.0, 0.0, 0.5, 1.0, 2.0]) == pytest.approx(
        [0.0, 0.0, 0.5, 1.0, 1.0], abs=1e-15)
    h = 1e-6
    for t in (0.0, 1.0):                                   # C1 at both ends
        assert abs(raised_cosine(t + h) - raised_cosine(t - h)) / (2 * h) < 1e-5


def test_relative_mode_keeps_the_source_perturbation():
    src = over_reference(pm.PREM(), {"vs": harmonic()}, kind="relative", name="src")
    (p,) = rereferenced(src, simple(4.5e3)).values()
    assert p.kind == "relative" and len(p.segments) == 8
    assert all(isinstance(s, pm.HarmonicField) for s in p.segments)
    held = [lay for lay in src.layers if "vs" in lay]
    assert all(s is lay["vs"].perturbation for s, lay in zip(p.segments, held))


def test_a_span_clips_a_perturbed_source():
    src = over_reference(pm.PREM(), {"vs": harmonic()}, kind="relative", name="src")
    onto = simple(4.5e3)
    whole = rereferenced(src, onto)["vs"]
    span = (4000e3, 6000e3)
    p = rereferenced(src, onto, span=span)["vs"]
    assert p.kind == "relative"
    assert p.support == pytest.approx(span)
    assert len(p.segments) == 5                    # the source's 8, clipped to 5
    inside = [s for s in p.segments if any(s is w for w in whole.segments)]
    assert len(inside) == 3                        # layers within the span, untouched
    rng = np.random.default_rng(6)
    for s in p.segments:
        r, t, f = points(rng, *s.interval, n=20)
        assert np.allclose(s(r, t, f), harmonic()(r, t, f), rtol=1e-12)


def test_absolute_mode_keeps_the_values():
    prem = pm.PREM()
    src = over_reference(prem, {"vs": harmonic()}, kind="relative", name="src")
    onto = simple(4.5e3, moho=6300e3)
    p = rereferenced(src, onto, mode="absolute")["vs"]
    cuts = {float(s.interval[0]) for s in p.segments}
    assert 6300e3 in cuts                          # onto's Moho, inside the support
    rng = np.random.default_rng(1)
    for s in p.segments:
        r, t, f = points(rng, *s.interval, n=20)
        lay = prem.layer(prem.skeleton.locate(float(np.mean(s.interval))).layer)
        rs = voigt_vs(lay["vsv"](r), lay["vsh"](r))
        mid = float(np.mean(s.interval))
        ro = onto.layer(onto.skeleton.locate(mid).layer)["vs"](r)
        want = rs * (1 + harmonic()(r, t, f)) / ro - 1
        assert np.allclose(s(r, t, f), want, rtol=1e-12, atol=1e-15)


def test_a_data_set_of_values_needs_its_reference_under_relative_mode():
    v = pm.AnalyticField((CMB, MOHO), lambda r, t, p: 4.6e3 + 10.0 * np.cos(t) + 0 * r)
    src = standalone({"vs": v}, surface=R, name="values")
    ref = simple(4.4e3)
    with pytest.raises(ValueError, match="needs source_reference="):
        rereferenced(src, ref)
    rng = np.random.default_rng(2)
    r, t, f = points(rng, CMB, MOHO)
    rel = rereferenced(src, simple(4.5e3), source_reference=ref)["vs"]
    assert np.allclose(rel.segments[0](r, t, f), v(r, t, f) / 4.4e3 - 1, rtol=1e-13)
    log = rereferenced(src, simple(4.5e3), source_reference=ref, kind="log")["vs"]
    assert log.kind == "log"
    assert np.allclose(log.segments[0](r, t, f), np.log(v(r, t, f) / 4.4e3), rtol=1e-13)
    ab = rereferenced(src, simple(4.5e3), mode="absolute", kind="absolute")["vs"]
    assert ab.segments[0].character == pm.SCALAR
    assert np.allclose(ab.segments[0](r, t, f), v(r, t, f) - 4.5e3, rtol=1e-13)


def test_a_source_in_other_scales_is_converted_to_the_model_s():
    """Onto a model in other scales, an SI source and its reference are
    converted first: the perturbation is the one the formulas give in SI,
    at the model's own radii and, where it is absolute, in its units."""
    from planetmodel.units import Scales
    prem = pm.PREM()
    src = over_reference(prem, {"vs": harmonic()}, kind="relative", name="src")
    v = pm.AnalyticField((CMB, MOHO), lambda r, t, p: 4.6e3 + 10.0 * np.cos(t) + 0 * r,
                         name="vs")
    values = standalone({"vs": v}, surface=R, name="values")
    onto = simple(4.5e3, moho=6300e3)
    rng = np.random.default_rng(8)
    for scales in (Scales(time=1000.0), Scales(length=1000.0, mass=1e12),
                   Scales.geophysical(R)):
        k, speed = scales.length, scales.length / scales.time
        other = onto.converted(scales)
        kept = rereferenced(src, other)["vs"]
        assert kept.support == pytest.approx((CMB / k, MOHO / k), rel=1e-12)
        for s in kept.segments:
            lo, hi = (k * x for x in s.interval)
            r, t, f = points(rng, lo, hi, n=20)
            assert np.allclose(s(r / k, t, f), harmonic()(r, t, f), rtol=1e-11,
                               atol=1e-15)
        against = rereferenced(src, other, mode="absolute")["vs"]
        assert any(np.isclose(k * s.interval[0], 6300e3) for s in against.segments)
        for s in against.segments:
            lo, hi = (k * x for x in s.interval)
            r, t, f = points(rng, lo, hi, n=20)
            mid = 0.5 * (lo + hi)
            lay = prem.layer(prem.skeleton.locate(mid).layer)
            rs = voigt_vs(lay["vsv"](r), lay["vsh"](r))
            ro = onto.layer(onto.skeleton.locate(mid).layer)["vs"](r)
            want = rs * (1 + harmonic()(r, t, f)) / ro - 1
            assert np.allclose(s(r / k, t, f), want, rtol=1e-10, atol=1e-13)
        r, t, f = points(rng, CMB, 6300e3, n=20)
        rel = rereferenced(values, other, source_reference=simple(4.4e3))["vs"]
        assert np.allclose(rel.segments[0](r / k, t, f), v(r, t, f) / 4.4e3 - 1,
                           rtol=1e-10)
        ab = rereferenced(values, other, mode="absolute", kind="absolute")["vs"]
        assert np.allclose(ab.segments[0](r / k, t, f), (v(r, t, f) - 4.5e3) / speed,
                           rtol=1e-10)


def test_a_log_perturbation_stays_log():
    prem = pm.PREM()
    src = over_reference(prem, {"vs": harmonic()}, kind="log", name="src")
    p = rereferenced(src, simple(4.5e3), mode="absolute")["vs"]
    assert p.kind == "log"
    s = p.segments[3]
    rng = np.random.default_rng(3)
    r, t, f = points(rng, *s.interval, n=20)
    lay = prem.layer(prem.skeleton.locate(float(np.mean(s.interval))).layer)
    want = harmonic()(r, t, f) + np.log(voigt_vs(lay["vsv"](r), lay["vsh"](r)) / 4.5e3)
    assert np.allclose(s(r, t, f), want, rtol=1e-12)


def test_spread_as_it_lies_keeps_the_fields_and_refuses_partial_cover():
    src = over_reference(pm.PREM(), {"vs": harmonic()}, kind="relative", name="src")
    (p,) = rereferenced(src, pm.PREM()).values()
    on = spread(p, pm.PREM())
    assert sorted(on) == list(range(2, 10))
    assert all(isinstance(f, pm.HarmonicField) for f in on.values())
    short = Perturbation("vs", "relative", (harmonic(lo=3600e3, hi=6000e3),))
    with pytest.raises(ValueError, match="only in part; ends=.*extend=True.*taper="):
        spread(short, pm.PREM())


def test_ends_map_the_support_onto_two_interfaces():
    p = Perturbation("vs", "relative", (harmonic(),))
    onto = simple(4.5e3, moho=6300e3)
    on = spread(p, onto, ends=("cmb", "moho"))
    assert sorted(on) == [2]
    rng = np.random.default_rng(4)
    r, t, f = points(rng, CMB, 6300e3)
    mapped = CMB + (r - CMB) * (MOHO - CMB) / (6300e3 - CMB)
    assert np.allclose(on[2](r, t, f), harmonic()(mapped, t, f), rtol=1e-12)
    with pytest.raises(ValueError, match="no interface 'd670'"):
        spread(p, onto, ends=("cmb", "d670"))


def test_extend_continues_and_taper_brings_to_zero():
    p = Perturbation("vs", "relative", (harmonic(lo=3600e3, hi=6000e3),))
    onto = simple(4.5e3)
    rng = np.random.default_rng(5)
    r, t, f = points(rng, CMB, MOHO, n=300)
    ext = spread(p, onto, extend=True)[2]
    clipped = np.clip(r, 3600e3, 6000e3)
    assert np.allclose(ext(r, t, f), harmonic(lo=3600e3, hi=6000e3)(clipped, t, f),
                       rtol=1e-12)
    tap = spread(p, onto, taper=(100e3, 200e3))[2]
    inside = (r > 3600e3) & (r < 6000e3)
    w = raised_cosine((r - 3600e3) / 100e3) * raised_cosine((6000e3 - r) / 200e3)
    want = np.where(inside, w * harmonic(lo=3600e3, hi=6000e3)(clipped, t, f), 0.0)
    assert np.allclose(tap(r, t, f), want, rtol=1e-12, atol=1e-18)
    deep = r > 3700e3
    assert np.all(tap(r, t, f)[r < 3600e3] == 0.0) and np.any(tap(r, t, f)[deep] != 0)
    with pytest.raises(ValueError, match="does not reach layer"):
        spread(p, onto, layers=["crust"], extend=True)
    assert sorted(spread(p, onto, layers=["mantle"], extend=True)) == [2]


def test_filtering_a_harmonic_field_by_degree():
    f = harmonic(lmax=10, seed=6)
    cut = filtered(f, lmax=6)
    assert cut.lmax == 6
    c, c0 = cut.coefficients.c, f.coefficients.c[..., :7, :7]
    assert np.array_equal(c, c0)
    rolled = filtered(f, lmax=6, lmin=1, roll_off=3)
    by_degree = rolled.coefficients.c[0, 0, 0, :, 0] / c0[0, 0, 0, :, 0]
    assert by_degree[0] == 0.0 and np.allclose(by_degree[1:4], 1.0)
    l = np.arange(7)
    assert np.allclose(by_degree[4:], raised_cosine((7 - l[4:]) / 4.0))
    assert np.all(np.diff(by_degree[3:]) < 0)
    mean = rolled.spherical_mean()
    assert abs(float(mean(5e6))) < 1e-18                    # lmin=1: no mean
    pert = pm.PerturbedField(pm.constant_field(f.interval, 4.5e3), f, kind="relative")
    assert filtered(pert, lmax=3).perturbation.lmax == 3


def test_filtering_a_gridded_field_by_degree():
    from planetmodel.harmonics import analyse_grid
    f = harmonic(lmax=6, seed=7)
    grid = pm.gauss_legendre(12)
    g = pm.GriddedField.sampled(f, grid, breaks=np.linspace(CMB, MOHO, 4), ngll=2,
                                radial="linear")
    same = filtered(g, lmax=12)
    assert np.allclose(same.values, g.values, rtol=0.0, atol=1e-14)
    low = filtered(g, lmax=3)
    c = analyse_grid(low.values, grid)
    assert np.abs(c[:, 4:]).max() < 1e-14
    assert np.allclose(c[:, :4], analyse_grid(g.values, grid)[:, :4], atol=1e-14)
    custom = pm.GriddedField.sampled(f, pm.AngularGrid(grid.colatitudes,
                                                       grid.longitudes),
                                     ngll=2, radial="linear")
    with pytest.raises(ValueError, match="shipped kind"):
        filtered(custom, lmax=3)
    with pytest.raises(TypeError, match="harmonic or a gridded field"):
        filtered(pm.AnalyticField((CMB, MOHO), lambda r, t, p: 0 * r), lmax=3)


def regional_box(prem, *, lat=(40.0, 60.0), lon=(-20.0, 20.0), depth=(25.0, 500.0),
                 vs=5000.0):
    """A regional grid of constant vsv over a box."""
    lats = np.arange(lat[0], lat[1] + 0.01, 1.0)
    lons = np.arange(lon[0], lon[1] + 0.01, 1.0)
    radius = R - np.array(depth[::-1]) * 1e3
    theta = np.radians(90.0 - lats[::-1])
    phi = np.radians(np.mod(lons, 360.0))
    phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
    values = np.full((radius.size, theta.size, phi.size), vs / 1e3)
    data = GridData(theta, phi, radius, {"vsv": values}, {"vsv": "km/s"})
    return RegionalGrid(data, {"vsv": ("vsv", None)})


def test_a_regional_grid_merges_inside_and_leaves_the_rest():
    prem = pm.PREM()
    box = regional_box(prem)
    grid = pm.equiangular(90, 180)
    m = merged(prem, box, name="vsv", width=np.radians(4.0), grid=grid,
               radial_width=(50e3, 0.0))
    lay = m.layer("low_velocity_zone")
    f = lay["vsv"]
    assert isinstance(f, pm.PerturbedField)
    r = 6200e3
    deep = pm.from_geographic(50.0, 0.0)
    # linear in radius between its nodes, while 5000 / R(r) - 1 is not
    assert float(f(r, *deep)) == pytest.approx(5000.0, rel=1e-4)
    node = lay.interval[0]
    assert float(f(node, *deep)) == pytest.approx(5000.0, rel=1e-12)
    away = pm.from_geographic(-30.0, 120.0)
    assert float(f(r, *away)) == pytest.approx(float(prem.layer(lay.index)["vsv"](r)),
                                               rel=1e-12)
    # across the box's southern edge: from outside to well inside, no jump
    lats = np.linspace(36.0, 50.0, 300)
    theta, phi = pm.from_geographic(lats, np.zeros_like(lats))
    across = np.asarray(f(np.full(lats.shape, r), theta, phi))
    assert np.max(np.abs(np.diff(across))) < 0.05 * abs(5000.0 - float(prem.layer(
        lay.index)["vsv"](r)))
    assert m.layer("ocean")["vsv"] is prem.layer("ocean")["vsv"]     # a fluid's left
    assert not isinstance(m.layer("lower_mantle")["vsv"], pm.PerturbedField)
    pm.testing.check_model(m)


def test_a_regional_grid_merges_into_a_model_in_other_scales():
    """A regional grid is SI: into a model in other scales its radii and
    its values go in the model's units, so the merged field is the SI
    model's in those units, the box's velocity at a node inside it."""
    from planetmodel.units import Scales
    prem = pm.PREM()
    box = regional_box(prem)
    grid = pm.equiangular(45, 90)
    si = merged(prem, box, name="vsv", width=np.radians(4.0), grid=grid,
                radial_width=(50e3, 0.0))
    rng = np.random.default_rng(9)
    for scales in (Scales(time=1000.0), Scales(length=1000.0, mass=1e12)):
        k, speed = scales.length, scales.length / scales.time
        m = merged(prem.converted(scales), box, name="vsv", width=np.radians(4.0),
                   grid=grid, radial_width=(50e3 / k, 0.0))
        lay = m.layer("low_velocity_zone")
        f = lay["vsv"]
        assert isinstance(f, pm.PerturbedField)
        deep = pm.from_geographic(50.0, 0.0)
        node = lay.interval[0]
        assert speed * float(f(node, *deep)) == pytest.approx(5000.0, rel=1e-12)
        lo, hi = prem.layer("low_velocity_zone").interval
        r = rng.uniform(lo, hi, 50)
        theta, phi = pm.from_geographic(rng.uniform(30.0, 70.0, 50),
                                        rng.uniform(-30.0, 30.0, 50))
        want = si.layer(lay.index)["vsv"](r, theta, phi)
        assert np.any(np.abs(want / prem.layer(lay.index)["vsv"](r) - 1) > 0.01)
        assert np.allclose(speed * f(r / k, theta, phi), want, rtol=1e-10, atol=0)
        assert not isinstance(m.layer("lower_mantle")["vsv"], pm.PerturbedField)


def test_merge_refusals():
    prem = pm.PREM()
    box = regional_box(prem)
    grid = pm.equiangular(18, 36)
    with pytest.raises(ValueError, match="not 'vs'"):
        merged(prem, box, name="vs", width=0.1, grid=grid)
    with pytest.raises(ValueError, match="non-negative angle"):
        merged(prem, box, name="vsv", width=-1.0, grid=grid)
    deep = regional_box(prem, depth=(3000.0, 3100.0))
    with pytest.raises(ValueError, match="reaches no layer"):
        merged(prem, deep, name="vsv", width=0.1, grid=grid)
