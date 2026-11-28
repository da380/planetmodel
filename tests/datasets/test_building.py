"""A data set's perturbations transplanted onto a model of one's own."""
import hashlib

import numpy as np
import pytest
from scipy.interpolate import PPoly

import planetmodel as pm
from planetmodel.datasets import (Dataset, Source, SphReader, demeaned, over_reference,
                                  resampled, standalone, transplanted)
from planetmodel.materials import voigt_vs

CMB, MOHO = 3480e3, 6346.6e3
R = 6371e3


def harmonic(lo=CMB, hi=MOHO, lmax=6, seed=0, scale=0.01):
    rng = np.random.default_rng(seed)
    c = scale * rng.standard_normal((2, 2, lmax + 1, lmax + 1))
    for l in range(lmax + 1):
        c[:, :, l, l + 1:] = 0.0
    c[:, 1, :, 0] = 0.0
    return pm.HarmonicField((lo, hi), PPoly(c[:, None] / np.array([hi - lo, 1.0])
                                            .reshape(2, 1, 1, 1, 1), [lo, hi]))


@pytest.fixture(scope="module")
def s40():
    """A data set like S40RTS: vs relative to PREM between the CMB and Moho."""
    return over_reference(pm.PREM(), {"vs": harmonic()}, kind="relative", name="s40")


def test_vs_lands_on_vsv_and_vsh_and_the_moduli_follow(s40):
    prem = pm.PREM()
    m = transplanted(prem, s40)
    mantle = [lay for lay in m.layers
              if lay.interval[0] >= CMB and lay.interval[1] <= MOHO]
    assert len(mantle) == 8
    rng = np.random.default_rng(1)
    for lay in mantle:
        lo, hi = lay.interval
        r, t = rng.uniform(lo, hi, 20), rng.uniform(0, np.pi, 20)
        p = rng.uniform(0, 6, 20)
        d = harmonic()(r, t, p)
        for name in ("vsv", "vsh"):
            assert isinstance(lay[name], pm.PerturbedField)
            want = prem.layer(lay.index)[name](r) * (1 + d)
            assert np.allclose(lay[name](r, t, p), want, rtol=1e-13)
        assert np.allclose(lay["L"](r, t, p), lay["rho"](r) * lay["vsv"](r, t, p) ** 2,
                           rtol=1e-12)
        assert np.allclose(lay["N"](r, t, p), lay["rho"](r) * lay["vsh"](r, t, p) ** 2,
                           rtol=1e-12)
    assert not isinstance(m.layer("inner_core")["vsv"], pm.PerturbedField)
    assert not isinstance(m.layer("upper_crust")["vsv"], pm.PerturbedField)
    assert isinstance(m, pm.PREM)
    pm.testing.check_model(m)


@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_a_model_in_other_scales_takes_the_source_in_its_units(s40, mode):
    """The source is SI and the model is not: the source is converted to
    the model's scales, so the perturbed velocity is the SI model's, in
    the model's units and at its radii, whether the source's perturbation
    or its values are kept."""
    from planetmodel.units import Scales
    prem = pm.PREM()
    rng = np.random.default_rng(11)
    for scales in (Scales(time=1000.0), Scales(length=1000.0, mass=1e12),
                   Scales.geophysical(R)):
        k, speed = scales.length, scales.length / scales.time
        m = transplanted(prem.converted(scales), s40, mode=mode)
        assert m.scales == scales
        for name in ("lower_mantle", "transition_zone_upper"):
            lay, ref = m.layer(name), prem.layer(name)
            lo, hi = ref.interval
            r, t = rng.uniform(lo, hi, 20), rng.uniform(0, np.pi, 20)
            p = rng.uniform(0, 6, 20)
            d = harmonic()(r, t, p)
            for vel in ("vsv", "vsh"):
                assert isinstance(lay[vel], pm.PerturbedField)
                # the source's own reference is PREM: its values over the
                # model's Voigt average are 1 + d again
                want = ref[vel](r) * (1 + d)
                assert np.allclose(speed * lay[vel](r / k, t, p), want, rtol=1e-10)


def test_the_model_writes_and_reads_back(s40, tmp_path):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.modelfile import read_model, write_model
    m = transplanted(pm.PREM(), s40, lmax=4)
    path = write_model(tmp_path / "prem_s40.nc", m, grid=pm.gauss_legendre(16))
    pm.testing.check_model_file(path, m)
    back = read_model(path)
    lay = back.layer("lower_mantle")
    r, t, p = 5e6, 1.1, 2.2
    assert float(lay["vsv"](r, t, p)) == pytest.approx(
        float(m.layer("lower_mantle")["vsv"](r, t, p)), rel=1e-6)


def test_names_land_where_they_are_told(s40):
    iso = pm.LayeredIsotropicElastic([0.0, 1221.5e3, CMB, MOHO, R],
                                     rho=[1.3e4, 1.1e4, 4.5e3, 2.8e3],
                                     vp=[1.1e4, 9e3, 1.2e4, 6.5e3],
                                     vs=[3.5e3, 0.0, 4.5e3, 3.6e3])
    m = transplanted(iso, s40)
    assert isinstance(m.layer(2)["vs"], pm.PerturbedField)
    only = transplanted(pm.PREM(), s40, names={"vs": "vsv"})
    lay = only.layer("lower_mantle")
    assert isinstance(lay["vsv"], pm.PerturbedField)
    assert not isinstance(lay["vsh"], pm.PerturbedField)
    pair = over_reference(pm.PREM(), {"vs": harmonic(), "vsv": harmonic(seed=1),
                                      "vsh": harmonic(seed=2)}, kind="relative")
    both = transplanted(pm.PREM(), pair)
    v = both.layer("lower_mantle")["vsv"]
    assert v.perturbation(5e6, 1.0, 1.0) == pytest.approx(
        float(harmonic(seed=1)(5e6, 1.0, 1.0)))          # vsv's own, not vs's
    density = pm.Model(pm.Geometry(pm.Skeleton([0.0, R])),
                       [{"rho": pm.constant_field((0.0, R), 5e3,
                                                  character=pm.DENSITY)}])
    with pytest.raises(ValueError, match="none of which the model holds"):
        transplanted(density, s40)


def test_layers_take_their_own_span_of_a_perturbed_source(s40):
    prem = pm.PREM()
    m = transplanted(prem, s40, layers=["lower_mantle"], ends=("d2741", "d771"))
    lay = m.layer("lower_mantle")
    rng = np.random.default_rng(7)
    r = rng.uniform(*lay.interval, 20)
    t, p = rng.uniform(0, np.pi, 20), rng.uniform(0, 6, 20)
    assert np.allclose(lay["vsv"].perturbation(r, t, p), harmonic()(r, t, p),
                       rtol=1e-12)                 # ends bound the span: identity
    assert not isinstance(m.layer("transition_zone_lower")["vsv"], pm.PerturbedField)


def test_the_mean_removed_and_a_band_kept(s40):
    m = transplanted(pm.PREM(), s40, remove_mean=True, lmax=3, lmin=1)
    pert = m.layer("lower_mantle")["vsv"].perturbation
    assert isinstance(pert, pm.HarmonicField) and pert.lmax == 3
    assert abs(float(pert.spherical_mean()(5e6))) < 1e-18
    with pytest.raises(ValueError, match="needs lmax="):
        transplanted(pm.PREM(), s40, lmin=1)


def gridded_values():
    """A data set of values: vs on a Gauss-Legendre grid, linear in radius."""
    grid = pm.gauss_legendre(8)
    v = pm.AnalyticField((CMB, MOHO), lambda r, t, p: 4.6e3 + 30.0 * np.cos(t)
                         + 20.0 * np.sin(t) ** 2 * np.cos(2 * p) + 1e-4 * (r - CMB))
    g = pm.GriddedField.sampled(v, grid, breaks=np.linspace(CMB, MOHO, 7), ngll=2,
                                radial="linear")
    return standalone({"vs": g}, surface=R, name="values"), v


def test_a_data_set_of_values_is_filtered_through_its_grid():
    src, v = gridded_values()
    ref = pm.PREM()
    m = transplanted(pm.PREM(), src, source_reference=ref, lmax=4, remove_mean=True)
    pert = m.layer("lower_mantle")["vsv"].perturbation
    pm.testing.check_model(m)
    lay = ref.layer("lower_mantle")
    r = lay.interval[0]                         # a node: the radial rule exact
    grid = pm.gauss_legendre(8)
    th, ph = grid.colatitudes[:, None], grid.longitudes[None, :]
    rel = v(r, th, ph) / voigt_vs(lay["vsv"](r), lay["vsh"](r)) - 1
    want = rel - np.sum(grid.weights[:, None] * rel) / (2 * grid.nphi)
    assert np.allclose(pert(r, th, ph), want, rtol=0.0, atol=1e-9)


def test_resampled_and_demeaned():
    src, v = gridded_values()
    g = src.layer(0)["vs"]
    ref = pm.constant_field((CMB, MOHO), 4.5e3)
    composed = pm.ComposedField(lambda a, b: a / b - 1, (g, ref), character=pm.SCALAR)
    rs = resampled(composed)
    assert isinstance(rs, pm.GriddedField) and rs.grid == g.grid
    nodes = rs.radii[:, :, None, None]
    th = g.grid.colatitudes[None, None, :, None]
    ph = g.grid.longitudes[None, None, None]
    assert np.allclose(rs.values, composed(nodes, th, ph), rtol=0.0, atol=1e-15)
    d = demeaned(rs)
    assert np.allclose(d.spherical_mean()(np.linspace(CMB, MOHO, 9)), 0.0, atol=1e-15)
    h = demeaned(harmonic())
    assert np.all(h.coefficients.c[..., 0, 0, 0] == 0.0)
    with pytest.raises(TypeError, match="no gridded field"):
        resampled(pm.AnalyticField((CMB, MOHO), lambda r, t, p: 0 * r))


def test_an_entry_is_loaded_and_its_reference_used(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANETMODEL_DATA", str(tmp_path / "cache"))
    lmax = 2
    rng = np.random.default_rng(3)
    lines = [f"{'':10}{lmax:5d} {'1' * (lmax + 1)} {24:3d} {'000' + '1' * 21}"]
    for _ in range(21):
        block = 1e-2 * rng.standard_normal((lmax + 1) ** 2)
        i = 0
        for l in range(lmax + 1):
            lines.append(" ".join(f"{x:.6e}" for x in block[i:i + 2 * l + 1]))
            i += 2 * l + 1
    path = tmp_path / "m.sph"
    path.write_text("\n".join(lines) + "\n")
    d = Dataset("m", (Source(path.as_uri(), hashlib.sha256(path.read_bytes())
                             .hexdigest()),),
                SphReader(), title="t", citation=("c",), licence="l", reference="PREM")
    m = transplanted(pm.PREM(), d)
    assert isinstance(m.layer("lower_mantle")["vsh"], pm.PerturbedField)
    box = path.parent / "box.nc"
    assert not box.exists()
    with pytest.raises(TypeError, match="not a model to transplant"):
        class Regional:
            def read(self, paths, dataset, **options):
                from planetmodel.datasets import RegionalGrid
                return RegionalGrid.__new__(RegionalGrid)
        transplanted(pm.PREM(), Dataset("r", d.sources, Regional(), title="t",
                                        citation=("c",), licence="l"))


def test_a_source_of_values_reaching_beyond_the_layers_asked_for():
    """A viscosity source reaching into layers that hold none is
    re-referenced only over the layers asked for."""
    prem = pm.PREM(ocean=False)
    for lay in prem.layers:
        lo, hi = lay.interval
        if CMB <= lo and hi <= 6291e3:
            prem = prem.with_field(lay.index, "viscosity", pm.constant_field(
                (lo, hi), 1e21, name="viscosity"))
    mantle = [lay.index for lay in prem.layers if "viscosity" in lay]

    def eta(r, t, p):
        return 1e21 * np.exp(np.cos(t) * np.sin(p) + (r - 5e6) / 2e6)

    src = standalone({"viscosity": pm.AnalyticField((3500e3, 6340e3), eta,
                                                    name="viscosity")}, surface=R)
    m = transplanted(prem, src, names={"viscosity": "viscosity"}, mode="absolute",
                     kind="log", layers=mantle, extend=True)
    i = m.layer("lower_mantle").index
    f = m.layer(i)["viscosity"]
    assert isinstance(f, pm.PerturbedField) and f.kind == "log"
    r, t, p = np.array([4000e3, 5200e3]), np.array([0.4, 2.1]), np.array([1.0, 3.0])
    assert np.allclose(f(r, t, p), eta(r, t, p), rtol=1e-12)
    assert "viscosity" not in m.layer("lid")
