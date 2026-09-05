"""Moduli conversions, the Voigt and Bond forms, ElasticField, and what a
layer's fields imply."""
import numpy as np
import pytest

from planetmodel.character import DENSITY, ELASTIC, SCALAR, VECTOR, Symmetry
from planetmodel.fields import AnalyticField, RadialField, constant_field
from planetmodel.frames import (VOIGT_PAIRS, bond_matrix, spherical_frame,
                                voigt_to_tensor)
from planetmodel.layerfunction import NumericLayer, PolynomialLayer, polynomial_layer
from planetmodel.materials import (ElasticField, elastic_moduli, is_fluid, kappa_mu,
                                   kappa_mu_from_moduli, moduli,
                                   moduli_from_velocities, velocities_from_moduli,
                                   voigt_matrix)
from planetmodel.testing import check_field

A_PREM = 6371e3
ISO, VTI = Symmetry.ISOTROPIC, Symmetry.VTI

# PREM's polynomials in the paper's units: the outer core (fluid), the
# lower mantle (isotropic), and the anisotropic zone below the lithosphere.
OC = (1221.5e3, 3480.0e3)
OC_POLY = {"rho": [12.5815, -1.2638, -3.6426, -5.5281],
           "vp": [11.0487, -4.0362, 4.8023, -13.5732], "vs": [0.0]}
LM = (3630.0e3, 5600.0e3)
LM_POLY = {"rho": [7.9565, -6.4761, 5.5283, -3.0807],
           "vp": [24.9520, -40.4673, 51.4832, -26.6419],
           "vs": [11.1671, -13.7818, 17.4575, -9.2777]}
LVZ = (6151.0e3, 6291.0e3)
LVZ_POLY = {"rho": [2.6910, 0.6924], "vpv": [0.8317, 7.2180],
            "vph": [3.5908, 4.6172], "vsv": [5.8582, -1.4678],
            "vsh": [-1.0839, 5.7176], "eta": [3.3687, -2.4778]}


def layer_of(interval, polys):
    """A dict layer of exact radial fields; rho has weight 1."""
    return {n: RadialField(interval, polynomial_layer(c, interval, scale=A_PREM),
                           character=DENSITY if n == "rho" else SCALAR, name=n)
            for n, c in polys.items()}


def values_of(layer, r):
    return {n: f(r) for n, f in layer.items()}


def is_polynomial(f) -> bool:
    return isinstance(f, RadialField) and isinstance(f.function, PolynomialLayer)


@pytest.fixture
def ti_inputs():
    rng = np.random.default_rng(11)
    n = 400
    rho = rng.uniform(1e3, 1.3e4, n)
    vpv = rng.uniform(2e3, 1.4e4, n)
    vsv = rng.uniform(1e3, 7e3, n)
    return dict(rho=rho, vpv=vpv, vsv=vsv, vph=vpv * rng.uniform(0.9, 1.1, n),
                vsh=vsv * rng.uniform(0.9, 1.1, n), eta=rng.uniform(0.8, 1.2, n))


# ----------------------------------------------------------- conversions

def test_moduli_velocity_round_trip(ti_inputs):
    m = moduli_from_velocities(**ti_inputs)
    back = velocities_from_moduli(ti_inputs["rho"], **m)
    for k in ("vpv", "vsv", "vph", "vsh", "eta"):
        assert np.allclose(back[k], ti_inputs[k], rtol=1e-13), k


def test_isotropic_defaults_collapse():
    rho, vp, vs = 3300.0, 8000.0, 4500.0
    m = moduli_from_velocities(rho, vp, vs)
    assert m["A"] == pytest.approx(m["C"]) and m["L"] == pytest.approx(m["N"])
    assert m["F"] == pytest.approx(m["A"] - 2.0 * m["L"])
    kappa, mu = kappa_mu_from_moduli(**m)
    assert mu == pytest.approx(rho * vs ** 2)
    assert kappa == pytest.approx(rho * (vp ** 2 - 4.0 * vs ** 2 / 3.0))


def test_voigt_average_reduces_to_isotropic(ti_inputs):
    rho, vp, vs = ti_inputs["rho"], ti_inputs["vpv"], ti_inputs["vsv"]
    kappa, mu = kappa_mu_from_moduli(**moduli_from_velocities(rho, vp, vs))
    assert np.allclose(mu, rho * vs ** 2, rtol=1e-13)
    assert np.allclose(kappa, rho * (vp ** 2 - 4.0 * vs ** 2 / 3.0), rtol=1e-13)


def test_velocities_from_moduli_guards_zero_density_and_undefined_eta():
    v = velocities_from_moduli(np.array([0.0, 3000.0]), 1e11, 1e11, 0.0, 5e10, 5e10)
    assert v["vpv"].tolist()[0] == 0.0 and v["vsv"].tolist()[0] == 0.0
    assert v["vpv"][1] == pytest.approx(np.sqrt(1e11 / 3000.0))
    assert np.all(v["eta"] == 1.0)                # A - 2 L = 0
    assert velocities_from_moduli(3000.0, 1e11, 1e11, -1e10, 0.0, 0.0)["vsv"] == 0.0


def test_conversions_run_on_fields_exactly():
    lay = layer_of(LVZ, LVZ_POLY)
    m = moduli_from_velocities(lay["rho"], lay["vpv"], lay["vsv"], vph=lay["vph"],
                               vsh=lay["vsh"], eta=lay["eta"])
    r = np.linspace(*LVZ, 40)
    want = moduli_from_velocities(**values_of(lay, r))
    for k in "ACFLN":
        assert is_polynomial(m[k]) and m[k].character == DENSITY
        assert np.allclose(m[k](r), want[k], rtol=1e-13)
    assert m["A"].function.degree == 3 and m["F"].function.degree == 4
    kappa, mu = kappa_mu_from_moduli(**m)
    k0, m0 = kappa_mu_from_moduli(**want)
    assert is_polynomial(kappa) and is_polynomial(mu)
    assert np.allclose(kappa(r), k0, rtol=1e-13) and np.allclose(mu(r), m0, rtol=1e-13)
    iso = moduli_from_velocities(lay["rho"], lay["vpv"], lay["vsv"])
    assert iso["A"] is iso["C"] or np.allclose(iso["A"](r), iso["C"](r))
    assert np.allclose(iso["F"](r), (iso["A"] - 2.0 * iso["L"])(r), rtol=1e-13)


# ----------------------------------------------------------------- Voigt

def test_isotropic_voigt_structure():
    kappa, mu = 1.3e11, 6.7e10
    v = voigt_matrix(ISO, {"kappa": kappa, "mu": mu})
    lam = kappa - 2.0 * mu / 3.0
    assert v.shape == (6, 6) and v.dtype == np.float64
    assert np.allclose(np.diag(v)[:3], lam + 2.0 * mu)
    assert np.allclose(np.diag(v)[3:], mu)
    assert v[0, 1] == pytest.approx(lam) and np.allclose(v[:3, 3:], 0.0)


def test_vti_voigt_reduces_to_isotropic():
    m = moduli_from_velocities(3300.0, 8000.0, 4500.0)
    kappa, mu = kappa_mu_from_moduli(**m)
    assert np.allclose(voigt_matrix(VTI, m),
                       voigt_matrix(ISO, {"kappa": kappa, "mu": mu}), rtol=1e-12)


@pytest.mark.parametrize("symmetry", [ISO, VTI])
def test_voigt_matrices_are_symmetric(symmetry):
    comps = ({"kappa": 1.3e11, "mu": 6.7e10} if symmetry is ISO
             else dict(zip("ACFLN", (3.1e11, 3.0e11, 1.1e11, 7e10, 7.4e10))))
    v = voigt_matrix(symmetry, comps)
    assert np.allclose(v, v.T)


def test_voigt_broadcasts_and_follows_the_dtype():
    v = voigt_matrix(ISO, {"kappa": np.ones((4, 2)) * 1e11, "mu": 5e10})
    assert v.shape == (4, 2, 6, 6)
    assert voigt_matrix(ISO, {"kappa": 2, "mu": 1}).dtype == np.float64
    c = voigt_matrix(VTI, dict(zip("ACFLN", (3.0, 3.0, 1.0, 1.0 + 0.1j, 1.0))))
    assert np.iscomplexobj(c) and c[4, 4] == 1.0 + 0.1j


def test_full_tensor_has_the_classical_symmetries():
    t = voigt_to_tensor(voigt_matrix(
        VTI, dict(zip("ACFLN", (3.1e11, 3.0e11, 1.1e11, 7e10, 7.4e10)))))
    assert t.shape == (3, 3, 3, 3)
    assert np.allclose(t, np.swapaxes(t, 0, 1))
    assert np.allclose(t, np.swapaxes(t, 2, 3))
    assert np.allclose(t, np.transpose(t, (2, 3, 0, 1)))


def test_unknown_symmetry_is_refused():
    with pytest.raises(ValueError, match="unknown symmetry"):
        voigt_matrix("cubic", {})


def vti_tensor(A, C, F, L, N, n):
    """The five-term invariant form of a transversely isotropic tensor
    with symmetry axis n."""
    d = np.eye(3)
    nn = np.outer(n, n)
    c1, c2, c3, c4, c5 = A - 2 * N, N, F - A + 2 * N, L - N, A + C - 2 * F - 4 * L
    return (c1 * np.einsum("ij,kl->ijkl", d, d)
            + c2 * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))
            + c3 * (np.einsum("ij,kl->ijkl", d, nn) + np.einsum("ij,kl->ijkl", nn, d))
            + c4 * (np.einsum("ik,jl->ijkl", nn, d) + np.einsum("il,jk->ijkl", nn, d)
                    + np.einsum("jk,il->ijkl", nn, d) + np.einsum("jl,ik->ijkl", nn, d))
            + c5 * np.einsum("i,j,k,l->ijkl", n, n, n, n))


def test_the_vti_voigt_matrix_has_its_axis_along_e_r():
    A, C, F, L, N = 3.1e11, 3.0e11, 1.1e11, 7.0e10, 7.4e10
    got = voigt_to_tensor(voigt_matrix(VTI, dict(zip("ACFLN", (A, C, F, L, N)))))
    want = vti_tensor(A, C, F, L, N, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(got, want, rtol=1e-15, atol=1e-15 * np.max(np.abs(want)))
    assert got[0, 0, 0, 0] == pytest.approx(C)
    assert got[1, 1, 1, 1] == pytest.approx(A)
    assert got[1, 2, 1, 2] == pytest.approx(N)
    assert got[0, 1, 0, 1] == pytest.approx(L)


# ------------------------------------------------------------------ Bond

def _random_rotation(rng, n):
    out = np.empty((n, 3, 3))
    for k in range(n):
        Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
        Q = Q * np.sign(np.diag(R))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        out[k] = Q
    return out


def _rotate_by_einsum(V, R):
    c = voigt_to_tensor(V)
    cr = np.einsum("...ia,...jb,...kc,...ld,...abcd->...ijkl", R, R, R, R, c)
    out = np.empty(V.shape)
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, m) in enumerate(VOIGT_PAIRS):
            out[..., a, b] = cr[..., i, j, k, m]
    return out


def _random_voigt(rng, n):
    M = rng.standard_normal((n, 6, 6))
    return 0.5 * (M + np.swapaxes(M, -1, -2))


def _relerr(got, want):
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def test_bond_matrix_matches_the_einsum_rotation():
    rng = np.random.default_rng(4)
    V, R = _random_voigt(rng, 64), _random_rotation(rng, 64)
    M = bond_matrix(R)
    assert _relerr(M @ V @ np.swapaxes(M, -1, -2), _rotate_by_einsum(V, R)) < 1e-13


def test_bond_matrix_matches_the_einsum_rotation_in_the_spherical_frame():
    rng = np.random.default_rng(5)
    R = spherical_frame(rng.uniform(0.0, np.pi, 48), rng.uniform(-np.pi, np.pi, 48))
    V = _random_voigt(rng, 48)
    M = bond_matrix(R)
    assert _relerr(M @ V @ np.swapaxes(M, -1, -2), _rotate_by_einsum(V, R)) < 1e-13


def test_an_isotropic_voigt_matrix_is_frame_invariant():
    rng = np.random.default_rng(6)
    V = voigt_matrix(ISO, {"kappa": rng.uniform(1e10, 4e11, 32),
                           "mu": rng.uniform(1e10, 2e11, 32)})
    M = bond_matrix(_random_rotation(rng, 32))
    assert _relerr(M @ V @ np.swapaxes(M, -1, -2), V) < 1e-13


# ---------------------------------------------------------- ElasticField

def iso_field(name="elastic_moduli"):
    lay = layer_of(LM, LM_POLY)
    mu = lay["rho"] * lay["vs"] ** 2
    kappa = lay["rho"] * lay["vp"] ** 2 - (4.0 / 3.0) * mu
    return ElasticField(ISO, {"kappa": kappa, "mu": mu}, name=name)


def vti_field(name="elastic_moduli"):
    lay = layer_of(LVZ, LVZ_POLY)
    m = moduli_from_velocities(lay["rho"], lay["vpv"], lay["vsv"], vph=lay["vph"],
                               vsh=lay["vsh"], eta=lay["eta"])
    return ElasticField(VTI, m, name=name)


@pytest.mark.parametrize("field", [iso_field(), vti_field()])
def test_elastic_fields_pass_the_contract(field):
    check_field(field)


def test_elastic_field_attributes():
    e = vti_field()
    assert e.character is ELASTIC and e.symmetry is VTI and e.interval == LVZ
    assert e.name == "elastic_moduli" and e.is_radial and e.stored_shape == (6, 6)
    assert list(e.moduli) == ["A", "C", "F", "L", "N"]
    assert list(iso_field().moduli) == ["kappa", "mu"]
    assert "vti" in repr(e) and "elastic_moduli" in repr(e)
    e2 = e.renamed("C")
    assert e2.name == "C" and e2.symmetry is VTI and e2.moduli["A"] is e.moduli["A"]
    assert e.renamed(None).name is None


def test_elastic_field_values_are_the_voigt_matrix_of_the_moduli():
    e = vti_field()
    r = np.linspace(*LVZ, 7)
    v = e(r, 0.3, 0.4)
    assert v.shape == (7, 6, 6) and v.dtype == np.float64
    want = voigt_matrix(VTI, {k: f(r) for k, f in e.moduli.items()})
    assert np.allclose(v, want, rtol=1e-13)
    assert e(r[:, None], np.zeros(3), 0.1).shape == (7, 3, 6, 6)
    with pytest.raises(ValueError, match="theta and phi"):
        e(r)


def test_voigt_false_expands_to_the_full_tensor_in_both_frames():
    e = vti_field()
    r, th, ph = 6.2e6, 0.7, 1.1
    for frame in ("spherical", "cartesian"):
        v = e.evaluate(r, th, ph, frame=frame)
        t = e.evaluate(r, th, ph, frame=frame, voigt=False)
        assert t.shape == (3, 3, 3, 3)
        assert np.allclose(t, voigt_to_tensor(v, rank=4))


def test_cartesian_elastic_tensor_matches_the_einsum_rotation():
    e = vti_field()
    r = 6.2e6
    assert not np.isclose(float(e.moduli["A"](r)), float(e.moduli["C"](r)))
    rng = np.random.default_rng(7)
    theta, phi = rng.uniform(0.0, np.pi, 12), rng.uniform(-np.pi, np.pi, 12)
    R = spherical_frame(theta, phi)
    sph = e.evaluate(r, theta, phi, voigt=False)
    want = np.einsum("...ia,...jb,...kc,...ld,...abcd->...ijkl", R, R, R, R, sph)
    got = e.evaluate(r, theta, phi, frame="cartesian", voigt=False)
    assert _relerr(got, want) < 1e-13
    assert _relerr(voigt_to_tensor(e.evaluate(r, theta, phi, frame="cartesian")),
                   want) < 1e-13
    iso = iso_field()
    r = np.linspace(*LM, 5)
    assert np.allclose(iso.evaluate(r, theta[:5], phi[:5], frame="cartesian"),
                       iso(r, theta[:5], phi[:5]), rtol=1e-12)


def test_isotropic_promotes_to_vti_exactly():
    iso = iso_field()
    vti = iso.as_symmetry(VTI)
    assert vti.symmetry is VTI and vti.name == iso.name
    assert list(vti.moduli) == ["A", "C", "F", "L", "N"]
    kappa, mu = iso.moduli["kappa"].function, iso.moduli["mu"].function
    for k, hand in (("A", kappa + (4.0 / 3.0) * mu), ("C", kappa + (4.0 / 3.0) * mu),
                    ("F", kappa - (2.0 / 3.0) * mu), ("L", mu), ("N", mu)):
        f = vti.moduli[k]
        assert is_polynomial(f) and f.name == k and f.character == DENSITY
        assert np.allclose(f.function.ppoly.c, hand.ppoly.c, rtol=1e-15)
    r = np.linspace(*LM, 9)
    assert np.allclose(iso(r, 0.2, 0.3), vti(r, 0.2, 0.3), rtol=1e-13)
    assert iso.as_symmetry(ISO) is iso and vti.as_symmetry(VTI) is vti
    with pytest.raises(ValueError, match="narrow"):
        vti.as_symmetry(ISO)
    with pytest.raises(ValueError, match="unknown symmetry"):
        vti.as_symmetry("cubic")


def test_on_interval_and_rescaled_go_through_the_moduli():
    e = vti_field()
    lo, hi = LVZ
    wider = e.on_interval(lo - 1e5, hi + 1e5)
    assert isinstance(wider, ElasticField) and wider.symmetry is VTI
    assert np.allclose(wider.interval, (lo - 1e5, hi + 1e5))
    assert all(is_polynomial(f) for f in wider.moduli.values())
    assert np.isfinite(wider(hi + 5e4, 0.1, 0.2)).all()
    g = e.rescaled(k=2.0, v=0.5)
    assert isinstance(g, ElasticField) and all(is_polynomial(f)
                                               for f in g.moduli.values())
    assert np.allclose(g(2.0 * 6.2e6, 0.1, 0.2), 0.5 * e(6.2e6, 0.1, 0.2), rtol=1e-12)


def test_elastic_field_refusals():
    lay = layer_of(LM, LM_POLY)
    k, m = lay["rho"], lay["vs"]
    with pytest.raises(ValueError, match="missing"):
        ElasticField(VTI, {"A": k})
    with pytest.raises(ValueError, match="only"):
        ElasticField(ISO, {"kappa": k, "mu": m, "A": k})
    with pytest.raises(ValueError, match="unknown symmetry"):
        ElasticField("cubic", {"kappa": k, "mu": m})
    with pytest.raises(TypeError, match="not a Field"):
        ElasticField(ISO, {"kappa": k, "mu": 3.0})
    v = RadialField(LM, [1.0, 0.0, 0.0], character=VECTOR)
    with pytest.raises(ValueError, match="rank"):
        ElasticField(ISO, {"kappa": k, "mu": v})
    other = constant_field(1.0, OC, character=DENSITY)
    with pytest.raises(ValueError, match="different intervals"):
        ElasticField(ISO, {"kappa": k, "mu": other})


def test_moduli_may_be_analytic():
    a = AnalyticField(LM, lambda r, t, p: 1e11 * (1 + 0.1 * np.cos(t)))
    b = constant_field(5e10, LM)
    e = ElasticField(ISO, {"kappa": a, "mu": b})
    assert not e.is_radial
    r = np.linspace(*LM, 5)
    v = e(r, 0.4, 0.0)
    assert np.allclose(v, voigt_matrix(ISO, {"kappa": a(r, 0.4, 0.0), "mu": 5e10}))
    check_field(e)


# ------------------------------------------------ the functions of a layer

class NamedLayer:
    """A layer with `names` and a `name`, as a model's layer has."""

    def __init__(self, name, fields):
        self.name = name
        self._fields = dict(fields)

    @property
    def names(self):
        return tuple(self._fields)

    def __contains__(self, name):
        return name in self._fields

    def __getitem__(self, name):
        return self._fields[name]


def test_is_fluid_reads_the_shear_fields():
    assert is_fluid(layer_of(OC, OC_POLY)) is True
    assert is_fluid(layer_of(LM, LM_POLY)) is False
    assert is_fluid(layer_of(LVZ, LVZ_POLY)) is False
    lay = layer_of(LVZ, LVZ_POLY)
    lay["vsh"] = RadialField(LVZ, 0.0)
    assert is_fluid(lay) is False                 # vsv still bears shear
    lay["vsv"] = RadialField(LVZ, 0.0)
    assert is_fluid(lay) is True
    assert is_fluid({"L": RadialField(OC, 0.0), "N": constant_field(0.0, OC)})
    assert not is_fluid({"mu": constant_field(1.0, OC)})


def test_is_fluid_samples_where_there_are_no_coefficients():
    numeric = RadialField(OC, lambda r: 0.0 * r)
    assert isinstance(numeric.function, NumericLayer)
    assert is_fluid({"vs": numeric}) is True
    assert is_fluid({"vs": RadialField(OC, lambda r: r / A_PREM)}) is False
    assert is_fluid({"mu": AnalyticField(OC, lambda r, t, p: 0.0 * r * t)}) is True
    assert is_fluid({"mu": AnalyticField(OC, lambda r, t, p: np.cos(t))}) is False


def test_is_fluid_refuses_a_layer_without_shear_fields():
    with pytest.raises(KeyError, match="holds none"):
        is_fluid({"rho": RadialField(OC, 1.0), "vp": RadialField(OC, 1.0)})
    with pytest.raises(KeyError, match="'outer core'"):
        is_fluid(NamedLayer("outer core", {"rho": RadialField(OC, 1.0)}))
    assert is_fluid(NamedLayer("outer core", layer_of(OC, OC_POLY))) is True


def test_moduli_from_the_five_held_are_the_fields_themselves():
    lay = layer_of(LVZ, LVZ_POLY)
    five = moduli_from_velocities(lay["rho"], lay["vpv"], lay["vsv"], vph=lay["vph"],
                                  vsh=lay["vsh"], eta=lay["eta"])
    got = moduli(five)
    assert list(got) == ["A", "C", "F", "L", "N"]
    assert all(got[k] is five[k] for k in got)
    e = elastic_moduli(five)
    assert e.symmetry is VTI and e.name == "elastic_moduli"
    assert all(e.moduli[k] is five[k] for k in got)


def test_moduli_from_kappa_and_mu_are_exact_and_isotropic():
    lay = layer_of(LM, LM_POLY)
    mu = lay["rho"] * lay["vs"] ** 2
    kappa = lay["rho"] * lay["vp"] ** 2 - (4.0 / 3.0) * mu
    iso = {"kappa": kappa, "mu": mu}
    m = moduli(iso)
    assert list(m) == ["A", "C", "F", "L", "N"]
    r = np.linspace(*LM, 20)
    for k, f in m.items():
        assert is_polynomial(f) and f.name == k
    assert np.allclose(m["A"](r), kappa(r) + 4.0 * mu(r) / 3.0, rtol=1e-13)
    assert np.allclose(m["C"](r), m["A"](r)) and np.allclose(m["L"](r), mu(r))
    assert np.allclose(m["F"](r), kappa(r) - 2.0 * mu(r) / 3.0, rtol=1e-13)
    e = elastic_moduli(iso)
    assert e.symmetry is ISO and e.moduli["kappa"] is kappa and e.moduli["mu"] is mu
    k2, m2 = kappa_mu(iso)
    assert is_polynomial(k2) and k2.name == "kappa" and m2.name == "mu"
    assert np.allclose(k2(r), kappa(r), rtol=1e-13)
    assert np.allclose(m2(r), mu(r), rtol=1e-13)


def test_moduli_from_the_ti_velocities():
    lay = layer_of(LVZ, LVZ_POLY)
    m = moduli(lay)
    r = np.linspace(*LVZ, 20)
    want = moduli_from_velocities(**values_of(lay, r))
    for k, f in m.items():
        assert is_polynomial(f) and f.name == k and f.character == DENSITY
        assert np.allclose(f(r), want[k], rtol=1e-13)
    e = elastic_moduli(lay)
    assert e.symmetry is VTI
    assert np.allclose(e(r, 0.1, 0.2), voigt_matrix(VTI, want), rtol=1e-13)
    k, mu = kappa_mu(lay)
    k0, mu0 = kappa_mu_from_moduli(**want)
    assert np.allclose(k(r), k0, rtol=1e-13) and np.allclose(mu(r), mu0, rtol=1e-13)


def test_moduli_from_rho_vp_vs_are_isotropic():
    for interval, polys in ((LM, LM_POLY), (OC, OC_POLY)):
        lay = layer_of(interval, polys)
        r = np.linspace(*interval, 20)
        rho, vp, vs = lay["rho"](r), lay["vp"](r), lay["vs"](r)
        m = moduli(lay)
        want = moduli_from_velocities(rho, vp, vs)
        for k, f in m.items():
            assert is_polynomial(f) and f.name == k
            assert np.allclose(f(r), want[k], rtol=1e-12)
        e = elastic_moduli(lay)
        assert e.symmetry is ISO and list(e.moduli) == ["kappa", "mu"]
        assert np.allclose(e(r, 0.1, 0.2), voigt_matrix(VTI, want), rtol=1e-12)
        k, mu = kappa_mu(lay)
        assert np.allclose(mu(r), rho * vs ** 2, rtol=1e-12)
        assert np.allclose(k(r), rho * (vp ** 2 - 4.0 * vs ** 2 / 3.0), rtol=1e-12)
    fluid = moduli(layer_of(OC, OC_POLY))
    assert fluid["L"].function.is_zero() and fluid["N"].function.is_zero()


def test_the_five_moduli_win_over_the_velocities():
    lay = layer_of(LVZ, LVZ_POLY)
    lay.update({k: constant_field(float(i + 1), LVZ, character=DENSITY)
                for i, k in enumerate("ACFLN")})
    assert moduli(lay)["N"] is lay["N"]
    iso = layer_of(LM, LM_POLY)
    iso["kappa"], iso["mu"] = constant_field(2.0, LM), constant_field(1.0, LM)
    assert elastic_moduli(iso).moduli["mu"] is iso["mu"]


def test_moduli_refuse_a_layer_without_enough():
    with pytest.raises(KeyError, match="rho with vp and vs"):
        moduli({"rho": RadialField(OC, 1.0)})
    with pytest.raises(KeyError, match="'core'"):
        elastic_moduli(NamedLayer("core", {"rho": RadialField(OC, 1.0),
                                           "vpv": RadialField(OC, 1.0)}))
    with pytest.raises(KeyError, match="kappa"):
        kappa_mu({"mu": RadialField(OC, 1.0)})
