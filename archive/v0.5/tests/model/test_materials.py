"""Moduli conversions, Voigt expansion, and ElasticField."""
import numpy as np
import pytest

from planetmodel import PREM, ELASTIC, RadialField, Skeleton, Symmetry
from planetmodel.model.units import Dimensions
from planetmodel.model.materials import (ElasticField, voigt_to_tensor,
                                    kappa_mu_from_moduli,
                                    moduli_from_velocities, voigt_matrix,
                                    velocities_from_moduli)


@pytest.fixture
def ti_inputs():
    rng = np.random.default_rng(11)
    n = 400
    rho = rng.uniform(1e3, 1.3e4, n)
    vpv = rng.uniform(2e3, 1.4e4, n)
    vsv = rng.uniform(1e3, 7e3, n)
    return dict(rho=rho, vpv=vpv, vsv=vsv,
                vph=vpv * rng.uniform(0.9, 1.1, n),
                vsh=vsv * rng.uniform(0.9, 1.1, n),
                eta=rng.uniform(0.8, 1.2, n))


# ------------------------------------------------------------- conversions

def test_moduli_velocity_round_trip(ti_inputs):
    """Velocities -> moduli -> velocities recovers the inputs."""
    m = moduli_from_velocities(**ti_inputs)
    back = velocities_from_moduli(ti_inputs["rho"], **m)
    for k in ("vpv", "vsv", "vph", "vsh", "eta"):
        assert np.allclose(back[k], ti_inputs[k], rtol=1e-13), k


def test_isotropic_defaults_collapse_correctly():
    """Omitting vph, vsh, eta gives A = C and F = A - 2L = lambda."""
    rho, vp, vs = 3300.0, 8000.0, 4500.0
    m = moduli_from_velocities(rho, vp, vs)
    assert m["A"] == pytest.approx(m["C"])
    assert m["L"] == pytest.approx(m["N"])
    assert m["F"] == pytest.approx(m["A"] - 2.0 * m["L"])
    kappa, mu = kappa_mu_from_moduli(**m)
    assert mu == pytest.approx(rho * vs ** 2)
    assert kappa == pytest.approx(rho * (vp ** 2 - 4.0 * vs ** 2 / 3.0))


def test_voigt_average_reduces_to_isotropic(ti_inputs):
    """kappa_mu_from_moduli on an isotropic medium returns it unchanged."""
    rho, vp, vs = ti_inputs["rho"], ti_inputs["vpv"], ti_inputs["vsv"]
    m = moduli_from_velocities(rho, vp, vs)
    kappa, mu = kappa_mu_from_moduli(**m)
    assert np.allclose(mu, rho * vs ** 2, rtol=1e-13)
    assert np.allclose(kappa, rho * (vp ** 2 - 4.0 * vs ** 2 / 3.0), rtol=1e-13)


def test_matches_the_loading_solvers_conventions():
    """The formulas here are the ones loading.py already uses."""
    prem = PREM()
    r = 5000e3
    vals = {n: float(prem[n].evaluate(r))
            for n in ("rho", "vpv", "vsv", "vph", "vsh", "eta")}
    m = moduli_from_velocities(**vals)
    assert m["A"] == pytest.approx(vals["rho"] * vals["vph"] ** 2)
    assert m["C"] == pytest.approx(vals["rho"] * vals["vpv"] ** 2)
    assert m["L"] == pytest.approx(vals["rho"] * vals["vsv"] ** 2)
    assert m["N"] == pytest.approx(vals["rho"] * vals["vsh"] ** 2)
    assert m["F"] == pytest.approx(vals["eta"] * (m["A"] - 2.0 * m["L"]))


# ------------------------------------------------------------------- Voigt

def test_isotropic_voigt_has_the_expected_structure():
    kappa, mu = 1.3e11, 6.7e10
    v = voigt_matrix(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu})
    lam = kappa - 2.0 * mu / 3.0
    assert v.shape == (6, 6)
    assert np.allclose(np.diag(v)[:3], lam + 2.0 * mu)
    assert np.allclose(np.diag(v)[3:], mu)
    assert v[0, 1] == pytest.approx(lam)
    assert np.allclose(v[:3, 3:], 0.0)


def test_vti_reduces_to_isotropic_when_the_medium_is():
    """The two constructions agree on an isotropic medium."""
    rho, vp, vs = 3300.0, 8000.0, 4500.0
    m = moduli_from_velocities(rho, vp, vs)
    kappa, mu = kappa_mu_from_moduli(**m)
    assert np.allclose(voigt_matrix(Symmetry.VTI, m),
                       voigt_matrix(Symmetry.ISOTROPIC,
                                    {"kappa": kappa, "mu": mu}), rtol=1e-12)


@pytest.mark.parametrize("symmetry", [Symmetry.ISOTROPIC, Symmetry.VTI])
def test_voigt_matrices_are_symmetric(symmetry):
    comps = ({"kappa": 1.3e11, "mu": 6.7e10} if symmetry is Symmetry.ISOTROPIC
             else dict(zip("ACFLN", (3.1e11, 3.0e11, 1.1e11, 7e10, 7.4e10))))
    v = voigt_matrix(symmetry, comps)
    assert np.allclose(v, v.T)


def test_voigt_broadcasts():
    v = voigt_matrix(Symmetry.ISOTROPIC,
                     {"kappa": np.ones((4, 2)) * 1e11, "mu": 5e10})
    assert v.shape == (4, 2, 6, 6)


def test_full_tensor_has_the_classical_symmetries():
    """The second elasticity tensor: minor and major symmetries both."""
    comps = dict(zip("ACFLN", (3.1e11, 3.0e11, 1.1e11, 7e10, 7.4e10)))
    t = voigt_to_tensor(voigt_matrix(Symmetry.VTI, comps))
    assert t.shape == (3, 3, 3, 3)
    assert np.allclose(t, np.swapaxes(t, 0, 1))          # minor, first pair
    assert np.allclose(t, np.swapaxes(t, 2, 3))          # minor, second pair
    assert np.allclose(t, np.transpose(t, (2, 3, 0, 1)))  # major


def test_unsupported_symmetry_says_so():
    with pytest.raises(NotImplementedError, match="ORTHOTROPIC|GENERAL|not yet"):
        voigt_matrix(Symmetry.GENERAL, {})


# ------------------------------------------------------------ ElasticField

@pytest.fixture
def prem_vti():
    prem = PREM()
    vals = {n: prem[n] for n in ("rho", "vpv", "vsv", "vph", "vsh", "eta")}
    sk = prem.skeleton
    m = {}
    for k in ("A", "C", "F", "L", "N"):
        def make(k=k):
            def fn(r, _v=vals):
                v = {n: np.asarray(_v[n].evaluate(r), dtype=float)
                     for n in ("rho", "vpv", "vsv", "vph", "vsh", "eta")}
                got = moduli_from_velocities(v["rho"], v["vpv"], v["vsv"],
                                             vph=v["vph"], vsh=v["vsh"],
                                             eta=v["eta"])
                return got[k]
            return fn
        f = make()
        m[k] = RadialField(sk, [(lambda fn=f, i=i: (lambda r: fn(r)))()
                                for i in range(sk.nlayers)], name=k)
    return ElasticField(Symmetry.VTI, m, name="elastic_moduli")


def test_elastic_field_character_and_shapes(prem_vti):
    assert prem_vti.character is ELASTIC
    r = np.linspace(1e5, 6e6, 7)
    assert prem_vti.evaluate(r).shape == (7, 6, 6)
    assert prem_vti.evaluate(r, voigt=False).shape == (7, 3, 3, 3, 3)
    assert prem_vti.evaluate(1e6).shape == (6, 6)


def test_elastic_field_values_match_direct_conversion(prem_vti):
    prem = PREM()
    r = 5000e3
    vals = {n: float(prem[n].evaluate(r))
            for n in ("rho", "vpv", "vsv", "vph", "vsh", "eta")}
    want = voigt_matrix(Symmetry.VTI, moduli_from_velocities(**vals))
    assert np.allclose(prem_vti.evaluate(r), want, rtol=1e-12)


def test_isotropic_promotes_to_vti():
    sk = Skeleton([0.0, 1.0])
    const = lambda v: RadialField(sk, [lambda r, v=v: np.full_like(
        np.asarray(r, dtype=float), v)])
    iso = ElasticField(Symmetry.ISOTROPIC,
                       {"kappa": const(1.3e11), "mu": const(6.7e10)})
    vti = iso.as_symmetry(Symmetry.VTI)
    assert vti.symmetry is Symmetry.VTI
    assert np.allclose(iso.evaluate(0.5), vti.evaluate(0.5), rtol=1e-12)


def test_as_symmetry_is_identity_for_the_same_class(prem_vti):
    assert prem_vti.as_symmetry(Symmetry.VTI) is prem_vti


def test_narrowing_symmetry_is_refused(prem_vti):
    with pytest.raises(ValueError, match="narrow"):
        prem_vti.as_symmetry(Symmetry.ISOTROPIC)


def test_wrong_moduli_are_rejected():
    sk = Skeleton([0.0, 1.0])
    f = RadialField(sk, [lambda r: np.zeros_like(np.asarray(r, dtype=float))])
    with pytest.raises(ValueError, match="missing"):
        ElasticField(Symmetry.VTI, {"A": f})
    with pytest.raises(ValueError, match="only"):
        ElasticField(Symmetry.ISOTROPIC, {"kappa": f, "mu": f, "A": f})


def test_moduli_must_share_a_skeleton():
    f1 = RadialField(Skeleton([0.0, 1.0]),
                     [lambda r: np.zeros_like(np.asarray(r, dtype=float))])
    f2 = RadialField(Skeleton([0.0, 2.0]),
                     [lambda r: np.zeros_like(np.asarray(r, dtype=float))])
    with pytest.raises(ValueError, match="one skeleton"):
        ElasticField(Symmetry.ISOTROPIC, {"kappa": f1, "mu": f2})


# --------------------------------------------------- Appendix B.9: the Bond
#
# The oracle, written before the closed form and kept: rotate the full
# (3,3,3,3) tensor by einsum and reduce, then demand that the 6x6 Bond
# matrix reproduce it.  A transposed index or a missing factor of two in
# the Bond matrix survives inspection easily and does not survive this.

def _random_rotation(rng, n=1):
    """n random proper rotations, by QR of a Gaussian with det fixed to +1."""
    out = np.empty((n, 3, 3))
    for k in range(n):
        Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
        Q = Q * np.sign(np.diag(R))          # fix QR's sign ambiguity
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        out[k] = Q
    return out


def _rotate_by_einsum(V, R):
    """Reduce -> rotate the full tensor -> read the six pairs back."""
    from planetmodel.model.materials import _VOIGT_PAIRS
    c = voigt_to_tensor(V)
    cr = np.einsum("...ia,...jb,...kc,...ld,...abcd->...ijkl", R, R, R, R, c)
    out = np.empty(V.shape)
    for a, (i, j) in enumerate(_VOIGT_PAIRS):
        for b, (k, m) in enumerate(_VOIGT_PAIRS):
            out[..., a, b] = cr[..., i, j, k, m]
    return out


def _random_voigt(rng, n):
    """Random symmetric 6x6 matrices: a second elasticity tensor's shape."""
    M = rng.standard_normal((n, 6, 6))
    return 0.5 * (M + np.swapaxes(M, -1, -2))


def _relerr(got, want):
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def test_bond_matrix_matches_the_einsum_rotation():
    """M V M^T is the Voigt reduction of R R R R : c, for random R."""
    from planetmodel.model.materials import bond_matrix
    rng = np.random.default_rng(4)
    n = 64
    V = _random_voigt(rng, n)
    R = _random_rotation(rng, n)
    M = bond_matrix(R)
    got = M @ V @ np.swapaxes(M, -1, -2)
    want = _rotate_by_einsum(V, R)
    assert _relerr(got, want) < 1e-13, _relerr(got, want)


def test_bond_matrix_matches_the_einsum_rotation_in_the_spherical_frame():
    """The same, with R the local (e_r, e_theta, e_phi) frame."""
    from planetmodel.model.frames import spherical_frame
    from planetmodel.model.materials import bond_matrix
    rng = np.random.default_rng(5)
    n = 48
    theta = rng.uniform(0.0, np.pi, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    R = spherical_frame(theta, phi)
    assert np.allclose(np.einsum("...ki,...kj->...ij", R, R),
                       np.eye(3), atol=1e-14), "the frame is not orthonormal"
    V = _random_voigt(rng, n)
    M = bond_matrix(R)
    got = M @ V @ np.swapaxes(M, -1, -2)
    want = _rotate_by_einsum(V, R)
    assert _relerr(got, want) < 1e-13, _relerr(got, want)


def test_a_physically_isotropic_voigt_matrix_is_frame_invariant():
    """Isotropy is a statement about every frame, so nothing may move."""
    from planetmodel.model.materials import bond_matrix
    rng = np.random.default_rng(6)
    n = 32
    kappa = rng.uniform(1e10, 4e11, n)
    mu = rng.uniform(1e10, 2e11, n)
    V = voigt_matrix(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu})
    M = bond_matrix(_random_rotation(rng, n))
    got = M @ V @ np.swapaxes(M, -1, -2)
    assert _relerr(got, V) < 1e-13, _relerr(got, V)


def test_prem_cartesian_elastic_tensor_matches_the_einsum_rotation():
    """PREM in the low-velocity zone, where A != C, so the frame matters."""
    from planetmodel.model.frames import spherical_frame
    prem = PREM()
    elastic = prem["elastic_moduli"]
    r = 6.2e6
    assert not np.isclose(float(prem.A.evaluate(r)), float(prem.C.evaluate(r)))

    rng = np.random.default_rng(7)
    theta = rng.uniform(0.0, np.pi, 12)
    phi = rng.uniform(-np.pi, np.pi, 12)
    R = spherical_frame(theta, phi)

    sph = elastic.evaluate(r, theta, phi, voigt=False)
    want = np.einsum("...ia,...jb,...kc,...ld,...abcd->...ijkl",
                     R, R, R, R, sph)
    got = elastic.evaluate(r, theta, phi, frame="cartesian", voigt=False)
    assert _relerr(got, want) < 1e-13, _relerr(got, want)

    # ... and the Voigt route agrees with expanding the rotated matrix.
    v_cart = elastic.evaluate(r, theta, phi, frame="cartesian")
    assert _relerr(voigt_to_tensor(v_cart), want) < 1e-13


def test_the_vti_voigt_matrix_has_its_axis_along_e_r():
    """The layout question, settled by the invariant form of B.8.3.

    In the frame (e_r, e_theta, e_phi) a radial symmetry axis is the
    first basis vector, so the tensor voigt_matrix(VTI) expands to must
    equal the five-term invariant form with n = (1, 0, 0) -- C at rrrr,
    A at tttt and pppp, N at tptp, L at rtrt and rprp.  The seismological
    tables put the axis at the third index because their third axis is
    vertical; copying that layout here made the axis e_phi, which is the
    error this test exists to keep out.
    """
    from .test_appendix_b8 import vti_tensor
    from planetmodel.model.materials import voigt_to_tensor

    A, C, F, L, N = 3.1e11, 3.0e11, 1.1e11, 7.0e10, 7.4e10
    got = voigt_to_tensor(voigt_matrix(
        Symmetry.VTI, dict(zip("ACFLN", (A, C, F, L, N)))), rank=4)
    want = vti_tensor(A, C, F, L, N, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(got, want, rtol=1e-15, atol=1e-15 * np.max(np.abs(want)))
    assert got[0, 0, 0, 0] == pytest.approx(C)
    assert got[1, 1, 1, 1] == pytest.approx(A)
    assert got[1, 2, 1, 2] == pytest.approx(N)
    assert got[0, 1, 0, 1] == pytest.approx(L)


def test_a_promoted_symmetry_keeps_its_dimensions():
    from planetmodel import RadialField, Skeleton, Symmetry
    sk = Skeleton([0.0, 1.0, 2.0])
    mod = {k: RadialField(sk, [lambda r: 1.0 + 0 * r] * 2, name=k,
                          dimensions=Dimensions.MASS / Dimensions.LENGTH
                          / Dimensions.TIME ** 2)
           for k in ("kappa", "mu")}
    iso = ElasticField(Symmetry.ISOTROPIC, mod, name="elastic_moduli")
    vti = iso.as_symmetry(Symmetry.VTI)
    assert all(f.dimensions is not None for f in vti.components.values())


def test_moduli_on_equal_but_distinct_skeletons_are_accepted():
    from planetmodel import RadialField, Skeleton, Symmetry
    a, b = Skeleton([0.0, 1.0, 2.0]), Skeleton([0.0, 1.0, 2.0])
    k = RadialField(a, [lambda r: 1.0 + 0 * r] * 2, name="kappa")
    m = RadialField(b, [lambda r: 1.0 + 0 * r] * 2, name="mu")
    ElasticField(Symmetry.ISOTROPIC, {"kappa": k, "mu": m})
