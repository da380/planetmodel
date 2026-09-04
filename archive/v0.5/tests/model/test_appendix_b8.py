"""The Appendix B.8 formulas, verified independently of any planetmodel code.

These are the derivations the library implements -- physical-symmetry
pull-backs and the first elasticity tensor -- built here from numpy
alone, directly from the appendix, so that the implementation is pinned
by constructions that do not depend on it.
"""
import numpy as np
import pytest

rng = np.random.default_rng(7)

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def random_deformation(scale=0.15):
    F = np.eye(3) + scale * rng.normal(size=(3, 3))
    return F, np.linalg.det(F)


def isotropic_tensor(kappa, mu):
    lam = kappa - 2.0 * mu / 3.0
    d = np.eye(3)
    return (lam * np.einsum("ij,kl->ijkl", d, d)
            + mu * (np.einsum("ik,jl->ijkl", d, d)
                    + np.einsum("il,jk->ijkl", d, d)))


def vti_tensor(A, C, F, L, N, n):
    """The five-term invariant form of B.8.3."""
    d = np.eye(3)
    nn = np.outer(n, n)
    c1, c2, c3, c4, c5 = A - 2 * N, N, F - A + 2 * N, L - N, A + C - 2 * F - 4 * L
    return (c1 * np.einsum("ij,kl->ijkl", d, d)
            + c2 * (np.einsum("ik,jl->ijkl", d, d)
                    + np.einsum("il,jk->ijkl", d, d))
            + c3 * (np.einsum("ij,kl->ijkl", d, nn)
                    + np.einsum("ij,kl->ijkl", nn, d))
            + c4 * (np.einsum("ik,jl->ijkl", nn, d)
                    + np.einsum("il,jk->ijkl", nn, d)
                    + np.einsum("jk,il->ijkl", nn, d)
                    + np.einsum("jl,ik->ijkl", nn, d))
            + c5 * np.einsum("i,j,k,l->ijkl", n, n, n, n))


def push4(T, F, J):
    """B.8.1, rank 4."""
    return np.einsum("iA,jB,kC,lD,ABCD->ijkl", F, F, F, F, T) / J


def random_second_tensor():
    """A random tensor with the full minor and major symmetries."""
    M6 = rng.normal(size=(6, 6))
    M6 = 0.5 * (M6 + M6.T)
    CC = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, l) in enumerate(VOIGT_PAIRS):
            for ii, jj in ((i, j), (j, i)):
                for kk, ll in ((k, l), (l, k)):
                    CC[ii, jj, kk, ll] = M6[a, b]
    return CC


def first_tensor(CC, S, F):
    """B.8.4: A_iAjB = d_ij S_AB + F_iC F_jD CC_CADB."""
    return (np.einsum("ij,AB->iAjB", np.eye(3), S)
            + np.einsum("iC,jD,CADB->iAjB", F, F, CC))


# ------------------------------------------------------------- B.8.1

def test_push_forward_preserves_the_full_symmetries():
    F, J = random_deformation()
    c = push4(random_second_tensor(), F, J)
    assert np.allclose(c, np.einsum("ijkl->jikl", c))
    assert np.allclose(c, np.einsum("ijkl->ijlk", c))
    assert np.allclose(c, np.einsum("ijkl->klij", c))


def test_pull_then_push_is_the_identity():
    F, J = random_deformation()
    T = random_second_tensor()
    Finv = np.linalg.inv(F)
    pulled = J * np.einsum("Ai,Bj,Ck,Dl,ijkl->ABCD", Finv, Finv, Finv, Finv, T)
    assert np.allclose(push4(pulled, F, J), T, rtol=1e-12)


# ------------------------------------------------------------- B.8.3 form

def test_vti_coefficients_reproduce_the_voigt_entries():
    """The six entries that define VTI, with the axis along e3, exactly."""
    A, C, F_, L, N = 3.1e11, 3.0e11, 1.1e11, 7.0e10, 7.4e10
    T = vti_tensor(A, C, F_, L, N, np.array([0.0, 0.0, 1.0]))
    assert T[0, 0, 0, 0] == pytest.approx(A, rel=1e-15)
    assert T[0, 0, 1, 1] == pytest.approx(A - 2 * N, rel=1e-15)
    assert T[0, 0, 2, 2] == pytest.approx(F_, rel=1e-15)
    assert T[2, 2, 2, 2] == pytest.approx(C, rel=1e-15)
    assert T[0, 2, 0, 2] == pytest.approx(L, rel=1e-15)
    assert T[0, 1, 0, 1] == pytest.approx(N, rel=1e-15)


def test_vti_collapses_to_isotropic():
    """c3 = c4 = c5 = 0 exactly at A = C, L = N, F = A - 2L."""
    A, L = 3.0e11, 7.0e10
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    T = vti_tensor(A, A, A - 2 * L, L, L, n)
    kappa = A - 4.0 * L / 3.0
    assert np.allclose(T, isotropic_tensor(kappa, L), rtol=1e-14)


# --------------------------------------------------- B.8.2 / B.8.3 pull-backs

def test_isotropic_pull_back_round_trips():
    """Two scalars with the density rule, structure carried by Cinv."""
    F, J = random_deformation()
    Cinv = np.linalg.inv(F.T @ F)
    kappa, mu = 1.3e11, 6.7e10
    lam = kappa - 2.0 * mu / 3.0
    Aref = J * (lam * np.einsum("AB,CD->ABCD", Cinv, Cinv)
                + mu * (np.einsum("AC,BD->ABCD", Cinv, Cinv)
                        + np.einsum("AD,BC->ABCD", Cinv, Cinv)))
    want = isotropic_tensor(kappa, mu)
    # atol scaled to the tensor: the structural zeros come back with
    # roundoff of order eps * |moduli|, which the default atol of
    # allclose (1e-8, absolute) would fail at these magnitudes.
    assert np.allclose(push4(Aref, F, J), want,
                       rtol=1e-12, atol=1e-12 * np.max(np.abs(want)))


def test_vti_pull_back_round_trips():
    """Five scalars, Cinv, and the pulled axis Ntil = F^-1 n."""
    F, J = random_deformation()
    Finv = np.linalg.inv(F)
    Cinv = np.linalg.inv(F.T @ F)
    A, C, F_, L, N = 3.1e11, 3.0e11, 1.1e11, 7.0e10, 7.4e10
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    Nt = Finv @ n
    NN = np.outer(Nt, Nt)
    c1, c2, c3 = A - 2 * N, N, F_ - A + 2 * N
    c4, c5 = L - N, A + C - 2 * F_ - 4 * L
    Aref = J * (c1 * np.einsum("AB,CD->ABCD", Cinv, Cinv)
                + c2 * (np.einsum("AC,BD->ABCD", Cinv, Cinv)
                        + np.einsum("AD,BC->ABCD", Cinv, Cinv))
                + c3 * (np.einsum("AB,CD->ABCD", Cinv, NN)
                        + np.einsum("AB,CD->ABCD", NN, Cinv))
                + c4 * (np.einsum("AC,BD->ABCD", NN, Cinv)
                        + np.einsum("AD,BC->ABCD", NN, Cinv)
                        + np.einsum("BC,AD->ABCD", NN, Cinv)
                        + np.einsum("BD,AC->ABCD", NN, Cinv))
                + c5 * np.einsum("A,B,C,D->ABCD", Nt, Nt, Nt, Nt))
    want = vti_tensor(A, C, F_, L, N, n)
    assert np.allclose(push4(Aref, F, J), want,
                       rtol=1e-12, atol=1e-12 * np.max(np.abs(want)))


# ------------------------------------------------------------- B.8.4

def test_action_formula_matches_the_materialised_contraction():
    """(A G) = G S + F K without building 81 components."""
    F, _ = random_deformation()
    CC, S = random_second_tensor(), rng.normal(size=(3, 3))
    S = 0.5 * (S + S.T)
    G = rng.normal(size=(3, 3))
    direct = np.einsum("iAjB,jB->iA", first_tensor(CC, S, F), G)
    K = np.einsum("CADB,DB->CA", CC, F.T @ G)
    assert np.allclose(G @ S + F @ K, direct, rtol=1e-13)


def test_first_tensor_has_major_symmetry_always():
    F, _ = random_deformation()
    S = rng.normal(size=(3, 3))
    A = first_tensor(random_second_tensor(), 0.5 * (S + S.T), F)
    assert np.allclose(A, np.einsum("iAjB->jBiA", A), rtol=1e-12)


def test_reduction_at_identity_and_zero_stress():
    """F = I, S = 0: the action is the classical CC : sym(G)."""
    CC = random_second_tensor()
    G = rng.normal(size=(3, 3))
    A0 = first_tensor(CC, np.zeros((3, 3)), np.eye(3))
    got = np.einsum("iAjB,jB->iA", A0, G)
    want = np.einsum("iAkl,kl->iA", CC, 0.5 * (G + G.T))
    assert np.allclose(got, want, rtol=1e-13)


def test_equilibrium_referred_tensor_is_c_plus_delta_sigma():
    """LAM = c + d_ik sigma_jl, the Maitra & Al-Attar decomposition."""
    F, J = random_deformation()
    CC, S = random_second_tensor(), rng.normal(size=(3, 3))
    S = 0.5 * (S + S.T)
    sigma = F @ S @ F.T / J
    A = first_tensor(CC, S, F)
    LAM = np.einsum("jA,lB,iAkB->ijkl", F, F, A) / J
    want = push4(CC, F, J) + np.einsum("ik,jl->ijkl", np.eye(3), sigma)
    assert np.allclose(LAM, want, rtol=1e-12)


def test_minor_symmetry_violation_is_exactly_the_stress_pattern():
    """The two-tensor distinction, quantitatively: with sigma /= 0 the
    minor symmetry fails by d_ik sigma_jl - d_jk sigma_il and nothing
    else."""
    F, J = random_deformation()
    CC, S = random_second_tensor(), rng.normal(size=(3, 3))
    S = 0.5 * (S + S.T)
    sigma = F @ S @ F.T / J
    A = first_tensor(CC, S, F)
    LAM = np.einsum("jA,lB,iAkB->ijkl", F, F, A) / J
    minor = LAM - np.einsum("ijkl->jikl", LAM)
    pattern = (np.einsum("ik,jl->ijkl", np.eye(3), sigma)
               - np.einsum("jk,il->ijkl", np.eye(3), sigma))
    assert np.allclose(minor, pattern, atol=1e-12 * np.max(np.abs(LAM)))


def test_stress_character_is_second_piola_kirchhoff():
    """The stored stress must be S: the rank-2 weight-1 push-forward of
    S is the Cauchy stress, which is what forces the identification."""
    F, J = random_deformation()
    S = rng.normal(size=(3, 3))
    S = 0.5 * (S + S.T)
    pushed = np.einsum("iA,jB,AB->ij", F, F, S) / J
    assert np.allclose(pushed, F @ S @ F.T / J, rtol=1e-14)
