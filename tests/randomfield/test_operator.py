"""The radial operator family: spectra against Bessel roots, orthonormality,
powers, the direct operators and white noise."""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import spherical_jn

from planetmodel import RadialMesh, PREM
from planetmodel.mesh1d import Mesh1D
from planetmodel.randomfield import RadialOperatorFamily


def _dense_from_band(band):
    q, n = band.shape[0] - 1, band.shape[1]
    A = np.zeros((n, n))
    for m in range(q + 1):
        u = q - m
        for j in range(u, n):
            A[j - u, j] = band[m, j]
    return A + np.triu(A, 1).T


def _neumann_wavenumbers(l, R, xmax=28.0):
    """Positive roots of j_l'(x) = 0 as wavenumbers k = x / R."""
    f = lambda x: spherical_jn(l, x, derivative=True)
    xs = np.linspace(0.25, xmax, 4000)
    v = f(xs)
    roots = [brentq(f, a, b) for a, b, va, vb
             in zip(xs[:-1], xs[1:], v[:-1], v[1:]) if va * vb < 0]
    return np.asarray(roots) / R


def test_ball_neumann_bessel_spectrum():
    """Discrete eigenvalues on a ball match 1 + kappa (x_nl / R)^2."""
    R, kap = 1.0, 0.04
    fam = RadialOperatorFamily(Mesh1D([0.0, R], ngll=6, drmax=0.05), kappa=kap)
    assert fam.is_ball
    for l in (0, 1, 5):
        k = _neumann_wavenumbers(l, R)
        exact = 1.0 + kap * k ** 2
        if l == 0:
            exact = np.concatenate(([1.0], exact))   # the constant mode
        got = fam.eigvalsh(l)[:exact.size]
        assert np.max(np.abs(got - exact) / exact) < 1e-8
    assert abs(fam.eigvalsh(0)[0] - 1.0) < 1e-12     # exactly the constant
    assert fam.embed(0, np.ones(fam.ndof(0))).shape == (fam.mesh.nglob,)
    assert fam.embed(2, np.ones(fam.ndof(2)))[0] == 0.0


def test_family_algebra():
    """Orthonormality, residual, powers, inner products, Robin shift."""
    mesh = Mesh1D([0.4, 0.55, 1.0], ngll=5, drmax=0.06)
    kap = lambda r: (0.05 + 0.02 * r) ** 2
    kap_h = lambda r: (0.08 + 0.01 * r) ** 2
    fam = RadialOperatorFamily(mesh, kappa=kap, kappa_h=kap_h, robin=(1.0, 2.0))
    assert not fam.is_ball
    for l in (0, 3):
        theta, Phi = fam.eig(l)
        M = fam.mass(l)
        assert np.min(theta) > 1.0                   # Robin shifts theta_min
        G = Phi.T @ (M[:, None] * Phi)
        assert np.max(np.abs(G - np.eye(theta.size))) < 1e-9
        band, _ = fam.pencil(l)
        A = _dense_from_band(band)
        assert np.max(np.abs(A @ Phi - (M[:, None] * Phi) * theta)) < 1e-8

        rng = np.random.default_rng(l)
        v = rng.standard_normal(fam.ndof(l))
        w = fam.apply_power(l, 0.7, fam.apply_power(l, -0.7, v))
        assert np.max(np.abs(w - v)) < 1e-9
        u = rng.standard_normal(fam.ndof(l))
        assert abs(fam.inner(l, 0.6, u, v) - fam.inner(l, 0.6, v, u)) < 1e-9
        assert fam.inner(l, -1.3, v, v) > 0.0
        assert np.isfinite(fam.logdet(l))

    theta_t, Phi_t = fam.eig(2, theta_max=3.0)       # truncated fetch
    theta_f, _ = fam.eig(2)                          # then full, cache upgrade
    assert theta_t.size < theta_f.size
    assert np.max(np.abs(theta_f[:theta_t.size] - theta_t)) < 1e-12


def test_apply_solve_and_white_noise():
    mesh = Mesh1D([0.0, 1.0], ngll=5, drmax=0.1)
    fam = RadialOperatorFamily(mesh, kappa=0.01)
    for l in (0, 4):
        rng = np.random.default_rng(l)
        v = rng.standard_normal(fam.ndof(l))
        band, mass = fam.pencil(l)
        assert np.allclose(fam.apply(l, v), _dense_from_band(band) @ v / mass)
        assert np.allclose(fam.apply(l, v), fam.apply_power(l, 1.0, v))
        assert np.allclose(fam.solve(l, v), fam.apply_power(l, -1.0, v))
        assert np.isclose(fam.inner(l, 0.0, v, fam.apply(l, v)),
                          fam.inner(l, 1.0, v, v))
        assert np.allclose(fam.solve(l, fam.apply(l, v)), v)
        V = rng.standard_normal((fam.ndof(l), 3))
        assert np.allclose(fam.solve(l, fam.apply(l, V)), V)
        wn = fam.white_noise(l, rng=rng, size=20000)
        assert wn.shape == (fam.ndof(l), 20000)
        assert np.max(np.abs((wn ** 2).mean(axis=1) * fam.mass(l) - 1.0)) < 0.06
        assert fam.white_noise(l, rng=1).shape == (fam.ndof(l),)
    with pytest.raises(ValueError):
        fam.apply(2, np.ones(3))


def test_refusals():
    mesh = Mesh1D([0.0, 1.0], ngll=4, drmax=0.5)
    with pytest.raises(ValueError):
        RadialOperatorFamily(mesh, weight="one")
    with pytest.raises(ValueError):
        RadialOperatorFamily(mesh, kappa=-1.0)
    with pytest.raises(ValueError):
        RadialOperatorFamily(mesh, robin=-1.0)
    with pytest.raises(TypeError):
        RadialOperatorFamily(np.linspace(0, 1, 5))
    with pytest.raises(ValueError):
        RadialOperatorFamily(mesh, kappa=np.ones(3))


def test_family_on_radialmesh():
    """The family composes with a model's RadialMesh."""
    mesh = RadialMesh(PREM(ocean=False), ngll=5, lmax=4)
    fam = RadialOperatorFamily(mesh, kappa=(2.0e5) ** 2)   # 200 km scale
    th = fam.eigvalsh(2)
    assert th[0] > 1.0 - 1e-8 and np.all(np.diff(th) >= -1e-6)
    v = np.ones(fam.ndof(2))
    assert fam.inner(2, 1.0, v, v) > fam.inner(2, 0.0, v, v) > 0.0
    assert "cached degrees: [2]" in repr(fam)
