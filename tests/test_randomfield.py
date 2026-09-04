"""Checks for the operator-family and random-field layers of planetmodel.

Run with `pytest` from the repository root, or directly with
`python tests/test_randomfield.py` (no pytest required).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn

from planetmodel import PREM, RadialMesh, Skeleton
from planetmodel.mesh1d import Mesh1D, lagrange_basis
from planetmodel.randomfield import LayeredGRF, RadialGRF, SphericalGRF
from planetmodel.sobolev import RadialOperatorFamily


def _dense_from_band(band):
    """Dense symmetric matrix from upper banded storage (test helper)."""
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


def test_mesh1d_and_ppoly():
    """Mesh1D geometry; to_ppoly reproduces the SEM interpolant exactly."""
    mesh = Mesh1D([0.0, 0.3, 1.0], ngll=5, drmax=0.15)
    assert mesh.nspec == 2 + 5 and mesh.nglob == mesh.nspec * 4 + 1
    assert 0.3 in mesh.left and 0.3 in mesh.right   # pinned breakpoint
    assert np.max(mesh.right - mesh.left) <= mesh.drmax + 1e-12
    assert mesh.element_of(0.3) == 2                # boundary -> upper element

    nodal = np.sin(3.0 * mesh.r) + mesh.r ** 2
    pp = mesh.to_ppoly(nodal)
    rng = np.random.default_rng(1)
    for e in range(mesh.nspec):
        x = rng.uniform(mesh.left[e], mesh.right[e], 7)
        exact = lagrange_basis(mesh.r[e], x) @ nodal[e]
        assert np.max(np.abs(pp(x) - exact)) < 1e-12
        dx = (mesh.deriv @ nodal[e]) / mesh.jac[e]
        assert np.max(np.abs(pp.derivative()(mesh.r[e, 1:-1]) - dx[1:-1])) < 1e-9

    sub = mesh.to_ppoly(nodal, elements=(2, 5))
    assert sub.x[0] == mesh.left[2] and sub.x[-1] == mesh.right[4]
    xm = 0.5 * (mesh.left[3] + mesh.right[3])
    assert abs(sub(xm) - pp(xm)) < 1e-13


def test_ball_neumann_bessel_spectrum():
    """Discrete eigenvalues on a ball match 1 + kappa (x_nl / R)^2."""
    R, kap = 1.0, 0.04
    fam = RadialOperatorFamily(Mesh1D([0.0, R], ngll=6, drmax=0.05),
                               kappa=kap)
    for l in (0, 1, 5):
        k = _neumann_wavenumbers(l, R)
        exact = 1.0 + kap * k ** 2
        if l == 0:
            exact = np.concatenate(([1.0], exact))   # the constant mode
        got = fam.eigvalsh(l)[:exact.size]
        assert np.max(np.abs(got - exact) / exact) < 1e-8
    assert abs(fam.eigvalsh(0)[0] - 1.0) < 1e-12     # exactly the constant


def test_family_algebra():
    """Orthonormality, residual, powers, inner products, Robin shift."""
    mesh = Mesh1D([0.4, 0.55, 1.0], ngll=5, drmax=0.06)
    kap = lambda r: (0.05 + 0.02 * r) ** 2
    kap_h = lambda r: (0.08 + 0.01 * r) ** 2
    fam = RadialOperatorFamily(mesh, kappa=kap, kappa_h=kap_h,
                               robin=(1.0, 2.0))
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


def test_radial_grf_normalization_and_stats():
    """Exact discrete sigma(r); Monte Carlo statistics agree."""
    sig = lambda r: 1.0 + r
    g = RadialGRF(0.5, 1.0, nu=1.5, lam=0.08, sigma=sig, tol=1e-10)
    assert np.max(np.abs(np.sqrt((g._B ** 2).sum(axis=1)) - sig(g.r))) < 1e-10
    assert np.max(np.abs(np.diag(g.covariance()) - sig(g.r) ** 2)) < 1e-10

    u = g.sample(rng=0, size=8000)
    assert u.shape == (8000, g.r.size)
    emp = u.std(axis=0)
    assert np.max(np.abs(emp / sig(g.r) - 1.0)) < 0.06
    assert np.max(np.abs(u.mean(axis=0)) / sig(g.r)) < 0.05

    pp = g.to_ppoly(u[0])
    assert np.max(np.abs(pp(g.r) - u[0])) < 1e-10
    assert pp.x[0] == 0.5 and pp.x[-1] == 1.0

    ball = RadialGRF(0.0, 1.0, nu=1.0, lam=0.2, sigma=2.0)
    assert ball.r[0] == 0.0                          # axis is a physical node
    assert np.max(np.abs(np.sqrt((ball._B ** 2).sum(axis=1)) - 2.0)) < 1e-10
    assert np.isfinite(ball.sample(rng=4)[0])


def test_padding_insensitivity():
    """Doubling the pad barely moves the physical-node covariance."""
    kw = dict(nu=1.0, lam=0.04, sigma=1.0, tol=1e-12)
    g2 = RadialGRF(0.6, 1.0, pad_factor=2.0, **kw)
    g4 = RadialGRF(0.6, 1.0, pad_factor=4.0, **kw)
    assert np.array_equal(g2.r, g4.r)                # identical physical nodes
    diff = np.max(np.abs(g2.covariance() - g4.covariance()))
    assert diff < 5e-4


def test_spherical_grf():
    """Degrees, axis behaviour, layout, and pole-synthesis MC variance."""
    g = SphericalGRF(0.7, 1.0, nu=2.0, lam=0.15, sigma=1.3, lmax=12)
    assert g.nmodes.size == 13 and np.all(g.nmodes >= 1)
    c = g.sample(rng=3)
    assert c.shape == (2, 13, 13, g.r.size)
    assert np.all(c[1, :, 0] == 0.0)                 # no sine at m = 0
    l_idx = np.arange(13)
    tri = np.arange(13)[None, :] > l_idx[:, None]
    assert np.all(c[:, tri] == 0.0)                  # m <= l only

    # variance of the field at the pole equals sigma^2 exactly in law:
    # u(pole) = sum_l c[0, l, 0] sqrt((2l+1)/4pi) for orthonormal harmonics
    yl0 = np.sqrt((2 * l_idx + 1) / (4.0 * np.pi))
    rng = np.random.default_rng(7)
    j = g.r.size // 2
    vals = np.array([g.sample(rng)[0, :, 0, j] @ yl0 for _ in range(2500)])
    assert abs(vals.var() / 1.3 ** 2 - 1.0) < 0.15

    ball = SphericalGRF(0.0, 0.8, nu=2.0, lam=0.15, lmax=6)
    cb = ball.sample(rng=5)
    assert ball.r[0] == 0.0
    assert np.all(cb[:, 1:, :, 0] == 0.0)            # l >= 1 vanish at centre
    assert cb[0, 0, 0, 0] != 0.0
    assert np.all(np.isfinite(cb))

    auto = SphericalGRF(0.7, 1.0, nu=2.0, lam=0.15, tol=1e-4)
    assert 2 < auto.lmax < 60                        # auto degree cut bites


def test_layered_grf():
    """Direct sums over a Skeleton: Field output, zeros, discontinuity."""
    sk = Skeleton([0.0, 0.4, 0.7, 1.0])
    g = LayeredGRF(sk, nu=1.5, lam=[0.10, 0.05, 0.03],
                   sigma=[1.0, 2.0, 0.5], name="dv")
    f = g.sample(rng=11)
    assert f.skeleton is sk and f.name == "dv"
    assert np.all(np.isfinite([f[i](0.5 * sum(sk.interval(i)))
                               for i in range(3)]))
    assert f[1](0.7) != f[2](0.7)                    # generic discontinuity
    assert np.isfinite(f[0](0.0))                    # ball layer at centre

    part = LayeredGRF(sk, nu=1.0, lam=0.05, layers=(1,))
    fp = part.sample(rng=2)
    assert fp[0](0.2) == 0.0 and fp[2](0.9) == 0.0
    assert fp[1](0.55) != 0.0
    assert part[1].nmodes >= 1


def test_family_on_radialmesh():
    """The family composes with a model RadialMesh (Sobolev use case)."""
    mesh = RadialMesh(PREM(ocean=False), ngll=5, lmax=4)
    fam = RadialOperatorFamily(mesh, kappa=(2.0e5) ** 2)   # 200 km scale
    th = fam.eigvalsh(2)
    assert th[0] > 1.0 - 1e-8 and np.all(np.diff(th) >= -1e-6)
    v = np.ones(fam.ndof(2))
    assert fam.inner(2, 1.0, v, v) > fam.inner(2, 0.0, v, v) > 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
