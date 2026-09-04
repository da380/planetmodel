"""Fast checks for the planetmodel package.

Run with `pytest` from the repository root, or directly with
`python tests/test_planetmodel.py` (no pytest required).
"""
import numpy as np

from planetmodel import PREM, RadialMesh
from planetmodel.loading import (love_number_table, love_numbers,
                                 reciprocity_residual, solve_degree, ti_moduli,
                                 voigt_moduli, write_love_numbers)
from planetmodel.mesh1d import (G_NEWTON, gll_points_weights, gravity,
                                lagrange_derivative_matrix)


def test_gll_exactness():
    """Weights, quadrature, and differentiation at machine precision."""
    rng = np.random.default_rng(0)
    for ngll in range(2, 9):
        N = ngll - 1
        x, w = gll_points_weights(ngll)
        assert abs(w.sum() - 2.0) < 1e-13
        k = 2 * N - 1
        assert abs(np.sum(w * x**k) - (1 - (-1.0) ** (k + 1)) / (k + 1)) < 1e-12
        p = np.polynomial.Polynomial(rng.standard_normal(N + 1))
        D = lagrange_derivative_matrix(x)
        assert np.max(np.abs(D @ p(x) - p.deriv()(x))) < 1e-11


def test_mesh_honours_skeleton():
    """Discontinuities on element boundaries; fluid layers identified."""
    prem = PREM()
    mesh = RadialMesh(prem, ngll=5, lmax=16)
    for b in prem.skeleton.inner_boundaries:
        assert np.any(np.isclose(mesh.left, b))
    assert np.max(mesh.right - mesh.left) <= mesh.drmax + 1e-9
    assert set(np.unique(mesh.layer[mesh.is_fluid])) == {1, 12}


def test_gravity():
    """g(0) = 0, the CMB peak, and the surface value GM/a^2."""
    prem = PREM()
    g = gravity(prem, np.array([0.0, 3480e3, 6371e3]))
    assert g[0] == 0.0
    assert abs(g[1] - 10.689) < 2e-3
    M = 5.9732e24
    assert abs(g[2] - G_NEWTON * M / 6371e3**2) < 1e-3


def test_love_numbers():
    """Low-degree PREM load Love numbers near their literature values."""
    prem = PREM()
    res = love_numbers(prem, lmax=8, rmax=6368e3)
    hp, lp, kp = res["hp"], res["lp"], res["kp"]
    assert kp[0] == -1.0                       # centre-of-mass frame, exact
    assert -1.32 < hp[0] < -1.25               # h'_1 (CM)
    assert -1.01 < hp[1] < -0.97               # h'_2
    assert -0.315 < kp[1] < -0.295             # k'_2
    assert 0.018 < lp[1] < 0.028               # l'_2


def test_degree_convergence():
    """The same degree on two discretizations agrees to high accuracy."""
    prem = PREM()
    m1 = RadialMesh(prem, ngll=5, lmax=8, rmax=6368e3)
    m2 = RadialMesh(prem, ngll=6, drmax=0.5 * m1.drmax, rmax=6368e3)
    a = solve_degree(m1, 4)
    b = solve_degree(m2, 4)
    for x, y in zip(a, b):
        assert abs(x - y) <= 1e-5 * abs(y)


def test_love_number_table():
    """Channel split, degree-0 mass conservation, and tidal benchmarks."""
    prem = PREM()
    tab = love_number_table(prem, 4, rmax=6368e3)
    mesh = tab["mesh"]
    res = love_numbers(prem, lmax=4, mesh=mesh)
    assert np.allclose(tab["h_u"][1:] + tab["h_phi"][1:], res["h"], atol=1e-15)
    assert np.allclose(tab["k_u"][1:] + tab["k_phi"][1:], res["k"], atol=1e-12)
    a = float(mesh.r[-1, -1])
    k0 = tab["k_u"][0] + tab["k_phi"][0]
    assert abs(k0 + 4 * np.pi * G_NEWTON * a) < 1e-9 * 4 * np.pi * G_NEWTON * a
    ga = float(mesh.nodal_gravity()[-1, -1])
    assert 0.28 < tab["k_t"][2] < 0.32            # PREM elastic k_2^T
    assert 0.57 < -ga * tab["h_t"][2] < 0.65      # PREM elastic h_2^T
    assert tab["h_t"][0] == 0.0 and np.all(tab["k_t"][:2] == 0.0)
    assert abs(tab["h_t"][1]) > 0.0          # degree-1 CM response is real


def test_write_love_numbers():
    """The written file round-trips through numpy.loadtxt."""
    import os
    import tempfile
    prem = PREM()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "love.dat")
        tab = write_love_numbers(path, prem, 2, rmax=6368e3)
        data = np.loadtxt(path)
    assert data.shape == (3, 7)
    assert np.array_equal(data[:, 0], [0, 1, 2])
    for j, name in enumerate(("h_u", "k_u", "h_phi", "k_phi", "h_t", "k_t")):
        assert np.allclose(data[:, j + 1], tab[name], rtol=1e-10)


def test_reciprocity_of_load_channels():
    """Eq. (64) of Al-Attar et al. (2024): g h^phi_l = k^u_l.

    A consequence of the symmetry of the bilinear form, so the four
    load columns carry only three independent numbers per degree.
    """
    prem = PREM()
    mesh = RadialMesh(prem, ngll=5, lmax=6, rmax=6368e3)
    tab = love_number_table(prem, 6, mesh=mesh)
    assert reciprocity_residual(tab).max() < 1e-10


def test_ti_isotropic_limit():
    """The TI kernel with Voigt input reproduces the (D76) isotropic form."""
    import numpy as np
    from planetmodel.loading import _as_five_moduli, _solid_local
    prem = PREM(ocean=False)
    mesh = RadialMesh(prem, ngll=5, lmax=4)
    kap, mu = voigt_moduli(mesh)
    m5 = _as_five_moduli(mesh, voigt_moduli)
    G = 6.6723e-11
    g = mesh.nodal_gravity(G=G)
    rho = mesh.nodal("rho")
    for l, e in ((1, 0), (2, 5), (6, mesh.nspec - 1)):
        if mesh.is_fluid[e]:
            continue
        r, w, D, jac = mesh.r[e], mesh.w, mesh.deriv, float(mesh.jac[e])
        n = r.size
        k2 = l * (l + 1.0)
        iU = 3 * np.arange(n); iV = iU + 1; iP = iU + 2
        B = r[:, None] * D / jac
        I = np.eye(n); Wq = w * jac
        Lref = np.zeros((3 * n, 3 * n))
        quad = lambda P, c: (P * (c * Wq)[:, None]).T @ P
        P = np.zeros((n, 3 * n)); P[:, iU] = B + 2 * I; P[:, iV] = -k2 * I
        Lref += quad(P, kap[e])
        P = np.zeros((n, 3 * n)); P[:, iU] = B - I; P[:, iV] = 0.5 * k2 * I
        Lref += (4.0 / 3.0) * quad(P, mu[e])
        P = np.zeros((n, 3 * n)); P[:, iV] = B - I; P[:, iU] = I
        Lref += k2 * quad(P, mu[e])
        Lref[iV, iV] += k2 * (k2 - 2.0) * mu[e] * Wq
        Lref[iU, iU] += 4.0 * rho[e] * (np.pi * G * rho[e] * r - g[e]) * r * Wq
        c = k2 * rho[e] * g[e] * r * Wq; Lref[iU, iV] += c; Lref[iV, iU] += c
        c = k2 * rho[e] * r * Wq; Lref[iV, iP] += c; Lref[iP, iV] += c
        M = (rho[e] * r * r * w)[:, None] * D
        Lref[np.ix_(iU, iP)] += M; Lref[np.ix_(iP, iU)] += M.T
        ifpg = 1.0 / (4 * np.pi * G)
        Lref[np.ix_(iP, iP)] += ifpg * ((D * (w * r * r)[:, None]).T @ D) / jac
        Lref[iP, iP] += ifpg * k2 * Wq
        Lnew = _solid_local(l, r, rho[e], g[e], *(m[e] for m in m5),
                            w, jac, D, G)
        assert np.abs(Lnew - Lref).max() <= 1e-12 * np.abs(Lref).max()


def test_ti_differs_from_voigt():
    """Full anisotropy changes the load response at the few-per-mille level."""
    prem = PREM(ocean=False)
    mesh = RadialMesh(prem, ngll=5, lmax=8)
    hi, _, _ = solve_degree(mesh, 8, ti_moduli)
    hv, _, _ = solve_degree(mesh, 8, voigt_moduli)
    rel = abs(hi - hv) / abs(hv)
    assert 1e-3 < rel < 2e-2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
