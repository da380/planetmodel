"""The separable operator family beyond the sphere: the disk and plane
annulus against Bessel roots, the flat layer against cosines, the centre
under either profile, every combination of weight, profile and index
against a dense assembly, the labels, and the bounded cache."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import jnp_zeros, jv, jvp, yvp

from planetmodel import testing
from planetmodel.mesh1d import Mesh1D
from planetmodel.randomfield import (
    RadialOperatorFamily,
    SeparableOperatorFamily,
    SpectralBasis,
    padded_mesh,
    restriction,
)
from planetmodel.randomfield.operator import CACHE_BYTES, INDICES, LATERALS, WEIGHTS


def _dense_from_band(band):
    q, n = band.shape[0] - 1, band.shape[1]
    A = np.zeros((n, n))
    for m in range(q + 1):
        u = q - m
        for j in range(u, n):
            A[j - u, j] = band[m, j]
    return A + np.triu(A, 1).T


def _dense_member(mesh, kappa, kappa_h, weight, lateral, lam):
    """M + K_r + lam K_h assembled densely, element by element, from the
    mesh's nodes, quadrature weights and derivative matrix."""
    n = mesh.nglob
    A, M = np.zeros((n, n)), np.zeros(n)
    for e in range(mesh.nspec):
        x, g = mesh.r[e], mesh.gmap[e]
        w = {"r2": x ** 2, "r": x, "one": np.ones_like(x)}[weight]
        p = {"inverse_square": 1.0 / x ** 2, "one": np.ones_like(x)}[lateral]
        dx = mesh.w * mesh.jac[e]
        D = mesh.deriv / mesh.jac[e]
        A[np.ix_(g, g)] += D.T @ np.diag(kappa(x) * w * dx) @ D
        A[g, g] += w * dx + lam * kappa_h(x) * w * p * dx
        M[g] += w * dx
    return A, M


def _disk_exact(m, count, a, kappa):
    """The first `count` eigenvalues of order m on a disk of radius a with
    a Neumann rim: 1 + kappa (j'_mk / a)^2, the constant mode leading at
    order 0."""
    roots = jnp_zeros(m, count)
    if m == 0:
        roots = np.concatenate(([0.0], roots[:-1]))
    return 1.0 + kappa * (roots / a) ** 2


# -- the class and its names ----------------------------------------------------

def test_radial_operator_family_is_the_separable_family():
    assert RadialOperatorFamily is SeparableOperatorFamily
    fam = RadialOperatorFamily(Mesh1D([0.0, 1.0], ngll=4, drmax=0.5))
    assert isinstance(fam, SeparableOperatorFamily)
    assert (fam.weight, fam.lateral, fam.index) == ("r2", "inverse_square", "degree")
    assert fam.cache_bytes == CACHE_BYTES == 2 ** 30
    assert fam.is_ball and fam.has_centre
    assert WEIGHTS == ("r2", "r", "one")
    assert LATERALS == ("inverse_square", "one")
    assert INDICES == ("degree", "order", "value")
    assert repr(fam).startswith("SeparableOperatorFamily(ball")


# -- every combination against a dense assembly ---------------------------------

COMBINATIONS = [
    ("r2", "inverse_square", "degree", 3, 12.0),
    ("one", "inverse_square", "degree", 3, 12.0),
    ("r", "inverse_square", "order", 3, 9.0),
    ("r2", "inverse_square", "value", 2.5, 2.5),
    ("r", "inverse_square", "value", 2.5, 2.5),
    ("one", "inverse_square", "value", 2.5, 2.5),
    ("r2", "one", "value", 2.5, 2.5),
    ("r", "one", "value", 2.5, 2.5),
    ("one", "one", "value", 2.5, 2.5),
]


@pytest.mark.parametrize("weight,lateral,index,label,lam", COMBINATIONS)
def test_member_is_the_dense_assembly(weight, lateral, index, label, lam):
    """Off a centre, each member's pencil is M + K_r + lambda K_h as a
    dense element-by-element assembly gives it, for every combination of
    weight, profile and index the family takes."""
    mesh = Mesh1D([0.4, 0.55, 1.0], ngll=5, drmax=0.2)
    kappa = lambda x: (0.3 + 0.2 * x) ** 2
    kappa_h = lambda x: 0.05 + 0.1 * x ** 2
    fam = SeparableOperatorFamily(mesh, kappa=kappa, kappa_h=kappa_h, weight=weight,
                                  lateral=lateral, index=index)
    assert not fam.has_centre
    assert fam.lateral_eigenvalue(label) == lam
    assert type(fam.label(label)) is (float if index == "value" else int)
    for l, lam_l in ((label, lam), (0, 0.0)):
        A, M = _dense_member(mesh, kappa, kappa_h, weight, lateral, lam_l)
        band, mass = fam.pencil(l)
        assert np.allclose(_dense_from_band(band), A, rtol=1e-12, atol=1e-13)
        assert np.allclose(mass, M, rtol=1e-13)
    theta = fam.eigvalsh(label)
    A, M = _dense_member(mesh, kappa, kappa_h, weight, lateral, lam)
    s = 1.0 / np.sqrt(M)
    assert np.allclose(theta, np.linalg.eigvalsh(s[:, None] * A * s), rtol=1e-10)


def test_refused_combinations_labels_and_meshes():
    mesh = Mesh1D([0.5, 1.0], ngll=4, drmax=0.25)
    for weight, index in (("r", "degree"), ("r2", "order"), ("one", "order")):
        with pytest.raises(ValueError, match=f"index '{index}' goes with weight"):
            SeparableOperatorFamily(mesh, weight=weight, index=index)
    for index in ("degree", "order"):
        with pytest.raises(ValueError, match="index='value'"):
            SeparableOperatorFamily(mesh, weight="one", lateral="one", index=index)
    with pytest.raises(ValueError, match="lateral must be one of"):
        SeparableOperatorFamily(mesh, lateral="inverse")
    with pytest.raises(ValueError, match="index must be one of"):
        SeparableOperatorFamily(mesh, index="wavenumber")
    with pytest.raises(ValueError, match="r measure needs a mesh in r >= 0"):
        SeparableOperatorFamily(Mesh1D([-1.0, 1.0]), weight="r", index="order")
    with pytest.raises(ValueError, match="r >= 0"):
        SeparableOperatorFamily(Mesh1D([-1.0, 1.0]), lateral="one", index="value")
    with pytest.raises(ValueError, match="cache_bytes"):
        SeparableOperatorFamily(mesh, cache_bytes=0)

    by_degree = SeparableOperatorFamily(mesh)
    by_order = SeparableOperatorFamily(mesh, weight="r", index="order")
    by_value = SeparableOperatorFamily(mesh, index="value")
    for fam, noun in ((by_degree, "degree"), (by_order, "order")):
        with pytest.raises(ValueError, match=f"{noun} must be an integer"):
            fam.eigvalsh(1.5)
        with pytest.raises(ValueError, match=f"{noun} must be non-negative"):
            fam.pencil(-1)
        assert fam.label(np.int64(3)) == 3 and type(fam.label(3.0)) is int
    for bad in (-0.5, np.nan, np.inf):
        with pytest.raises(ValueError, match="non-negative and finite"):
            by_value.eigvalsh(bad)
    assert by_degree.lateral_eigenvalue(4) == 20.0
    assert by_order.lateral_eigenvalue(4) == 16.0
    assert by_value.lateral_eigenvalue(4) == 4.0


def test_plain_measure_inverse_square_needs_positive_coordinates_above_zero():
    """An interval of the line reaching x <= 0 has its member lambda = 0
    alone, whichever index labels it; in x > 0 it has them all."""
    for a in (-1.0, 0.0):
        mesh = Mesh1D([a, 1.0], ngll=5, drmax=0.25)
        by_degree = SeparableOperatorFamily(mesh, weight="one")
        by_value = SeparableOperatorFamily(mesh, weight="one", index="value")
        assert not by_degree.has_centre and by_degree.ndof(0) == mesh.nglob
        assert np.array_equal(by_degree.eigvalsh(0), by_value.eigvalsh(0.0))
        for call in (by_degree.pencil, by_degree.eigvalsh, by_degree.ndof,
                     by_degree.mass, by_degree.eig):
            with pytest.raises(ValueError, match="degree l >= 1 needs r > 0"):
                call(1)
        for call in (by_value.pencil, by_value.eigvalsh, by_value.ndof):
            with pytest.raises(ValueError, match="above zero needs x > 0"):
                call(2.0)
    mesh = Mesh1D([0.5, 1.5], ngll=5, drmax=0.25)
    by_degree = SeparableOperatorFamily(mesh, weight="one")
    by_value = SeparableOperatorFamily(mesh, weight="one", index="value")
    assert np.array_equal(by_degree.eigvalsh(2), by_value.eigvalsh(6.0))
    assert by_degree.eigvalsh(2)[0] > 1.0


# -- the disk and the plane annulus -----------------------------------------------

def test_disk_neumann_bessel_spectrum_converges():
    """Order m on a disk of radius a, constant kappa, Neumann rim: the
    eigenvalues are 1 + kappa (j'_mk / a)^2, reached spectrally as the
    mesh is refined."""
    a, kap, orders = 1.3, 0.04, (0, 1, 2, 5)
    worst = []
    for drmax in (0.4, 0.2, 0.1, 0.05):
        fam = SeparableOperatorFamily(Mesh1D([0.0, a], ngll=6, drmax=drmax),
                                      kappa=kap, weight="r", index="order")
        assert fam.has_centre and not fam.is_ball and "disk" in repr(fam)
        err = 0.0
        for m in orders:
            exact = _disk_exact(m, 8, a, kap)
            err = max(err, np.max(np.abs(fam.eigvalsh(m)[:8] - exact) / exact))
        worst.append(err)
    assert worst[0] < 1e-2 and worst[-1] < 1e-9
    assert all(fine < coarse / 100.0 for coarse, fine in zip(worst, worst[1:]))
    assert abs(fam.eigvalsh(0)[0] - 1.0) < 1e-12          # exactly the constant
    assert "cached orders: [0, 1, 2, 5]" in repr(fam)

    # more nodes per element at a fixed four elements
    by_order = []
    for ngll in (4, 6, 8, 10):
        fam = SeparableOperatorFamily(Mesh1D([0.0, a], ngll=ngll, drmax=0.33),
                                      kappa=kap, weight="r", index="order")
        exact = _disk_exact(2, 8, a, kap)
        by_order.append(np.max(np.abs(fam.eigvalsh(2)[:8] - exact) / exact))
    assert all(fine < coarse / 50.0 for coarse, fine in zip(by_order, by_order[1:]))


def test_disk_modes_and_the_centre():
    """The modes are the Bessel functions J_m(j'_mk r / a), M-orthonormal;
    those of order m >= 1 vanish at the centre, and at order 0 the
    condensed centre carries J_0's value there."""
    a, kap = 1.3, 0.04
    mesh = Mesh1D([0.0, a], ngll=6, drmax=0.05)
    fam = SeparableOperatorFamily(mesh, kappa=kap, weight="r", index="order")
    r = mesh.rglob
    full_mass = np.concatenate(([0.0], fam.mass(0)))
    for m in (0, 1, 3):
        assert fam.ndof(m) == mesh.nglob - 1
        theta, Phi = fam.eig(m, count=6)
        M = fam.mass(m)
        assert np.allclose(Phi.T @ (M[:, None] * Phi), np.eye(6), atol=1e-12)
        assert np.allclose(fam.apply(m, Phi), Phi * theta, atol=1e-9)
        full = fam.embed(m, Phi)
        assert full.shape == (mesh.nglob, 6)
        roots = jnp_zeros(m, 6)
        if m == 0:
            roots = np.concatenate(([0.0], roots[:-1]))
        for k in range(6):
            exact = jv(m, roots[k] * r / a)
            exact /= np.sqrt(full_mass @ exact ** 2)
            exact *= np.sign(exact @ (full_mass * full[:, k]))
            # the condensed centre is the least accurate node
            assert np.max(np.abs(full[:, k] - exact)) < (1e-9 if k < 3 else 1e-6)
            assert np.max(np.abs(full[1:, k] - exact[1:])) < 1e-7
        if m == 0:
            assert np.all(np.abs(full[0]) > 0.5)           # J_0(0) = 1, normalised
            assert abs(full[0, 0] - np.sqrt(2.0) / a) < 1e-12   # the constant
        else:
            assert np.all(full[0] == 0.0)


def test_plane_annulus_cross_product_spectrum():
    """Order m on a < r < b with Neumann ends: 1 + kappa k^2 over the roots
    of J_m'(k a) Y_m'(k b) - J_m'(k b) Y_m'(k a)."""
    ra, rb, kap = 0.5, 1.5, 0.04

    def wavenumbers(m, count):
        f = lambda k: (jvp(m, k * ra) * yvp(m, k * rb)
                       - jvp(m, k * rb) * yvp(m, k * ra))
        ks = np.linspace(1e-3, 40.0, 20000)
        v = f(ks)
        return np.array([brentq(f, k0, k1) for k0, k1, v0, v1
                         in zip(ks[:-1], ks[1:], v[:-1], v[1:]) if v0 * v1 < 0][:count])

    worst = []
    for drmax in (0.25, 0.125, 0.0625):
        fam = SeparableOperatorFamily(Mesh1D([ra, rb], ngll=6, drmax=drmax),
                                      kappa=kap, weight="r", index="order")
        assert not fam.has_centre and "annulus" in repr(fam)
        err = 0.0
        for m in (0, 1, 4):
            exact = 1.0 + kap * wavenumbers(m, 6) ** 2
            if m == 0:
                exact = np.concatenate(([1.0], exact))
            assert fam.ndof(m) == fam.mesh.nglob
            err = max(err, np.max(np.abs(fam.eigvalsh(m)[:exact.size] - exact) / exact))
        worst.append(err)
    assert worst[0] < 1e-3 and worst[-1] < 1e-9
    assert all(fine < coarse / 100.0 for coarse, fine in zip(worst, worst[1:]))


def test_disk_basis_holds_the_contract():
    lam = 0.15
    mesh = Mesh1D([0.0, 0.5, 1.0, 1.6], ngll=5, drmax=0.1)
    fam = SeparableOperatorFamily(mesh, kappa=lambda r: (lam * (1.0 + 0.3 * r)) ** 2,
                                  kappa_h=lam ** 2, weight="r", index="order",
                                  robin=(None, 2.0))
    for m, R in ((0, restriction(mesh, 0.0, 1.0)), (3, restriction(mesh, 0.5, 1.0))):
        basis = SpectralBasis(fam, m, restrict=R, power=2.0, tol=1e-8)
        assert basis.degree == m and type(basis.degree) is int
        assert basis.lateral_eigenvalue == float(m * m)
        testing.check_spectral_basis(basis)
        r1, r2 = R.interval
        assert basis.weights().sum() == pytest.approx((r2 ** 2 - r1 ** 2) / 2.0,
                                                      rel=1e-12)
        assert np.allclose(basis.galerkin(fam), np.diag(basis.theta), atol=1e-9)


# -- the flat layer ---------------------------------------------------------------

def _flat(mesh, **kw):
    return SeparableOperatorFamily(mesh, weight="one", lateral="one", index="value",
                                   **kw)


def test_flat_neumann_cosine_spectrum_converges():
    """Constant coefficients on [a, b], a < 0 < b, Neumann ends: the
    eigenvalues are 1 + kappa (j pi / H)^2 + kappa_h lambda and the modes
    the cosines, for every lateral eigenvalue."""
    za, zb, kap, kap_h = -0.7, 1.9, 0.04, 0.09
    H = zb - za
    j = np.arange(8)
    lams = (0.0, 0.37, 12.5, 400.0)
    worst = []
    for drmax in (0.65, 0.33, 0.17, 0.09):
        fam = _flat(Mesh1D([za, zb], ngll=6, drmax=drmax), kappa=kap, kappa_h=kap_h)
        assert not fam.has_centre and "interval" in repr(fam)
        err = 0.0
        for lam in lams:
            exact = 1.0 + kap * (j * np.pi / H) ** 2 + kap_h * lam
            assert fam.ndof(lam) == fam.mesh.nglob
            err = max(err, np.max(np.abs(fam.eigvalsh(lam)[:8] - exact) / exact))
        worst.append(err)
    assert worst[0] < 1e-3 and worst[-1] < 1e-11
    assert all(fine < coarse / 100.0 for coarse, fine in zip(worst, worst[1:]))
    assert fam.cached() == list(lams)

    z, M = fam.mesh.rglob, fam.mass(0.0)
    theta, Phi = fam.eig(12.5, count=6)
    assert np.allclose(Phi.T @ (M[:, None] * Phi), np.eye(6), atol=1e-12)
    for k in range(6):
        cosine = np.cos(k * np.pi * (z - za) / H)
        cosine /= np.sqrt(M @ cosine ** 2)
        assert np.max(np.abs(Phi[:, k] - cosine)) < 1e-9     # positive at z = a
    # the lateral term only shifts the spectrum when kappa_h is constant
    assert np.allclose(fam.eigvalsh(12.5) - fam.eigvalsh(0.0), kap_h * 12.5,
                       rtol=0.0, atol=1e-9 * fam.eigvalsh(0.0)[-1])


def test_flat_member_zero_is_the_plain_interval_exactly():
    """lambda = 0 of the flat family is degree 0 of the plain-measure
    family on the same mesh, array for array."""
    mesh = Mesh1D([-0.7, 0.4, 1.9], ngll=6, drmax=0.1)
    kw = dict(kappa=lambda z: 0.04 + 0.01 * z ** 2, kappa_h=0.3, robin=(0.5, None))
    flat = _flat(mesh, **kw)
    line = SeparableOperatorFamily(mesh, weight="one", **kw)
    assert np.array_equal(flat.pencil(0.0)[0], line.pencil(0)[0])
    assert np.array_equal(flat.mass(0.0), line.mass(0))
    assert np.array_equal(flat.eigvalsh(0.0), line.eigvalsh(0))
    for a, b in zip(flat.eig(0.0), line.eig(0)):
        assert np.array_equal(a, b)
    assert np.array_equal(flat.eig(0.0, count=5)[1], line.eig(0, count=5)[1])


def test_flat_depth_varying_lateral_coefficient():
    """With kappa_h varying in depth: to first order in a small variation
    each eigenvalue moves by lambda <phi_j, delta kappa_h phi_j>, phi_j the
    normalised cosine; and a strong variation agrees with a much finer
    mesh."""
    za, zb, kap, kap_h, lam = -0.7, 1.9, 0.04, 0.09, 12.5
    H = zb - za
    mesh = Mesh1D([za, zb], ngll=6, drmax=0.09)
    for eps in (1e-4, 1e-5):
        delta = lambda z: kap_h * eps * (np.cos(np.pi * (z - za) / H)
                                         + 0.5 * (z - za) / H)
        fam = _flat(mesh, kappa=kap, kappa_h=lambda z: kap_h + delta(z))
        got = fam.eigvalsh(lam)
        for j in range(4):
            norm = H if j == 0 else H / 2.0
            shift = lam / norm * quad(
                lambda z: delta(z) * np.cos(j * np.pi * (z - za) / H) ** 2,
                za, zb, epsabs=1e-15)[0]
            exact = 1.0 + kap * (j * np.pi / H) ** 2 + kap_h * lam
            assert shift > 1e-7
            # second order: (lambda delta)^2 over the gap to the next mode
            assert abs(got[j] - exact - shift) < 1e3 * eps * shift

    strong = lambda z: kap_h * (1.0 + 0.6 * np.tanh(3.0 * z))
    coarse = _flat(Mesh1D([za, zb], ngll=6, drmax=0.17), kappa=kap, kappa_h=strong)
    fine = _flat(Mesh1D([za, zb], ngll=10, drmax=0.03), kappa=kap, kappa_h=strong)
    assert np.allclose(coarse.eigvalsh(lam)[:8], fine.eigvalsh(lam)[:8], rtol=1e-8)
    # bounded by the extremes of kappa_h
    theta0 = coarse.eigvalsh(0.0)[0]
    lowest = coarse.eigvalsh(lam)[0]
    assert theta0 + 0.4 * kap_h * lam < lowest < theta0 + 1.6 * kap_h * lam


def test_flat_basis_holds_the_contract_with_float_labels():
    lam_scale = 0.2
    mesh = padded_mesh(0.0, 1.0, pad=0.8, weight="one", ngll=5, drmax=0.1)
    assert mesh.rglob[0] == -0.8                     # above the free surface
    fam = _flat(mesh, kappa=lam_scale ** 2,
                kappa_h=lambda z: (lam_scale * (1.0 + 0.5 * z)) ** 2)
    other = _flat(mesh, kappa=0.3, kappa_h=0.1)
    R = restriction(mesh, 0.0, 1.0)
    for label in (0, 0.37, np.float64(41.5)):
        basis = SpectralBasis(fam, label, restrict=R, power=2.0, tol=1e-8)
        assert type(basis.degree) is float and basis.degree == label
        assert basis.lateral_eigenvalue == float(label)
        assert repr(basis).startswith(f"SpectralBasis(l={float(label)!r}")
        testing.check_spectral_basis(basis)
        assert basis.weights().sum() == pytest.approx(1.0, rel=1e-12)
        assert np.allclose(basis.galerkin(fam), np.diag(basis.theta), atol=1e-9)
        G = basis.galerkin(other)
        Phi = basis.modes()
        assert np.allclose(G, Phi.T @ (other.mass(label)[:, None]
                                       * other.apply(label, Phi)), atol=1e-10)
    # equal floats are one member, however they are spelt
    assert fam.cached() == [0.0, 0.37, 41.5]
    kx, ky = 3.0, 4.0
    fam.eigvalsh(kx ** 2 + ky ** 2); fam.eigvalsh(25.0); fam.eigvalsh(np.float64(25))
    assert fam.cached() == [0.0, 0.37, 25.0, 41.5]
    assert "cached values" in repr(fam)


def test_galerkin_refuses_another_weight_profile_or_index():
    mesh = Mesh1D([0.5, 1.0], ngll=4, drmax=0.25)
    basis = SpectralBasis(SeparableOperatorFamily(mesh, index="value"), 2.0)
    for what, kw in (("weight", dict(weight="r", index="value")),
                     ("lateral", dict(lateral="one", index="value")),
                     ("index", dict())):
        with pytest.raises(ValueError, match=f"the family's {what} is"):
            basis.galerkin(SeparableOperatorFamily(mesh, **kw))
    assert basis.galerkin(SeparableOperatorFamily(mesh, index="value")).shape == (
        basis.nmodes, basis.nmodes)


# -- a centre under the profile one ---------------------------------------------

def test_centre_under_profile_one_is_condensed_in_every_member():
    """With the lateral profile one a centre is no zero of the modes: every
    member is the member lambda = 0 shifted (kappa_h constant), its axis
    value condensed.  On a disk that is a cylinder's family along its
    axis: 1 + kappa (j'_0k / a)^2 + kappa_h lambda."""
    a, kap, kap_h = 1.3, 0.04, 0.09
    mesh = Mesh1D([0.0, a], ngll=6, drmax=0.05)
    cyl = SeparableOperatorFamily(mesh, kappa=kap, kappa_h=kap_h, weight="r",
                                  lateral="one", index="value")
    exact = _disk_exact(0, 8, a, kap)
    for lam in (0.0, 3.3):
        assert cyl.ndof(lam) == mesh.nglob - 1
        assert np.allclose(cyl.eigvalsh(lam)[:8], exact + kap_h * lam, rtol=1e-10)
        theta, Phi = cyl.eig(lam, count=3)
        full = cyl.embed(lam, Phi)
        assert abs(full[0, 0] - np.sqrt(2.0) / a) < 1e-10      # the constant
        assert np.all(np.abs(full[0]) > 0.5)
        assert np.allclose(cyl.solve(lam, cyl.apply(lam, Phi)), Phi)
    ball = SeparableOperatorFamily(mesh, kappa=kap, kappa_h=kap_h)
    shifted = SeparableOperatorFamily(mesh, kappa=kap, kappa_h=kap_h, lateral="one",
                                      index="value")
    assert shifted.is_ball
    assert np.allclose(shifted.eigvalsh(2.0), ball.eigvalsh(0) + 2.0 * kap_h,
                       rtol=1e-10)
    assert np.array_equal(shifted.eigvalsh(0.0), ball.eigvalsh(0))


# -- the cache --------------------------------------------------------------------

def test_cache_evicts_the_least_recently_used_and_recomputes_identically():
    mesh = Mesh1D([0.0, 1.0], ngll=5, drmax=0.05)
    n = mesh.nglob - 1
    per_member = 8 * (n * n + n + 5 * n)       # eigenvectors, values, the band
    fam = SeparableOperatorFamily(mesh, kappa=0.01, cache_bytes=3 * per_member)
    first = {l: fam.eig(l) for l in (0, 1, 2)}
    assert fam.cached() == [0, 1, 2] and fam.cache_nbytes == 3 * per_member
    fam.eig(0)                                  # degree 0 is now the most recent
    fam.eig(3)                                  # so degree 1 goes
    assert fam.cached() == [0, 2, 3]
    fam.eig(4); fam.eig(5)
    assert fam.cached() == [3, 4, 5] and fam.cache_nbytes <= fam.cache_bytes
    for l, (theta, Phi) in first.items():       # each recomputed, bit for bit
        again = fam.eig(l)
        assert again[0] is not theta
        assert np.array_equal(again[0], theta) and np.array_equal(again[1], Phi)
    band = fam.pencil(4)[0]
    assert np.array_equal(band, SeparableOperatorFamily(mesh, kappa=0.01).pencil(4)[0])

    # a member larger than the budget is still served, alone
    tiny = SeparableOperatorFamily(mesh, kappa=0.01, cache_bytes=1)
    for l in (0, 1, 0):
        theta, Phi = tiny.eig(l)
        assert np.array_equal(theta, first[l][0]) and np.array_equal(Phi, first[l][1])
        assert tiny.cached() == [l]
    assert np.allclose(tiny.apply_power(2, -1.0, Phi[:, 0]), tiny.solve(2, Phi[:, 0]))

    unbounded = SeparableOperatorFamily(mesh, kappa=0.01, cache_bytes=None)
    for l in range(12):
        unbounded.eig(l)
    assert unbounded.cached() == list(range(12))
    assert unbounded.cache_nbytes == 12 * per_member


def test_bounded_cache_leaves_a_basis_and_a_sampler_unchanged():
    """Bases built from a family whose cache holds one member at a time are
    those of an unbounded one."""
    mesh = Mesh1D([0.3, 1.0], ngll=5, drmax=0.05)
    R = restriction(mesh, 0.3, 1.0)
    roomy = SeparableOperatorFamily(mesh, kappa=0.01)
    tight = SeparableOperatorFamily(mesh, kappa=0.01, cache_bytes=1)
    for l in (0, 3, 1, 3, 0):
        a = SpectralBasis(roomy, l, restrict=R, power=2.0, tol=1e-6)
        b = SpectralBasis(tight, l, restrict=R, power=2.0, tol=1e-6)
        assert np.array_equal(a.theta, b.theta)
        assert np.array_equal(a.modes(), b.modes())
    assert len(tight.cached()) == 1 and len(roomy.cached()) == 3
