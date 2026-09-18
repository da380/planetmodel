"""The truncated eigenbases: the contracts on intervals, annuli and balls,
the truncation rules, and evaluation and integration against oracles."""
import numpy as np
import pytest
from scipy.interpolate import BarycentricInterpolator

from planetmodel import testing
from planetmodel.harmonics import packing
from planetmodel.randomfield import (RadialOperatorFamily, SpectralBasis,
                                     SphericalBasis, matern_reach, padded_mesh,
                                     restriction)


def _setup(r1, r2, weight, *, robin=None):
    """A padded mesh, its family and the restriction to [r1, r2]."""
    mesh = padded_mesh(r1, r2, pad=0.3, weight=weight, ngll=5, drmax=0.1)
    fam = RadialOperatorFamily(mesh, kappa=lambda r: 0.02 + 0.01 * r ** 2,
                               weight=weight, robin=robin)
    return fam, restriction(mesh, r1, r2)


CASES = {"interval": (-0.4, 0.6, "one", (0,)),
         "line in r > 0": (0.5, 1.0, "one", (0, 3)),
         "annulus": (0.5, 1.0, "r2", (0, 3)),
         "ball": (0.0, 0.8, "r2", (0, 3))}
RULES = [dict(), dict(nmodes=7), dict(theta_max=3.0), dict(power=2.0, tol=1e-3)]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rule", RULES, ids=lambda d: "+".join(d) or "all")
def test_the_contract(case, rule):
    r1, r2, weight, degrees = CASES[case]
    fam, R = _setup(r1, r2, weight)
    for l in degrees:
        basis = SpectralBasis(fam, l, restrict=R, **rule)
        testing.check_spectral_basis(basis)
        assert basis.degree == l and basis.family is fam and basis.restriction is R
        assert repr(basis).startswith("SpectralBasis(l=")


def test_the_contract_with_robin_conditions_and_no_restriction():
    fam, _ = _setup(0.5, 1.0, "r2", robin=(2.0, None))
    basis = SpectralBasis(fam, 2, nmodes=9)
    assert basis.restriction.elements == (0, fam.mesh.nspec)
    assert basis.r.size == fam.mesh.nglob
    testing.check_spectral_basis(basis)


def test_truncation_rules():
    fam, R = _setup(0.0, 0.8, "r2")
    every = SpectralBasis(fam, 1, restrict=R)
    assert every.nmodes == fam.ndof(1)
    assert np.array_equal(every.theta, fam.eigvalsh(1))
    seven = SpectralBasis(fam, 1, restrict=R, nmodes=7)
    assert seven.nmodes == 7 and np.allclose(seven.theta, every.theta[:7])
    assert np.allclose(np.abs(seven.modes()), np.abs(every.modes()[:, :7]))
    below = SpectralBasis(fam, 1, restrict=R, theta_max=float(every.theta[4]) * 1.001)
    assert below.nmodes == 5
    # the variance-trace rule: the fewest modes whose omitted trace is within tol
    for p, tol in ((2.0, 1e-3), (1.0, 0.2), (3.0, 0.0)):
        b = SpectralBasis(fam, 1, restrict=R, power=p, tol=tol)
        w = fam.eigvalsh(1) ** -p
        k = b.nmodes
        assert w[k:].sum() <= tol * w.sum() * (1 + 1e-12)
        assert k == 1 or w[k - 1:].sum() > tol * w.sum()
    assert SpectralBasis(fam, 1, restrict=R, power=2.0, tol=0.999).nmodes == 1


def test_refusals():
    fam, R = _setup(0.5, 1.0, "r2")
    for kw in (dict(nmodes=3, theta_max=2.0), dict(nmodes=3, power=2.0, tol=0.1),
               dict(power=2.0), dict(tol=0.1), dict(nmodes=0),
               dict(nmodes=fam.ndof(0) + 1), dict(theta_max=0.5),
               dict(power=2.0, tol=1.0)):
        with pytest.raises(ValueError):
            SpectralBasis(fam, 0, restrict=R, **kw)
    with pytest.raises(TypeError):
        SpectralBasis(fam.mesh, 0)
    other = padded_mesh(0.5, 1.0, pad=0.2, drmax=0.05)
    with pytest.raises(ValueError, match="restriction"):
        SpectralBasis(fam, 0, restrict=restriction(other, 0.5, 1.0))
    line, Rl = _setup(-0.4, 0.6, "one")
    with pytest.raises(ValueError, match="degree l >= 1 needs r > 0"):
        SpectralBasis(line, 1, restrict=Rl)
    b = SpectralBasis(fam, 0, restrict=R, nmodes=4)
    for bad in (np.ones(3), np.ones((4, 2, 2))):
        with pytest.raises(ValueError):
            b.synthesise(bad)
    with pytest.raises(ValueError):
        b.analyse(np.ones(b.r.size))              # physical nodes are not enough


def test_evaluate_against_the_lagrange_interpolant():
    """Within an element a mode is the polynomial through its GLL nodes."""
    fam, R = _setup(0.0, 0.8, "r2")
    mesh = fam.mesh
    for l in (0, 2):
        basis = SpectralBasis(fam, l, restrict=R, nmodes=6)
        Phi = basis.modes()
        for e in range(*R.elements):
            x = np.linspace(mesh.left[e], mesh.right[e], 7)[1:-1]
            want = BarycentricInterpolator(mesh.r[e], Phi[mesh.gmap[e]])(x)
            assert np.allclose(basis.evaluate(x), want, rtol=0.0, atol=1e-11)
        c = np.arange(1.0, 7.0)
        assert np.allclose(basis.evaluate(basis.r) @ c, basis.synthesise(c))
    assert basis.evaluate(0.8).shape == (6,)
    assert basis.evaluate(0.8 * (1 + 1e-15)).shape == (6,)   # an endpoint, to roundoff


@pytest.mark.parametrize("weight", ["r2", "one"])
def test_weights_integrate_a_synthesised_field(weight):
    fam, R = _setup(0.5, 1.0, weight)
    basis = SpectralBasis(fam, 0, restrict=R, nmodes=6)
    c = np.array([1.0, -0.5, 0.3, 0.2, -0.1, 0.05])
    x = np.linspace(0.5, 1.0, 20001)
    w = x ** 2 if weight == "r2" else np.ones_like(x)
    want = np.trapezoid(w * (basis.evaluate(x) @ c), x)
    assert basis.weights() @ basis.synthesise(c) == pytest.approx(want, rel=1e-7)
    # the constant is the first Neumann mode, so the projection of one is one
    ones = basis.synthesise(basis.analyse(np.ones(fam.mesh.nglob)))
    assert np.allclose(ones, 1.0, atol=1e-10)


@pytest.mark.parametrize("a, b, weight", [(0.6, 1.0, "r2"), (-0.2, 0.2, "one")])
def test_padding_insensitivity_of_the_pointwise_variance(a, b, weight):
    nu, lam = 1.0, 0.04

    def variance(pad_factor):
        mesh = padded_mesh(a, b, pad=pad_factor * matern_reach(nu, lam),
                           weight=weight, drmax=lam / 2.0)
        fam = RadialOperatorFamily(mesh, kappa=lam ** 2, weight=weight)
        basis = SpectralBasis(fam, 0, restrict=restriction(mesh, a, b),
                              power=nu + 0.5, tol=1e-12)
        return basis.pointwise_variance(nu + 0.5)

    v2, v4 = variance(2.0), variance(4.0)
    assert np.max(np.abs(v2 - v4) / v4) < 5e-4


@pytest.mark.parametrize("case", ["annulus", "ball"])
def test_spherical_basis(case):
    r1, r2, weight, _ = CASES[case]
    fam, R = _setup(r1, r2, weight)
    bases = [SpectralBasis(fam, l, restrict=R, nmodes=9 - l) for l in range(5)]
    sph = SphericalBasis(bases)
    testing.check_spherical_basis(sph)
    assert sph.lmax == 4 and list(sph.nmodes) == [9, 8, 7, 6, 5]
    assert sph.size == sum((2 * l + 1) * (9 - l) for l in range(5))
    assert sph[3] is bases[3] and sph.r is R.r
    # the vector of a coefficient array, harmonic by harmonic, is packing's
    v = np.random.default_rng(1).standard_normal(sph.size)
    c = sph.synthesise(v)
    s, l, m = packing(sph.lmax)
    assert c[s, l, m].shape == (25, R.r.size)
    assert np.allclose(c[0, 2, 1], bases[2].synthesise(v[sph.block(0, 2, 1)]))
    assert repr(sph).startswith("SphericalBasis(lmax=4")


def test_spherical_basis_refusals():
    fam, R = _setup(0.5, 1.0, "r2")
    b = [SpectralBasis(fam, l, restrict=R, nmodes=4) for l in range(3)]
    with pytest.raises(ValueError, match="degree 0"):
        SphericalBasis([])
    with pytest.raises(ValueError, match="degree"):
        SphericalBasis([b[0], b[2]])
    with pytest.raises(TypeError):
        SphericalBasis([b[0], fam])
    other, Ro = _setup(0.5, 1.0, "r2")
    with pytest.raises(ValueError, match="one family"):
        SphericalBasis([b[0], SpectralBasis(other, 1, restrict=Ro, nmodes=4)])
    with pytest.raises(ValueError, match="one restriction"):
        SphericalBasis([b[0], SpectralBasis(fam, 1, nmodes=4)])
    sph = SphericalBasis(b)
    with pytest.raises(ValueError):
        sph.synthesise(np.ones(sph.size + 1))
    with pytest.raises(ValueError):
        sph.analyse(np.ones((2, 3, 3, R.r.size)))
