"""Mappings: the identity, the radial stretch, a general mapping, scaling."""
import numpy as np
import pytest

from planetmodel import (CallableDisplacement, IdentityMapping, MappingBase,
                         RadialStretch, ScaledMapping, Skeleton, ValidityReport,
                         ZeroDisplacement, testing, validity_lattice)
from planetmodel.frames import cartesian_points, spherical_coordinates

SK = Skeleton([0.0, 0.4, 0.8, 1.0])
A = 1.0


def points():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    radii = rng.uniform(0.05, 0.9, 40) * A
    X *= (radii / np.linalg.norm(X, axis=1))[:, None]
    return X


def flattening_h(f):
    def h(r, theta, phi):
        return -f * r * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)

    def dr(r, theta, phi):
        return -f * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0) + 0.0 * r

    def grad(r, theta, phi):
        return (3.0 * f * r * np.cos(theta) * np.sin(theta),
                np.zeros(np.broadcast(r, theta, phi).shape))

    return CallableDisplacement(h, radial_derivative=dr, angular_gradient=grad)


def kinked_h(amp):
    def h(r, theta, phi):
        return amp * np.maximum(r - 0.8, 0.0) * np.cos(2.0 * phi) * np.sin(theta) ** 2

    return CallableDisplacement(h, knots=[0.8])


class Squash(MappingBase):
    """A non-radial analytic mapping: x -> (x, y, c z + b x y)."""

    def __init__(self, c, b):
        self.c, self.b = c, b

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        out = X.copy()
        out[..., 2] = self.c * X[..., 2] + self.b * X[..., 0] * X[..., 1]
        return out

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        F = np.broadcast_to(np.eye(3), X.shape[:-1] + (3, 3)).copy()
        F[..., 2, 2] = self.c
        F[..., 2, 0] = self.b * X[..., 1]
        F[..., 2, 1] = self.b * X[..., 0]
        return F

    def jacobian(self, X):
        X = np.asarray(X, dtype=float)
        return np.full(X.shape[:-1], self.c)


def test_identity():
    m = IdentityMapping()
    X = points()
    assert np.array_equal(m(X), X)
    assert m.is_identity and m.is_valid()
    assert np.all(m.jacobian(X) == 1.0)
    testing.check_mapping(m, X)


def swirl_h():
    return CallableDisplacement(lambda r, t, p: 0.03 * r * np.sin(t) * np.cos(p))


@pytest.mark.parametrize("h", [flattening_h(0.05), kinked_h(0.1), swirl_h()])
def test_radial_stretch_passes_the_contract(h):
    m = RadialStretch(h, rmax=A)
    testing.check_mapping(m, points())
    assert m.is_valid(sample=validity_lattice(SK))


def test_radial_stretch_closed_forms():
    m = RadialStretch(flattening_h(0.05), rmax=A)
    X = points()
    r, theta, phi, R = spherical_coordinates(X)
    F_sph = m.deformation_gradient_spherical(X)
    F = m.deformation_gradient(X)
    assert np.allclose(F, np.einsum("...ik,...kl,...jl->...ij", R, F_sph, R))
    assert np.allclose(m.jacobian(X), np.linalg.det(F))
    assert np.allclose(m.jacobian(X), F_sph[..., 0, 0] * F_sph[..., 1, 1] ** 2)
    assert m.knots == ()
    assert RadialStretch(kinked_h(0.1), rmax=A).knots == (0.8,)


def test_radial_stretch_needs_rmax_and_moves_along_rays():
    with pytest.raises(TypeError):
        RadialStretch(flattening_h(0.05))
    with pytest.raises(ValueError, match="positive"):
        RadialStretch(flattening_h(0.05), rmax=0.0)
    m = RadialStretch(flattening_h(0.05), rmax=A)
    X = points()
    x = m(X)
    assert np.allclose(np.cross(x, X), 0.0, atol=1e-12)
    assert np.allclose(m(np.zeros((2, 3))), 0.0)
    assert not m.is_identity
    assert RadialStretch(ZeroDisplacement(), rmax=A).is_identity


def test_validity_reports_the_failing_factor():
    lattice = validity_lattice(SK)
    ok = RadialStretch(kinked_h(0.1), rmax=A).is_valid(sample=lattice)
    assert ok and isinstance(ok, ValidityReport) and ok.margin > 0
    fold = RadialStretch(kinked_h(-12.0), rmax=A).is_valid(sample=lattice)
    assert not fold and "folds radially" in fold.reason
    assert fold.worst_point is not None and fold.worst_point[0] > 0.8
    crossing = RadialStretch(CallableDisplacement(lambda r, t, p: -0.3 * (1.0 - r)),
                             rmax=A)
    report = crossing.is_valid(sample=lattice)
    assert not report and "cross through the origin" in report.reason
    centre = RadialStretch(CallableDisplacement(lambda r, t, p: 0.1 + 0.0 * r), rmax=A)
    report = centre.is_valid(X=np.zeros((1, 3)))
    assert not report and "at the centre" in report.reason
    assert report.margin == pytest.approx(0.1 / A)
    with pytest.raises(ValueError, match="give either"):
        RadialStretch(kinked_h(0.1), rmax=A).is_valid()


def test_generic_validity_on_a_general_mapping():
    lattice = validity_lattice(SK)
    assert Squash(0.9, 0.1).is_valid(sample=lattice)
    bad = Squash(-0.5, 0.0).is_valid(sample=lattice)
    assert not bad and "J = -0.5" in bad.reason


def test_inverse_and_linearisation():
    m = RadialStretch(flattening_h(0.05), rmax=A)
    X = points()
    assert np.allclose(m.inverse(m(X)), X, atol=1e-9)
    delta = flattening_h(0.01)
    base = RadialStretch(flattening_h(0.05), rmax=A)
    lin = base.linearise(delta, X=X)
    s = 1e-4
    plus = RadialStretch(CallableDisplacement(
        lambda r, t, p: base.h(r, t, p) + s * delta(r, t, p)), rmax=A)
    minus = RadialStretch(CallableDisplacement(
        lambda r, t, p: base.h(r, t, p) - s * delta(r, t, p)), rmax=A)
    dF = (plus.deformation_gradient(X) - minus.deformation_gradient(X)) / (2 * s)
    dJ = (plus.jacobian(X) - minus.jacobian(X)) / (2 * s)
    assert np.allclose(lin.dF, dF, atol=1e-6)
    assert np.allclose(lin.dJ, dJ, atol=1e-6)
    with pytest.raises(NotImplementedError):
        Squash(0.9, 0.1).inverse(X)


def test_general_mapping_passes_the_contract():
    testing.check_mapping(Squash(0.9, 0.1), points())


def test_scaled_mapping():
    k = 6.371e6
    m = RadialStretch(flattening_h(0.05), rmax=A)
    big = ScaledMapping(m, k)
    X = points()
    assert np.allclose(big(k * X), k * m(X))
    assert np.allclose(big.deformation_gradient(k * X), m.deformation_gradient(X))
    assert np.allclose(big.jacobian(k * X), m.jacobian(X))
    assert np.allclose(big.inverse(big(k * X)), k * X, rtol=1e-9)
    assert big.is_valid(sample=validity_lattice(Skeleton(k * SK.boundaries)))
    assert ScaledMapping(RadialStretch(kinked_h(0.1), rmax=A), k).knots == (0.8 * k,)
    assert ScaledMapping(IdentityMapping(), k).is_identity
    with pytest.raises(ValueError, match="positive"):
        ScaledMapping(m, -1.0)
    testing.check_mapping(big, k * X)
    hand = RadialStretch(CallableDisplacement(
        lambda r, t, p: k * m.h(r / k, t, p)), rmax=k * A)
    assert np.allclose(hand(k * X), big(k * X))


def test_validity_lattice_covers_every_layer_and_both_poles():
    r, theta, phi = validity_lattice(SK, n_r=3, n_theta=5, n_phi=4)
    assert r.shape == (9, 1, 1) and theta.shape == (1, 5, 1) and phi.shape == (1, 1, 4)
    assert theta[0, 0, 0] == 0.0 and theta[0, -1, 0] == np.pi
    for lo, hi in zip(SK.boundaries[:-1], SK.boundaries[1:]):
        inside = (r.ravel() > lo) & (r.ravel() < hi)
        assert inside.sum() == 3
    X = cartesian_points(r, theta, phi)
    assert X.shape == (9, 5, 4, 3)
