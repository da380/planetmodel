"""Mappings: the identity, the radial stretch, a general mapping, scaling."""
import numpy as np
import pytest

from planetmodel import (CallableDisplacement, Geometry, IdentityMapping,
                         MappingBase, RadialStretch, ScaledMapping, Skeleton,
                         ValidityReport, ZeroDisplacement, outer_radius_of, testing,
                         validity_lattice)
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
    assert m.is_identity and m.validity()
    assert np.all(m.jacobian(X) == 1.0)
    testing.check_mapping(m, X)


def swirl_h():
    return CallableDisplacement(lambda r, t, p: 0.03 * r * np.sin(t) * np.cos(p))


@pytest.mark.parametrize("h", [flattening_h(0.05), kinked_h(0.1), swirl_h()])
def test_radial_stretch_passes_the_contract(h):
    m = RadialStretch(h, rmax=A)
    testing.check_mapping(m, points())
    assert m.validity(sample=validity_lattice(SK))


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
    assert RadialStretch(flattening_h(0.05), rmax=SK).rmax == 1.0
    assert RadialStretch(flattening_h(0.05), rmax=Geometry(SK)).rmax == 1.0
    assert outer_radius_of(2.5) == 2.5 and outer_radius_of(SK) == 1.0
    m = RadialStretch(flattening_h(0.05), rmax=A)
    X = points()
    x = m(X)
    assert np.allclose(np.cross(x, X), 0.0, atol=1e-12)
    assert np.allclose(m(np.zeros((2, 3))), 0.0)
    assert not m.is_identity
    assert RadialStretch(ZeroDisplacement(), rmax=A).is_identity


def test_validity_reports_the_failing_factor():
    lattice = validity_lattice(SK)
    ok = RadialStretch(kinked_h(0.1), rmax=A).validity(sample=lattice)
    assert ok and isinstance(ok, ValidityReport) and ok.margin > 0
    fold = RadialStretch(kinked_h(-12.0), rmax=A).validity(sample=lattice)
    assert not fold and "folds radially" in fold.reason
    assert fold.worst_point is not None and fold.worst_point[0] > 0.8
    # with h(0) = 0, 1 + dh/dr > 0 implies 1 + h/r > 0: shells cross only
    # in a hollow body, or after a tear at the centre
    crossing = RadialStretch(CallableDisplacement(lambda r, t, p: -0.3 * (1.0 - r)),
                             rmax=A)
    report = crossing.validity(sample=validity_lattice(Skeleton([0.1, A])))
    assert not report and "cross through the origin" in report.reason
    assert not crossing.validity(sample=lattice)              # torn at the centre
    centre = RadialStretch(CallableDisplacement(lambda r, t, p: 0.1 + 0.0 * r), rmax=A)
    report = centre.validity(X=np.zeros((1, 3)))
    assert not report and "at the centre" in report.reason
    assert report.margin == pytest.approx(0.1 / A)
    with pytest.raises(ValueError, match="give either"):
        RadialStretch(kinked_h(0.1), rmax=A).validity()


def test_generic_validity_on_a_general_mapping():
    lattice = validity_lattice(SK)
    assert Squash(0.9, 0.1).validity(sample=lattice)
    bad = Squash(-0.5, 0.0).validity(sample=lattice)
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


def test_the_inverse_is_the_root_along_each_ray():
    """Against brentq one point at a time: dh/dr jumps at a knot, h does
    not change with r above it (a Newton step lands on the root), points
    lie beyond the body and at the origin, and a batch keeps its shape."""
    from scipy.optimize import brentq

    def h(r, theta, phi):
        return 0.03 * np.sin(theta) ** 2 * np.cos(2 * phi) * np.minimum(r, 0.6) / 0.6

    def dr(r, theta, phi):
        return np.where(r < 0.6, 0.03 * np.sin(theta) ** 2 * np.cos(2 * phi) / 0.6,
                        0.0)

    m = RadialStretch(CallableDisplacement(h, radial_derivative=dr, knots=[0.6]),
                      rmax=A)
    rng = np.random.default_rng(3)
    r = np.append(rng.uniform(0.0, 1.3, 59), 0.0)
    t, p = np.arccos(rng.uniform(-1, 1, 60)), rng.uniform(-np.pi, np.pi, 60)
    Y = m(cartesian_points(r, t, p)).reshape(6, 10, 3)
    rho, theta, phi, _ = spherical_coordinates(Y)
    want = np.array([brentq(lambda s, a=a, b=b, c=c: s + h(s, b, c) - a, 0.0, 2.0,
                            xtol=1e-14) if a > 0 else 0.0
                     for a, b, c in zip(rho.ravel(), theta.ravel(), phi.ravel())])
    got = m.inverse(Y)
    assert got.shape == (6, 10, 3)
    assert np.allclose(spherical_coordinates(got)[0].ravel(), want, rtol=0, atol=1e-12)
    assert np.allclose(got.reshape(-1, 3), cartesian_points(r, t, p), rtol=0,
                       atol=1e-12)
    # a torn centre: a physical point inside h(0) goes to the origin
    torn = RadialStretch(CallableDisplacement(lambda r, t, p: 0.01 + 0.0 * r),
                         rmax=A)
    back = torn.inverse(cartesian_points(np.array([0.005, 0.5]), 1.0, 2.0))
    assert np.allclose(np.linalg.norm(back, axis=-1), [0.0, 0.49], rtol=0, atol=1e-12)


def test_general_mapping_passes_the_contract():
    testing.check_mapping(Squash(0.9, 0.1), points())


def test_scaling_keeps_a_mapping_in_its_own_form():
    """A stretch of boundary reliefs stays one, gridded reliefs gridded; a
    stretch of a callable stays a stretch; the identity stays itself; a
    mapping with no `scaled` is wrapped, and a wrapper scaled again
    multiplies its factor; each is k m(X / k)."""
    from planetmodel import GriddedFunction, gauss_legendre, layer_linear
    from planetmodel.displacement import LayerLinear
    from planetmodel.mapping import scaled_mapping

    k = 6.371e6
    X = points()
    gf = GriddedFunction.sampled(lambda t, p: 0.01 * np.sin(t) ** 2 * np.cos(2 * p),
                                 gauss_legendre(8))
    stack = RadialStretch(layer_linear(SK, [None, gf, lambda t, p: 0.02 + 0 * t, gf]),
                          rmax=A)
    big = scaled_mapping(stack, k)
    assert isinstance(big, RadialStretch) and isinstance(big.h, LayerLinear)
    assert np.allclose(big.h.boundaries, k * SK.boundaries) and big.rmax == k * A
    assert isinstance(big.h.reliefs[1], GriddedFunction)
    assert np.allclose(big.h.reliefs[1].values, k * gf.values)
    assert big.knots == tuple(k * x for x in stack.knots)
    flat = scaled_mapping(RadialStretch(flattening_h(0.05), rmax=A), k)
    assert isinstance(flat, RadialStretch) and isinstance(flat.h, CallableDisplacement)
    for m, mk in ((stack, big), (RadialStretch(flattening_h(0.05), rmax=A), flat)):
        assert np.allclose(mk(k * X), k * m(X), rtol=1e-13, atol=0.0)
        assert np.allclose(mk.deformation_gradient(k * X), m.deformation_gradient(X),
                           rtol=1e-12, atol=1e-14)
        testing.check_mapping(mk, k * X)
    assert scaled_mapping(IdentityMapping(), k).is_identity
    wrapped = scaled_mapping(Squash(0.9, 0.1), 2.0)
    assert isinstance(wrapped, ScaledMapping) and wrapped.k == 2.0
    again = wrapped.scaled(3.0)
    assert isinstance(again, ScaledMapping) and again.k == 6.0
    assert again.mapping is wrapped.mapping
    with pytest.raises(ValueError, match="positive"):
        scaled_mapping(stack, 0.0)


def test_scaled_mapping():
    k = 6.371e6
    m = RadialStretch(flattening_h(0.05), rmax=A)
    big = ScaledMapping(m, k)
    X = points()
    assert np.allclose(big(k * X), k * m(X))
    assert np.allclose(big.deformation_gradient(k * X), m.deformation_gradient(X))
    assert np.allclose(big.jacobian(k * X), m.jacobian(X))
    assert np.allclose(big.inverse(big(k * X)), k * X, rtol=1e-9)
    assert big.validity(sample=validity_lattice(Skeleton(k * SK.boundaries)))
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
    assert r.shape == (10, 1, 1) and theta.shape == (1, 5, 1) and phi.shape == (1, 1, 4)
    assert r.ravel()[0] == 0.0                                  # the centre
    assert validity_lattice(Skeleton([0.5, 1.0]), n_r=3)[0].shape == (3, 1, 1)
    assert theta[0, 0, 0] == 0.0 and theta[0, -1, 0] == np.pi
    for lo, hi in zip(SK.boundaries[:-1], SK.boundaries[1:]):
        inside = (r.ravel() > lo) & (r.ravel() < hi)
        assert inside.sum() == 3
    X = cartesian_points(r, theta, phi)
    assert X.shape == (10, 5, 4, 3)


def test_a_stretch_at_the_centre_takes_its_radial_limit():
    """h = 0.1 r is the uniform dilation by 1.1, at the centre too."""
    m = RadialStretch(CallableDisplacement(lambda r, t, p: 0.1 * r), rmax=A)
    X = np.array([[0.0, 0.0, 0.0], [5e-10, 0.0, 0.0], [0.3, 0.2, 0.1]])
    assert np.allclose(m(X), 1.1 * X, rtol=1e-12, atol=0.0)
    assert np.allclose(m.deformation_gradient(X), 1.1 * np.eye(3), rtol=1e-9)
    assert np.allclose(m.jacobian(X), 1.1 ** 3, rtol=1e-9)
    dF = m.linearise(lambda r, t, p: 0.01 * r, X=X).dF
    assert np.allclose(dF, 0.01 * np.eye(3), rtol=1e-6, atol=1e-12)


def test_a_uniform_uplift_tears_the_centre_and_is_refused():
    uplift = RadialStretch(CallableDisplacement(lambda r, t, p: 0.01 + 0.0 * r),
                           rmax=A)
    with pytest.raises(ValueError, match="at the centre"):
        Geometry(SK, mapping=uplift)
    Geometry(Skeleton([0.5, 1.0]), mapping=uplift)        # a hollow body has none


class Shear:
    """A bare mapping: the protocol's three methods and nothing else."""

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        return X + 0.1 * X[..., [1]] * np.array([1.0, 0.0, 0.0])

    def deformation_gradient(self, X):
        F = np.broadcast_to(np.eye(3), np.shape(X)[:-1] + (3, 3)).copy()
        F[..., 0, 1] = 0.1
        return F

    def jacobian(self, X):
        return np.ones(np.shape(X)[:-1])


def test_a_scaled_mapping_asks_what_the_mapping_has():
    scaled = ScaledMapping(Shear(), 2.0)
    assert scaled.validity(sample=validity_lattice(SK))
    assert Geometry(SK, mapping=Shear()).scaled(2.0).validity()
    with pytest.raises(NotImplementedError):
        scaled.inverse(np.zeros((1, 3)))
    with pytest.raises(NotImplementedError):
        scaled.linearise(lambda r, t, p: 0.0 * r, X=np.ones((1, 3)))


def test_a_scaled_stretch_linearises_as_the_stretch_of_the_scaled_displacement():
    k = 3.0

    def h(r, t, p):
        return 0.02 * r ** 2 * np.cos(t)

    def delta(r, t, p):
        return 0.01 * r * np.sin(t) * np.cos(p)

    scaled = ScaledMapping(RadialStretch(CallableDisplacement(h), rmax=1.0), k)
    direct = RadialStretch(CallableDisplacement(lambda r, t, p: k * h(r / k, t, p)),
                           rmax=k)
    X = k * points()[:20]
    a, b = scaled.linearise(delta, X=X), direct.linearise(delta, X=X)
    assert np.allclose(a.dF, b.dF, rtol=1e-6, atol=1e-9)
    assert np.allclose(a.dJ, b.dJ, rtol=1e-6, atol=1e-9)


def test_the_stretch_s_inverse_breaks_a_newton_cycle_between_layers():
    """Linear in s on each layer, s + h(s) can send Newton from the outer
    layer into the inner one and straight back, each step landing on an end
    of the bracket; the bisection takes over there and the root is found."""
    from planetmodel import RadialStretch, Skeleton, layer_linear
    sk = Skeleton([0.0, 0.5, 0.8, 1.0])

    def up(t, p):
        return 0.15 + 0.0 * t

    def down(t, p):
        return -0.02 + 0.0 * t

    m = RadialStretch(layer_linear(sk, [None, None, up, down]), rmax=1.0)
    x = cartesian_points(np.array([0.85, 0.9, 0.6]), 1.0, 2.0)
    X = m.inverse(x)
    assert np.allclose(m(X), x, rtol=0, atol=1e-12)
    assert np.allclose(np.linalg.norm(X, axis=-1), [0.5 + 0.35 / 1.5, 0.5 + 0.4 / 1.5,
                                                    0.5 + 0.1 / 1.5], atol=1e-12)
