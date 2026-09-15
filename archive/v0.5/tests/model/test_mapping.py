"""Mappings: the closed forms, checked against numerical differentiation.

The formulas in RadialStretch are transcribed algebra, and a sign or a
transpose error in transcribed algebra survives inspection easily.  So
nothing here is checked against the formula it implements: F is compared
with a central difference of the mapping itself, J with the determinant
of that numerical F, and the linearisation with a difference in the
perturbation amplitude.  Only then are the sparse spherical-frame
entries checked, as documentation of the shape rather than as evidence.
"""
import numpy as np
import pytest

from planetmodel import PREM, Skeleton
from planetmodel.model.displacement import (CallableDisplacement, ZeroDisplacement,
                                       layer_linear)
from planetmodel.model.frames import spherical_coordinates
from planetmodel.model.mapping import (IdentityMapping, Mapping,
                                  MappingBase, MappingPerturbation,
                                  RadialStretch, ValidityReport,
                                  validity_lattice)
from planetmodel.model.character import ELASTIC
from planetmodel.model.topography import AnalyticTopography
from planetmodel.testing import check_mapping

A = 6.371e6


def points(n=200, lo=0.2, hi=0.9, seed=11):
    """Random points in a shell, avoiding the origin and the surface."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v * rng.uniform(lo * A, hi * A, size=(n, 1))


def smooth_h(amp=3.0e4, kr=2.0):
    """A smooth displacement with analytic derivatives throughout."""
    return CallableDisplacement(
        lambda r, t, p: amp * np.sin(kr * np.pi * r / A) * np.cos(t),
        radial_derivative=lambda r, t, p: (
            amp * (kr * np.pi / A) * np.cos(kr * np.pi * r / A) * np.cos(t)),
        angular_gradient=lambda r, t, p: (
            -amp * np.sin(kr * np.pi * r / A) * np.sin(t),
            np.zeros_like(np.asarray(t, dtype=float))))


def numerical_F(m, X, step=None):
    """d m_i / d X_j by central differences -- the oracle."""
    step = step if step is not None else 1e-6 * float(np.max(np.abs(X)))
    F = np.empty(X.shape[:-1] + (3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = step
        F[..., :, j] = (np.asarray(m(X + e), dtype=float)
                        - np.asarray(m(X - e), dtype=float)) / (2.0 * step)
    return F


# ------------------------------------------------------------- contracts

@pytest.mark.parametrize("build", [
    IdentityMapping,
    lambda: RadialStretch(ZeroDisplacement()),
    lambda: RadialStretch(smooth_h(), rmax=A),
    lambda: RadialStretch(smooth_h(amp=-2.0e4, kr=1.0), rmax=A),
])
def test_shipped_mappings_satisfy_the_contract(build):
    check_mapping(build(), points())


def test_the_contract_catches_a_displacement_that_is_not_m_minus_X():
    """The clause has teeth: a displacement computed a second way."""
    class Sloppy(RadialStretch):
        def displacement(self, X):
            return 1.001 * super().displacement(X)

    with pytest.raises(AssertionError, match=r"m\(X\) - X"):
        check_mapping(Sloppy(smooth_h(), rmax=A), points(20))


def test_the_contract_catches_a_wrong_linearisation():
    """And so does the amplitude difference: dJ scaled by a half."""
    class Sloppy(RadialStretch):
        def linearise(self, delta, X=None):
            lin = super().linearise(delta, X=X)
            return MappingPerturbation(lin.dF, 0.5 * lin.dJ)

    with pytest.raises(AssertionError, match="dJ disagrees"):
        check_mapping(Sloppy(smooth_h(), rmax=A), points(20))


def test_protocol_is_structural():
    assert isinstance(IdentityMapping(), Mapping)
    assert isinstance(RadialStretch(smooth_h()), Mapping)

    class Foreign:
        """Anything with the three methods is a Mapping."""
        def __call__(self, X): return np.asarray(X, dtype=float)
        def deformation_gradient(self, X, *, frame="cartesian"):
            return np.broadcast_to(np.eye(3), np.shape(X)[:-1] + (3, 3))
        def jacobian(self, X): return np.ones(np.shape(X)[:-1])

    assert isinstance(Foreign(), Mapping)


# ------------------------------------------- F against the numerical oracle

def test_F_matches_a_central_difference_of_the_mapping():
    """The primary oracle: F is the gradient of m, however it was derived."""
    m = RadialStretch(smooth_h(), rmax=A)
    X = points()
    F = m.deformation_gradient(X)
    fd = numerical_F(m, X)
    assert np.max(np.abs(F - fd)) / np.max(np.abs(fd)) < 1e-8


def test_F_converges_at_second_order_in_the_step():
    """A wrong closed form would not track the difference as h shrinks."""
    m = RadialStretch(smooth_h(), rmax=A)
    X = points(60)
    F = m.deformation_gradient(X)
    coarse = np.max(np.abs(F - numerical_F(m, X, step=1e-4 * A)))
    finer = np.max(np.abs(F - numerical_F(m, X, step=1e-5 * A)))
    assert finer < 0.05 * coarse          # ~100x for second order


def test_the_convention_is_dm_i_by_dX_j():
    """F[i, j] differentiates component i along direction j."""
    m = RadialStretch(smooth_h(), rmax=A)
    X = points(20)
    F = m.deformation_gradient(X)
    step = 1e-6 * A
    e = np.array([step, 0.0, 0.0])
    column0 = (m(X + e) - m(X - e)) / (2.0 * step)
    assert np.allclose(F[..., :, 0], column0, rtol=1e-6, atol=1e-9)


def test_jacobian_is_the_determinant_of_the_numerical_F():
    m = RadialStretch(smooth_h(), rmax=A)
    X = points()
    assert np.allclose(m.jacobian(X), np.linalg.det(numerical_F(m, X)),
                       rtol=1e-7, atol=1e-9)


def test_identity_is_exactly_the_identity():
    X = points()
    m = IdentityMapping()
    assert np.array_equal(m(X), X)
    assert np.allclose(m.deformation_gradient(X), np.eye(3))
    assert np.allclose(m.jacobian(X), 1.0)
    assert m.is_identity
    assert RadialStretch(ZeroDisplacement()).is_identity


# ------------------------------------------ the closed forms, documented

def test_spherical_frame_entries_have_the_stated_form():
    """F in the local frame, as written in the design.

    Documentation of the shape; the evidence that it is *right* is the
    finite-difference comparison above.
    """
    h = smooth_h()
    m = RadialStretch(h, rmax=A)
    X = points(150)
    r, th, ph, _ = spherical_coordinates(X)
    F = m.deformation_gradient(X, frame="spherical")

    hv = h(r, th, ph)
    hr = h.radial_derivative(r, th, ph)
    ht, hp = h.angular_gradient(r, th, ph)

    assert np.allclose(F[..., 0, 0], 1.0 + hr)
    assert np.allclose(F[..., 0, 1], ht / r)
    assert np.allclose(F[..., 0, 2], hp / (r * np.sin(th)))
    assert np.allclose(F[..., 1, 1], 1.0 + hv / r)
    assert np.allclose(F[..., 2, 2], 1.0 + hv / r)
    for i, j in ((1, 0), (2, 0), (1, 2), (2, 1)):
        assert np.allclose(F[..., i, j], 0.0)


def test_jacobian_matches_its_closed_form():
    """J = (1 + dh/dr)(1 + h/r)^2."""
    h = smooth_h()
    m = RadialStretch(h, rmax=A)
    X = points(150)
    r, th, ph, _ = spherical_coordinates(X)
    want = (1.0 + h.radial_derivative(r, th, ph)) * (1.0 + h(r, th, ph) / r) ** 2
    assert np.allclose(m.jacobian(X), want, rtol=1e-14)


def test_the_frame_is_orthonormal_and_right_handed():
    _, _, _, R = spherical_coordinates(points(100))
    assert np.allclose(np.einsum("...ik,...jk->...ij", R, R), np.eye(3),
                       atol=1e-14)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-14)


def test_a_radial_map_moves_points_along_their_own_radius():
    m = RadialStretch(smooth_h(), rmax=A)
    X = points(50)
    x = m(X)
    cross = np.cross(X, x)
    assert np.max(np.linalg.norm(cross, axis=-1)) < 1e-6 * A ** 2


# ------------------------------------------------------------ validity

@pytest.mark.parametrize("k,valid", [(0.5, True), (0.9, True), (0.99, True),
                                     (1.0, False), (1.01, False), (2.0, False)])
def test_validity_flips_exactly_where_the_condition_does(k, valid):
    """1 + dh/dr > 0 is the condition, and the report tracks it exactly."""
    h = CallableDisplacement(
        lambda r, t, p: -k * np.asarray(r, dtype=float),
        radial_derivative=lambda r, t, p: np.full_like(
            np.asarray(r, dtype=float), -k))
    r = np.linspace(1e5, A, 40)
    rep = RadialStretch(h).is_valid(
        sample=(r, np.full_like(r, 1.0), np.full_like(r, 0.5)))
    assert bool(rep) is valid
    assert rep.margin == pytest.approx(1.0 - k, abs=1e-12)


def test_an_invalid_report_says_where_and_why():
    h = CallableDisplacement(
        lambda r, t, p: -2.0 * np.asarray(r, dtype=float),
        radial_derivative=lambda r, t, p: np.full_like(
            np.asarray(r, dtype=float), -2.0))
    r = np.linspace(1e5, A, 20)
    rep = RadialStretch(h).is_valid(
        sample=(r, np.full_like(r, 0.4), np.full_like(r, 0.1)))
    assert not rep
    assert "folds radially" in rep.reason
    assert rep.worst_point is not None
    assert "INVALID" in repr(rep)


def test_exaggeration_sweep_finds_the_folding_threshold():
    """The mesher's question: how much exaggeration is too much?

    The outermost PREM span is 12 km and the relief is 3 km, so dh/dr
    reaches -1 at exaggeration 4 exactly.
    """
    body = (PREM(ocean=False).name_interface(-1, "surface")
            .with_surface("surface",
                          AnalyticTopography(lambda t, p: 3.0e3 * np.cos(t))))
    b = body.skeleton.boundaries
    r = np.linspace(float(b[-2]) + 1.0, float(b[-1]) - 1.0, 40)
    th = np.linspace(0.0, np.pi, 30)
    R, T = np.meshgrid(r, th, indexing="ij")
    sample = (R, T, np.zeros_like(R))

    def valid_at(exag):
        bo = body.with_surface(
            "surface", AnalyticTopography(lambda t, p: 3.0e3 * np.cos(t)) * exag)
        return bool(RadialStretch(layer_linear()(bo)).is_valid(sample=sample))

    assert valid_at(1) and valid_at(2) and valid_at(3.9)
    assert not valid_at(4.1) and not valid_at(10)


def test_a_partial_sample_can_miss_a_fold():
    """Documented caveat: the verdict is only as good as the sample.

    Relief that folds the mapping near one pole is harmless near the
    other, so a sample covering one colatitude reports a folding
    mapping valid.  The mesher passes its actual nodes, which cover the
    sphere; anyone else must too.
    """
    body = (PREM(ocean=False).name_interface(-1, "surface")
            .with_surface("surface",
                          AnalyticTopography(lambda t, p: 3.0e4 * np.cos(t))))
    m = RadialStretch(layer_linear()(body))
    b = body.skeleton.boundaries
    r = np.linspace(float(b[-2]) + 1.0, float(b[-1]) - 1.0, 40)

    narrow = m.is_valid(sample=(r, np.full_like(r, 0.2), np.zeros_like(r)))
    th = np.linspace(0.0, np.pi, 30)
    R, T = np.meshgrid(r, th, indexing="ij")
    wide = m.is_valid(sample=(R, T, np.zeros_like(R)))

    assert bool(narrow) and not bool(wide)


def test_generic_validity_uses_the_jacobian():
    """MappingBase's default test, used by mappings without closed forms."""
    m = IdentityMapping()
    assert bool(MappingBase.is_valid(m, X=points(20)))


def test_validity_needs_something_to_check():
    with pytest.raises(ValueError, match="either points"):
        RadialStretch(smooth_h()).is_valid()


# ------------------------------------------------------------- inverse

def test_inverse_round_trips():
    m = RadialStretch(smooth_h(), rmax=A)
    X = points(60)
    assert np.allclose(m.inverse(m(X)), X, rtol=1e-9, atol=1e-5)


def test_inverse_round_trips_near_the_centre():
    m = RadialStretch(smooth_h(amp=1.0e3), rmax=A)
    X = points(30, lo=0.001, hi=0.05)
    assert np.allclose(m.inverse(m(X)), X, rtol=1e-8, atol=1e-4)


def test_identity_inverts_trivially():
    X = points(10)
    assert np.allclose(IdentityMapping().inverse(X), X)


def test_mappings_without_an_inverse_say_so():
    class Bare(MappingBase):
        def __call__(self, X): return np.asarray(X, dtype=float)
        def deformation_gradient(self, X, *, frame="cartesian"):
            return np.broadcast_to(np.eye(3), np.shape(X)[:-1] + (3, 3))
        def jacobian(self, X): return np.ones(np.shape(X)[:-1])

    with pytest.raises(NotImplementedError, match="does not provide an inverse"):
        Bare().inverse(points(3))


# -------------------------------------------------------- linearisation

def test_linearise_matches_a_finite_difference_in_amplitude():
    """dF and dJ against differencing the mapping in the perturbation.

    Both h and the perturbation carry analytic derivatives, so the only
    numerical step in the comparison is the amplitude difference itself.
    """
    h, d = smooth_h(3.0e4, 2.0), smooth_h(5.0e3, 1.0)
    m = RadialStretch(h, rmax=A)
    X = points(120)
    lin = m.linearise(d, X=X)

    def shifted(s):
        def value(r, t, p):
            return h(r, t, p) + s * d(r, t, p)

        def radial(r, t, p):
            return (h.radial_derivative(r, t, p)
                    + s * d.radial_derivative(r, t, p))

        def angular(r, t, p):
            a1, b1 = h.angular_gradient(r, t, p)
            a2, b2 = d.angular_gradient(r, t, p)
            return a1 + s * a2, b1 + s * b2

        return RadialStretch(CallableDisplacement(
            value, radial_derivative=radial, angular_gradient=angular), rmax=A)

    eps = 1e-3
    dF = (shifted(eps).deformation_gradient(X)
          - shifted(-eps).deformation_gradient(X)) / (2.0 * eps)
    dJ = (shifted(eps).jacobian(X) - shifted(-eps).jacobian(X)) / (2.0 * eps)

    assert np.max(np.abs(lin.dF - dF)) / np.max(np.abs(dF)) < 1e-8
    assert np.max(np.abs(lin.dJ - dJ)) / np.max(np.abs(dJ)) < 1e-8


def test_linearise_is_linear_in_the_perturbation():
    m = RadialStretch(smooth_h(), rmax=A)
    X = points(40)
    d = smooth_h(5.0e3, 1.0)
    single = m.linearise(d, X=X)
    double = m.linearise(CallableDisplacement(
        lambda r, t, p: 2.0 * d(r, t, p),
        radial_derivative=lambda r, t, p: 2.0 * d.radial_derivative(r, t, p),
        angular_gradient=lambda r, t, p: tuple(
            2.0 * g for g in d.angular_gradient(r, t, p))), X=X)
    assert np.allclose(double.dF, 2.0 * single.dF, rtol=1e-12)
    assert np.allclose(double.dJ, 2.0 * single.dJ, rtol=1e-12)


def test_linearise_needs_points():
    with pytest.raises(ValueError, match="points X"):
        RadialStretch(smooth_h()).linearise(smooth_h())


# ----------------------------------------------------------- displacement

def test_displacement_is_the_mapping_minus_the_point():
    """u(X) = m(X) - X, for both shipped mappings.

    Trivial by definition, which is the point: it is defined in terms of
    m rather than read off h, so a mapping cannot report a displacement
    its own __call__ disagrees with -- and the exporters write this
    array, not h.
    """
    X = points(50)
    assert np.allclose(IdentityMapping().displacement(X), 0.0)
    m = RadialStretch(smooth_h(), rmax=A)
    u = m.displacement(X)
    assert u.shape == X.shape
    assert np.allclose(u, m(X) - X, rtol=0, atol=1e-9)


def test_a_radial_displacement_is_h_along_e_r():
    """For a radial stretch the vector field is h e_r, and only that."""
    h = smooth_h()
    m = RadialStretch(h, rmax=A)
    X = points(60)
    r, theta, phi, R = spherical_coordinates(X)
    want = np.asarray(h(r, theta, phi))[..., None] * R[..., :, 0]
    assert np.allclose(m.displacement(X), want, rtol=1e-12,
                       atol=1e-9 * float(np.max(np.abs(want))))


# ---------------------------------------------------- derived quantities

def test_cauchy_green_and_gravity_tensor_are_consistent():
    """a = J C^-1, from the [extra] tier."""
    m = RadialStretch(smooth_h(), rmax=A)
    X = points(40)
    F = m.deformation_gradient(X)
    C = m.right_cauchy_green(X)
    assert np.allclose(C, np.einsum("...ki,...kj->...ij", F, F))
    a = m.gravity_tensor(X)
    assert np.allclose(np.einsum("...ij,...jk->...ik", a, C),
                       m.jacobian(X)[..., None, None] * np.eye(3), atol=1e-10)


def test_a_mapping_pushes_a_rank_four_field_forward():
    """The convenience carries tensors, not only scalars.

    The scalar cases and the generic rule itself live in
    test_pushforward.py; what is checked here is only that the mapping's
    own convenience reaches them, with F and J it formed itself.
    """
    m = RadialStretch(smooth_h(), rmax=A)
    X = points(5)
    T = np.broadcast_to(np.eye(3)[:, :, None, None] * np.eye(3),
                        X.shape[:-1] + (3, 3, 3, 3))
    c = m.push_forward(T, X, ELASTIC)
    assert c.shape == X.shape[:-1] + (3, 3, 3, 3)
    assert np.all(np.isfinite(c))
    assert np.allclose(c, np.einsum("...ijkl->...klij", c))


# ------------------------------------------------- piecewise displacements

def test_layer_linear_mapping_passes_away_from_knots():
    """A piecewise-linear h is smooth inside a span, and F is right there."""
    body = (PREM(ocean=False).name_interface(-1, "surface")
            .with_surface("surface",
                          AnalyticTopography(lambda t, p: 3.0e3 * np.cos(t))))
    b = body.skeleton.boundaries
    lo, hi = float(b[-2]), float(b[-1])
    rng = np.random.default_rng(5)
    v = rng.normal(size=(60, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    X = v * rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), size=(60, 1))
    check_mapping(RadialStretch(layer_linear()(body), rmax=hi), X, rtol=1e-4)


# ------------------------------------------------------------ validation

def test_bad_point_shapes_are_rejected():
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        RadialStretch(smooth_h()).deformation_gradient(np.zeros((4, 2)))


def test_bad_frame_is_rejected():
    with pytest.raises(ValueError, match="frame must be"):
        RadialStretch(smooth_h()).deformation_gradient(points(3),
                                                       frame="polar")


def test_validity_report_is_falsey_when_invalid():
    assert not ValidityReport(False, -1.0)
    assert ValidityReport(True, 1.0)


# ----------------------------------------------- review fixes, pinned

def test_identity_linearise_works_and_equals_the_zero_stretch():
    """Regression: IdentityMapping.linearise neither accepted points nor
    passed them through, so it raised unconditionally."""
    d = smooth_h(1.0e3, 1.0)
    X = points(30)
    a = IdentityMapping().linearise(d, X=X)
    b = RadialStretch(ZeroDisplacement()).linearise(d, X=X)
    assert np.allclose(a.dF, b.dF)
    assert np.allclose(a.dJ, b.dJ)
    # and it is the truth: differencing the identity perturbed by eps*d
    eps = 1e-3
    up = RadialStretch(CallableDisplacement(
        lambda r, t, p: eps * d(r, t, p),
        radial_derivative=lambda r, t, p: eps * d.radial_derivative(r, t, p),
        angular_gradient=lambda r, t, p: tuple(
            eps * g for g in d.angular_gradient(r, t, p))))
    dF_fd = (up.deformation_gradient(X) - np.eye(3)) / eps
    assert np.max(np.abs(a.dF - dF_fd)) / max(np.max(np.abs(dF_fd)), 1e-30) < 1e-2


def test_validity_lattice_covers_thin_spans_and_both_poles():
    """A global uniform radial sample would put no point in PREM's 12 km
    outermost span, though thin spans are where dh/dr is largest; the
    lattice lays points per span, so every span gets its share."""

    body = PREM(ocean=False)
    r, th, ph = validity_lattice(body.skeleton, n_r=8)
    b = body.skeleton.boundaries
    for lo, hi in zip(b[:-1], b[1:]):
        inside = (r > lo) & (r < hi)
        assert inside.sum() == 8, f"span {lo}-{hi} undersampled"
    assert th.min() == 0.0 and th.max() == pytest.approx(np.pi)


def test_validity_lattice_finds_the_fold_a_narrow_sample_missed():
    from planetmodel.model.displacement import layer_linear

    body = (PREM(ocean=False).name_interface(-1, "surface")
            .with_surface("surface",
                          AnalyticTopography(lambda t, p: 3.0e3 * np.cos(t))
                          * 10.0))
    m = RadialStretch(layer_linear()(body))
    rep = m.is_valid(sample=validity_lattice(body.skeleton))
    assert not rep
    assert "folds radially" in rep.reason
    # the report names the folding span, not just a verdict
    assert rep.worst_point[0] == pytest.approx(6.356e6, rel=1e-3)


def test_is_valid_accepts_a_broadcastable_sample():
    """Regression: the failure report indexed the raw sample arrays with
    a flat index into the broadcast result, so a lattice of (n,1,1),
    (1,m,1), (1,1,k) axes crashed exactly when the verdict was invalid
    -- the one time the report matters."""
    h = CallableDisplacement(
        lambda r, t, p: -2.0 * np.asarray(r, dtype=float),
        radial_derivative=lambda r, t, p: np.full_like(
            np.asarray(r, dtype=float), -2.0))
    r = np.linspace(1e5, A, 12)[:, None, None]
    th = np.linspace(0.0, np.pi, 5)[None, :, None]
    ph = np.linspace(-np.pi, np.pi, 4, endpoint=False)[None, None, :]
    rep = RadialStretch(h).is_valid(sample=(r, th, ph))
    assert not rep and rep.worst_point is not None


# -------------------------------------------- validity is dimensionless

def test_the_origin_is_the_fixed_point_not_a_fold():
    """Regression: is_valid tested r + h > 0, which fails at r = 0 for
    every radial map -- the centre is the fixed point and stays put, so
    r + h is identically zero there.  Any node set containing the origin
    was therefore reported invalid, which the mesher hit immediately."""
    m = RadialStretch(smooth_h(), rmax=A)
    X = np.vstack([points(20), np.zeros((1, 3))])
    report = m.is_valid(X=X)
    assert report, repr(report)


def test_the_validity_margin_does_not_depend_on_units():
    """Regression: min(r + h, 1 + dh/dr) mixes a length with a pure
    number, so the reported margin changed by six orders between SI and
    non-dimensional units.  J = (1 + dh/dr)(1 + h/r)^2, and both of
    those factors are dimensionless."""
    from planetmodel.model.topography import AnalyticTopography

    body = (PREM(ocean=False).name_interface(-1, "surface")
            .with_surface("surface",
                          AnalyticTopography(lambda t, p: 3.0e3 * np.cos(t))))
    nd = body.nondimensionalised()
    X = points(50, lo=0.1, hi=0.95, seed=3) * (6.368e6 / A)

    si_margin = body.mapping(rule=layer_linear()).is_valid(X=X).margin
    nd_margin = nd.mapping(rule=layer_linear()).is_valid(
        X=X / nd.scales.length).margin
    assert si_margin == pytest.approx(nd_margin, rel=1e-12)


def test_a_displacement_at_the_centre_is_its_own_failure():
    """h(0) != 0 has no radial direction to act along; h/r diverges.
    That is a broken displacement, not a fold, and says so."""
    h = CallableDisplacement(
        lambda r, t, p: np.full_like(np.asarray(r, dtype=float), 1.0e5),
        radial_derivative=lambda r, t, p: np.zeros_like(
            np.asarray(r, dtype=float)))
    report = RadialStretch(h).is_valid(X=np.zeros((1, 3)))
    assert not report
    assert "centre" in report.reason


def test_shells_crossing_the_origin_are_caught_separately():
    """1 + h/r <= 0 means a shell has been pushed through the centre."""
    # A negative slope would trip the radial test first, so this one has
    # positive slope and a strongly negative value: only 1 + h/r fails.
    h2 = CallableDisplacement(
        lambda r, t, p: -1.5 * A + 0.5 * np.asarray(r, dtype=float),
        radial_derivative=lambda r, t, p: np.full_like(
            np.asarray(r, dtype=float), 0.5))
    report = RadialStretch(h2).is_valid(X=points(10, lo=0.2, hi=0.4))
    assert not report
    assert "1 + h/r" in report.reason


def test_a_generic_mapping_accepts_a_sample():
    import numpy as np
    import pytest
    from planetmodel.model.mapping import MappingBase, validity_lattice

    class Squash(MappingBase):
        def __call__(self, X):
            return np.asarray(X) * np.array([1.0, 1.0, 0.5])

        def deformation_gradient(self, X, *, frame="cartesian"):
            return np.broadcast_to(np.diag([1.0, 1.0, 0.5]),
                                   np.shape(X)[:-1] + (3, 3))

        def jacobian(self, X):
            return np.full(np.shape(X)[:-1], 0.5)

    verdict = Squash().is_valid(sample=validity_lattice(Skeleton([0.0, 1.0])))
    assert verdict and verdict.margin == pytest.approx(0.5)


# ------------------------------------------------------------------ frames

def test_spherical_coordinates_has_no_length_floor():
    """The same geometric statement in metres and in scaled units: the
    origin maps to a finite direction, and a point a nanometre off it
    is not the origin."""
    from planetmodel.model.frames import spherical_coordinates
    r, th, ph, R = spherical_coordinates(np.zeros(3))
    assert r == 0.0 and np.isfinite(th) and np.isfinite(ph)
    assert np.allclose(R @ R.T, np.eye(3))
    r, th, ph, _ = spherical_coordinates(np.array([0.0, 0.0, 1e-9]))
    assert r == pytest.approx(1e-9) and th == 0.0


def test_the_origin_floor_scales_with_the_body():
    """A stretch on a unit body and the same stretch scaled to the Earth
    agree, point for point, because the floor is a fraction of rmax."""
    h_unit = CallableDisplacement(lambda r, t, p: 0.01 * r ** 2, knots=(1.0,))
    h_earth = CallableDisplacement(lambda r, t, p: 0.01 * r ** 2 / A,
                                   knots=(A,))
    X = points(50, lo=1e-12, hi=0.9) / A          # a unit body's points
    unit = RadialStretch(h_unit, rmax=1.0)
    earth = RadialStretch(h_earth, rmax=A)
    assert np.allclose(earth(X * A) / A, unit(X), rtol=1e-12, atol=1e-14)
    assert np.allclose(earth.jacobian(X * A), unit.jacobian(X), rtol=1e-9)
    inv = earth.inverse(earth(X * A))
    assert np.allclose(inv, X * A, rtol=1e-9, atol=1e-6)
