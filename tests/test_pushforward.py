"""Push-forward and pull-back: the array rule, and the fields carried by it.

The rule is one contraction built from the rank, so the construction is
what is checked: an explicit quadruple loop at rank 4, the matrix
products at ranks 1 and 2, rho / J at rank 0, the symmetries preserved,
pull of push the identity at every rank and weight, and a uniform
dilation where the answer is a power of k by eye.  The field layer is
then held to the array layer at the same points, to the field contract,
and to the frame at the physical point.
"""
from itertools import product

import numpy as np
import pytest

from planetmodel import (CallableDisplacement, IdentityMapping, MappingBase,
                         RadialStretch)
from planetmodel.character import DENSITY, ELASTIC, SCALAR, STRESS, VECTOR, Character
from planetmodel.fields import AnalyticField, RadialField, constant_field
from planetmodel.frames import (cartesian_points, rotate_slots, spherical_coordinates,
                                spherical_frame, tensor_to_voigt, voigt_to_tensor)
from planetmodel.pushforward import (PulledBackField, PushedForwardField,
                                     full_components, pull_back, push_forward)
from planetmodel.testing import check_field

rng = np.random.default_rng(7)

IV = (0.5, 1.0)
RMAX = 1.0
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


# ------------------------------------------------------------ ingredients

def random_deformation(scale=0.15):
    F = np.eye(3) + scale * rng.normal(size=(3, 3))
    return F, np.linalg.det(F)


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


def push4(T, F, J):
    """The rank-4 rule written out, independent of the library."""
    return np.einsum("iA,jB,kC,lD,ABCD->ijkl", F, F, F, F, T) / J


def relerr(got, want):
    got, want = np.asarray(got), np.asarray(want)
    scale = float(np.max(np.abs(want))) or 1.0
    return float(np.max(np.abs(got - want))) / scale


def flattening_h(f):
    def h(r, theta, phi):
        return -f * r * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)

    def dr(r, theta, phi):
        return -f * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0) + 0.0 * r

    def grad(r, theta, phi):
        return (3.0 * f * r * np.cos(theta) * np.sin(theta),
                np.zeros(np.broadcast(r, theta, phi).shape))

    return CallableDisplacement(h, radial_derivative=dr, angular_gradient=grad)


def uniform_dilation(c):
    """h = c r: F = (1 + c) I and J = (1 + c)^3 everywhere."""
    def zero(r, t, p):
        return np.zeros(np.broadcast(r, t, p).shape)

    return CallableDisplacement(
        lambda r, t, p: c * r + zero(r, t, p),
        radial_derivative=lambda r, t, p: c + zero(r, t, p),
        angular_gradient=lambda r, t, p: (zero(r, t, p), zero(r, t, p)))


class Squash(MappingBase):
    """A non-radial mapping, x -> (x, y, c z + b x y), so the frames at X
    and at m(X) differ."""

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


def stretch():
    return RadialStretch(flattening_h(0.05), rmax=RMAX)


MAPPINGS = {"identity": IdentityMapping, "stretch": stretch}


def grid():
    r = np.linspace(0.55, 0.95, 4)[:, None, None]
    th = np.array([0.4, 1.3, 2.7])[None, :, None]
    ph = np.array([-1.1, 0.9])[None, None, :]
    return r, th, ph


def rho():
    return RadialField(IV, lambda r: 3.0 - r ** 2, character=DENSITY, name="rho")


def vector():
    return RadialField(IV, [lambda r: r, 0.3, lambda r: 0.2 * r ** 2],
                       character=VECTOR, name="v")


ELASTIC_VOIGT = (lambda M: 0.5 * (M + M.T) + 6.0 * np.eye(6))(
    np.random.default_rng(3).normal(size=(6, 6)))


def elastic():
    return constant_field(ELASTIC_VOIGT, IV, character=ELASTIC, name="c")


def stress():
    def fn(r, t, p):
        return np.stack([r, r * np.cos(t), 1.0 + 0.0 * r, 0.1 * np.sin(t),
                         0.2 * r, 0.3 * np.cos(p)], axis=-1)

    return AnalyticField(IV, fn, character=STRESS, name="S")


FIELDS = {"rho": rho, "vector": vector, "elastic": elastic, "stress": stress}


# ------------------------------------------------------- the array level

def test_density_pushes_forward_as_rho_over_J():
    F, J = random_deformation()
    rho = np.full((5,), 3300.0)
    assert np.allclose(push_forward(rho, F, J, DENSITY), rho / J, rtol=1e-14)
    assert np.array_equal(push_forward(rho, F, J, SCALAR), rho)


def test_a_vector_is_wrapped_once():
    F, J = random_deformation()
    v = rng.normal(size=3)
    assert relerr(push_forward(v, F, J, VECTOR), F @ v) < 1e-15


def test_a_stress_pushes_to_F_S_FT_over_J():
    F, J = random_deformation()
    S = rng.normal(size=(3, 3))
    S = 0.5 * (S + S.T)
    assert relerr(push_forward(S, F, J, STRESS), F @ S @ F.T / J) < 1e-14


def test_rank_four_preserves_the_full_symmetries():
    F, J = random_deformation()
    T = random_second_tensor()
    c = push_forward(T, F, J, ELASTIC)
    assert np.allclose(c, np.einsum("ijkl->jikl", c))
    assert np.allclose(c, np.einsum("ijkl->ijlk", c))
    assert np.allclose(c, np.einsum("ijkl->klij", c))
    assert relerr(c, push4(T, F, J)) < 1e-15


def test_the_contraction_agrees_with_an_explicit_quadruple_loop():
    F, J = random_deformation()
    T = rng.normal(size=(3, 3, 3, 3))
    want = np.zeros((3, 3, 3, 3))
    for i, j, k, l in product(range(3), repeat=4):
        total = 0.0
        for a, b, c, d in product(range(3), repeat=4):
            total += F[i, a] * F[j, b] * F[k, c] * F[l, d] * T[a, b, c, d]
        want[i, j, k, l] = total / J
    assert relerr(push_forward(T, F, J, ELASTIC), want) < 1e-13


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("weight", [0, 1])
def test_push_of_pull_is_the_identity(rank, weight):
    char = Character(rank, weight)
    T = rng.normal(size=(3,) * rank)
    near, _ = random_deformation(0.05)
    big, det = random_deformation(0.4)
    big = big * np.cbrt(3.0 / det)
    assert abs(np.linalg.det(big) - 3.0) < 1e-12
    for F in (near, big):
        J = np.linalg.det(F)
        back = push_forward(pull_back(T, F, J, char), F, J, char)
        assert relerr(back, T) < 1e-12
        there = pull_back(push_forward(T, F, J, char), F, J, char)
        assert relerr(there, T) < 1e-12


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("weight", [0, 1])
def test_a_uniform_dilation_scales_by_k_to_the_rank_minus_three_weights(rank,
                                                                          weight):
    """F = k I, J = k^3: a weight-w rank-n tensor scales by k^(n - 3w)."""
    k = 1.7
    char = Character(rank, weight)
    T = rng.normal(size=(4,) + (3,) * rank)
    F = np.broadcast_to(k * np.eye(3), (4, 3, 3))
    J = np.full(4, k ** 3)
    assert relerr(push_forward(T, F, J, char), k ** (rank - 3 * weight) * T) < 1e-14
    assert relerr(pull_back(T, F, J, char), k ** (3 * weight - rank) * T) < 1e-14


def test_complex_values_pass_through_the_array_level():
    F, J = random_deformation()
    T = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    got = push_forward(T, F, J, STRESS)
    assert got.dtype == np.complex128
    want = push_forward(T.real, F, J, STRESS) + 1j * push_forward(T.imag, F, J, STRESS)
    assert relerr(got, want) < 1e-14


def test_voigt_input_is_refused_by_name():
    F, J = random_deformation()
    with pytest.raises(ValueError, match="Voigt"):
        push_forward(np.zeros((6, 6)), F, J, ELASTIC)
    with pytest.raises(ValueError, match="Voigt"):
        pull_back(np.zeros((4, 6)), F, J, STRESS)
    with pytest.raises(ValueError, match="trailing shape"):
        push_forward(np.zeros((3, 4)), F, J, VECTOR)


# ---------------------------------------------------------- full_components

def test_full_components_expands_voigt_and_leaves_the_rest():
    r, th, ph = grid()
    c = elastic()
    full = full_components(c, r, th, ph, frame="spherical")
    assert full.shape == (4, 3, 2, 3, 3, 3, 3)
    assert relerr(full, voigt_to_tensor(c.evaluate(r, th, ph))) == 0.0
    cart = full_components(c, r, th, ph)
    R = spherical_frame(th, ph)
    assert relerr(cart, rotate_slots(full, R, 4)) < 1e-14
    v = vector()
    assert relerr(full_components(v, r, th, ph, frame="spherical"),
                  v.evaluate(r, th, ph)) == 0.0
    assert full_components(rho(), r, th, ph).shape == (4, 3, 2)


# --------------------------------------------------------- the field level

@pytest.mark.parametrize("which", FIELDS)
@pytest.mark.parametrize("mapping", MAPPINGS)
def test_pushed_fields_satisfy_the_field_contract(which, mapping):
    check_field(PushedForwardField(FIELDS[which](), MAPPINGS[mapping]()))


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_pulled_back_fields_satisfy_the_field_contract(mapping):
    m = MAPPINGS[mapping]()
    scalar = PulledBackField(IV, lambda r, t, p: 2.0 + r * np.sin(t), m,
                             character=DENSITY, name="rho")
    check_field(scalar)
    S = 0.5 * (lambda A: A + A.T)(rng.normal(size=(3, 3)))
    rank2 = PulledBackField(IV, lambda r, t, p: S * (1.0 + r)[..., None, None], m,
                            character=STRESS, name="S")
    check_field(rank2)
    voigt = PulledBackField(
        IV, lambda r, t, p: np.broadcast_to(tensor_to_voigt(S, rank=2),
                                            np.shape(r) + (6,)), m,
        character=STRESS)
    assert voigt.evaluate(*grid()).shape == (4, 3, 2, 6)


@pytest.mark.parametrize("which", FIELDS)
def test_under_the_identity_a_pushed_field_is_its_source(which):
    src = FIELDS[which]()
    pushed = PushedForwardField(src, IdentityMapping())
    r, th, ph = grid()
    for frame in ("spherical", "cartesian"):
        assert relerr(pushed.evaluate(r, th, ph, frame=frame),
                      src.evaluate(r, th, ph, frame=frame)) < 1e-14


@pytest.mark.parametrize("which", ["rho", "vector", "elastic", "stress"])
@pytest.mark.parametrize("mapping", MAPPINGS)
def test_pulling_back_the_pushed_values_reproduces_the_source(which, mapping):
    """The physical values of the pushed field, read back at their
    physical points through the inverse, pulled back again."""
    src = FIELDS[which]()
    m = MAPPINGS[mapping]()
    pushed = PushedForwardField(src, m)

    def physical(r, t, p):
        ref = m.inverse(cartesian_points(r, t, p))
        r0, t0, p0, _ = spherical_coordinates(ref)
        return pushed.evaluate(r0, t0, p0)

    back = PulledBackField(IV, physical, m, character=src.character)
    r = np.linspace(0.55, 0.95, 3)[:, None]
    th, ph = np.array([0.4, 2.7])[None, :], 0.9
    for frame in ("spherical", "cartesian"):
        assert relerr(back.evaluate(r, th, ph, frame=frame),
                      src.evaluate(r, th, ph, frame=frame)) < 1e-8


@pytest.mark.parametrize("which", ["vector", "elastic", "stress"])
def test_the_cartesian_frame_is_the_array_rule_on_the_source(which):
    src = FIELDS[which]()
    m = stretch()
    r, th, ph = grid()
    X = cartesian_points(r, th, ph)
    F, J = m.deformation_gradient(X), m.jacobian(X)
    want = push_forward(full_components(src, r, th, ph), F, J, src.character)
    if src.character.voigt_shape:
        want = tensor_to_voigt(want, rank=src.character.rank)
    got = PushedForwardField(src, m).evaluate(r, th, ph, frame="cartesian")
    assert relerr(got, want) < 1e-13


def test_pushed_density_is_rho_over_J_at_the_reference_point():
    m = stretch()
    r, th, ph = grid()
    J = m.jacobian(cartesian_points(r, th, ph))
    got = PushedForwardField(rho(), m).evaluate(r, th, ph)
    assert relerr(got, rho().evaluate(r, th, ph) / J) < 1e-14


@pytest.mark.parametrize("which", ["vector", "elastic", "stress"])
def test_the_spherical_frame_is_the_one_at_the_physical_point(which):
    """Under a non-radial mapping the two frames differ, and the spherical
    components are the Cartesian ones rotated by the frame at m(X)."""
    src = FIELDS[which]()
    m = Squash(0.9, 0.1)
    pushed = PushedForwardField(src, m)
    r, th, ph = grid()
    cart = pushed.evaluate(r, th, ph, frame="cartesian")
    sph = pushed.evaluate(r, th, ph)
    rank = src.character.rank
    full = voigt_to_tensor(cart, rank=rank) if src.character.voigt_shape else cart

    _, _, _, Rp = spherical_coordinates(m(cartesian_points(r, th, ph)))
    want = rotate_slots(full, np.swapaxes(Rp, -1, -2), rank)
    wrong = rotate_slots(full, np.swapaxes(spherical_frame(th, ph), -1, -2), rank)
    if src.character.voigt_shape:
        want, wrong = (tensor_to_voigt(x, rank=rank) for x in (want, wrong))
    assert relerr(sph, want) < 1e-13
    assert relerr(sph, wrong) > 1e-3


def test_a_uniform_dilation_through_the_field_class():
    c = 0.03
    m = RadialStretch(uniform_dilation(c), rmax=RMAX)
    r, th, ph = grid()
    got = PushedForwardField(elastic(), m).evaluate(r, th, ph)
    assert relerr(got, (1.0 + c) * elastic().evaluate(r, th, ph)) < 1e-13
    got = PushedForwardField(rho(), m).evaluate(r, th, ph)
    assert relerr(got, rho().evaluate(r, th, ph) / (1.0 + c) ** 3) < 1e-14


def test_the_pushed_field_keeps_interval_character_and_names_itself():
    pushed = PushedForwardField(rho(), stretch())
    assert pushed.interval == IV and pushed.character == DENSITY
    assert pushed.name == "rho_phys" and pushed.is_radial is False
    assert pushed.renamed("x").name == "x" and pushed.renamed(None).name is None
    assert PushedForwardField(rho(), stretch(), name="d").name == "d"
    assert PushedForwardField(rho().renamed(None), stretch()).name is None
    assert pushed.source is not None and pushed.mapping is not None
    with pytest.raises(ValueError, match="theta and phi"):
        pushed(0.7)
    with pytest.raises(ValueError, match="outside the interval"):
        pushed.evaluate(1.2, 0.5, 0.5)
    with pytest.raises(ValueError, match="frame"):
        pushed.evaluate(0.7, 0.5, 0.5, frame="geographic")
    wider = pushed.on_interval(0.0, 2.0)
    assert wider.interval == (0.0, 2.0) and wider.name == "rho_phys"
    assert np.isfinite(wider.evaluate(1.5, 0.5, 0.5))


def test_the_pulled_back_field_is_a_field_and_refuses_complex():
    m = stretch()
    f = PulledBackField(IV, lambda r, t, p: 1.0 + 0.0 * r, m, character=SCALAR,
                        name="q")
    assert f.interval == IV and f.name == "q" and f.is_radial is False
    assert f.renamed("p").name == "p" and f.on_interval(0.0, 2.0).interval == (0, 2)
    assert f.physical is not None and f.mapping is m
    with pytest.raises(ValueError, match="theta and phi"):
        f(0.7)
    with pytest.raises(TypeError, match="callable"):
        PulledBackField(IV, 1.0, m, character=SCALAR)
    bad = PulledBackField(IV, lambda r, t, p: (1.0 + 1j) * r, m, character=SCALAR)
    with pytest.raises(TypeError, match="complex"):
        bad.evaluate(0.7, 0.5, 0.5)
    with pytest.raises(ValueError, match="trailing shape"):
        PulledBackField(IV, lambda r, t, p: np.zeros(np.shape(r) + (4,)), m,
                        character=STRESS).evaluate(0.7, 0.5, 0.5)


@pytest.mark.parametrize("cls", ["pushed", "pulled"])
def test_rescaled_round_trips_through_the_scaled_mapping(cls):
    m = stretch()
    if cls == "pushed":
        f = PushedForwardField(elastic(), m)
    else:
        f = PulledBackField(IV, lambda r, t, p: 2.0 + r * np.cos(t) ** 2, m,
                            character=DENSITY)
    r, th, ph = grid()
    k, v = 6.371e6, 0.3
    g = f.rescaled(k=k, v=v)
    assert np.allclose(g.interval, (k * IV[0], k * IV[1]))
    assert relerr(g.evaluate(k * r, th, ph), v * f.evaluate(r, th, ph)) < 1e-12
    back = g.rescaled(k=1.0 / k, v=1.0 / v)
    assert relerr(back.evaluate(r, th, ph), f.evaluate(r, th, ph)) < 1e-12
