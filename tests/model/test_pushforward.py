"""Push-forward and pull-back: the generic rule, and fields carried by it.

The rule is one einsum built from the rank (Appendix B.8.1), so what has
to be checked is that the *construction* is right, not that four hand
written cases are.  So the einsum is checked against an explicit
quadruple loop at rank 4, against the matrix products F v and F S F^T / J
at ranks 1 and 2, and against rho/J at rank 0; the symmetry preservation
and round-trip helpers are imported from test_appendix_b8, which built
them from numpy alone the day the appendix was written, rather than
written again here.

The field layer is then checked against the array layer at the same
points, and against the one case where the answer is known by hand: a
uniform dilation, where F = (1 + c) I and a weight-1 rank-4 field is
scaled by (1 + c)^4 / (1 + c)^3.
"""
from itertools import product

import numpy as np
import pytest

from planetmodel import PREM
from planetmodel.model.character import (DENSITY, ELASTIC, SCALAR, STRESS, VECTOR,
                                    Character)
from planetmodel.model.displacement import CallableDisplacement, ZeroDisplacement
from planetmodel.model.fields.composite import ComposedField
from planetmodel.model.frames import spherical_frame
from planetmodel.model.mapping import IdentityMapping, RadialStretch
from planetmodel.model.materials import bond_matrix, voigt_to_tensor
from planetmodel.model.pushforward import (PushedForwardField,
                                      check_tensor_symmetries, pull_back,
                                      push_forward, push_forward_field)
from planetmodel.testing import check_field

from .test_appendix_b8 import push4, random_deformation, random_second_tensor
from .test_mapping import A, points, smooth_h

rng = np.random.default_rng(19)


def grid(n_r=4):
    """A (r, theta, phi) lattice inside PREM, as three broadcast axes."""
    r = np.linspace(0.15 * A, 0.97 * A, n_r)[:, None, None]
    th = np.array([0.4, 1.3, 2.7])[None, :, None]
    ph = np.array([-1.1, 0.9])[None, None, :]
    return r, th, ph


def cartesian(r, th, ph):
    """The Cartesian points of a broadcast (r, theta, phi) lattice."""
    r, th, ph = np.broadcast_arrays(r, th, ph)
    return np.stack([r * np.sin(th) * np.cos(ph),
                     r * np.sin(th) * np.sin(ph), r * np.cos(th)], axis=-1)


def uniform_dilation(c):
    """h = c r: every point moves out by the same fraction of its radius.

    The one mapping whose push-forward can be written down by eye --
    F = (1 + c) I and J = (1 + c)^3 -- and therefore the one place a
    factor of J or a missing slot has nowhere to hide.
    """
    def zero(r, t, p):
        return np.zeros(np.broadcast(np.asarray(r, dtype=float),
                                     np.asarray(t, dtype=float),
                                     np.asarray(p, dtype=float)).shape)

    return CallableDisplacement(
        lambda r, t, p: c * np.asarray(r, dtype=float) + zero(r, t, p),
        radial_derivative=lambda r, t, p: c + zero(r, t, p),
        angular_gradient=lambda r, t, p: (zero(r, t, p), zero(r, t, p)))


def relerr(got, want):
    """Max absolute difference, scaled by the magnitude of the answer."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    scale = float(np.max(np.abs(want))) or 1.0
    return float(np.max(np.abs(got - want))) / scale


# ------------------------------------------------------- the scalar cases

def test_density_pushes_forward_as_rho_over_J():
    """Where AAC16 (71) and MMA26 agree, checked on a real mapping."""
    m = RadialStretch(smooth_h(), rmax=A)
    X = points(50)
    J = m.jacobian(X)
    rho = np.full(X.shape[:-1], 3300.0)
    assert np.allclose(push_forward(rho, m.deformation_gradient(X), J, DENSITY),
                       rho / J, rtol=1e-14)
    assert np.allclose(push_forward(rho, m.deformation_gradient(X), J, SCALAR),
                       rho)


def test_identity_pushes_density_through_unchanged():
    X = points(20)
    m = IdentityMapping()
    rho = np.full(X.shape[:-1], 3300.0)
    assert np.allclose(m.push_forward(rho, X, DENSITY), rho)


# ------------------------------------------------- ranks 1, 2 and 4 by hand

def test_a_vector_is_wrapped_once():
    F, J = random_deformation()
    v = rng.normal(size=3)
    assert relerr(push_forward(v, F, J, VECTOR), F @ v) < 1e-15


def test_a_stress_pushes_to_F_S_FT_over_J():
    """Rank 2 weight 1: the second Piola-Kirchhoff to the Cauchy stress."""
    F, J = random_deformation()
    S = rng.normal(size=(3, 3))
    S = 0.5 * (S + S.T)
    assert relerr(push_forward(S, F, J, STRESS), F @ S @ F.T / J) < 1e-14


def test_rank_four_preserves_the_full_symmetries():
    """Identical wrapping on every slot, so Voigt stays faithful."""
    F, J = random_deformation()
    T = random_second_tensor()
    c = push_forward(T, F, J, ELASTIC)
    check_tensor_symmetries(c)
    assert relerr(c, push4(T, F, J)) < 1e-15


def test_the_einsum_agrees_with_an_explicit_quadruple_loop():
    """The oracle for the subscript construction itself.

    Written index by index from B.8.1, with no einsum anywhere: if the
    generated subscript string wrapped a slot with the wrong factor of
    F, or transposed one, this is what would notice.
    """
    F, J = random_deformation()
    T = rng.normal(size=(3, 3, 3, 3))
    want = np.zeros((3, 3, 3, 3))
    for i, j, k, l in product(range(3), repeat=4):
        total = 0.0
        for a, b, c, d in product(range(3), repeat=4):
            total += F[i, a] * F[j, b] * F[k, c] * F[l, d] * T[a, b, c, d]
        want[i, j, k, l] = total / J
    assert relerr(push_forward(T, F, J, ELASTIC), want) < 1e-13


# ------------------------------------------------------------- round trips

@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("weight", [0, 1])
def test_push_of_pull_is_the_identity(rank, weight):
    """Both a near-identity F and one with det about 3."""
    char = Character(rank, weight)
    T = rng.normal(size=(3,) * rank)
    near, _ = random_deformation(0.05)
    big, _ = random_deformation(0.4)
    big = big * (3.0 / abs(np.linalg.det(big))) ** (1.0 / 3.0)
    for F in (near, big):
        J = np.linalg.det(F)
        back = push_forward(pull_back(T, F, J, char), F, J, char)
        assert relerr(back, T) < 1e-12, f"det F = {J}"


def test_the_stretched_case_really_is_stretched():
    """The det-3 mapping of the round trip is not a near-identity."""
    big, _ = random_deformation(0.4)
    big = big * (3.0 / abs(np.linalg.det(big))) ** (1.0 / 3.0)
    assert abs(np.linalg.det(big) - 3.0) < 1e-12


# ------------------------------------------------------------ what is refused

def test_voigt_input_is_refused_by_name():
    """The tensor-level rule takes tensors; Voigt has the wrong shape."""
    F, J = random_deformation()
    with pytest.raises(ValueError, match="Voigt"):
        push_forward(np.zeros((6, 6)), F, J, ELASTIC)
    with pytest.raises(ValueError, match="Voigt"):
        pull_back(np.zeros((4, 6)), F, J, STRESS)


def test_a_wrong_component_shape_is_refused():
    F, J = random_deformation()
    with pytest.raises(ValueError, match="trailing shape"):
        push_forward(np.zeros((3, 4)), F, J, VECTOR)


def test_an_absurd_rank_is_refused_rather_than_built():
    F, J = random_deformation()
    with pytest.raises(ValueError, match="rank 7"):
        push_forward(np.zeros((3,) * 7), F, J, Character(7, 1))


def test_rank_six_is_still_allowed():
    """The ceiling is a guard, not the library's rank limit."""
    F, J = random_deformation()
    char = Character(6, 1)
    T = rng.normal(size=(3,) * 6)
    assert relerr(push_forward(pull_back(T, F, J, char), F, J, char), T) < 1e-11


# --------------------------------------------------------- the field layer

def test_pushed_density_is_rho_over_J_at_the_reference_point():
    m = RadialStretch(smooth_h(), rmax=A)
    prem = PREM()
    r, th, ph = grid()
    J = m.jacobian(cartesian(r, th, ph))
    got = push_forward_field(prem.rho, m).evaluate(r, th, ph)
    assert relerr(got, np.asarray(prem.rho.evaluate(r, th, ph)) / J) < 1e-14


def test_pushed_elastic_is_the_B81_einsum_on_the_cartesian_source():
    """The field layer against the array rule, at the same points."""
    m = RadialStretch(smooth_h(), rmax=A)
    prem = PREM()
    r, th, ph = grid()
    X = cartesian(r, th, ph)
    F, J = m.deformation_gradient(X), m.jacobian(X)
    src = prem.elastic_moduli.evaluate(r, th, ph, frame="cartesian", voigt=False)
    want = np.einsum("...iA,...jB,...kC,...lD,...ABCD->...ijkl",
                     F, F, F, F, src) / J[..., None, None, None, None]
    got = push_forward_field(prem.elastic_moduli, m).evaluate(
        r, th, ph, frame="cartesian", voigt=False)
    assert relerr(got, want) < 1e-12
    check_tensor_symmetries(got)


def test_the_spherical_frame_is_the_bond_rotation_back():
    """V_sph = bond(R^T) V_cart bond(R^T)^T.

    Note that the back-rotation is the Bond matrix of R^T and *not* the
    transpose of the Bond matrix of R: this convention's M carries
    factors of two on the shear columns and is not orthogonal, so
    M^T V M is not V_sph.  Pinned here because it is an easy thing to
    write down and a hard thing to notice.
    """
    m = RadialStretch(smooth_h(), rmax=A)
    prem = PREM()
    r, th, ph = grid()
    pushed = push_forward_field(prem.elastic_moduli, m)
    v_cart = pushed.evaluate(r, th, ph, frame="cartesian")
    v_sph = pushed.evaluate(r, th, ph)

    R = np.broadcast_to(spherical_frame(th, ph), v_sph.shape[:-2] + (3, 3))
    Mi = bond_matrix(np.swapaxes(R, -1, -2))
    assert relerr(Mi @ v_cart @ np.swapaxes(Mi, -1, -2), v_sph) < 1e-12
    assert relerr(bond_matrix(R) @ v_sph @ np.swapaxes(bond_matrix(R), -1, -2),
                  v_cart) < 1e-12


def test_the_voigt_reduction_loses_nothing():
    """Reduce, expand, and the full tensor comes back."""
    m = RadialStretch(smooth_h(), rmax=A)
    pushed = push_forward_field(PREM().elastic_moduli, m)
    r, th, ph = grid()
    full = pushed.evaluate(r, th, ph, voigt=False)
    assert relerr(voigt_to_tensor(pushed.evaluate(r, th, ph)), full) < 1e-14


def test_a_uniform_dilation_scales_a_rank_four_field_by_one_plus_c():
    """The case the answer is known by hand: (1 + c)^4 / (1 + c)^3.

    A pure radial stretch with h = c r has F = (1 + c) I in every frame,
    so the pushed-forward tensor is the source's own spherical Voigt
    matrix times (1 + c), with no rotation and no mixing of components.
    """
    c = 0.03
    m = RadialStretch(uniform_dilation(c))
    prem = PREM()
    r, th, ph = grid()
    X = cartesian(r, th, ph)
    assert relerr(m.deformation_gradient(X), (1.0 + c) * np.eye(3)) < 1e-15
    assert relerr(m.jacobian(X), (1.0 + c) ** 3) < 1e-15

    got = push_forward_field(prem.elastic_moduli, m).evaluate(r, th, ph)
    want = (1.0 + c) * np.asarray(prem.elastic_moduli.evaluate(r, th, ph))
    assert relerr(got, want) < 1e-13


def test_a_uniform_dilation_divides_density_by_one_plus_c_cubed():
    c = -0.02
    m = RadialStretch(uniform_dilation(c))
    prem = PREM()
    r, th, ph = grid()
    got = push_forward_field(prem.rho, m).evaluate(r, th, ph)
    want = np.asarray(prem.rho.evaluate(r, th, ph)) / (1.0 + c) ** 3
    assert relerr(got, want) < 1e-14


# ------------------------------------------------------------- the contract

def velocity_view(prem):
    """A ComposedField view, vph = sqrt(A / rho): weight 0, rank 0."""
    return ComposedField(lambda a, rho: np.sqrt(a / rho),
                         (prem.A, prem.rho), name="vph_view",
                         character=SCALAR)


@pytest.mark.parametrize("which", ["rho", "elastic_moduli", "velocity"])
def test_pushed_fields_satisfy_the_field_contract(which):
    prem = PREM()
    src = velocity_view(prem) if which == "velocity" else prem[which]
    check_field(push_forward_field(src, RadialStretch(smooth_h(), rmax=A)))


def test_the_identity_mapping_returns_the_source_itself():
    prem = PREM()
    assert push_forward_field(prem.rho, IdentityMapping()) is prem.rho
    pushed = push_forward_field(prem.elastic_moduli, IdentityMapping())
    assert pushed is prem.elastic_moduli


def test_a_zero_stretch_is_the_identity_too():
    """F = I and J = 1 exactly, so there is nothing to carry."""
    prem = PREM()
    assert push_forward_field(prem.rho, RadialStretch(ZeroDisplacement())) is prem.rho


def test_the_pushed_field_keeps_skeleton_character_and_dimensions():
    prem = PREM()
    pushed = push_forward_field(prem.rho, RadialStretch(smooth_h(), rmax=A))
    assert isinstance(pushed, PushedForwardField)
    assert pushed.skeleton == prem.skeleton
    assert pushed.character == prem.rho.character
    assert pushed.dimensions is prem.rho.dimensions
    assert pushed.name == "rho_phys"
    assert pushed.is_radial is False


def test_the_angles_are_required():
    """F depends on direction, so a radial source does not excuse them."""
    pushed = push_forward_field(PREM().rho, RadialStretch(smooth_h(), rmax=A))
    with pytest.raises(ValueError, match="theta"):
        pushed.evaluate(3.0e6)
    with pytest.raises(ValueError, match="theta"):
        pushed.evaluate(3.0e6, 0.7)


def test_an_unknown_frame_is_refused():
    pushed = push_forward_field(PREM().rho, RadialStretch(smooth_h(), rmax=A))
    with pytest.raises(ValueError, match="frame"):
        pushed.evaluate(3.0e6, 0.7, 0.4, frame="geographic")


# ------------------------------------- a rank-2 field that is not ElasticField

class ConstantStress:
    """A constant equilibrium stress, Voigt-reduced in the spherical frame.

    Nothing shipped is rank 2 yet -- the stress fields arrive with
    FirstElasticField -- so this stands in for one, and it is the path
    where push_forward_field has to expand the Voigt vector itself
    rather than ask for `voigt=False`.  It is deliberately minimal: a
    Field is whatever has the four attributes and evaluates.
    """

    character = STRESS
    dimensions = None

    def __init__(self, skeleton, v, name="S"):
        self.skeleton = skeleton
        self.name = name
        self._v = np.asarray(v, dtype=float)

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side="upper", frame="spherical"):
        b = np.asarray(self.skeleton.boundaries, dtype=float)
        r = np.asarray(r, dtype=float)
        if np.any(r < b[0] - 1e-9) or np.any(r > b[-1] + 1e-9):
            raise ValueError(f"radius outside [{b[0]}, {b[-1]}]")
        shape = np.broadcast(r, 0.0 if theta is None else np.asarray(theta),
                             0.0 if phi is None else np.asarray(phi)).shape
        out = np.broadcast_to(self._v, shape + (6,))
        if frame == "cartesian":
            if theta is None or phi is None:
                raise ValueError("frame='cartesian' needs theta and phi")
            M = np.broadcast_to(bond_matrix(spherical_frame(theta, phi)),
                                shape + (6, 6))
            out = np.einsum("...ab,...b->...a", M, out)
        return np.array(out)


def test_a_generic_rank_two_field_is_expanded_from_voigt():
    """The branch for a source with no voigt= keyword of its own."""
    prem = PREM()
    m = RadialStretch(smooth_h(), rmax=A)
    src = ConstantStress(prem.skeleton, [3.0e9, 2.0e9, 1.0e9, 5.0e8, -4.0e8,
                                         7.0e8])
    r, th, ph = grid()
    X = cartesian(r, th, ph)
    F, J = m.deformation_gradient(X), m.jacobian(X)
    S = voigt_to_tensor(src.evaluate(r, th, ph, frame="cartesian"), rank=2)
    want = np.einsum("...iA,...jB,...AB->...ij", F, F, S) / J[..., None, None]

    pushed = push_forward_field(src, m)
    got = pushed.evaluate(r, th, ph, frame="cartesian", voigt=False)
    assert relerr(got, want) < 1e-13
    assert relerr(got, np.swapaxes(got, -1, -2)) < 1e-14
    assert pushed.evaluate(r, th, ph).shape == (4, 3, 2, 6)
    check_field(pushed)


# ---------------------------------------------------------- complex values

def test_a_complex_field_pushes_forward_as_its_two_real_parts_do():
    """The rule is linear, so the dtype follows the input and nothing is
    dropped: the imaginary part of a frozen viscoelastic modulus rides
    across the mapping with its real part."""
    from planetmodel.model.fields.frequency import at_frequency
    from planetmodel.model.rheology import constant_q

    prem = PREM()
    law = constant_q(prem["elastic_moduli"], prem["qkappa"], prem["qmu"],
                     reference_period=1.0)
    omega = 2.0 * np.pi / 20.0
    both = at_frequency(law, omega, part="complex")
    real = at_frequency(law, omega, part="real")
    imag = at_frequency(law, omega, part="imag")
    m = RadialStretch(smooth_h(), rmax=A)
    r, th, ph = np.array([5.0e6, 6.0e6]), np.array([0.7, 1.9]), np.array([0.3, -2.0])
    got = push_forward_field(both, m).evaluate(r, th, ph)
    assert got.dtype == np.complex128
    want = (push_forward_field(real, m).evaluate(r, th, ph)
            + 1j * push_forward_field(imag, m).evaluate(r, th, ph))
    assert relerr(got, want) < 1e-13
    assert np.max(np.abs(np.imag(got))) > 0.0
