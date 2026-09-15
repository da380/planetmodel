"""The first elasticity tensor: the action against the materialised tensor.

Nothing here checks a formula against itself.  `apply` -- the route a
weak form actually takes, six multiply-adds per point through the Voigt
matrix -- is measured against the full `(3, 3, 3, 3)` tensor contracted
by a plain einsum, and that tensor is in turn measured against
`test_appendix_b8.first_tensor`, the numpy-only transcription of B.8.4
written the day the appendix was.  The equilibrium form is checked both
ways round: assembled from the pushed-forward second tensor and the
Cauchy stress, and obtained by pushing the first tensor's two reference
slots forward.

The two symmetries are the point of the whole class, so they are
asserted in both directions: the major symmetry holds always, and the
minor symmetry is required to *fail* by a measurable amount as soon as
the body is stressed or the mapping is not the identity.  A test that
only checked what holds would pass on a second elasticity tensor
returned by mistake.

PREM is VTI between the Moho and 220 km depth, so the radii are chosen
there as well as in the isotropic mantle: a bug in the frame handling
has nowhere to hide in a tensor whose components depend on direction.
"""
import warnings

import numpy as np
import pytest

from planetmodel import PREM
from planetmodel.io.deck import read_isotropic_deck
from planetmodel.model.character import (ELASTIC, FIRST_ELASTIC, STRESS,
                                    Character, Symmetry)
from planetmodel.model.fields.composite import FieldBase, SumField
from planetmodel.model.firstelastic import FirstElasticField
from planetmodel.model.frames import spherical_frame
from planetmodel.model.mapping import IdentityMapping, RadialStretch
from planetmodel.model.materials import tensor_to_voigt, voigt_to_tensor
from planetmodel.model.pullback import pulled_back_elastic
from planetmodel.model.frames import rotate_slots
from planetmodel.model.pushforward import push_forward
from planetmodel.model.units import Dimensions
from planetmodel.testing import check_field

from .test_appendix_b8 import first_tensor, push4, random_second_tensor
from .test_mapping import A, smooth_h
from .test_pushforward import relerr

rng = np.random.default_rng(31)

#: PREM's anisotropic shell: Moho (6346.6 km) down to 220 km depth
#: (6151 km).  Two radii there, one in the isotropic lower mantle, one
#: in the outer core, so both symmetry classes are exercised.
RADII = np.array([3.0e6, 5.0e6, 6.18e6, 6.30e6])


def grid():
    """A (r, theta, phi) lattice as three broadcast axes."""
    return (RADII[:, None, None],
            np.array([0.4, 1.3, 2.7])[None, :, None],
            np.array([-1.1, 0.9])[None, None, :])


def cartesian(r, th, ph):
    """The Cartesian points of a broadcast (r, theta, phi) lattice."""
    r, th, ph = np.broadcast_arrays(r, th, ph)
    return np.stack([r * np.sin(th) * np.cos(ph),
                     r * np.sin(th) * np.sin(ph), r * np.cos(th)], axis=-1)


def gradient(shape):
    """A random referential displacement gradient G_jB = du_j/dX_B."""
    return rng.normal(size=shape + (3, 3))


# ------------------------------------------------------- a synthetic stress


class SphericalStress(FieldBase):
    """A smooth symmetric stress, Voigt (..., 6) in the spherical frame.

    A stand-in for the equilibrium stress a self-gravitating body
    carries, written out by hand so the test depends on numpy alone.
    It is deliberately not hydrostatic and not
    radial: every one of the six components varies with all three
    coordinates, so a term dropped or a slot transposed shows up.

    `frame` is honoured the way the model layer honours it everywhere:
    the components are native to the spherical frame, and the Cartesian
    ones are obtained by expanding, conjugating with
    R = [e_r, e_theta, e_phi] and reducing again.
    """

    character = STRESS

    def __init__(self, skeleton, scale=2.0e10, name="S"):
        self.skeleton = skeleton
        self.scale = float(scale)
        self.name = name

    @property
    def dimensions(self):
        return Dimensions.MODULUS

    @property
    def is_radial(self) -> bool:
        return False

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side="upper", frame="spherical"):
        if frame not in ("spherical", "cartesian"):
            raise ValueError(f"unknown frame {frame!r}: 'spherical' or "
                             "'cartesian'")
        if theta is None or phi is None:
            raise ValueError("SphericalStress needs theta and phi: its "
                             "components depend on angle")
        r, theta, phi = np.broadcast_arrays(
            np.asarray(r, dtype=float), np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float))
        u = r / A
        v = self.scale * np.stack([
            -1.0 - 0.4 * u,
            -0.9 - 0.3 * u * np.cos(theta),
            -0.8 - 0.2 * np.sin(theta) * np.cos(phi),
            0.15 * np.sin(2.0 * theta) * np.sin(phi),
            0.10 * np.cos(theta) * np.cos(phi) * u,
            0.05 * np.sin(theta) * np.sin(2.0 * phi),
        ], axis=-1)
        if frame == "cartesian":
            R = spherical_frame(theta, phi)
            v = tensor_to_voigt(rotate_slots(voigt_to_tensor(v, rank=2), R, 2), rank=2)
        return v


@pytest.fixture(scope="module")
def prem():
    return PREM()


@pytest.fixture(scope="module")
def iso():
    """PREM without a crust, read as an isotropic deck.

    A different path through `ElasticField.evaluate`: a physically
    isotropic Voigt matrix is the same in every frame, so the Cartesian
    components come back unrotated, and the first tensor must be built
    from them just the same.
    """
    with warnings.catch_warnings():
        # the file's header says 220 knots and carries 214.
        warnings.simplefilter("ignore", UserWarning)
        return read_isotropic_deck("tests/data/prem.nocrust")


@pytest.fixture(scope="module")
def stretch():
    return RadialStretch(smooth_h(), rmax=A)


@pytest.fixture(scope="module")
def stress(prem):
    return SphericalStress(prem.skeleton)


def pieces(prem, mapping, stress, r, th, ph):
    """CC, S and F in Cartesian components, read from the objects."""
    X = cartesian(r, th, ph)
    CC = np.asarray(prem.elastic_moduli.evaluate(r, th, ph, frame="cartesian",
                                          voigt=False), dtype=float)
    R = spherical_frame(*np.broadcast_arrays(
        np.broadcast_to(th, np.broadcast(r, th, ph).shape),
        np.broadcast_to(ph, np.broadcast(r, th, ph).shape)))
    S = (np.zeros(CC.shape[:-4] + (3, 3)) if stress is None else
         rotate_slots(voigt_to_tensor(np.asarray(stress.evaluate(r, th, ph),
                                          dtype=float), rank=2), R, 2))
    F = np.asarray(mapping.deformation_gradient(X), dtype=float)
    return CC, S, F, np.asarray(mapping.jacobian(X), dtype=float)


# ------------------------------------------------------------- the character


def test_the_first_tensor_has_its_own_character():
    assert FIRST_ELASTIC != ELASTIC
    assert (FIRST_ELASTIC.rank, FIRST_ELASTIC.weight) == (4, 1)
    assert FIRST_ELASTIC.voigt_shape is None
    assert ELASTIC.voigt_shape == (6, 6)
    assert FIRST_ELASTIC.component_shape == (3, 3, 3, 3)
    assert Character(4, 1) == ELASTIC          # positional (rank, weight)
    assert "no Voigt form" in str(FIRST_ELASTIC)


def test_a_first_and_a_second_elastic_field_cannot_be_added(prem, stretch):
    first = FirstElasticField(prem.elastic_moduli, stretch)
    with pytest.raises(ValueError, match="character"):
        SumField((first, prem.elastic_moduli))
    with pytest.raises(ValueError, match="character"):
        first + prem.elastic_moduli


def test_the_field_reports_what_it_is(prem, stretch, stress):
    f = FirstElasticField(prem.elastic_moduli, stretch, stress=stress)
    assert f.character is FIRST_ELASTIC
    assert f.dimensions is Dimensions.MODULUS
    assert f.skeleton == prem.elastic_moduli.skeleton
    assert f.is_radial is False
    assert "FirstElasticField" in repr(f)


def test_the_source_and_the_stress_are_type_checked(prem, stretch, stress):
    with pytest.raises(ValueError, match="second"):
        FirstElasticField(prem.rho, stretch)
    with pytest.raises(ValueError, match="STRESS"):
        FirstElasticField(prem.elastic_moduli, stretch, stress=prem.rho)


def test_the_angles_are_required(prem, stretch):
    f = FirstElasticField(prem.elastic_moduli, stretch)
    with pytest.raises(ValueError, match="theta and phi"):
        f.evaluate(5.0e6)
    with pytest.raises(ValueError, match="theta and phi"):
        f.apply(np.eye(3), 5.0e6)


def test_an_unknown_frame_is_refused_by_name(prem, stretch):
    f = FirstElasticField(prem.elastic_moduli, stretch)
    r, th, ph = grid()
    for call in (lambda: f.evaluate(r, th, ph, frame="polar"),
                 lambda: f.apply(gradient((1,)), r, th, ph, frame="polar"),
                 lambda: f.equilibrium_form(r, th, ph, frame="polar")):
        with pytest.raises(ValueError, match="frame"):
            call()


# -------------------------------------------- (i) apply against the tensor


@pytest.mark.parametrize("with_stress", [False, True])
@pytest.mark.parametrize("frame", ["cartesian", "spherical"])
def test_apply_is_the_materialised_contraction(prem, stretch, stress,
                                               with_stress, frame):
    """(A G)_iA = A_iAjB G_jB, in whichever frame it is asked for."""
    f = FirstElasticField(prem.elastic_moduli, stretch,
                          stress=stress if with_stress else None)
    r, th, ph = grid()
    shape = np.broadcast(r, th, ph).shape
    G = gradient(shape)
    Amat = f.evaluate(r, th, ph, frame=frame)
    want = np.einsum("...iAjB,...jB->...iA", Amat, G)
    got = f.apply(G, r, th, ph, frame=frame)
    assert relerr(got, want) < 1e-13


def test_apply_broadcasts_a_single_gradient_over_the_points(prem, stretch,
                                                            stress):
    """One G at many points is the same as that G repeated."""
    f = FirstElasticField(prem.elastic_moduli, stretch, stress=stress)
    r, th, ph = grid()
    shape = np.broadcast(r, th, ph).shape
    G = gradient(())
    one = f.apply(G, r, th, ph)
    many = f.apply(np.broadcast_to(G, shape + (3, 3)), r, th, ph)
    assert relerr(one, many) < 1e-15


def test_a_misshapen_gradient_is_refused(prem, stretch):
    f = FirstElasticField(prem.elastic_moduli, stretch)
    r, th, ph = grid()
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        f.apply(np.zeros((4, 2)), r, th, ph)


# ------------------------------ (ii) the identity mapping and no stress


def test_at_the_identity_the_action_is_the_classical_one(prem):
    """F = I, S = 0: apply is CC : sym(G), the textbook contraction."""
    f = FirstElasticField(prem.elastic_moduli, IdentityMapping())
    r, th, ph = grid()
    shape = np.broadcast(r, th, ph).shape
    G = gradient(shape)
    CC = np.asarray(prem.elastic_moduli.evaluate(r, th, ph, frame="cartesian",
                                          voigt=False), dtype=float)
    want = np.einsum("...iAkl,...kl->...iA", CC,
                     0.5 * (G + np.swapaxes(G, -1, -2)))
    assert relerr(f.apply(G, r, th, ph, frame="cartesian"), want) < 1e-13


def test_at_the_identity_the_tensor_is_the_second_one(prem):
    """A_iAjB = CC_CADB with C = i and D = j, which is CC_iAjB."""
    f = FirstElasticField(prem.elastic_moduli, IdentityMapping())
    r, th, ph = grid()
    CC = np.asarray(prem.elastic_moduli.evaluate(r, th, ph, frame="cartesian",
                                          voigt=False), dtype=float)
    assert relerr(f.evaluate(r, th, ph, frame="cartesian"), CC) < 1e-14

    # and in the spherical frame it is the source's own Voigt matrix,
    # since an unstressed identity map leaves a second tensor alone.
    got = f.evaluate(r, th, ph)
    assert relerr(tensor_to_voigt(got, rank=4),
                  np.asarray(prem.elastic_moduli.evaluate(r, th, ph))) < 1e-13


# ------------------------------------------------ (iii) the two symmetries


@pytest.mark.parametrize("mapping,with_stress", [
    ("identity", False), ("identity", True),
    ("stretch", False), ("stretch", True),
])
def test_the_major_symmetry_holds_always(prem, stretch, stress, mapping,
                                         with_stress):
    """A_iAjB = A_jBiA, from the major symmetry of CC and of S."""
    m = IdentityMapping() if mapping == "identity" else stretch
    f = FirstElasticField(prem.elastic_moduli, m,
                          stress=stress if with_stress else None)
    r, th, ph = grid()
    for frame in ("cartesian", "spherical"):
        Amat = f.evaluate(r, th, ph, frame=frame)
        swapped = np.einsum("...iAjB->...jBiA", Amat)
        assert relerr(Amat, swapped) < 1e-12


@pytest.mark.parametrize("mapping,with_stress", [
    ("identity", True), ("stretch", False), ("stretch", True),
])
def test_the_minor_symmetry_fails_once_S_or_F_is_nontrivial(
        prem, stretch, stress, mapping, with_stress):
    """The two-tensor distinction: A_iAjB is not A_AijB.

    This is the assertion that a second elasticity tensor returned by
    mistake would fail, so it is written as a lower bound rather than a
    tolerance.
    """
    m = IdentityMapping() if mapping == "identity" else stretch
    f = FirstElasticField(prem.elastic_moduli, m,
                          stress=stress if with_stress else None)
    r, th, ph = grid()
    Amat = f.evaluate(r, th, ph, frame="cartesian")
    swapped = np.einsum("...iAjB->...AijB", Amat)
    assert relerr(Amat, swapped) > 1e-3


def test_the_minor_symmetry_holds_at_the_identity_without_stress(prem):
    """The one case where A is a second elasticity tensor after all."""
    f = FirstElasticField(prem.elastic_moduli, IdentityMapping())
    r, th, ph = grid()
    Amat = f.evaluate(r, th, ph, frame="cartesian")
    assert relerr(Amat, np.einsum("...iAjB->...AijB", Amat)) < 1e-14


# --------------------------------------------- (iv) the equilibrium form


@pytest.mark.parametrize("with_stress", [False, True])
def test_equilibrium_form_is_c_plus_delta_sigma(prem, stretch, stress,
                                                with_stress):
    """LAM = push_forward(CC) + d_ik sigma_jl, assembled independently."""
    src = stress if with_stress else None
    f = FirstElasticField(prem.elastic_moduli, stretch, stress=src)
    r, th, ph = grid()
    CC, S, F, J = pieces(prem, stretch, src, r, th, ph)
    sigma = np.einsum("...iA,...AB,...jB->...ij", F, S, F) / J[..., None, None]
    want = (push_forward(CC, F, J, ELASTIC)
            + np.einsum("ik,...jl->...ijkl", np.eye(3), sigma))
    assert relerr(f.equilibrium_form(r, th, ph, frame="cartesian"), want) < 1e-12


@pytest.mark.parametrize("with_stress", [False, True])
def test_equilibrium_form_is_A_with_both_reference_slots_pushed(
        prem, stretch, stress, with_stress):
    """LAM_ijkl = F_jA F_lB A_iAkB / J -- the other way to the same tensor."""
    src = stress if with_stress else None
    f = FirstElasticField(prem.elastic_moduli, stretch, stress=src)
    r, th, ph = grid()
    _, _, F, J = pieces(prem, stretch, src, r, th, ph)
    Amat = f.evaluate(r, th, ph, frame="cartesian")
    want = (np.einsum("...jA,...lB,...iAkB->...ijkl", F, F, Amat)
            / J[..., None, None, None, None])
    assert relerr(f.equilibrium_form(r, th, ph, frame="cartesian"), want) < 1e-12


def test_the_equilibrium_minor_violation_is_exactly_the_stress_pattern(
        prem, stretch, stress):
    """LAM_ijkl - LAM_jikl = d_ik sigma_jl - d_jk sigma_il, and nothing else."""
    f = FirstElasticField(prem.elastic_moduli, stretch, stress=stress)
    r, th, ph = grid()
    _, S, F, J = pieces(prem, stretch, stress, r, th, ph)
    sigma = np.einsum("...iA,...AB,...jB->...ij", F, S, F) / J[..., None, None]
    LAM = f.equilibrium_form(r, th, ph, frame="cartesian")
    got = LAM - np.einsum("...ijkl->...jikl", LAM)
    want = (np.einsum("ik,...jl->...ijkl", np.eye(3), sigma)
            - np.einsum("jk,...il->...ijkl", np.eye(3), sigma))
    scale = float(np.max(np.abs(LAM)))
    assert float(np.max(np.abs(got - want))) / scale < 1e-12
    assert float(np.max(np.abs(want))) / scale > 1e-4


def test_the_equilibrium_form_is_minor_symmetric_without_stress(prem, stretch):
    """c alone: pushing all four slots identically keeps the symmetries."""
    f = FirstElasticField(prem.elastic_moduli, stretch)
    r, th, ph = grid()
    LAM = f.equilibrium_form(r, th, ph, frame="cartesian")
    assert relerr(LAM, np.einsum("...ijkl->...jikl", LAM)) < 1e-13
    assert relerr(LAM, np.einsum("...ijkl->...klij", LAM)) < 1e-13


# ------------------------------------------------- the appendix's own oracle


def test_the_field_agrees_with_the_appendix_construction(prem, stretch,
                                                         stress):
    """`first_tensor` from test_appendix_b8, point by point.

    CC, S and F are read from the very objects the field reads them
    from, so what is compared is the assembly and the slot order, which
    is the thing B.8.4 is easy to get wrong.
    """
    f = FirstElasticField(prem.elastic_moduli, stretch, stress=stress)
    for r in RADII:
        for th, ph in ((0.4, -1.1), (1.3, 0.9), (2.7, 2.0)):
            CC, S, F, _ = pieces(prem, stretch, stress, r, th, ph)
            want = first_tensor(CC, S, F)
            got = f.evaluate(r, th, ph, frame="cartesian")
            assert relerr(got, want) < 1e-13


def test_the_appendix_action_holds_for_a_random_second_tensor(prem, stretch,
                                                              stress):
    """The action route, on tensors the appendix built rather than PREM's."""
    r, th, ph = 6.2e6, 0.7, 0.3
    _, S, F, _ = pieces(prem, stretch, stress, r, th, ph)
    CC = random_second_tensor()
    G = gradient(())
    direct = np.einsum("iAjB,jB->iA", first_tensor(CC, S, F), G)
    K = np.einsum("CADB,DB->CA", CC, F.T @ G)
    assert relerr(G @ S + F @ K, direct) < 1e-13


def test_push4_of_the_second_tensor_is_the_elastic_part_of_LAM(prem, stretch):
    """The unstressed equilibrium form is B.8.1 applied to CC, point by point."""
    f = FirstElasticField(prem.elastic_moduli, stretch)
    for r in (5.0e6, 6.30e6):
        CC, _, F, J = pieces(prem, stretch, None, r, 1.1, -0.6)
        got = f.equilibrium_form(r, 1.1, -0.6, frame="cartesian")
        assert relerr(got, push4(CC, F, float(J))) < 1e-12


# ------------------------------------------- an isotropic source, and a pulled-back one


def test_the_isotropic_action_is_lambda_trE_plus_two_mu_E(iso):
    """A closed form owing nothing to Voigt: sigma = lam tr(E) I + 2 mu E.

    The classical isotropic constitutive law, written out rather than
    contracted, against `apply` at the identity mapping with no stress.
    It is the one case where the answer can be read off the moduli.
    """
    f = FirstElasticField(iso.elastic_moduli, IdentityMapping())
    assert iso.elastic_moduli.symmetry is Symmetry.ISOTROPIC
    r = np.array([2.0e6, 4.0e6, 6.0e6])[:, None]
    th, ph = np.array([0.5, 1.9])[None, :], 0.3
    shape = np.broadcast(r, th, np.asarray(ph)).shape
    G = gradient(shape)
    E = 0.5 * (G + np.swapaxes(G, -1, -2))
    kappa = np.asarray(iso.elastic_moduli.components["kappa"].evaluate(r), dtype=float)
    mu = np.asarray(iso.elastic_moduli.components["mu"].evaluate(r), dtype=float)
    lam = (kappa - 2.0 * mu / 3.0)[..., None, None]
    trE = np.einsum("...ii->...", E)[..., None, None]
    want = lam * trE * np.eye(3) + 2.0 * mu[..., None, None] * E
    assert relerr(f.apply(G, r, th, ph, frame="cartesian"), want) < 1e-13


def test_an_isotropic_source_agrees_with_the_appendix(iso, stretch):
    """The same B.8.4 assembly, with the isotropic branch of the source."""
    f = FirstElasticField(iso.elastic_moduli, stretch,
                          stress=SphericalStress(iso.elastic_moduli.skeleton))
    for r in (2.0e6, 5.5e6, 6.2e6):
        CC = np.asarray(iso.elastic_moduli.evaluate(r, 0.9, -0.3, frame="cartesian",
                                             voigt=False), dtype=float)
        R = spherical_frame(0.9, -0.3)
        S = rotate_slots(voigt_to_tensor(np.asarray(
            SphericalStress(iso.elastic_moduli.skeleton).evaluate(r, 0.9, -0.3),
            dtype=float), rank=2), R, 2)
        X = cartesian(r, 0.9, -0.3)
        F = np.asarray(stretch.deformation_gradient(X), dtype=float)
        got = f.evaluate(r, 0.9, -0.3, frame="cartesian")
        assert relerr(got, first_tensor(CC, S, F)) < 1e-13


def test_a_pulled_back_source_is_accepted(prem, stretch, stress):
    """A physically isotropic medium, pulled back: still a second tensor.

    The pull-back is where a referentially fully anisotropic source
    comes from, and the first tensor must be built from it with no
    special case -- character ELASTIC and a Voigt matrix are all this
    class asks of a source.
    """
    src = pulled_back_elastic(
        Symmetry.ISOTROPIC,
        {"kappa": lambda r, t, p: 1.3e11 + 0.0 * np.asarray(r, dtype=float),
         "mu": lambda r, t, p: 6.7e10 + 0.0 * np.asarray(r, dtype=float)},
        stretch, skeleton=prem.skeleton, name="phys_iso")
    assert src.character == ELASTIC
    f = FirstElasticField(src, stretch, stress=stress)
    r, th, ph = 5.0e6, 1.2, 0.4
    CC = np.asarray(src.evaluate(r, th, ph, frame="cartesian", voigt=False),
                    dtype=float)
    R = spherical_frame(th, ph)
    S = rotate_slots(voigt_to_tensor(np.asarray(stress.evaluate(r, th, ph),
                                         dtype=float), rank=2), R, 2)
    F = np.asarray(stretch.deformation_gradient(cartesian(r, th, ph)),
                   dtype=float)
    assert relerr(f.evaluate(r, th, ph, frame="cartesian"),
                  first_tensor(CC, S, F)) < 1e-13

    G = gradient(())
    want = np.einsum("iAjB,jB->iA", f.evaluate(r, th, ph), G)
    assert relerr(f.apply(G, r, th, ph), want) < 1e-13


# ---------------------------------------------------------- frames and shapes


@pytest.mark.parametrize("method", ["evaluate", "equilibrium_form"])
def test_the_spherical_frame_rotates_every_slot(prem, stretch, stress, method):
    """Spherical components are R^T on all four slots of the Cartesian ones."""
    f = FirstElasticField(prem.elastic_moduli, stretch, stress=stress)
    r, th, ph = grid()
    cart = getattr(f, method)(r, th, ph, frame="cartesian")
    sph = getattr(f, method)(r, th, ph)
    shape = np.broadcast(r, th, ph).shape
    R = np.broadcast_to(spherical_frame(np.broadcast_to(th, shape),
                                         np.broadcast_to(ph, shape)),
                        shape + (3, 3))
    assert relerr(sph, rotate_slots(cart, np.swapaxes(R, -1, -2), 4)) < 1e-13
    assert relerr(cart, rotate_slots(sph, R, 4)) < 1e-13


def test_shapes_are_the_points_followed_by_three_cubed(prem, stretch, stress):
    f = FirstElasticField(prem.elastic_moduli, stretch, stress=stress)
    r, th, ph = grid()
    shape = np.broadcast(r, th, ph).shape
    assert f.evaluate(r, th, ph).shape == shape + (3, 3, 3, 3)
    assert f.equilibrium_form(r, th, ph).shape == shape + (3, 3, 3, 3)
    assert f.apply(gradient(shape), r, th, ph).shape == shape + (3, 3)
    assert f.evaluate(6.0e6, 0.5, 0.2).shape == (3, 3, 3, 3)


def test_a_stress_given_in_full_components_is_accepted(prem, stretch, stress):
    """(..., 3, 3) and the Voigt (..., 6) must give the same field."""
    class FullStress(FieldBase):
        character = STRESS
        skeleton = stress.skeleton
        name = "S_full"
        dimensions = Dimensions.MODULUS
        is_radial = False

        def evaluate(self, r, theta=None, phi=None, **kw):
            return voigt_to_tensor(stress.evaluate(r, theta, phi, **kw), rank=2)

    r, th, ph = grid()
    a = FirstElasticField(prem.elastic_moduli, stretch, stress=stress)
    b = FirstElasticField(prem.elastic_moduli, stretch, stress=FullStress())
    assert relerr(b.evaluate(r, th, ph), a.evaluate(r, th, ph)) < 1e-15


# ------------------------------------------------------------- the contract


@pytest.mark.parametrize("with_stress", [False, True])
def test_the_field_satisfies_the_field_contract(prem, stretch, stress,
                                                with_stress):
    check_field(FirstElasticField(prem.elastic_moduli, stretch,
                                  stress=stress if with_stress else None))


def test_the_contract_sees_the_full_component_shape(prem, stretch):
    """check_field must read (3, 3, 3, 3), not a Voigt shape that is None."""
    f = FirstElasticField(prem.elastic_moduli, stretch)
    char = f.character
    trailing = (char.voigt_shape if char.voigt_shape is not None
                else char.component_shape)
    assert trailing == (3, 3, 3, 3)
    assert f.evaluate(np.array([5.0e6]), 0.7, 0.4).shape == (1,) + trailing


def test_a_radius_outside_the_skeleton_is_refused(prem, stretch):
    f = FirstElasticField(prem.elastic_moduli, stretch)
    b = np.asarray(prem.skeleton.boundaries, dtype=float)
    for bad in (b[0] - 1.0, b[-1] + 1.0):
        with pytest.raises(ValueError):
            f.evaluate(bad, 0.7, 0.4)
