"""Pull-back: the closed forms of B.8.2 and B.8.3 against the generic rule.

Nothing here checks a fast path against the formula it implements.  The
isotropic and VTI closed forms are checked against `PulledBackField`,
which knows nothing about symmetry -- it evaluates the physical Voigt
matrix, rotates it and applies F^-1 to every slot -- and both are then
checked against the appendix's own oracle: push the referential tensor
forward and the physical tensor must come back.  The numpy-only
constructions of `test_appendix_b8` (`isotropic_tensor`, `vti_tensor`,
`push4`) are imported rather than written again, so the thing the
implementation is measured against is the thing the appendix pinned.

Two shortcuts are taken for a `RadialStretch` -- the physical point is
(r + h, theta, phi) with the frame unchanged, and `Ntil = e_r/(1+dh/dr)`
with no inverse -- and both are checked against the general route by
wrapping the same mapping in an object that is not a `RadialStretch` and
so cannot be recognised.
"""
import numpy as np
import pytest

from planetmodel import PREM
from planetmodel.model.character import DENSITY, ELASTIC, SCALAR, Symmetry
from planetmodel.model.frames import spherical_frame
from planetmodel.model.mapping import IdentityMapping, MappingBase, RadialStretch
from planetmodel.model.materials import MODULI_NAMES, voigt_matrix, voigt_to_tensor
from planetmodel.model.pullback import (PulledBackElasticField, PulledBackField,
                                   pulled_back_elastic)
from planetmodel.model.pushforward import check_tensor_symmetries, push_forward_field
from planetmodel.model.units import Dimensions
from planetmodel.testing import check_field

from .test_appendix_b8 import isotropic_tensor, push4, vti_tensor
from .test_mapping import A, smooth_h
from .test_pushforward import relerr, uniform_dilation

SKELETON = PREM().skeleton


# ------------------------------------------------- the physical medium

def _shape(r, t, p):
    """Zeros of the broadcast shape, so every modulus is point-shaped."""
    return np.zeros(np.broadcast(np.asarray(r, dtype=float),
                                 np.asarray(t, dtype=float),
                                 np.asarray(p, dtype=float)).shape)


def phys_kappa(r, t, p):
    """A smooth physical bulk modulus, varying in all three coordinates."""
    r = np.asarray(r, dtype=float)
    return 1.3e11 * (1.0 + 0.3 * np.sin(2.0 * r / A) * np.cos(t)
                     + 0.1 * np.cos(p)) + _shape(r, t, p)


def phys_mu(r, t, p):
    r = np.asarray(r, dtype=float)
    return 6.7e10 * (1.0 + 0.2 * np.cos(3.0 * r / A) * np.sin(t) * np.sin(p))


def _modulus(scale, kr, ka):
    """A distinct smooth modulus for each of the five VTI slots."""
    def f(r, t, p):
        r = np.asarray(r, dtype=float)
        return scale * (1.0 + ka * np.sin(kr * r / A + t) * np.cos(p))
    return f


VTI_MODULI = {
    "A": _modulus(3.1e11, 2.0, 0.25),
    "C": _modulus(3.0e11, 1.5, 0.15),
    "F": _modulus(1.1e11, 2.5, 0.20),
    "L": _modulus(7.0e10, 3.0, 0.10),
    "N": _modulus(7.4e10, 1.0, 0.30),
}
ISO_MODULI = {"kappa": phys_kappa, "mu": phys_mu}


def iso_voigt(r, t, p):
    """The physical isotropic Voigt matrix at a physical point."""
    return voigt_matrix(Symmetry.ISOTROPIC,
                        {k: f(r, t, p) for k, f in ISO_MODULI.items()})


def radial_axis_voigt(A, C, F, L, N):
    """The VTI Voigt matrix with the axis along e_r: materials' own layout.

    Kept as a name because the tests below read as statements about a
    medium whose axis is unambiguously radial; the layout itself is
    `materials.voigt_matrix`, which `test_materials.py` pins against the
    appendix's invariant form with n = e_r.
    """
    return voigt_matrix(Symmetry.VTI, dict(zip(MODULI_NAMES, (A, C, F, L, N))))


def vti_voigt(r, t, p):
    """The physical VTI Voigt matrix, axis = e_r at that point."""
    return radial_axis_voigt(*(VTI_MODULI[k](r, t, p) for k in MODULI_NAMES))


PHYSICAL_VOIGT = {Symmetry.ISOTROPIC: iso_voigt, Symmetry.VTI: vti_voigt}
MODULI = {Symmetry.ISOTROPIC: ISO_MODULI, Symmetry.VTI: VTI_MODULI}


# --------------------------------------------------------------- helpers

def rtp(n=24, seed=5):
    """Random reference points, away from the origin and the poles."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.15 * A, 0.95 * A, size=n),
            rng.uniform(0.25, np.pi - 0.25, size=n),
            rng.uniform(-np.pi, np.pi, size=n))


def mapping():
    """The smooth radial stretch the mapping tests already exercise."""
    return RadialStretch(smooth_h(), rmax=A)


def physical_of(m, r, t, p):
    """The physical spherical coordinates of m(X), for a radial m."""
    return r + np.asarray(m.h(r, t, p), dtype=float), t, p


class Opaque(MappingBase):
    """The same mapping, wearing a different type.

    A `RadialStretch` is recognised by `isinstance` and gets the two
    shortcuts; this delegates every method to one and is not a
    RadialStretch, so the general route runs on identical geometry.  It
    is the only honest way to compare the two: monkeypatching the test
    would change what is being tested.
    """

    def __init__(self, m):
        self._m = m

    def __call__(self, X):
        return self._m(X)

    def deformation_gradient(self, X, *, frame="cartesian"):
        return self._m.deformation_gradient(X, frame=frame)

    def jacobian(self, X):
        return self._m.jacobian(X)


def generic(symmetry, m):
    """The pull-back by the generic rule: no symmetry knowledge at all."""
    return PulledBackField(PHYSICAL_VOIGT[symmetry], m, skeleton=SKELETON,
                           character=ELASTIC, dimensions=Dimensions.MODULUS,
                           name="A_ref_generic")


def fast(symmetry, m):
    """The pull-back by the closed form of B.8.2 or B.8.3."""
    return pulled_back_elastic(symmetry, MODULI[symmetry], m,
                               skeleton=SKELETON, name="A_ref")


# ------------------------------------------- the fast paths against generic

@pytest.mark.parametrize("symmetry", [Symmetry.ISOTROPIC, Symmetry.VTI])
@pytest.mark.parametrize("frame", ["cartesian", "spherical"])
def test_the_closed_form_agrees_with_the_generic_rule(symmetry, frame):
    """B.8.2 and B.8.3 against one evaluation, one rotation and F^-1."""
    m = mapping()
    r, t, p = rtp()
    want = generic(symmetry, m).evaluate(r, t, p, frame=frame)
    got = fast(symmetry, m).evaluate(r, t, p, frame=frame)
    assert relerr(got, want) < 1e-13, f"{symmetry} in the {frame} frame"


@pytest.mark.parametrize("symmetry", [Symmetry.ISOTROPIC, Symmetry.VTI])
def test_the_closed_form_agrees_at_full_rank_too(symmetry):
    """Voigt keeps one slot per class; this compares all eighty-one."""
    m = mapping()
    r, t, p = rtp()
    want = generic(symmetry, m).evaluate(r, t, p, frame="cartesian",
                                         voigt=False)
    got = fast(symmetry, m).evaluate(r, t, p, frame="cartesian", voigt=False)
    assert relerr(got, want) < 1e-13
    check_tensor_symmetries(got)


def test_the_radial_axis_layout_is_the_invariant_form_with_n_along_e_r():
    """The test medium's own definition, against Appendix B.8.3.

    In the spherical frame e_r has components (1, 0, 0), so a VTI medium
    with a radial symmetry axis has, in that frame, exactly the tensor
    `vti_tensor(..., n = (1, 0, 0))`.  Everything below reads
    `radial_axis_voigt` as that medium, so this is what says the reading
    is right.
    """
    moduli = (3.1e11, 3.0e11, 1.1e11, 7.0e10, 7.4e10)
    got = voigt_to_tensor(radial_axis_voigt(*moduli), rank=4)
    want = vti_tensor(*moduli, np.array([1.0, 0.0, 0.0]))
    assert relerr(got, want) < 1e-15


def test_the_generic_route_reads_the_voigt_matrix_as_the_radial_frame():
    """What the generic route assumes, stated as its own check.

    `vti_voigt` is a Voigt matrix with the symmetry axis at index 3, and
    the generic route treats that as components in the spherical frame
    at the physical point.  Expanded and rotated by R(x) it must be
    exactly `vti_tensor(..., n = e_r(x))`, the invariant form B.8.3
    starts from.  If this fails, the comparison above is comparing two
    different media.
    """
    m = mapping()
    r, t, p = rtp(8)
    rp, tp, pp = physical_of(m, r, t, p)
    R = spherical_frame(tp, pp)
    V = voigt_to_tensor(vti_voigt(rp, tp, pp), rank=4)
    got = np.einsum("...iA,...jB,...kC,...lD,...ABCD->...ijkl", R, R, R, R, V)
    for k in range(r.size):
        want = vti_tensor(*(VTI_MODULI[n](rp[k], tp[k], pp[k])
                            for n in MODULI_NAMES), R[k][:, 0])
        assert relerr(got[k], want) < 1e-14


# ----------------------------------------------- the appendix's own oracle

@pytest.mark.parametrize("symmetry", [Symmetry.ISOTROPIC, Symmetry.VTI])
def test_pushing_the_referential_tensor_forward_recovers_the_physical_one(
        symmetry):
    """The B.8.2/B.8.3 oracle: push o pull is the identity, pointwise."""
    m = mapping()
    r, t, p = rtp(12)
    X = np.stack([r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p),
                  r * np.cos(t)], axis=-1)
    F = m.deformation_gradient(X)
    J = m.jacobian(X)
    ref = fast(symmetry, m).evaluate(r, t, p, frame="cartesian", voigt=False)
    rp, tp, pp = physical_of(m, r, t, p)
    n = spherical_frame(tp, pp)[..., :, 0]

    for k in range(r.size):
        if symmetry is Symmetry.ISOTROPIC:
            want = isotropic_tensor(phys_kappa(rp[k], tp[k], pp[k]),
                                    phys_mu(rp[k], tp[k], pp[k]))
        else:
            want = vti_tensor(*(VTI_MODULI[q](rp[k], tp[k], pp[k])
                                for q in MODULI_NAMES), n[k])
        assert relerr(push4(ref[k], F[k], J[k]), want) < 1e-12


def test_vti_with_isotropic_moduli_is_the_isotropic_fast_path():
    """A = C, L = N, F = A - 2L: c3 = c4 = c5 = 0 and the axis drops out."""
    m = mapping()
    r, t, p = rtp()

    def a(r, t, p):
        return phys_kappa(r, t, p) + 4.0 * phys_mu(r, t, p) / 3.0

    def f(r, t, p):
        return a(r, t, p) - 2.0 * phys_mu(r, t, p)

    five = {"A": a, "C": a, "F": f, "L": phys_mu, "N": phys_mu}
    got = PulledBackElasticField(Symmetry.VTI, five, m,
                                 skeleton=SKELETON).evaluate(r, t, p)
    want = fast(Symmetry.ISOTROPIC, m).evaluate(r, t, p)
    assert relerr(got, want) < 1e-13


# ------------------------------------------------------ the radial shortcut

@pytest.mark.parametrize("symmetry", [Symmetry.ISOTROPIC, Symmetry.VTI])
def test_the_radial_shortcut_agrees_with_the_general_route(symmetry):
    """(r + h, theta, phi) and e_r/(1 + dh/dr) against m(X) and F^-1 n."""
    m = mapping()
    r, t, p = rtp()
    short = fast(symmetry, m).evaluate(r, t, p, frame="cartesian", voigt=False)
    long = fast(symmetry, Opaque(m)).evaluate(r, t, p, frame="cartesian",
                                              voigt=False)
    assert relerr(short, long) < 1e-13


def test_the_radial_shortcut_agrees_in_the_generic_route_too():
    """The same shortcut in PulledBackField: one displacement evaluation."""
    m = mapping()
    r, t, p = rtp()
    short = generic(Symmetry.VTI, m).evaluate(r, t, p)
    long = generic(Symmetry.VTI, Opaque(m)).evaluate(r, t, p)
    assert relerr(short, long) < 1e-13


# ----------------------------------------------------- scalars and density

def phys_rho(r, t, p):
    r = np.asarray(r, dtype=float)
    return 3300.0 * (1.0 + 0.1 * np.cos(4.0 * r / A) * np.sin(t) * np.cos(p))


def test_a_physical_density_pulls_back_to_J_rho_composed_with_m():
    """Weight 1, rank 0: the density rule, and gplspec's homogeneous body."""
    m = mapping()
    r, t, p = rtp()
    X = np.stack([r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p),
                  r * np.cos(t)], axis=-1)
    field = PulledBackField(phys_rho, m, skeleton=SKELETON, character=DENSITY,
                            dimensions=Dimensions.DENSITY, name="rho")
    rp, tp, pp = physical_of(m, r, t, p)
    want = m.jacobian(X) * phys_rho(rp, tp, pp)
    assert relerr(field.evaluate(r, t, p), want) < 1e-14


def test_a_weight_zero_scalar_is_just_composition():
    """Q_kappa and friends: invariant, so the pull-back is q o m."""
    m = mapping()
    r, t, p = rtp()
    field = PulledBackField(phys_rho, m, skeleton=SKELETON, character=SCALAR,
                            name="q")
    rp, tp, pp = physical_of(m, r, t, p)
    assert relerr(field.evaluate(r, t, p), phys_rho(rp, tp, pp)) < 1e-15


# ---------------------------------------------------------- known answers

@pytest.mark.parametrize("symmetry", [Symmetry.ISOTROPIC, Symmetry.VTI])
def test_the_identity_mapping_pulls_back_to_the_physical_voigt_matrix(symmetry):
    """F = I, J = 1, Cinv = I: the referential tensor IS the physical one."""
    r, t, p = rtp()
    got = fast(symmetry, IdentityMapping()).evaluate(r, t, p)
    assert relerr(got, PHYSICAL_VOIGT[symmetry](r, t, p)) < 1e-14


def test_a_uniform_dilation_divides_a_referential_tensor_by_one_plus_c():
    """h = c r: Cinv = (1 + c)^-2 I and J = (1 + c)^3, so A_ref = A/(1 + c).

    The one case the factor can be written down by eye, and therefore
    the one place a stray J or a squared Cinv has nowhere to hide.
    """
    c = 0.04
    m = RadialStretch(uniform_dilation(c))
    r, t, p = rtp()
    rp, tp, pp = physical_of(m, r, t, p)
    for symmetry in (Symmetry.ISOTROPIC, Symmetry.VTI):
        got = fast(symmetry, m).evaluate(r, t, p)
        want = PHYSICAL_VOIGT[symmetry](rp, tp, pp) / (1.0 + c)
        assert relerr(got, want) < 1e-13, symmetry


# ---------------------------------------------------------- the full circle

@pytest.mark.parametrize("symmetry", [Symmetry.ISOTROPIC, Symmetry.VTI])
def test_pull_back_then_push_forward_returns_the_physical_medium(symmetry):
    """Pull back, push forward, and the model comes back as it went in.

    The mapping is radial, so the frames at X and m(X) coincide and the
    spherical Voigt matrix at the reference point is directly comparable
    with the physical moduli at the physical point.
    """
    m = mapping()
    r, t, p = rtp()
    circle = push_forward_field(fast(symmetry, m), m)
    rp, tp, pp = physical_of(m, r, t, p)
    assert relerr(circle.evaluate(r, t, p),
                  PHYSICAL_VOIGT[symmetry](rp, tp, pp)) < 1e-12


# -------------------------------------------------------------- contracts

@pytest.mark.parametrize("symmetry", [Symmetry.ISOTROPIC, Symmetry.VTI])
def test_the_fast_paths_satisfy_the_field_contract(symmetry):
    check_field(fast(symmetry, mapping()))


def test_the_generic_scalar_satisfies_the_field_contract():
    check_field(PulledBackField(phys_rho, mapping(), skeleton=SKELETON,
                                character=DENSITY,
                                dimensions=Dimensions.DENSITY, name="rho"))


def test_the_generic_tensor_satisfies_the_field_contract():
    check_field(generic(Symmetry.VTI, mapping()))


def test_the_circle_satisfies_the_field_contract():
    check_field(push_forward_field(fast(Symmetry.VTI, mapping()), mapping()))


def test_the_pulled_back_field_reports_what_it_is():
    m = mapping()
    field = fast(Symmetry.VTI, m)
    assert field.skeleton == SKELETON
    assert field.character == ELASTIC
    assert field.dimensions is Dimensions.MODULUS
    assert field.symmetry is Symmetry.VTI
    assert field.moduli_names == MODULI_NAMES
    assert field.name == "A_ref"
    assert field.is_radial is False
    assert generic(Symmetry.VTI, m).is_radial is False


# --------------------------------------------------------- what is refused

def test_the_angles_are_required():
    for field in (fast(Symmetry.VTI, mapping()),
                  generic(Symmetry.ISOTROPIC, mapping())):
        with pytest.raises(ValueError, match="theta"):
            field.evaluate(3.0e6)
        with pytest.raises(ValueError, match="theta"):
            field.evaluate(3.0e6, 0.7)


def test_a_radius_outside_the_skeleton_is_refused():
    """The physical callable would extrapolate; the skeleton is ours."""
    with pytest.raises(ValueError, match="skeleton"):
        fast(Symmetry.ISOTROPIC, mapping()).evaluate(1.1 * A, 0.7, 0.4)


def test_an_unknown_frame_is_refused():
    with pytest.raises(ValueError, match="frame"):
        fast(Symmetry.ISOTROPIC, mapping()).evaluate(3.0e6, 0.7, 0.4,
                                                     frame="geographic")


def test_the_wrong_moduli_are_refused_by_name():
    m = mapping()
    with pytest.raises(ValueError, match="missing"):
        PulledBackElasticField(Symmetry.VTI, ISO_MODULI, m, skeleton=SKELETON)
    with pytest.raises(ValueError, match="takes only"):
        PulledBackElasticField(Symmetry.ISOTROPIC,
                               dict(ISO_MODULI, N=phys_mu), m,
                               skeleton=SKELETON)


def test_an_unsupported_symmetry_names_the_generic_route():
    for symmetry in (Symmetry.ORTHOTROPIC, Symmetry.GENERAL):
        with pytest.raises(NotImplementedError, match="PulledBackField"):
            pulled_back_elastic(symmetry, {}, mapping(), skeleton=SKELETON)


def test_a_non_callable_physical_quantity_is_refused():
    with pytest.raises(TypeError, match="callable"):
        PulledBackField(3300.0, mapping(), skeleton=SKELETON,
                        character=DENSITY)


def test_a_wrong_physical_shape_is_refused_by_name():
    """A rank-4 callable returning something that is neither shape."""
    field = PulledBackField(lambda r, t, p: np.zeros(np.shape(r) + (3, 3)),
                            mapping(), skeleton=SKELETON, character=ELASTIC)
    with pytest.raises(ValueError, match="Voigt|trailing shape"):
        field.evaluate(3.0e6, 0.7, 0.4)
