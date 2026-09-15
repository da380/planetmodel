"""AnalyticField: a formula presented as a field, in either frame.

The class is thin, so what has to be checked is the two things thinness
hides.  First the frame: the components a formula gives in the spherical
frame must come back Cartesian as R v for a vector and as the Bond
rotation for a Voigt matrix, and the round trip must return what went
in -- nothing here is compared with a second copy of the rotation
written in this file, only with the library's own R and with the
explicit `R v R^T` einsum on full components.  Second the contract: the
angles are required, radii outside the skeleton are refused, and a
Voigt-shaped return is understood as Voigt rather than as a strange
array of numbers.

Every rank the library uses -- 0, 1, 2, 4, and the first elasticity
tensor's rank 4 without a Voigt form -- goes through `check_field`,
which is what makes this the field the other contracts can be written
against.
"""
import numpy as np
import pytest

from planetmodel import (DENSITY, ELASTIC, FIRST_ELASTIC, SCALAR, STRESS, VECTOR,
                    AnalyticField, RadialStretch, Skeleton, push_forward_field)
from planetmodel.model.materials import bond_matrix, voigt_to_tensor
from planetmodel.model.frames import spherical_frame
from planetmodel.model.units import Dimensions
from planetmodel.testing import check_field

from .test_mapping import A, smooth_h

SK = Skeleton([0.2 * A, 0.6 * A, A])


def grid():
    """A (r, theta, phi) lattice as three broadcast axes."""
    return (np.linspace(0.25 * A, 0.95 * A, 4)[:, None, None],
            np.array([0.4, 1.3, 2.6])[None, :, None],
            np.array([-1.7, 0.3])[None, None, :])


def scalar_fn(r, t, p):
    """A genuinely three-dimensional scalar."""
    return 3.0e3 + 200.0 * np.sin(3.0 * r / A) * np.cos(t) * np.cos(p)


def vector_fn(r, t, p):
    """Spherical-frame components (v_r, v_theta, v_phi)."""
    return np.stack([r / A + 0.0 * t, np.sin(t) * np.cos(p),
                     0.5 * np.cos(t) + 0.0 * p], axis=-1)


def stress_fn(r, t, p):
    """A symmetric rank-2 field, in Voigt order (11, 22, 33, 23, 13, 12)."""
    s = np.stack([1.0 + r / A, 2.0 - np.cos(t), 3.0 * np.ones_like(p * t),
                  0.4 * np.sin(t), 0.3 * np.cos(p), 0.2 + 0.0 * r], axis=-1)
    return 1.0e9 * s


def elastic_fn(r, t, p):
    """A Voigt 6x6 with the full symmetries, varying with position."""
    base = np.eye(6) * np.array([4.0, 4.0, 5.0, 1.0, 1.0, 1.5])
    base[0, 1] = base[1, 0] = 2.0
    base[0, 2] = base[2, 0] = base[1, 2] = base[2, 1] = 1.8
    w = 1.0 + 0.1 * np.sin(2.0 * r / A) * np.cos(t) * np.sin(p)
    return 1.0e11 * w[..., None, None] * base


#: A fixed rank-4 tensor with the major symmetry and neither minor one:
#: what a first elasticity tensor looks like, and why FIRST_ELASTIC has
#: no Voigt form.  Built once, so the field is a function of position.
_FIRST = (lambda M: 0.5 * (M + M.T))(
    np.random.default_rng(5).normal(size=(9, 9))).reshape(3, 3, 3, 3)


def first_fn(r, t, p):
    """Full (3, 3, 3, 3) components with major symmetry only."""
    w = 1.0 + 0.05 * np.cos(t) * np.sin(p) * r / A
    return 1.0e11 * w[..., None, None, None, None] * _FIRST


CASES = {
    "scalar": (scalar_fn, SCALAR),
    "density": (scalar_fn, DENSITY),
    "vector": (vector_fn, VECTOR),
    "stress": (stress_fn, STRESS),
    "elastic_moduli": (elastic_fn, ELASTIC),
    "first": (first_fn, FIRST_ELASTIC),
}


def field(kind, **kw):
    """One of the case fields, on the shared skeleton."""
    fn, char = CASES[kind]
    return AnalyticField(fn, SK, character=char, name=kind,
                         dimensions=Dimensions.DIMENSIONLESS, **kw)


# ------------------------------------------------------------- contracts

@pytest.mark.parametrize("kind", list(CASES))
def test_analytic_fields_of_every_rank_satisfy_the_contract(kind):
    check_field(field(kind))


def test_the_angles_are_required_by_name():
    """A formula is not asked which of its arguments it ignores."""
    f = field("scalar")
    assert f.is_radial is False
    with pytest.raises(ValueError, match="theta and phi"):
        f.evaluate(0.5 * A)


def test_radii_outside_the_skeleton_are_refused():
    f = field("scalar")
    with pytest.raises(ValueError, match="outside the skeleton"):
        f.evaluate(1.5 * A, 0.7, 0.4)


def test_an_unknown_frame_is_refused_at_both_ends():
    with pytest.raises(ValueError, match="unknown frame"):
        AnalyticField(scalar_fn, SK, frame="geographic")
    with pytest.raises(ValueError, match="unknown frame"):
        field("scalar").evaluate(0.5 * A, 0.7, 0.4, frame="geographic")


def test_a_rank_two_field_must_bring_its_component_axes():
    """One number per point is not a rank-2 value, and is said so."""
    f = AnalyticField(lambda r, t, p: r, SK, character=STRESS)
    with pytest.raises(ValueError, match="trailing shape"):
        f.evaluate(0.5 * A, 0.7, 0.4)


def test_a_constant_tensor_is_broadcast_over_the_points():
    """The component axes must be there; the point axes need not be."""
    v = np.arange(6.0)
    f = AnalyticField(lambda r, t, p: v, SK, character=STRESS)
    r, th, ph = grid()
    out = f.evaluate(r, th, ph)
    assert out.shape == np.broadcast_shapes(r.shape, th.shape, ph.shape) + (6,)
    assert np.allclose(out, v)


def test_voigt_and_full_input_agree():
    """The trailing shapes distinguish themselves, as in PulledBackField."""
    r, th, ph = grid()
    voigt = field("elastic_moduli")
    full = AnalyticField(lambda r, t, p: voigt_to_tensor(elastic_fn(r, t, p), rank=4),
                         SK, character=ELASTIC)
    assert np.allclose(voigt.evaluate(r, th, ph), full.evaluate(r, th, ph))
    assert np.allclose(voigt.evaluate(r, th, ph, frame="cartesian"),
                       full.evaluate(r, th, ph, frame="cartesian"))


def test_layer_and_side_are_accepted_and_ignored():
    """An analytic function has no sides: both give the same number."""
    f = field("scalar")
    b = float(SK.boundaries[1])
    assert np.allclose(f.evaluate(b, 0.7, 0.4, side="lower"),
                       f.evaluate(b, 0.7, 0.4, side="upper"))
    assert np.allclose(f.evaluate(b, 0.7, 0.4, layer=0),
                       f.evaluate(b, 0.7, 0.4, layer=1))


# ---------------------------------------------------------------- frames

def test_a_vector_rotates_as_R_v():
    """The rank-1 analogue of the Bond clause, written out explicitly."""
    r, th, ph = grid()
    f = field("vector")
    sph = f.evaluate(r, th, ph)
    cart = f.evaluate(r, th, ph, frame="cartesian")
    R = spherical_frame(*np.broadcast_arrays(th, ph))
    want = np.einsum("...ij,...j->...i", R, sph)
    err = np.max(np.abs(cart - want)) / np.max(np.abs(want))
    assert err < 1e-15, err
    assert np.allclose(sph, vector_fn(*np.broadcast_arrays(r, th, ph)))


def test_a_voigt_matrix_rotates_by_the_bond_matrix():
    """And the Bond matrix is itself checked against the full einsum."""
    r, th, ph = grid()
    f = field("elastic_moduli")
    sph = f.evaluate(r, th, ph)
    cart = f.evaluate(r, th, ph, frame="cartesian")
    R = spherical_frame(*np.broadcast_arrays(th, ph))
    M = bond_matrix(R)
    assert np.allclose(cart, M @ sph @ np.swapaxes(M, -1, -2), rtol=1e-13)

    # The independent construction: rotate all four slots of the full
    # tensor and read the Voigt entries off the result.
    T = voigt_to_tensor(sph, rank=4)
    rot = np.einsum("...ia,...jb,...kc,...ld,...abcd->...ijkl", R, R, R, R, T)
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    want = np.stack([np.stack([rot[..., i, j, k, l] for k, l in pairs], axis=-1)
                     for i, j in pairs], axis=-2)
    err = np.max(np.abs(cart - want)) / np.max(np.abs(want))
    assert err < 1e-13, err


@pytest.mark.parametrize("kind", ["vector", "stress", "elastic_moduli", "first"])
def test_the_two_frames_are_inverse_to_one_another(kind):
    """A field whose formula speaks Cartesian gives the same values."""
    r, th, ph = grid()
    sph = field(kind)
    fn, char = CASES[kind]
    cart_values = sph.evaluate(r, th, ph, frame="cartesian")

    def cartesian_fn(rr, tt, pp):
        """The same field, its formula written in the Cartesian frame."""
        return AnalyticField(fn, SK, character=char).evaluate(
            rr, tt, pp, frame="cartesian")

    other = AnalyticField(cartesian_fn, SK, character=char, frame="cartesian")
    assert np.allclose(other.evaluate(r, th, ph, frame="cartesian"),
                       cart_values)
    err = np.max(np.abs(other.evaluate(r, th, ph) - sph.evaluate(r, th, ph)))
    assert err < 1e-13 * max(1.0, np.max(np.abs(sph.evaluate(r, th, ph)))), err


# ------------------------------------------------------- with the mapping

@pytest.mark.parametrize("kind", ["density", "vector", "stress", "elastic_moduli"])
def test_an_analytic_field_pushed_forward_satisfies_the_contract(kind):
    """Analytic sources for the push-forward and pull-back machinery."""
    m = RadialStretch(smooth_h(1.0e4, 1.0), rmax=A)
    check_field(push_forward_field(field(kind), m))


def test_evaluate_at_is_evaluate_in_cartesian_coordinates():
    r, th, ph = grid()
    r, th, ph = np.broadcast_arrays(r, th, ph)
    X = np.stack([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph),
                  r * np.cos(th)], axis=-1)
    f = field("vector")
    assert np.allclose(f.evaluate_at(X),
                       f.evaluate(r, th, ph, frame="cartesian"))
    assert np.allclose(f.evaluate_at(X, frame="spherical"),
                       f.evaluate(r, th, ph))
