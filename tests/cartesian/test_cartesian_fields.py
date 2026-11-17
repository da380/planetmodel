"""The flat system and its fields: (depth, x, y) with the frame (e_up, e_x,
e_y), the field kinds through their contract, the algebra keeping the
class, and messages that speak of depth."""
import numpy as np
import pytest

from planetmodel import (DENSITY, SCALAR, STRESS, VECTOR, ComposedField,
                         PerturbedField, PulledBackField, PushedForwardField,
                         RadialField, Symmetry)
from planetmodel import cartesian as flat
from planetmodel.coordinates import (SPHERICAL, Coordinates, increases_outward,
                                     layer_samples)
from planetmodel.fields import HeldField, held
from planetmodel.layerfunction import polynomial_layer
from planetmodel.mapping import MappingBase
from planetmodel.materials import ElasticField
from planetmodel.skeleton import Skeleton
from planetmodel.testing import check_field

FLAT = flat.FLAT
IV = (2.0, 5.0)


class Shear(MappingBase):
    """m(X) = A X, det A > 0: not the identity."""

    A = np.array([[1.0, 0.1, 0.05], [0.0, 1.1, -0.1], [0.02, 0.0, 0.9]])

    def __call__(self, X):
        return np.asarray(X, dtype=float) @ self.A.T

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(self.A, X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        X = np.asarray(X, dtype=float)
        return np.full(X.shape[:-1], np.linalg.det(self.A))


def cubic():
    return polynomial_layer(IV, [1.0, 0.3, -0.02, 0.001])


def flat_fields():
    """One of each kind, in the flat system."""
    rho = flat.DepthField(IV, cubic(), name="rho")
    vec = flat.DepthField(IV, [cubic(), 2.0, lambda d: 0.5 * d], character=VECTOR)
    stress = flat.constant_field(IV, np.arange(1.0, 7.0), character=STRESS)
    lateral = flat.AnalyticField(IV, lambda d, x, y: 1.0 + 0.1 * d + 0.2 * x * y)
    turning = flat.AnalyticField(
        IV, lambda d, x, y: np.stack(np.broadcast_arrays(x, y * d, 1.0 + 0 * d),
                                     axis=-1),
        character=VECTOR, frame="cartesian")
    kappa = flat.DepthField(IV, polynomial_layer(IV, [3.0, 0.5]))
    mu = flat.AnalyticField(IV, lambda d, x, y: 1.0 + 0.1 * x ** 2 + 0.0 * d)
    iso = ElasticField(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu})
    return {
        "depth": rho, "vector": vec, "stress": stress, "analytic": lateral,
        "analytic cartesian": turning,
        "composed": ComposedField(lambda a, b: a * b, (rho, lateral), character=SCALAR),
        "sum": rho + rho * 2.0, "product": rho * lateral, "power": rho ** 2,
        "number": 1.0 + lateral, "negated": -vec, "derivative": rho.derivative(),
        "perturbed": PerturbedField(rho, lateral - 1.0, kind="relative"),
        "elastic": iso, "vti": iso.as_symmetry(Symmetry.VTI),
        "pushed": PushedForwardField(vec, Shear()),
        "pushed elastic": PushedForwardField(iso, Shear()),
        "pulled": PulledBackField(
            IV, lambda d, x, y: np.stack(np.broadcast_arrays(d, x, 1.0 + y), axis=-1),
            Shear(), character=VECTOR, coordinates=FLAT),
        "held depth": held(vec, IV[1], (5.0, 8.0)),
        "held analytic": held(lateral, IV[1], (5.0, 8.0)),
    }


# ------------------------------------------------------------ the system

def test_flat_is_a_coordinate_system_of_depth_x_and_y():
    assert isinstance(FLAT, Coordinates) and flat.FlatCoordinates() == FLAT
    assert FLAT.frames == ("local", "cartesian")
    assert FLAT.labels == ("depth", "x", "y") and repr(FLAT) == "FLAT"
    d, x, y = np.array([0.0, 1.5, 7.0]), np.array([2.0, -1.0, 0.5]), np.array(
        [3.0, 4.0, -2.0])
    X = FLAT.points(d, x, y)
    assert np.array_equal(X, np.stack([x, y, -d], axis=-1))
    dd, xx, yy, R = FLAT.split(X)
    assert np.array_equal(dd, d) and np.array_equal(xx, x) and np.array_equal(yy, y)
    assert R.shape == (3, 3, 3) and np.array_equal(R[1], flat.coordinates.LOCAL_FRAME)
    assert all(np.allclose(a, 2.5 * b) for a, b in zip(FLAT.scaled(d, x, y, 2.5),
                                                       (d, x, y)))


def test_the_local_frame_is_up_x_y_right_handed_with_depth_running_against_it():
    R = FLAT.frame(0.3, -0.2)
    up, ex, ey = R[:, 0], R[:, 1], R[:, 2]
    assert np.array_equal(up, [0, 0, 1]) and np.array_equal(ex, [1, 0, 0])
    assert np.array_equal(ey, [0, 1, 0])
    assert np.isclose(np.linalg.det(R), 1.0)
    # the frame with its first vector along increasing depth would be left-handed
    assert np.isclose(np.linalg.det(np.stack([-up, ex, ey], axis=-1)), -1.0)
    assert increases_outward(SPHERICAL) and not increases_outward(FLAT)


def test_a_vectors_first_component_is_upward_and_the_derivative_downward():
    down = flat.DepthField(IV, [-1.0, 0.0, 0.0], character=VECTOR)
    assert np.allclose(down.evaluate(3.0, 1.0, 2.0, frame="cartesian"), [0, 0, -1],
                       rtol=0, atol=1e-15)
    # rho = 1 + d grows downward: d rho / d depth = +1, so -1 along +z
    rho = flat.DepthField(IV, polynomial_layer(IV, [1.0, 1.0]))
    assert np.allclose(rho.derivative()(np.linspace(*IV, 5)), 1.0, rtol=1e-14)
    z, h = -3.0, 1e-3
    dz = (rho.evaluate_at([0.0, 0.0, z + h]) - rho.evaluate_at([0.0, 0.0, z - h])) / (
        2 * h)
    assert np.isclose(dz, -1.0, rtol=1e-9)


def test_the_vertical_is_the_axis_of_a_transversely_isotropic_layer():
    five = {k: flat.constant_field(IV, v)
            for k, v in dict(A=9.0, C=7.0, F=3.0, L=2.0, N=2.5).items()}
    C = ElasticField(Symmetry.VTI, five).evaluate(3.0, 0.4, -0.7, frame="cartesian",
                                                  voigt=False)
    assert np.isclose(C[2, 2, 2, 2], 7.0) and np.isclose(C[0, 0, 0, 0], 9.0)
    assert np.isclose(C[1, 1, 1, 1], 9.0) and np.isclose(C[0, 2, 0, 2], 2.0)
    assert np.isclose(C[0, 1, 0, 1], 2.5)


def test_the_lattice_lies_inside_every_layer_over_the_bounds():
    sk = Skeleton([0.0, 1.0, 4.0])
    d, x, y = FLAT.lattice(sk, n_c=3, n_a=4, n_b=5, bounds=((0.0, 6.0), (-1.0, 1.0)))
    assert d.shape == (6, 1, 1) and x.shape == (1, 4, 1) and y.shape == (1, 1, 5)
    assert np.array_equal(d.ravel(), layer_samples(sk, 3))
    assert np.all((d > 0) & (d < 4)) and not np.any(np.isin(d, [1.0]))
    assert x.min() == 0.0 and x.max() == 6.0 and y.min() == -1.0 and y.max() == 1.0
    _, x, _ = FLAT.lattice(sk)
    assert x.min() == -4.0 and x.max() == 4.0      # unbounded: about the origin


# ------------------------------------------------------------ the fields

def test_flat_fields_pass_the_contract():
    for label, field in flat_fields().items():
        assert field.coordinates is FLAT, label
        check_field(field)


def test_the_algebra_and_every_copy_keep_the_flat_class():
    rho = flat.DepthField(IV, cubic(), name="rho", character=DENSITY)
    plain = RadialField(IV, cubic(), character=DENSITY, coordinates=FLAT)
    for got in (rho + rho, rho - plain, plain + rho, 2.0 * rho, -rho, rho / 2.0,
                rho * flat.constant_field(IV, 2.0), flat.constant_field(IV, 1.0) ** 3,
                rho.derivative(), rho.on_interval(1.0, 6.0), rho.renamed("x"),
                rho.rescaled(k=2.0, v=3.0), rho.held(5.0, (5.0, 9.0)),
                held(rho, 5.0, (5.0, 9.0))):
        assert type(got) is flat.DepthField and got.coordinates is FLAT, got
    assert type(plain + plain) is RadialField and (plain + plain).coordinates is FLAT
    f = flat.AnalyticField(IV, lambda d, x, y: d + x * y, name="f")
    for got in (f.on_interval(1.0, 6.0), f.renamed("g"), f.rescaled(k=2.0, v=1.0)):
        assert type(got) is flat.AnalyticField and got.coordinates is FLAT
    # the sphere's classes are as they were
    sph = RadialField(IV, cubic())
    assert type(sph + sph) is RadialField and (sph * 2.0).coordinates is SPHERICAL
    assert "DepthField(" in repr(rho) and "AnalyticField(" in repr(f)


def test_a_constant_field_of_depth_is_exact():
    s = flat.constant_field(IV, np.arange(1.0, 7.0), character=STRESS, name="s")
    assert type(s) is flat.DepthField and s.name == "s" and s.radial_degree == 0
    assert np.array_equal(s.evaluate(np.linspace(*IV, 4), 0.0, 0.0),
                          np.tile(np.arange(1.0, 7.0), (4, 1)))


def test_a_held_field_is_the_source_at_its_depth_exactly():
    lateral = flat.AnalyticField(IV, lambda d, x, y: d * (1.0 + x - 2.0 * y))
    h = held(lateral, 5.0, (5.0, 9.0))
    assert isinstance(h, HeldField) and h.at == 5.0 and h.source is lateral
    rng = np.random.default_rng(3)
    d, x, y = rng.uniform(5.0, 9.0, 50), rng.uniform(-9, 9, 50), rng.uniform(-9, 9, 50)
    assert np.array_equal(h.evaluate(d, x, y), lateral.evaluate(np.full(50, 5.0), x, y))
    vec = flat.DepthField(IV, [cubic(), 2.0, lambda d: 0.5 * d], character=VECTOR)
    hv = held(vec, 5.0, (5.0, 9.0))
    assert np.array_equal(hv.evaluate(d, x, y), np.broadcast_to(vec.evaluate(
        5.0, 0.0, 0.0), (50, 3)))
    with pytest.raises(ValueError, match="depth 6 is outside the interval"):
        HeldField(lateral, 6.0, (5.0, 9.0))
    with pytest.raises(ValueError, match="depth 6 is outside the interval"):
        vec.held(6.0, (5.0, 9.0))


def test_messages_speak_of_depth_and_x_and_y():
    rho = flat.DepthField(IV, cubic())
    f = flat.AnalyticField(IV, lambda d, x, y: d + x)
    with pytest.raises(ValueError, match="depth 1 is outside the interval"):
        rho(1.0)
    with pytest.raises(ValueError, match="needs x and y as well as depth"):
        f(3.0)
    with pytest.raises(TypeError, match=r"\(depth, x, y\) or with depth alone"):
        f(3.0, 1.0)
    with pytest.raises(ValueError, match="unknown frame 'spherical'"):
        rho.evaluate(3.0, 0.0, 0.0, frame="spherical")


def test_a_mix_of_flat_and_spherical_fields_is_refused_by_name():
    rho = flat.DepthField(IV, cubic())
    sph = RadialField(IV, cubic())
    with pytest.raises(ValueError, match="'flat' and 'spherical'"):
        rho + sph
    with pytest.raises(ValueError, match="'spherical' and 'flat'"):
        sph * flat.AnalyticField(IV, lambda d, x, y: x + 0 * d)
