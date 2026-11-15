"""The coordinate-system seam: SPHERICAL is the frames module, and a toy
system with a constant frame carries the shared fields through their
contract."""
import numpy as np
import pytest

from planetmodel.character import ELASTIC, SCALAR, STRESS, VECTOR, Symmetry
from planetmodel.coordinates import (SPHERICAL, Coordinates, SphericalCoordinates,
                                     common_coordinates, coordinates_of, resolve_frame)
from planetmodel.fields import (FRAMES, AnalyticField, ComposedField, RadialField,
                                constant_field)
from planetmodel.frames import cartesian_points, spherical_coordinates, spherical_frame
from planetmodel.geometry import Geometry
from planetmodel.layerfunction import polynomial_layer
from planetmodel.mapping import MappingBase, validity_lattice
from planetmodel.materials import ElasticField
from planetmodel.model import Model
from planetmodel.perturbation import PerturbedField
from planetmodel.pushforward import PulledBackField, PushedForwardField
from planetmodel.skeleton import Skeleton
from planetmodel.testing import check_field

IV = (0.5, 2.0)


class Toy:
    """A constant-frame system, X = P (c, a, b), P orthogonal with det 1:
    the columns of P are the local basis (e_c, e_a, e_b).  No centre, no
    pole; the lateral coordinates are lengths, so a change of scale
    multiplies all three."""

    frames = ("local", "cartesian")

    def __init__(self, name: str, P: list) -> None:
        self.name = name
        self.P = np.asarray(P, dtype=float)

    def points(self, c, a, b) -> np.ndarray:
        q = np.stack(np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                           for x in (c, a, b))), axis=-1)
        return q @ self.P.T

    def split(self, X, *, near=None) -> tuple:
        q = np.asarray(X, dtype=float) @ self.P
        a, b = q[..., 1], q[..., 2]
        return q[..., 0], a, b, self.frame(a, b)

    def frame(self, a, b) -> np.ndarray:
        shape = np.broadcast(np.asarray(a), np.asarray(b)).shape
        return np.broadcast_to(self.P, shape + (3, 3)).copy()

    def scaled(self, c, a, b, k) -> tuple:
        return tuple(np.asarray(x, dtype=float) * k for x in (c, a, b))

    def lattice(self, skeleton, *, n_c=4, n_a=4, n_b=4) -> tuple:
        edges = skeleton.boundaries
        inset = [(lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))
                 for lo, hi in zip(edges[:-1], edges[1:])]
        c = np.concatenate([np.linspace(lo, hi, n_c) for lo, hi in inset])
        return (c[:, None, None], np.linspace(-1.0, 1.0, n_a)[None, :, None],
                np.linspace(-1.0, 1.0, n_b)[None, None, :])

    def probe(self, rng, n) -> tuple:
        return rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n)


# points (c, a, b) -> (a, b, c): the layered coordinate is the height
UP = Toy("toy-up", [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
# points (c, a, b) -> (b, a, -c): the layered coordinate is a depth below
# z = 0, decreasing upward, with (e_c, e_a, e_b) = (down, y, x) right-handed
DOWN = Toy("toy-down", [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
TOYS = pytest.mark.parametrize("toy", [UP, DOWN], ids=["up", "down"])


class Linear(MappingBase):
    """m(X) = A X, with det A > 0."""

    A = np.array([[1.1, 0.1, 0.0], [0.05, 0.9, 0.2], [0.0, -0.1, 1.2]])

    def __call__(self, X):
        return np.asarray(X, dtype=float) @ self.A.T

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(self.A, X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        X = np.asarray(X, dtype=float)
        return np.full(X.shape[:-1], np.linalg.det(self.A))


def cubic():
    return polynomial_layer(IV, [1.0, 0.3, -0.2, 0.1])


def toy_fields(toy):
    """One of each shipped kind that is not spherical by nature, on `toy`."""
    rho = RadialField(IV, cubic(), name="rho", coordinates=toy)
    vec = RadialField(IV, [cubic(), 2.0, lambda c: 0.5 * c], character=VECTOR,
                      coordinates=toy)
    stress = constant_field(IV, np.arange(1.0, 7.0), character=STRESS,
                            coordinates=toy)
    lateral = AnalyticField(IV, lambda c, a, b: 1.0 + 0.1 * c + 0.2 * a * b,
                            coordinates=toy)
    turning = AnalyticField(
        IV, lambda c, a, b: np.stack(np.broadcast_arrays(a, b * c, 1.0 + 0 * c),
                                     axis=-1),
        character=VECTOR, frame="cartesian", coordinates=toy)
    composed = ComposedField(lambda x, y: x * y, (rho, lateral), character=SCALAR)
    kappa = RadialField(IV, polynomial_layer(IV, [3.0, 0.5]), coordinates=toy)
    mu = AnalyticField(IV, lambda c, a, b: 1.0 + 0.1 * a ** 2 + 0.0 * c,
                       coordinates=toy)
    return {
        "radial": rho, "vector": vec, "stress": stress, "analytic": lateral,
        "analytic cartesian": turning, "composed": composed,
        "sum": rho + rho * 2.0, "product": rho * lateral, "power": rho ** 2,
        "number": 1.0 + lateral, "negated": -vec,
        "perturbed": PerturbedField(rho, lateral - 1.0, kind="relative"),
        "elastic": ElasticField(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu}),
        "vti": ElasticField(Symmetry.ISOTROPIC,
                            {"kappa": kappa, "mu": mu}).as_symmetry(Symmetry.VTI),
        "pushed": PushedForwardField(vec, Linear()),
        "pushed elastic": PushedForwardField(
            ElasticField(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu}), Linear()),
        "pulled": PulledBackField(
            IV, lambda c, a, b: np.stack(np.broadcast_arrays(c, a, 1.0 + b), axis=-1),
            Linear(), character=VECTOR, coordinates=toy),
    }


# --------------------------------------------------------------- SPHERICAL

def test_spherical_is_the_frames_module():
    assert isinstance(SPHERICAL, Coordinates)
    assert SphericalCoordinates() == SPHERICAL and FRAMES == SPHERICAL.frames
    rng = np.random.default_rng(1)
    r, theta, phi = rng.uniform(0.0, 2.0, 7), rng.uniform(0.0, np.pi, 7), rng.uniform(
        -np.pi, np.pi, 7)
    X = cartesian_points(r, theta, phi)
    assert np.array_equal(SPHERICAL.points(r, theta, phi), X)
    for got, want in zip(SPHERICAL.split(X), spherical_coordinates(X)):
        assert np.array_equal(got, want)
    assert np.array_equal(SPHERICAL.frame(theta, phi), spherical_frame(theta, phi))
    sk = Skeleton([0.0, 1.0, 2.0])
    for got, want in zip(SPHERICAL.lattice(sk, n_c=3, n_a=5, n_b=4),
                         validity_lattice(sk, n_r=3, n_theta=5, n_phi=4)):
        assert np.array_equal(got, want)
    assert np.allclose(SPHERICAL.scaled(r, theta, phi, 2.5)[0], 2.5 * r, rtol=1e-15)


def test_split_near_takes_the_angles_where_the_point_fixes_none():
    X = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    theta, phi = np.full(4, 0.7), np.full(4, 1.3)
    _, t, p, R = SPHERICAL.split(X, near=(theta, phi))
    assert np.allclose(p[:3], 1.3) and np.isclose(p[3], np.pi / 4)
    assert t[0] == 0.0 and t[1] == np.pi and t[2] == 0.7
    assert np.allclose(R, spherical_frame(t, p))


def test_the_defaults_are_spherical():
    rho = RadialField(IV, cubic())
    assert rho.coordinates is SPHERICAL and coordinates_of(rho) is SPHERICAL
    assert (rho + 1.0).coordinates is SPHERICAL
    assert Geometry(Skeleton([0.0, 1.0])).coordinates is SPHERICAL
    assert coordinates_of(object()) is SPHERICAL
    assert resolve_frame(SPHERICAL, None) == "spherical"


# --------------------------------------------------------------- the toy

@TOYS
def test_toy_fields_pass_the_contract(toy):
    assert isinstance(toy, Coordinates)
    for label, field in toy_fields(toy).items():
        assert field.coordinates is toy, label
        check_field(field)


@TOYS
def test_a_rank_one_field_presents_the_toys_components(toy):
    v = RadialField(IV, [1.0, 2.0, 3.0], character=VECTOR, coordinates=toy)
    c, a, b = np.linspace(*IV, 5), np.linspace(-1.0, 1.0, 5), np.full(5, 0.3)
    want = toy.P @ np.array([1.0, 2.0, 3.0])
    assert np.allclose(v.evaluate(c, a, b), [1.0, 2.0, 3.0], rtol=0, atol=1e-15)
    assert np.allclose(v.evaluate(c, a, b, frame="cartesian"), want, rtol=0, atol=1e-15)
    assert np.allclose(v(c, a, b), [1.0, 2.0, 3.0], rtol=0, atol=1e-15)
    X = toy.points(c, a, b)
    assert np.allclose(v.evaluate_at(X), want, rtol=0, atol=1e-15)
    assert np.allclose(v.evaluate_at(X, frame="local"), [1.0, 2.0, 3.0],
                       rtol=0, atol=1e-15)
    back = AnalyticField(IV, lambda c, a, b: want, character=VECTOR, frame="cartesian",
                         coordinates=toy)
    assert np.allclose(back.evaluate(c, a, b), [1.0, 2.0, 3.0], rtol=0, atol=1e-15)


def test_the_toy_orders_its_points_as_stated():
    assert np.array_equal(UP.points(1.0, 2.0, 3.0), [2.0, 3.0, 1.0])
    assert np.array_equal(DOWN.points(1.0, 2.0, 3.0), [3.0, 2.0, -1.0])
    for toy in (UP, DOWN):
        assert np.isclose(np.linalg.det(toy.P), 1.0)


@TOYS
def test_frames_are_the_systems_own(toy):
    v = RadialField(IV, [1.0, 2.0, 3.0], character=VECTOR, coordinates=toy)
    with pytest.raises(ValueError, match="unknown frame 'spherical'"):
        v.evaluate(1.0, 0.0, 0.0, frame="spherical")
    with pytest.raises(ValueError, match="unknown frame 'local'"):
        RadialField(IV, [1.0, 2.0, 3.0], character=VECTOR).evaluate(
            1.0, 0.0, 0.0, frame="local")
    with pytest.raises(ValueError, match="unknown frame 'spherical'"):
        AnalyticField(IV, lambda c, a, b: c, frame="spherical", coordinates=toy)
    assert AnalyticField(IV, lambda c, a, b: c, coordinates=toy).frame == "local"


def test_a_mix_of_systems_is_refused_by_name():
    up = RadialField(IV, cubic(), coordinates=UP)
    down = RadialField(IV, cubic(), coordinates=DOWN)
    sph = RadialField(IV, cubic())
    lateral = AnalyticField(IV, lambda c, a, b: a + 0.0 * c, coordinates=UP)
    names = "'toy-up' and 'spherical'"
    with pytest.raises(ValueError, match=names):
        up + sph
    with pytest.raises(ValueError, match=names):
        up * sph
    with pytest.raises(ValueError, match="'spherical' and 'toy-up'"):
        sph - lateral
    with pytest.raises(ValueError, match="'toy-up' and 'toy-down'"):
        up / down
    with pytest.raises(ValueError, match="different coordinate systems"):
        ComposedField(np.add, (lateral, sph), character=SCALAR)
    with pytest.raises(ValueError, match="different coordinate systems"):
        PerturbedField(sph, lateral, kind="relative")
    with pytest.raises(ValueError, match="different coordinate systems"):
        ElasticField(Symmetry.ISOTROPIC, {"kappa": up, "mu": sph})
    with pytest.raises(ValueError, match="'toy-up', the geometry in 'spherical'"):
        Model(Geometry(Skeleton(list(IV))), [{"q": up}])
    with pytest.raises(ValueError, match="no fields"):
        common_coordinates(())


def test_a_system_is_checked():
    with pytest.raises(TypeError, match="expected a Coordinates"):
        RadialField(IV, cubic(), coordinates="flat")
    with pytest.raises(TypeError, match="expected a Coordinates"):
        constant_field(IV, 1.0, character=SCALAR, coordinates=object())

    class Stated:
        coordinates = "flat"

    with pytest.raises(TypeError, match="not a Coordinates"):
        coordinates_of(Stated())
    assert constant_field(IV, np.eye(6), character=ELASTIC,
                          coordinates=UP).coordinates is UP
