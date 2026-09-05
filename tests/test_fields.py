"""Fields on one interval: evaluation, frames, the algebra, and the contract."""
import numpy as np
import pytest

from planetmodel.character import DENSITY, ELASTIC, SCALAR, STRESS, VECTOR, Character
from planetmodel.fields import (AnalyticField, ComposedField, Field, RadialField,
                                constant_field)
from planetmodel.frames import spherical_frame, voigt_to_tensor
from planetmodel.layerfunction import NumericLayer, polynomial_layer
from planetmodel.testing import check_field

IV = (1221.5e3, 3480.0e3)              # PREM's outer core
A = 6371e3


def rho():
    return RadialField(IV, polynomial_layer([12.5815, -1.2638, -3.6426, -5.5281],
                                            IV, scale=A), character=DENSITY, name="rho")


def vp():
    return RadialField(IV, polynomial_layer([11.0487, -4.0362, 4.8023, -13.5732],
                                            IV, scale=A), name="vp")


def r_in(n=50):
    return np.linspace(*IV, n)


# --------------------------------------------------------------- radial

def test_radial_scalar_evaluates_with_or_without_angles():
    f = rho()
    r = r_in()
    assert f(r).shape == (50,) and f(r).dtype == np.float64
    assert np.allclose(f(r), f.evaluate(r, 0.3, 0.4))
    assert f(r[:, None], 0.3, np.zeros(3)).shape == (50, 3)
    assert f.is_radial and f.character == DENSITY and f.name == "rho"
    assert f.function.degree == 3


def test_radial_refuses_outside_its_interval_to_rtol():
    f = rho()
    lo, hi = IV
    width = hi - lo
    assert np.isfinite(f(hi + 0.5e-9 * width))
    with pytest.raises(ValueError, match="outside the interval"):
        f(hi + 1e-6 * width)
    with pytest.raises(ValueError, match="outside the interval"):
        f(np.array([lo, lo - 1.0]))
    assert np.isfinite(f.on_interval(0.0, A)(hi + 1e5))


def test_both_ends_are_reached_exactly():
    f = rho()
    lo, hi = IV
    p = f.function
    assert f(lo) == p(lo) and f(hi) == p(hi)


def test_radial_vector_components_are_spherical_and_rotate():
    v = RadialField(IV, [polynomial_layer([1.0], IV), 0.0, 2.0], character=VECTOR)
    th, ph = 0.7, 1.1
    s = v(2e6, th, ph)
    assert s.tolist() == [1.0, 0.0, 2.0]
    assert np.allclose(v.evaluate(2e6, th, ph, frame="cartesian"),
                       spherical_frame(th, ph) @ s)
    with pytest.raises(ValueError, match="theta and phi"):
        v(2e6)
    with pytest.raises(ValueError, match="nested sequence"):
        RadialField(IV, [1.0, 2.0], character=VECTOR)


def test_derivative_and_integral():
    f = rho()
    r = r_in()
    d = f.derivative()
    assert d.character == DENSITY and d.name is None
    assert np.allclose(d(r), f.function.derivative()(r))
    assert np.isclose(f.integrate(*IV), f.function.integrate(*IV))
    v = RadialField(IV, [1.0, 2.0, 3.0], character=VECTOR)
    assert np.allclose(v.integrate(*IV), np.array([1.0, 2.0, 3.0]) * (IV[1] - IV[0]))


def test_renamed_and_repr():
    f = rho().renamed("density")
    assert f.name == "density" and "density" in repr(f)
    assert rho().renamed(None).name is None


# --------------------------------------------------------------- algebra

def test_products_of_polynomials_are_exact_polynomials():
    mu = rho() * vp() ** 2
    assert isinstance(mu, RadialField) and mu.character == Character(0, 1)
    assert mu.function.degree == 9
    r = r_in(500)
    want = rho()(r) * vp()(r) ** 2
    assert np.allclose(mu(r), want, rtol=1e-13, atol=1e-13 * np.max(np.abs(want)))
    hand = rho().function * vp().function * vp().function
    assert np.allclose(mu.function.ppoly.c, hand.ppoly.c, rtol=1e-14)


def test_sums_need_one_character_and_one_interval():
    with pytest.raises(ValueError, match="cannot add"):
        rho() + vp()
    with pytest.raises(ValueError, match="cannot subtract"):
        rho() - vp()
    other = RadialField((0.0, IV[0]), 1.0, character=DENSITY)
    with pytest.raises(ValueError, match="different intervals"):
        rho() + other
    s = rho() + rho()
    assert isinstance(s, RadialField) and np.allclose(s(r_in()), 2 * rho()(r_in()))
    assert (rho() - rho()).function.is_zero()


def test_scaling_and_negation_and_division():
    f = vp()
    r = r_in()
    assert np.allclose((2.0 * f)(r), 2 * f(r)) and np.allclose((f * 2.0)(r), 2 * f(r))
    assert np.allclose((f / 2.0)(r), f(r) / 2) and np.allclose((-f)(r), -f(r))
    assert np.allclose((np.float64(2.0) * f)(r), 2 * f(r))
    q = rho() / rho()
    assert q.character == SCALAR and np.allclose(q(r), 1.0)
    assert (rho() / vp() / vp()).character == Character(0, 1)
    with pytest.raises(ValueError, match="weight"):
        vp() / rho()                 # weight -1 is not a character
    with pytest.raises(ValueError, match="weight"):
        rho() * rho()
    with pytest.raises(ValueError, match="rank 0"):
        rho() ** 2


def test_powers():
    f = vp()
    r = r_in()
    assert np.allclose((f ** 3)(r), f(r) ** 3, rtol=1e-13)
    with pytest.raises(ValueError, match="non-negative integer"):
        f ** 0.5


def test_a_numeric_operand_makes_a_radial_field_of_a_numeric_layer():
    g = RadialField(IV, np.sin)
    h = rho() * g
    assert isinstance(h, RadialField) and isinstance(h.function, NumericLayer)
    r = r_in()
    assert np.allclose(h(r), rho()(r) * np.sin(r))


def test_an_analytic_operand_makes_a_composed_field():
    a = AnalyticField(IV, lambda r, t, p: np.cos(t))
    h = vp() * a
    assert isinstance(h, ComposedField) and not h.is_radial
    r = r_in()
    assert np.allclose(h(r, 0.3, 0.1), vp()(r) * np.cos(0.3))
    with pytest.raises(ValueError, match="theta and phi"):
        h(r)


# ------------------------------------------------------------- analytic

def test_analytic_scalar():
    a = AnalyticField(IV, lambda r, t, p: r * np.cos(t), name="a")
    r = r_in()
    assert np.allclose(a(r, 0.3, 0.0), r * np.cos(0.3))
    assert a(r[:, None], np.zeros(4), 0.0).shape == (50, 4)
    with pytest.raises(ValueError, match="theta and phi"):
        a(r)
    z = AnalyticField(IV, lambda r, t, p: 1j * r)
    assert z(r, 0.1, 0.2).dtype == np.complex128 and z.dtype == np.complex128
    with pytest.raises(TypeError, match="callable"):
        AnalyticField(IV, 3.0)


def test_analytic_tensor_in_either_constructor_frame():
    T = np.diag([1.0, 2.0, 3.0])
    th, ph = 0.7, 1.1
    R = spherical_frame(th, ph)
    cart = AnalyticField(IV, lambda r, t, p: T, character=STRESS, frame="cartesian")
    sph = AnalyticField(IV, lambda r, t, p: T, character=STRESS)
    assert np.allclose(voigt_to_tensor(cart(2e6, th, ph), rank=2), R.T @ T @ R)
    assert np.allclose(voigt_to_tensor(cart.evaluate(2e6, th, ph, frame="cartesian"),
                                       rank=2), T)
    assert np.allclose(voigt_to_tensor(sph.evaluate(2e6, th, ph, frame="cartesian"),
                                       rank=2), R @ T @ R.T)
    voigt = AnalyticField(IV, lambda r, t, p: np.array([1.0, 2.0, 3.0, 0, 0, 0]),
                          character=STRESS)
    assert np.allclose(voigt(2e6, th, ph), sph(2e6, th, ph))


def test_analytic_shape_refusals():
    bad = AnalyticField(IV, lambda r, t, p: np.ones(4), character=VECTOR)
    with pytest.raises(ValueError, match="trailing shape"):
        bad(2e6, 0.1, 0.2)
    wrong_points = AnalyticField(IV, lambda r, t, p: np.ones((7, 3)), character=VECTOR)
    with pytest.raises(ValueError, match="should return"):
        wrong_points(np.linspace(*IV, 5), 0.1, 0.2)
    with pytest.raises(ValueError, match="unknown frame"):
        AnalyticField(IV, lambda r, t, p: r, frame="polar")


# ------------------------------------------------------------- composed

def test_composed_field_is_pointwise_and_never_sampled():
    calls = []

    def bulk(rho_, vp_):
        calls.append(rho_.shape)
        return rho_ * vp_ ** 2

    c = ComposedField(bulk, (rho(), vp()), character=Character(0, 1), name="kappa")
    r = r_in(7)
    assert np.allclose(c(r), rho()(r) * vp()(r) ** 2)
    assert calls == [(7,)] and c.is_radial and c.name == "kappa"
    with pytest.raises(ValueError, match="at least one"):
        ComposedField(bulk, (), character=SCALAR)
    with pytest.raises(TypeError, match="not a Field"):
        ComposedField(bulk, (rho(), np.sin), character=SCALAR)
    with pytest.raises(ValueError, match="different intervals"):
        ComposedField(bulk, (rho(), RadialField((0.0, 1.0), 1.0)), character=SCALAR)


# -------------------------------------------------------------- constants

def test_constant_field():
    c = constant_field(3.0, IV, name="c")
    assert c(r_in()).tolist() == [3.0] * 50 and c.function.is_zero() is False
    e = constant_field(np.eye(6), IV, character=ELASTIC)
    assert e(2e6, 0.1, 0.2).shape == (6, 6)
    with pytest.raises(ValueError, match="shape"):
        constant_field(1.0, IV, character=VECTOR)


# -------------------------------------------------------------- contracts

def _voigt_rank4():
    C = np.random.default_rng(1).normal(size=(6, 6))
    C = C + C.T
    return [[polynomial_layer([C[i, j], 0.1 * C[i, j]], IV, scale=A) for j in range(6)]
            for i in range(6)]


@pytest.mark.parametrize("field", [
    rho(), vp(), rho() * vp() ** 2,
    RadialField(IV, lambda r: np.sin(r / A), name="numeric"),
    RadialField(IV, [polynomial_layer([1.0, 1.0], IV, scale=A), 0.0, 2.0],
                character=VECTOR),
    RadialField(IV, _voigt_rank4(), character=ELASTIC),
    constant_field(np.array([1.0, 2.0, 3.0, 0, 0, 0]), IV, character=STRESS),
    AnalyticField(IV, lambda r, t, p: r * np.cos(t)),
    AnalyticField(IV, lambda r, t, p: np.stack([r, np.sin(t), np.cos(p)], axis=-1),
                  character=VECTOR),
    AnalyticField(IV, lambda r, t, p: np.diag([1.0, 2.0, 3.0]), character=STRESS,
                  frame="cartesian"),
    AnalyticField(IV, lambda r, t, p: np.eye(6) * r[..., None, None],
                  character=ELASTIC),
    ComposedField(lambda a, b: np.sqrt(np.abs(a * b)), (rho(), vp()), character=SCALAR),
    vp() * AnalyticField(IV, lambda r, t, p: np.cos(t)),
    2.0 * AnalyticField(IV, lambda r, t, p: r * np.cos(t)),
])
def test_shipped_fields_pass_the_contract(field):
    check_field(field)


def test_protocol_is_structural():
    assert isinstance(rho(), Field)
    assert not isinstance(np.sin, Field)
