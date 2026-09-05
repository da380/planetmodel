"""Layer functions: the exact polynomial algebra, and the numeric fallback.

The oracle for every exact claim is the pointwise operation itself:
whatever is done on coefficients must agree with f(r) * g(r), with the
derivative of the interpolant, with quadrature, to machine precision.
"""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, PPoly

from planetmodel.layerfunction import (LayerFunction, NumericLayer, PolynomialLayer,
                                       as_layer_function, constant_layer,
                                       polynomial_fit, polynomial_layer,
                                       same_interval)
from planetmodel.testing import check_layer_function

IV = (1.0, 2.0)


def _poly(seed, n=4, interval=IV):
    rng = np.random.default_rng(seed)
    return polynomial_layer(rng.normal(size=n), interval, scale=interval[1])


# ------------------------------------------------------------- construction

def test_polynomial_layer_is_the_prem_form():
    a = 6371e3
    f = polynomial_layer([13.0885, 0.0, -8.8381], (0.0, 1221.5e3), scale=a)
    r = np.linspace(0.0, 1221.5e3, 7)
    assert np.allclose(f(r), 13.0885 - 8.8381 * (r / a) ** 2, rtol=1e-14)
    assert f.degree == 2 and f.interval == (0.0, 1221.5e3)


def test_polynomial_layer_refusals():
    with pytest.raises(ValueError, match="increase"):
        polynomial_layer([1.0], (2.0, 1.0))
    with pytest.raises(ValueError, match="coefficient"):
        polynomial_layer([], IV)
    with pytest.raises(ValueError, match="scale"):
        polynomial_layer([1.0], IV, scale=0.0)
    with pytest.raises(TypeError, match="PPoly"):
        PolynomialLayer(lambda r: r)


def test_constant_layer_is_exact_and_flat():
    c = constant_layer(2.5, IV)
    assert c.degree == 0 and np.all(c(np.linspace(*IV, 9)) == 2.5)
    assert c.derivative().is_zero() and not c.is_zero()
    assert constant_layer(0.0, IV).is_zero()


def test_spline_and_ppoly_are_accepted():
    x = np.linspace(0.0, 4.0, 5)
    s = CubicSpline(x, np.sin(x))
    f = as_layer_function(s, (0.0, 4.0))
    assert isinstance(f, PolynomialLayer)
    assert np.allclose(f(np.linspace(0, 4, 33)), s(np.linspace(0, 4, 33)))
    g = PolynomialLayer(PPoly(s.c, s.x))
    assert g.interval == (0.0, 4.0)


def test_as_layer_function_adapts_everything():
    f = _poly(0)
    assert as_layer_function(f, IV) is f
    assert as_layer_function(f, (0.5, 3.0)).interval == (0.5, 3.0)
    assert isinstance(as_layer_function(3.0, IV), PolynomialLayer)
    n = as_layer_function(np.sin, IV)
    assert isinstance(n, NumericLayer) and n.fn is np.sin
    with pytest.raises(TypeError, match="callable"):
        as_layer_function("rho", IV)


def test_protocol_is_structural():
    assert isinstance(_poly(0), LayerFunction)
    assert isinstance(NumericLayer(np.sin, IV), LayerFunction)
    assert not isinstance(np.sin, LayerFunction)


# ---------------------------------------------------------- the exact algebra

@pytest.mark.parametrize("seed", range(4))
def test_products_and_sums_are_pointwise_exact(seed):
    f, g = _poly(seed, 4), _poly(seed + 10, 3)
    r = np.linspace(*IV, 1000)
    for op in (lambda a, b: a * b, lambda a, b: a + b, lambda a, b: a - b):
        want = op(f(r), g(r))
        got = op(f, g)
        assert isinstance(got, PolynomialLayer)
        assert np.allclose(got(r), want, rtol=1e-14,
                           atol=1e-14 * np.max(np.abs(want)))


def test_product_degree_is_the_sum_of_degrees():
    f = polynomial_layer([0.0, 0.0, 0.0, 1.0], IV)
    g = polynomial_layer([0.0, 0.0, 1.0], IV)
    fg = f * g
    assert fg.degree == 5
    r = np.linspace(*IV, 101)
    assert np.allclose(fg(r), r ** 5, rtol=1e-14)
    assert (f ** 2).degree == 6 and np.allclose((f ** 2)(r), r ** 6, rtol=1e-14)


def test_scalar_arithmetic_stays_polynomial():
    f = _poly(1)
    r = np.linspace(*IV, 51)
    for g, want in ((2.0 * f, 2.0 * f(r)), (f * 2.0, 2.0 * f(r)),
                    (f / 4.0, f(r) / 4.0), (f + 1.5, f(r) + 1.5),
                    (1.5 + f, f(r) + 1.5), (f - 1.5, f(r) - 1.5),
                    (1.5 - f, 1.5 - f(r)), (-f, -f(r))):
        assert isinstance(g, PolynomialLayer)
        assert np.allclose(g(r), want, rtol=1e-14)


def test_mismatched_breakpoints_are_refined_to_the_union_exactly():
    x5, x7 = np.linspace(0.0, 4.0, 5), np.linspace(0.0, 4.0, 7)
    a = PolynomialLayer(CubicSpline(x5, np.sin(x5)))
    b = PolynomialLayer(CubicSpline(x7, np.exp(-x7)))
    r = np.linspace(0.0, 4.0, 777)
    ab = a * b
    assert isinstance(ab, PolynomialLayer)
    assert np.allclose(ab(r), a(r) * b(r), rtol=1e-13, atol=1e-13)
    assert np.allclose((a + b)(r), a(r) + b(r), rtol=1e-13, atol=1e-13)


def test_product_integral_matches_quadrature():
    f = polynomial_layer([1.0, 2.0, -0.5], (0.5, 3.0))
    g = polynomial_layer([0.0, 1.5, 0.25], (0.5, 3.0))
    fine = np.linspace(0.5, 3.0, 200001)
    want = np.trapezoid(f(fine) * g(fine), fine)
    assert abs((f * g).integrate(0.5, 3.0) - want) < 1e-9 * abs(want)


def test_derivative_and_integral_are_exact():
    f = polynomial_layer([1.0, 2.0, 3.0], IV)             # 1 + 2r + 3r^2
    r = np.linspace(*IV, 11)
    assert np.allclose(f.derivative()(r), 2.0 + 6.0 * r, rtol=1e-14)
    assert np.allclose(f.derivative(nu=2)(r), 6.0, rtol=1e-14)
    assert f.derivative(nu=3).is_zero()
    assert f.derivative(nu=0) is f
    assert np.isclose(f.integrate(1.0, 2.0), (2 + 4 + 8) - (1 + 1 + 1), rtol=1e-14)
    assert f.integrate(2.0, 1.0) == -f.integrate(1.0, 2.0)


def test_division_by_a_polynomial_goes_numeric():
    f, g = _poly(2), _poly(3)
    h = f / g
    assert isinstance(h, NumericLayer)
    r = np.linspace(*IV, 21)
    assert np.allclose(h(r), f(r) / g(r), rtol=1e-14)


def test_different_intervals_are_refused():
    f = _poly(0)
    g = _poly(1, interval=(1.0, 3.0))
    with pytest.raises(ValueError, match="different intervals"):
        f + g
    with pytest.raises(ValueError, match="different intervals"):
        f * g


def test_powers_take_non_negative_integers():
    f = _poly(0)
    with pytest.raises(ValueError, match="non-negative integer"):
        f ** -1
    with pytest.raises(ValueError, match="non-negative integer"):
        f ** 0.5
    assert (f ** 0).degree == 0 and (f ** 0)(1.5) == 1.0


# -------------------------------------------------------------- rescaling

def test_rescaled_is_exact_and_round_trips():
    f = _poly(4)
    k, v = 6.371e6, 1e-3
    g = f.rescaled(k=k, v=v)
    r = np.linspace(*IV, 101)
    assert isinstance(g, PolynomialLayer)
    assert np.allclose(g(k * r), v * f(r), rtol=1e-14)
    assert np.allclose(g.interval, (k, 2.0 * k))
    back = g.rescaled(k=1.0 / k, v=1.0 / v)
    assert np.allclose(back.ppoly.c, f.ppoly.c, rtol=1e-14)
    with pytest.raises(ValueError, match="positive"):
        f.rescaled(k=0.0, v=1.0)


def test_rescaled_zero_stays_zero():
    assert constant_layer(0.0, IV).rescaled(k=3.0, v=2.0).is_zero()


# ------------------------------------------------------- beyond the interval

def test_on_interval_continues_the_polynomial():
    f = polynomial_layer([0.0, 1.0], IV)                  # r
    g = f.on_interval(0.0, 5.0)
    assert g.interval == (0.0, 5.0) and g(4.0) == 4.0 and g(0.0) == 0.0
    assert f(np.array([0.0, 5.0])).tolist() == [0.0, 5.0]   # continues silently too


def test_numeric_on_interval_keeps_the_callable():
    n = NumericLayer(np.sin, IV)
    m = n.on_interval(0.0, 3.0)
    assert m.fn is np.sin and m.interval == (0.0, 3.0)


# ------------------------------------------------------------- the numeric kind

def test_numeric_layer_calculus_is_honest():
    n = NumericLayer(np.sin, (0.1, 3.0))
    r = np.linspace(0.2, 2.9, 40)
    assert np.allclose(n.derivative()(r), np.cos(r), atol=1e-6)
    assert np.allclose(n.derivative(nu=2)(r), -np.sin(r), atol=1e-3)
    assert abs(n.integrate(0.0, np.pi) - 2.0) < 1e-8
    exact = NumericLayer(np.sin, (0.1, 3.0), derivative=np.cos)
    assert np.allclose(exact.derivative()(r), np.cos(r), rtol=1e-15)


def test_numeric_layer_broadcasts_a_constant_callable():
    n = NumericLayer(lambda r: 2.0, IV)
    assert n(np.linspace(*IV, 5)).shape == (5,)
    assert n(1.5).shape == ()


def test_mixed_arithmetic_is_pointwise_exact_and_numeric():
    f = _poly(5, interval=(0.5, 2.0))
    g = NumericLayer(np.sin, (0.5, 2.0))
    r = np.linspace(0.6, 1.9, 40)
    for h, want in ((f + 2.0 * g, f(r) + 2.0 * np.sin(r)),
                    (g + f, f(r) + np.sin(r)),
                    (f * g, f(r) * np.sin(r)),
                    (g * f, f(r) * np.sin(r)),
                    (g / f, np.sin(r) / f(r)),
                    (f - g, f(r) - np.sin(r)),
                    (g - f, np.sin(r) - f(r)),
                    (1.0 / g, 1.0 / np.sin(r)),
                    (g ** 2, np.sin(r) ** 2),
                    (-g, -np.sin(r))):
        assert isinstance(h, NumericLayer)
        assert np.allclose(h(r), want, rtol=1e-13)


def test_numeric_rescaled_is_v_f_of_r_over_k():
    n = NumericLayer(np.sin, (0.1, 3.0), derivative=np.cos)
    g = n.rescaled(k=2.0, v=3.0)
    r = np.linspace(0.2, 2.9, 20)
    assert np.allclose(g(2.0 * r), 3.0 * np.sin(r))
    assert np.allclose(g.derivative()(2.0 * r), 1.5 * np.cos(r))
    assert g.interval == (0.2, 6.0)


# ------------------------------------------------------------------ fitting

def test_polynomial_fit_recovers_a_polynomial_exactly():
    f = _poly(6)
    g = polynomial_fit(f, IV, degree=3)
    assert np.allclose(g.ppoly.c, f.ppoly.c, rtol=1e-12, atol=1e-12)


def test_polynomial_fit_of_a_smooth_function_converges():
    r = np.linspace(*IV, 101)
    errors = [np.max(np.abs(polynomial_fit(np.exp, IV, degree=d)(r) - np.exp(r)))
              for d in (2, 4, 8)]
    assert errors[0] > errors[1] > errors[2] and errors[2] < 1e-8
    with pytest.raises(ValueError, match="more points"):
        polynomial_fit(np.exp, IV, degree=3, n=3)


# ---------------------------------------------------------------- contracts

def test_same_interval_is_relative():
    assert same_interval((0.0, 1.0), (0.0, 1.0 + 1e-12))
    assert not same_interval((0.0, 1.0), (0.0, 1.0 + 1e-6))


@pytest.mark.parametrize("fn", [
    _poly(7), constant_layer(1.0, IV), _poly(8) * _poly(9),
    NumericLayer(np.exp, IV), NumericLayer(np.exp, IV, derivative=np.exp),
    _poly(7) + NumericLayer(np.exp, IV), polynomial_fit(np.exp, IV, degree=6),
])
def test_shipped_layer_functions_pass_the_contract(fn):
    check_layer_function(fn)
