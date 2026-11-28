"""Layer functions: the exact polynomial algebra, the exponential of a
polynomial and the logarithm it declares, and the numeric fallback.

The oracle for every exact claim is the pointwise operation itself:
whatever is done on coefficients must agree with f(r) * g(r), with the
derivative of the interpolant, with quadrature, to machine precision.
"""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, PPoly

from planetmodel.layerfunction import (ExponentialLayer, LayerFunction, NumericLayer,
                                       PolynomialLayer, as_layer_function,
                                       constant_layer, polynomial_fit,
                                       polynomial_layer, same_interval)
from planetmodel.testing import check_layer_function

IV = (1.0, 2.0)


def _poly(seed, n=4, interval=IV):
    rng = np.random.default_rng(seed)
    return polynomial_layer(interval, rng.normal(size=n), scale=interval[1])


# ------------------------------------------------------------- construction

def test_polynomial_layer_is_the_prem_form():
    a = 6371e3
    f = polynomial_layer((0.0, 1221.5e3), [13.0885, 0.0, -8.8381], scale=a)
    r = np.linspace(0.0, 1221.5e3, 7)
    assert np.allclose(f(r), 13.0885 - 8.8381 * (r / a) ** 2, rtol=1e-14)
    assert f.degree == 2 and f.interval == (0.0, 1221.5e3)


def test_polynomial_layer_refusals():
    with pytest.raises(ValueError, match="increase"):
        polynomial_layer((2.0, 1.0), [1.0])
    with pytest.raises(ValueError, match="coefficient"):
        polynomial_layer(IV, [])
    with pytest.raises(ValueError, match="scale"):
        polynomial_layer(IV, [1.0], scale=0.0)
    with pytest.raises(TypeError, match="PPoly"):
        PolynomialLayer(IV, lambda r: r)


def test_constant_layer_is_exact_and_flat():
    c = constant_layer(IV, 2.5)
    assert c.degree == 0 and np.all(c(np.linspace(*IV, 9)) == 2.5)
    assert c.derivative().is_zero() and not c.is_zero()
    assert constant_layer(IV, 0.0).is_zero()


def test_spline_and_ppoly_are_accepted():
    x = np.linspace(0.0, 4.0, 5)
    s = CubicSpline(x, np.sin(x))
    f = as_layer_function((0.0, 4.0), s)
    assert isinstance(f, PolynomialLayer)
    assert np.allclose(f(np.linspace(0, 4, 33)), s(np.linspace(0, 4, 33)))
    g = PolynomialLayer((0.0, 4.0), PPoly(s.c, s.x))
    assert g.interval == (0.0, 4.0)


def test_as_layer_function_adapts_everything():
    f = _poly(0)
    assert as_layer_function(IV, f) is f
    assert as_layer_function((0.5, 3.0), f).interval == (0.5, 3.0)
    assert isinstance(as_layer_function(IV, 3.0), PolynomialLayer)
    n = as_layer_function(IV, np.sin)
    assert isinstance(n, NumericLayer) and n.fn is np.sin
    with pytest.raises(TypeError, match="callable"):
        as_layer_function(IV, "rho")


def test_protocol_is_structural():
    assert isinstance(_poly(0), LayerFunction)
    assert isinstance(NumericLayer(IV, np.sin), LayerFunction)
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
    f = polynomial_layer(IV, [0.0, 0.0, 0.0, 1.0])
    g = polynomial_layer(IV, [0.0, 0.0, 1.0])
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
    a = PolynomialLayer((0.0, 4.0), CubicSpline(x5, np.sin(x5)))
    b = PolynomialLayer((0.0, 4.0), CubicSpline(x7, np.exp(-x7)))
    r = np.linspace(0.0, 4.0, 777)
    ab = a * b
    assert isinstance(ab, PolynomialLayer)
    assert np.allclose(ab(r), a(r) * b(r), rtol=1e-13, atol=1e-13)
    assert np.allclose((a + b)(r), a(r) + b(r), rtol=1e-13, atol=1e-13)


def test_product_integral_matches_quadrature():
    f = polynomial_layer((0.5, 3.0), [1.0, 2.0, -0.5])
    g = polynomial_layer((0.5, 3.0), [0.0, 1.5, 0.25])
    fine = np.linspace(0.5, 3.0, 200001)
    want = np.trapezoid(f(fine) * g(fine), fine)
    assert abs((f * g).integrate(0.5, 3.0) - want) < 1e-9 * abs(want)


def test_derivative_and_integral_are_exact():
    f = polynomial_layer(IV, [1.0, 2.0, 3.0])             # 1 + 2r + 3r^2
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


def test_a_number_divided_by_a_polynomial_goes_numeric():
    g = _poly(3)
    r = np.linspace(*IV, 21)
    for c in (1.0, 3, np.float64(2.5), 1.0 + 2.0j):
        h = c / g
        assert isinstance(h, NumericLayer) and h.interval == g.interval
        assert np.allclose(h(r), c / g(r), rtol=1e-14)
    with pytest.raises(TypeError):
        "one" / g


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
    assert constant_layer(IV, 0.0).rescaled(k=3.0, v=2.0).is_zero()


# ------------------------------------------------------- beyond the interval

def test_on_interval_continues_the_polynomial():
    f = polynomial_layer(IV, [0.0, 1.0])                  # r
    g = f.on_interval(0.0, 5.0)
    assert g.interval == (0.0, 5.0) and g(4.0) == 4.0 and g(0.0) == 0.0
    assert f(np.array([0.0, 5.0])).tolist() == [0.0, 5.0]   # continues silently too


def test_numeric_on_interval_keeps_the_callable():
    n = NumericLayer(IV, np.sin)
    m = n.on_interval(0.0, 3.0)
    assert m.fn is np.sin and m.interval == (0.0, 3.0)


# ------------------------------------------------------------- the numeric kind

def test_numeric_layer_calculus_is_honest():
    n = NumericLayer((0.1, 3.0), np.sin)
    r = np.linspace(0.2, 2.9, 40)
    assert np.allclose(n.derivative()(r), np.cos(r), atol=1e-6)
    assert np.allclose(n.derivative(nu=2)(r), -np.sin(r), atol=1e-3)
    assert abs(n.integrate(0.0, np.pi) - 2.0) < 1e-8
    exact = NumericLayer((0.1, 3.0), np.sin, derivative=np.cos)
    assert np.allclose(exact.derivative()(r), np.cos(r), rtol=1e-15)


def test_numeric_layer_broadcasts_a_constant_callable():
    n = NumericLayer(IV, lambda r: 2.0)
    assert n(np.linspace(*IV, 5)).shape == (5,)
    assert n(1.5).shape == ()


def test_mixed_arithmetic_is_pointwise_exact_and_numeric():
    f = _poly(5, interval=(0.5, 2.0))
    g = NumericLayer((0.5, 2.0), np.sin)
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
    n = NumericLayer((0.1, 3.0), np.sin, derivative=np.cos)
    g = n.rescaled(k=2.0, v=3.0)
    r = np.linspace(0.2, 2.9, 20)
    assert np.allclose(g(2.0 * r), 3.0 * np.sin(r))
    assert np.allclose(g.derivative()(2.0 * r), 1.5 * np.cos(r))
    assert g.interval == (0.2, 6.0)


# ------------------------------------------------------------------ fitting

def test_polynomial_fit_recovers_a_polynomial_exactly():
    f = _poly(6)
    g = polynomial_fit(IV, f, degree=3)
    assert np.allclose(g.ppoly.c, f.ppoly.c, rtol=1e-12, atol=1e-12)


def test_polynomial_fit_of_a_smooth_function_converges():
    r = np.linspace(*IV, 101)
    errors = [np.max(np.abs(polynomial_fit(IV, np.exp, degree=d)(r) - np.exp(r)))
              for d in (2, 4, 8)]
    assert errors[0] > errors[1] > errors[2] and errors[2] < 1e-8
    with pytest.raises(ValueError, match="more points"):
        polynomial_fit(IV, np.exp, degree=3, n=3)


# ---------------------------------------------------------------- contracts

def test_same_interval_is_relative():
    assert same_interval((0.0, 1.0), (0.0, 1.0 + 1e-12))
    assert not same_interval((0.0, 1.0), (0.0, 1.0 + 1e-6))


@pytest.mark.parametrize("fn", [
    _poly(7), constant_layer(IV, 1.0), _poly(8) * _poly(9),
    NumericLayer(IV, np.exp), NumericLayer(IV, np.exp, derivative=np.exp),
    _poly(7) + NumericLayer(IV, np.exp), polynomial_fit(IV, np.exp, degree=6),
    ExponentialLayer(IV, _poly(3), factor=2.5),
    ExponentialLayer(IV, _poly(4), factor=_poly(5)),
])
def test_shipped_layer_functions_pass_the_contract(fn):
    check_layer_function(fn)


def test_a_fit_on_a_thin_shell_far_from_the_centre_keeps_its_digits():
    """The coefficients are held about the shell's inner radius."""
    from planetmodel.layerfunction import polynomial_fit

    lo, hi = 6346.6e3, 6356e3

    def f(r):
        return 3.0 + 0.5 * np.tanh((r - 6351e3) / 2e3)

    r = np.linspace(lo, hi, 101)
    n = 64
    rc = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.pi * (np.arange(n) + 0.5) / n)
    for degree in (5, 7, 9):
        fit = polynomial_fit((lo, hi), f, degree=degree, n=n)
        # the same least squares, evaluated in the scaled window
        same = np.polynomial.Polynomial.fit(rc, f(rc), degree, domain=[lo, hi])
        assert np.max(np.abs(fit(r) - same(r))) < 1e-10


def test_numeric_derivatives_stay_in_the_interval_and_scale_their_step():
    def strict(r):
        r = np.asarray(r)
        if np.any((r < 0.0) | (r > 1.0)):
            raise ValueError("outside")
        return np.sin(r)

    f = NumericLayer((0.0, 1.0), strict)
    r = np.array([0.0, 0.5, 1.0])
    assert np.allclose(f.derivative()(r), np.cos(r), atol=1e-8)
    assert np.allclose(f.derivative(nu=2)(r), -np.sin(r), atol=1e-6)
    assert np.allclose(f.derivative(nu=3)(r), -np.cos(r), atol=1e-4)


def test_division_with_numeric_layers_and_a_relative_quadrature():
    p = polynomial_layer((0.0, 1.0), [1.0, 2.0])
    n = NumericLayer((0.0, 1.0), lambda r: 1.0 + np.asarray(r) ** 2)
    assert np.isclose((p / n)(0.5), 2.0 / 1.25)
    assert np.isclose((1j / n)(0.5), 1j / 1.25)
    tiny = NumericLayer((0.0, 1.0), lambda r: 1e-12 * np.exp(-((np.asarray(r) - 0.5)
                                                               / 0.01) ** 2))
    exact = 1e-12 * 0.01 * np.sqrt(np.pi)
    assert np.isclose(tiny.integrate(0.0, 1.0), exact, rtol=1e-8)


# ------------------------------------------------- the exponential of a polynomial

def _spline_of_logs():
    """A cubic spline through the logarithms of values spanning six decades,
    as a deck's column joined by `log-cubic` holds it."""
    r = np.array([1.0, 1.2, 1.45, 1.6, 2.0])
    return PolynomialLayer(IV, CubicSpline(r, np.log(10.0) * np.array([0.0, 3.0, 1.0,
                                                                         6.0, 4.0])))


def test_an_exponential_layer_is_its_factor_times_the_exponential():
    p, q = _spline_of_logs(), _poly(11)
    f = ExponentialLayer(IV, p, factor=q)
    r = np.linspace(*IV, 37)
    assert np.allclose(f(r), q(r) * np.exp(p(r)), rtol=1e-15, atol=0)
    assert f(1.5).shape == () and f(r.reshape(-1, 1)).shape == (37, 1)
    assert f.exponent.interval == f.factor.interval == IV
    assert ExponentialLayer(IV, p.ppoly)(1.3) == np.exp(p(1.3))
    with pytest.raises(TypeError, match="a constant one is a factor"):
        ExponentialLayer(IV, 2.0)
    with pytest.raises(ValueError, match="real"):
        ExponentialLayer(IV, _poly(1) * 1j)


def test_exponential_derivatives_are_exact_to_every_order():
    """(q' + q p') exp(p) each time, against a central difference of the one
    before and against the closed second derivative."""
    p, q = _spline_of_logs(), _poly(12)
    f = ExponentialLayer(IV, p, factor=q)
    x = np.array([1.05, 1.3, 1.5, 1.7, 1.95])
    h = 1e-6
    for nu in (1, 2, 3):
        d, before = f.derivative(nu=nu), f.derivative(nu=nu - 1)
        assert isinstance(d, ExponentialLayer)
        fd = (before(x + h) - before(x - h)) / (2.0 * h)
        assert np.allclose(d(x), fd, rtol=1e-6, atol=0), nu
    dp, ddp = p.derivative()(x), p.derivative(nu=2)(x)
    dq, ddq = q.derivative()(x), q.derivative(nu=2)(x)
    closed = (ddq + 2.0 * dq * dp + q(x) * (ddp + dp * dp)) * np.exp(p(x))
    assert np.allclose(f.derivative(nu=2)(x), closed, rtol=1e-13, atol=0)
    assert f.derivative(nu=0) is f
    with pytest.raises(ValueError):
        f.derivative(nu=-1)


def test_exponential_integrals_are_exact_on_lines_and_quadrature_elsewhere():
    """A line in the exponent and a constant factor are integrated in closed
    form piece by piece, the rest by quadrature: both against a 40-point
    Gauss-Legendre rule on every piece, signed, and a derivative's integral
    against the difference of the function."""
    knots = np.array([1.0, 1.1, 1.35, 1.6, 2.0])
    logs = np.log(np.array([1e20, 1e23, 1e21, 1e24, 1e19]) / 1e20)
    line = PolynomialLayer(IV, PPoly(np.vstack([np.diff(logs) / np.diff(knots),
                                                logs[:-1]]), knots))
    x, w = np.polynomial.legendre.leggauss(40)
    lines = ExponentialLayer(IV, line, factor=1e20)
    for f in (lines, ExponentialLayer(IV, line, factor=3.0).derivative(),
              ExponentialLayer(IV, _spline_of_logs(), factor=_poly(2))):
        for a, b in ((1.0, 2.0), (1.03, 1.52), (1.7, 1.2)):
            lo, hi = min(a, b), max(a, b)
            breaks = np.union1d(f.exponent.ppoly.x, f.factor.ppoly.x)
            inside = breaks[(breaks > lo) & (breaks < hi)]
            cuts = np.unique(np.concatenate([[lo, hi], inside]))
            parts = [0.5 * (v - u) * np.sum(w * f(0.5 * (v - u) * x + 0.5 * (u + v)))
                     for u, v in zip(cuts[:-1], cuts[1:])]
            want = sum(parts) if b >= a else -sum(parts)
            scale = sum(abs(part) for part in parts)    # the rule rounds to ~2e-14
            assert abs(f.integrate(a, b) - want) <= 5e-14 * scale, (a, b)
    d = lines.derivative()
    for a, b in ((1.03, 1.52), (1.0, 2.0)):
        scale = sum(abs(d.integrate(u, v)) for u, v in zip(knots[:-1], knots[1:]))
        assert abs(d.integrate(a, b) - (lines(b) - lines(a))) <= 1e-14 * scale


def test_exponential_algebra_stays_exponential_where_it_can():
    p, q = _spline_of_logs(), _poly(13)
    f, g = ExponentialLayer(IV, p, factor=2.0), ExponentialLayer(IV, -0.5 * p, factor=q)
    poly = _poly(14)
    r = np.linspace(*IV, 29)
    exact = [(f * g, f(r) * g(r)), (f * poly, f(r) * poly(r)),
             (poly * f, f(r) * poly(r)), (3.0 * f, 3.0 * f(r)), (f / 4.0, f(r) / 4.0),
             (-f, -f(r)), (f ** 3, f(r) ** 3), (1.0 / f, 1.0 / f(r)),
             (g / f, g(r) / f(r))]
    for h, want in exact:
        assert isinstance(h, ExponentialLayer)
        assert np.allclose(h(r), want, rtol=1e-13, atol=0)
    numeric = [(f + g, f(r) + g(r)), (f - poly, f(r) - poly(r)),
               (poly + f, f(r) + poly(r)), (poly - f, poly(r) - f(r)),
               (f + 1.0, f(r) + 1.0), (2.0 - f, 2.0 - f(r)), (f / g, f(r) / g(r)),
               (1.0 / g, 1.0 / g(r)), (poly / f, poly(r) / f(r)),
               (f * NumericLayer(IV, np.sin), f(r) * np.sin(r))]
    for h, want in numeric:
        assert isinstance(h, NumericLayer)
        assert np.allclose(h(r), want, rtol=1e-13, atol=0)
    with pytest.raises(ValueError, match="different intervals"):
        f * ExponentialLayer((1.0, 3.0), _poly(1, interval=(1.0, 3.0)))
    with pytest.raises(ValueError, match="non-negative integer"):
        f ** -1


def test_exponential_rescaled_and_on_interval():
    p, q = _spline_of_logs(), _poly(15)
    f = ExponentialLayer(IV, p, factor=q)
    g = f.rescaled(k=3.0, v=-2.0)
    r = np.linspace(*IV, 13)
    assert g.interval == (3.0, 6.0)
    assert np.allclose(g(3.0 * r), -2.0 * f(r), rtol=1e-13, atol=0)
    wider = f.on_interval(0.5, 2.5)
    assert wider.interval == (0.5, 2.5)
    assert np.allclose(wider(r), f(r), rtol=1e-15, atol=0)
    assert np.isfinite(wider(np.array([0.6, 2.4]))).all()
    assert isinstance(as_layer_function(IV, f), ExponentialLayer)
    assert isinstance(f, LayerFunction)


def test_a_logarithm_that_is_a_polynomial_is_declared_and_no_other():
    """An exponential layer of a positive constant factor, on every piece,
    has log q + p, its exponential the function; so has a polynomial layer
    that is a positive constant on every piece.  A factor that varies or
    is not positive, and a polynomial that varies, has none."""
    p = _spline_of_logs()
    r = np.linspace(*IV, 31)
    f = ExponentialLayer(IV, p, factor=2.5e20)
    log = f.logarithm()
    assert isinstance(log, PolynomialLayer) and log.degree == 3
    assert np.allclose(log(r), np.log(2.5e20) + p(r), rtol=1e-15, atol=0)
    assert np.allclose(np.exp(log(r)), f(r), rtol=1e-13, atol=0)
    assert np.array_equal(log.ppoly.x, p.ppoly.x)
    for g in (f * f, 1.0 / f, f / 3.0, f.rescaled(k=2.0, v=4.0),
              ExponentialLayer(IV, p, factor=PPoly([[1.0, 3.0]], [1.0, 1.3, 2.0]))):
        h = g.logarithm()
        x = np.linspace(*g.interval, 31)
        assert h is not None and np.allclose(np.exp(h(x)), g(x), rtol=1e-12, atol=0)
        check_layer_function(g)
    for g in (ExponentialLayer(IV, p, factor=_poly(5)), -f,
              ExponentialLayer(IV, p, factor=PPoly([[1.0, -3.0]], [1.0, 1.3, 2.0])),
              ExponentialLayer(IV, p, factor=1.0 + 2.0j)):
        assert g.logarithm() is None
    c = constant_layer(IV, 7.0).logarithm()
    assert c.degree == 0 and np.array_equal(c(r), np.full(r.size, np.log(7.0)))
    for g in (_poly(6), constant_layer(IV, 0.0), constant_layer(IV, -1.0),
              constant_layer(IV, 2.0 + 1.0j)):
        assert g.logarithm() is None
    assert not hasattr(NumericLayer(IV, np.exp), "logarithm")
