"""The layer-function protocol and the exact polynomial product.

The oracle for multiply_layer_functions is the pointwise product itself:
whatever the implementation does with coefficients, evaluating the
result must agree with f(r) * g(r) to machine precision, and its
integral must agree with quadrature.  That is what makes the exactness
claim testable rather than asserted.
"""
import warnings

import numpy as np
import pytest
from scipy.interpolate import CubicSpline, PPoly

from planetmodel import PREM, polynomial_layer
from planetmodel.model.fields.layer_function import (LayerFunction, as_layer_function,
                                           multiply_layer_functions)
from planetmodel.testing import check_layer_function


# ----------------------------------------------------------- exact products

@pytest.mark.parametrize("seed", range(5))
def test_single_piece_polynomial_product_is_exact(seed):
    """The oracle: coefficients out, pointwise product in."""
    rng = np.random.default_rng(seed)
    lo, hi = 1.0e3, 6.371e6
    f = polynomial_layer(rng.normal(size=4), (lo, hi), scale=hi)
    g = polynomial_layer(rng.normal(size=3), (lo, hi), scale=hi)

    fg = multiply_layer_functions(f, g)
    r = np.linspace(lo, hi, 1000)
    want = np.asarray(f(r)) * np.asarray(g(r))
    scale = np.max(np.abs(want))
    assert np.allclose(fg(r), want, rtol=1e-14, atol=1e-14 * scale)


def test_product_degree_is_the_sum_of_degrees():
    """A cubic times a quadratic is a quintic, exactly."""
    f = polynomial_layer([0.0, 0.0, 0.0, 1.0], (1.0, 2.0))   # r^3
    g = polynomial_layer([0.0, 0.0, 1.0], (1.0, 2.0))        # r^2
    fg = multiply_layer_functions(f, g)
    assert isinstance(fg, PPoly)
    assert fg.c.shape[0] == 6                                 # degree 5
    r = np.linspace(1.0, 2.0, 101)
    assert np.allclose(fg(r), r ** 5, rtol=1e-14)


def test_product_integral_matches_quadrature():
    """.integrate on the product agrees with fine quadrature."""
    lo, hi = 0.5, 3.0
    f = polynomial_layer([1.0, 2.0, -0.5], (lo, hi))
    g = polynomial_layer([0.0, 1.5, 0.25], (lo, hi))
    fg = multiply_layer_functions(f, g)
    fine = np.linspace(lo, hi, 200001)
    want = np.trapezoid(np.asarray(f(fine)) * np.asarray(g(fine)), fine)
    assert abs(fg.integrate(lo, hi) - want) < 1e-9 * abs(want)


def test_multi_piece_ppolys_multiply_piecewise():
    """Shared breakpoints multiply exactly, piece by piece."""
    x = np.linspace(0.0, 4.0, 5)
    f = CubicSpline(x, np.sin(x))
    g = CubicSpline(x, np.exp(-x))
    fg = multiply_layer_functions(f, g)
    r = np.linspace(0.0, 4.0, 777)
    assert np.allclose(fg(r), f(r) * g(r), rtol=1e-13, atol=1e-13)


def test_the_product_is_still_a_layer_function():
    f = polynomial_layer([1.0, 1.0], (1.0, 2.0))
    check_layer_function(multiply_layer_functions(f, f), (1.0, 2.0))


# ------------------------------------------------------- inexact, and loud

def test_mismatched_breakpoints_warn_and_name_the_operands():
    f = CubicSpline(np.linspace(0.0, 1.0, 5), np.arange(5.0))
    g = CubicSpline(np.linspace(0.0, 1.0, 7), np.arange(7.0))
    with pytest.warns(UserWarning, match="rho.*vp|approximation"):
        fg = multiply_layer_functions(f, g, names=("rho", "vp"))
    r = np.linspace(0.05, 0.95, 50)
    assert np.allclose(fg(r), f(r) * g(r), rtol=1e-3)


def test_non_overlapping_operands_raise():
    f = polynomial_layer([1.0], (0.0, 1.0))
    g = polynomial_layer([1.0], (2.0, 3.0))
    with pytest.raises(ValueError, match="do not overlap"):
        multiply_layer_functions(f, g)


def test_opaque_callables_refuse_rather_than_guess():
    with pytest.raises(ValueError, match="cannot infer an interval"):
        multiply_layer_functions(lambda r: r, lambda r: r)


# ---------------------------------------------------------------- adapters

def test_as_layer_function_leaves_exact_functions_alone():
    f = polynomial_layer([1.0, 2.0], (1.0, 2.0))
    assert as_layer_function(f) is f


def test_as_layer_function_supplies_derivative_and_integral():
    f = as_layer_function(lambda r: np.sin(np.asarray(r, dtype=float)))
    r = np.linspace(0.2, 3.0, 40)
    assert np.allclose(f(r), np.sin(r))
    assert np.allclose(f.derivative()(r), np.cos(r), atol=1e-6)
    assert np.allclose(f.derivative(2)(r), -np.sin(r), atol=1e-3)
    assert abs(f.integrate(0.0, np.pi) - 2.0) < 1e-8


def test_as_layer_function_result_satisfies_the_contract():
    check_layer_function(as_layer_function(lambda r: np.asarray(r) ** 2),
                         (1.0, 2.0))


def test_as_layer_function_rejects_non_callables():
    with pytest.raises(TypeError, match="callable"):
        as_layer_function(42)


def test_protocol_is_structural():
    assert isinstance(polynomial_layer([1.0], (0.0, 1.0)), LayerFunction)
    assert isinstance(lambda r: r, LayerFunction)


# ------------------------------------------------- the reason this exists

def test_prem_moduli_products_stay_exact():
    """rho * vpv^2, computed on coefficients, is exact for PREM.

    This is the property the whole module exists to protect: PREM's
    fields are exact polynomials, so the moduli built from them must be
    exact polynomials too, not resampled approximations.
    """
    prem = PREM()
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any refit here is a failure
        for i in range(prem.skeleton.nlayers):
            rho, v = prem.rho[i], prem.vpv[i]
            v2 = multiply_layer_functions(v, v)
            c = multiply_layer_functions(rho, v2)
            lo, hi = prem.skeleton.interval(i)
            r = np.linspace(lo, hi, 500)
            want = np.asarray(rho(r)) * np.asarray(v(r)) ** 2
            assert np.allclose(c(r), want, rtol=1e-13,
                               atol=1e-13 * np.max(np.abs(want)))


# --------------------------------------------------- linear combinations

def test_combine_of_polynomials_is_exact():
    from planetmodel.model.fields.layer_function import combine_layer_functions
    f = polynomial_layer([1.0, 2.0, 3.0], (1.0, 2.0))
    g = polynomial_layer([0.0, -1.0], (1.0, 2.0))
    h = combine_layer_functions([(2.0, f), (-0.5, g)])
    assert isinstance(h, PPoly)
    r = np.linspace(1.0, 2.0, 501)
    assert np.allclose(h(r), 2.0 * f(r) - 0.5 * g(r), rtol=1e-14)


def test_combine_handles_a_repeated_operand():
    """kappa and mu may be the same object in a degenerate medium."""
    from planetmodel.model.fields.layer_function import combine_layer_functions
    f = polynomial_layer([1.0, 1.0], (1.0, 2.0))
    h = combine_layer_functions([(1.0, f), (4.0 / 3.0, f)])
    r = np.linspace(1.0, 2.0, 51)
    assert np.allclose(h(r), (1.0 + 4.0 / 3.0) * f(r), rtol=1e-14)


def test_combine_of_mixed_operands_is_pointwise_exact():
    from planetmodel.model.fields.layer_function import combine_layer_functions
    f = polynomial_layer([1.0, 1.0], (0.5, 2.0))
    g = as_layer_function(lambda r: np.sin(np.asarray(r, dtype=float)))
    h = combine_layer_functions([(1.0, f), (2.0, g)])
    r = np.linspace(0.6, 1.9, 40)
    assert np.allclose(h(r), f(r) + 2.0 * np.sin(r), rtol=1e-13)


def test_combine_needs_terms():
    from planetmodel.model.fields.layer_function import combine_layer_functions
    with pytest.raises(ValueError, match="at least one"):
        combine_layer_functions([])
