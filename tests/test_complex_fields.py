"""Fields and layer functions with complex values."""
import numpy as np
import pytest

from planetmodel import (DENSITY, VECTOR, AnalyticField, ComposedField, NumericLayer,
                         RadialField, RadialMesh, constant_field, gauss_legendre,
                         LayeredIsotropicElastic, polynomial_layer, sample, testing)
from planetmodel.layerfunction import as_scalar, as_values, constant_layer


def test_complex_layer_functions():
    iv = (0.5, 1.0)
    p = polynomial_layer([1.0, 2.0j, -0.5 + 0.25j], iv)
    testing.check_layer_function(p)
    r = np.linspace(0.5, 1.0, 5)
    assert p(r).dtype == np.complex128
    assert np.allclose(p(r), 1.0 + 2.0j * r + (-0.5 + 0.25j) * r ** 2)
    assert isinstance(p.integrate(0.5, 1.0), complex)
    q = polynomial_layer([3.0, 1.0], iv)
    for f in (p + q, p * q, q - p, p * 2.0j, q / (1.0 + 1.0j), p ** 2, -p):
        testing.check_layer_function(f)
    assert np.allclose((p * q)(r), p(r) * q(r))
    assert np.allclose((q + 1.0j)(r), q(r) + 1.0j)
    d = p.derivative()
    assert np.allclose(d(r), 2.0j + 2.0 * (-0.5 + 0.25j) * r)
    n = NumericLayer(lambda x: np.exp(1.0j * x), iv)
    testing.check_layer_function(n)
    assert isinstance(n.integrate(0.5, 1.0), complex)
    assert np.isclose(n.integrate(0.5, 1.0), (np.exp(1.0j) - np.exp(0.5j)) / 1.0j)
    assert constant_layer(2.0j, iv)(0.7) == 2.0j
    assert as_scalar(np.float64(1.0)) == 1.0 and as_scalar(1.0j) == 1.0j
    assert as_values([1, 2]).dtype == np.float64
    assert as_values(np.array([1.0j])).dtype == np.complex128


def test_complex_fields():
    iv = (0.5, 1.0)
    f = RadialField(iv, polynomial_layer([1.0, 2.0j], iv), name="z")
    testing.check_field(f)
    assert f.dtype == np.complex128 and f(0.75) == 1.0 + 1.5j
    g = constant_field(2.0 - 1.0j, iv)
    assert g.dtype == np.complex128
    h = constant_field([1.0j, 2.0, 3.0], iv, character=VECTOR)
    testing.check_field(h)
    assert h.evaluate(0.7, 1.0, 0.5).dtype == np.complex128
    real = constant_field(3.0, iv)
    assert real.dtype == np.float64
    for x in (f + g, f * real, real * 1.0j, f / real, real / 2.0j):
        testing.check_field(x)
        assert x.dtype == np.complex128
    a = AnalyticField(iv, lambda r, t, p: np.exp(1.0j * r) * np.cos(t))
    testing.check_field(a)
    assert a.dtype == np.complex128
    c = ComposedField(lambda u, v: u * v, [f, real], character=f.character)
    testing.check_field(c)
    assert np.allclose(c(0.8), f(0.8) * 3.0)


def test_complex_nodal_values_and_the_sampler():
    model = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0],
                                    vs=[1.0, 1.0])
    model = model.with_field(1, "mu", constant_field(1.0 + 0.1j, (0.5, 1.0),
                                                     character=DENSITY, name="mu"))
    model = model.with_field(0, "mu", constant_field(2.0, (0.0, 0.5),
                                                     character=DENSITY, name="mu"))
    mesh = RadialMesh(model, ngll=4, drmax=0.25)
    mu = mesh.nodal(model, "mu")
    assert mu.dtype == np.complex128
    assert np.all(mu[mesh.layer == 0] == 2.0)
    assert np.all(mu[mesh.layer == 1] == 1.0 + 0.1j)
    assert mesh.nodal(model, "rho").dtype == np.float64
    with pytest.raises(TypeError, match="complex"):
        sample(model, gauss_legendre(2), fields=["mu"])
