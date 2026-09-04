"""Radial displacements: the protocol, the adapter and the contract."""
import numpy as np
import pytest

from planetmodel import (CallableDisplacement, RadialDisplacement, Skeleton,
                         ZeroDisplacement, as_displacement, testing)

SK = Skeleton([0.0, 0.5, 1.0])


def flattening(r, theta, phi):
    """h = -f r P2(cos theta): an oblate radial displacement."""
    return -0.05 * r * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)


def flattening_dr(r, theta, phi):
    return -0.05 * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0) + 0.0 * r


def flattening_grad(r, theta, phi):
    return (-0.05 * r * 0.5 * (-6.0 * np.cos(theta) * np.sin(theta)),
            np.zeros(np.broadcast(r, theta, phi).shape))


def kinked(r, theta, phi):
    """Zero below r = 0.5, growing linearly above it: a kink at 0.5."""
    return 0.02 * np.maximum(r - 0.5, 0.0) * np.cos(theta) ** 2


def test_zero_displacement_is_a_displacement():
    z = ZeroDisplacement()
    assert isinstance(z, RadialDisplacement)
    assert np.all(z(np.linspace(0, 1, 4), 0.3, 0.1) == 0.0)
    assert z.bounds() == (0.0, 0.0) and z.knots == ()
    testing.check_displacement(z, SK)


def test_callable_with_exact_derivatives():
    h = CallableDisplacement(flattening, radial_derivative=flattening_dr,
                             angular_gradient=flattening_grad, name="f")
    r, t, p = 0.7, 0.4, 1.1
    assert h(r, t, p) == pytest.approx(flattening(r, t, p))
    assert h.radial_derivative(r, t, p) == pytest.approx(flattening_dr(r, t, p))
    gt, _ = h.angular_gradient(r, t, p)
    assert gt == pytest.approx(flattening_grad(r, t, p)[0])
    assert "f" in repr(h)
    testing.check_displacement(h, SK)


def test_callable_with_differenced_derivatives():
    h = CallableDisplacement(flattening)
    r, t, p = 0.7, 0.4, 1.1
    assert h.radial_derivative(r, t, p) == pytest.approx(flattening_dr(r, t, p),
                                                          rel=1e-6)
    gt, gp = h.angular_gradient(r, t, p)
    want_t, want_p = flattening_grad(r, t, p)
    assert gt == pytest.approx(want_t, rel=1e-6)
    assert gp == pytest.approx(want_p, abs=1e-9)
    testing.check_displacement(h, SK)


def test_derivatives_carried_by_the_function_are_used():
    class WithDerivatives:
        def __call__(self, r, theta, phi):
            return flattening(r, theta, phi)

        def radial_derivative(self, r, theta, phi):
            return flattening_dr(r, theta, phi)

        def angular_gradient(self, r, theta, phi):
            return flattening_grad(r, theta, phi)

    h = CallableDisplacement(WithDerivatives())
    assert h.radial_derivative(0.3, 0.2, 0.1) == flattening_dr(0.3, 0.2, 0.1)


def test_declared_knot_passes_and_undeclared_kink_is_caught():
    testing.check_displacement(CallableDisplacement(kinked, knots=[0.5]), SK)
    with pytest.raises(AssertionError, match="not declared as a knot"):
        testing.check_displacement(CallableDisplacement(kinked), SK)


def test_a_displacement_multivalued_at_the_pole_is_caught():
    bad = CallableDisplacement(lambda r, t, p: 0.01 * r * np.cos(p))
    with pytest.raises(AssertionError, match="single-valued at the pole"):
        testing.check_displacement(bad, SK)


def test_as_displacement():
    bare = as_displacement(flattening, knots=[0.5])
    assert isinstance(bare, CallableDisplacement) and bare.knots == (0.5,)
    ready = ZeroDisplacement()
    assert as_displacement(ready) is ready
    with pytest.raises(ValueError, match="already declares"):
        as_displacement(ready, knots=[0.5])
    with pytest.raises(TypeError, match="callable"):
        CallableDisplacement(3.0)


def test_broadcasting():
    h = CallableDisplacement(flattening)
    r = np.linspace(0.1, 1.0, 4)[:, None, None]
    t = np.linspace(0.1, 3.0, 3)[None, :, None]
    p = np.linspace(-3.0, 3.0, 2)[None, None, :]
    assert h(r, t, p).shape == (4, 3, 2)
    assert h.radial_derivative(r, t, p).shape == (4, 3, 2)
