"""Fields that depend on frequency or on time.

A field has a `kind`: "static", "frequency" or "time".  A static field
is lifted to either dependent kind by wrapping it as one that ignores
the extra argument; a dependent field is composed from static fields
and a formula of `omega` (angular frequency, rad/s in an SI body) or
`t`; and `at_frequency`/`at_time` freeze it at one argument as a static
field again, which is what a sample, a push-forward or a file sees.
Values are complex by default -- the complex tensor is the object --
and the real or imaginary part is an explicit request.  A field says
where `omega` may lie: `omega_domain` "real" refuses a complex
frequency; "complex" accepts the Laplace variable `s = i omega`.

This script lifts, composes, freezes and checks one field of each kind.
"""
import numpy as np

from planetmodel import (SCALAR, Dimensions, RadialField, Skeleton, at_frequency,
                         at_time, lifted_to_frequency, lifted_to_time, testing)
from planetmodel.model.fields.frequency import ComposedFrequencyField
from planetmodel.model.fields.time import ComposedTimeField

sk = Skeleton([0.0, 1.0, 2.0])
mu = RadialField(sk, [lambda r: 6.0e10 + 0.0 * r, lambda r: 7.0e10 + 1e9 * r],
                 name="mu", character=SCALAR, dimensions=Dimensions.MODULUS)
r = np.array([0.5, 1.5])

# -- lifting -----------------------------------------------------------------
static_as_frequency = lifted_to_frequency(mu)
assert mu.kind == "static" and static_as_frequency.kind == "frequency"
assert static_as_frequency.omega_domain == "complex"      # nothing to refuse
values = static_as_frequency.evaluate(r, omega=3.0)
assert values.dtype == np.complex128                       # complex by default
assert np.allclose(values, mu.evaluate(r))                 # and equal to the source
assert np.allclose(static_as_frequency.evaluate(r, omega=3.0, part="real"),
                   mu.evaluate(r))

# -- composing: a Kelvin-Voigt-like modulus mu (1 + i omega tau) -------------
# The formula takes the argument first, then the operands' values.
tau = 2.0
kv = ComposedFrequencyField(lambda omega, m: m * (1.0 + 1j * omega * tau), [mu],
                            character=SCALAR, dimensions=Dimensions.MODULUS,
                            name="kv", omega_domain="real")
omega = 0.25
assert np.allclose(kv.evaluate(r, omega=omega), mu.evaluate(r) * (1 + 0.5j))
assert np.allclose(kv.evaluate(r, omega=omega, part="imag"), 0.5 * mu.evaluate(r))

# A real-domain field refuses a frequency off the real axis by name.
try:
    kv.evaluate(r, omega=1.0 + 1.0j)
except ValueError as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("omega_domain='real' must refuse a complex omega")
assert np.allclose(kv.evaluate(r, omega=complex(omega, 0.0)),   # real, spelt complex
                   kv.evaluate(r, omega=omega))

# `evaluate_with` is the same evaluation with the argument positional; it
# is what the views and the algebra call.
assert np.allclose(kv.evaluate_with(r, None, None, omega, layer=None,
                                    side="upper", frame="spherical"),
                   kv.evaluate(r, omega=omega))

# -- the algebra works on dependent fields too --------------------------------
mixed = kv + static_as_frequency                    # frequency plus frequency
assert mixed.kind == "frequency"
assert np.allclose(mixed.evaluate(r, omega=omega), mu.evaluate(r) * (2 + 0.5j))
scaled = 0.5 * kv
assert np.allclose(scaled.evaluate(r, omega=omega), 0.5 * kv.evaluate(r, omega=omega))

# -- freezing -----------------------------------------------------------------
frozen = at_frequency(kv, omega)                   # a static field, complex128
assert frozen.kind == "static"
assert frozen.evaluate(r).dtype == np.complex128
assert np.allclose(frozen.evaluate(r), kv.evaluate(r, omega=omega))
real_part = kv.at(omega, part="real")              # the same, spelt on the field
assert real_part.evaluate(r).dtype == np.float64
assert np.allclose(real_part.evaluate(r), mu.evaluate(r))
assert np.isclose(frozen.restricted(1)(1.5), kv.evaluate(1.5, omega=omega))

# -- time: a relaxation function mu exp(-t / tau) -----------------------------
relax = ComposedTimeField(lambda t, m: m * np.exp(-t / tau), [mu],
                          character=SCALAR, dimensions=Dimensions.MODULUS,
                          name="relaxation")
assert relax.kind == "time"
assert np.allclose(relax.evaluate(r, t=2.0), mu.evaluate(r) * np.exp(-1.0))
assert np.allclose(at_time(relax, 0.0).evaluate(r), mu.evaluate(r))
assert lifted_to_time(mu).kind == "time"

# Frequency and time never mix in one expression.
try:
    kv + relax
except (TypeError, ValueError) as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("a frequency field and a time field must not add")

# -- the contracts ------------------------------------------------------------
testing.check_frequency_dependent_field(kv, omegas=[0.1, 1.0, 10.0])
testing.check_frequency_dependent_field(static_as_frequency, omegas=[0.1, 1.0])
testing.check_time_dependent_field(relax, ts=[0.0, 1.0, 5.0])

print("ok: lifted, composed and frozen fields of both kinds agree with their formulas")
