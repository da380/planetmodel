"""The algebra of fields: sum, scale, compose, and the checks it makes.

Fields on one skeleton add, scale and compose into new fields lazily:
nothing is sampled until the result is evaluated.  A sum needs the same
character on both sides and the same dimensions; a composition through
`ComposedField` states the character and dimensions of its result,
since a function of fields can be anything.  An AnalyticField is a
formula of (r, theta, phi) on a skeleton, the way angular dependence
enters, and it takes part in the same algebra.

This script builds a few fields, combines them, and shows what the
algebra accepts and what it refuses.
"""
import numpy as np

from planetmodel import (DENSITY, SCALAR, VECTOR, AnalyticField, ComposedField,
                         Dimensions, RadialField, Skeleton)

sk = Skeleton([0.0, 1.0, 2.0])
rho = RadialField(sk, [lambda r: 3.0 + 0.0 * r, lambda r: 1.0 + r],
                  name="rho", character=DENSITY, dimensions=Dimensions.DENSITY)
vs = RadialField(sk, [lambda r: 2.0 + 0.0 * r, lambda r: 4.0 + 0.0 * r],
                 name="vs", character=SCALAR, dimensions=Dimensions.VELOCITY)

r = np.array([0.5, 1.5])

# -- scaling and sums ----------------------------------------------------------
twice = 2.0 * rho
assert np.allclose(twice.evaluate(r), [6.0, 5.0])
assert twice.character == DENSITY and twice.dimensions == Dimensions.DENSITY

total = rho + twice
assert np.allclose(total.evaluate(r), [9.0, 7.5])
assert total.skeleton == sk

try:
    rho + vs                                       # density plus a scalar
except ValueError as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("fields of different character must not add")

# -- composition ------------------------------------------------------------
# mu = rho vs^2, whose dimensions are those of a modulus: say so.
mu = ComposedField(lambda d, v: d * v * v, [rho, vs], name="mu",
                   character=SCALAR, dimensions=Dimensions.MODULUS)
assert np.allclose(mu.evaluate(r), [3.0 * 4.0, 2.5 * 16.0])
assert mu.is_radial                                 # its operands are radial

# The composites remember what they were made from, which is what lets a
# body split them per layer and re-express them in other units.
assert mu.operands() == (rho, vs)
assert mu.restricted(1).skeleton == Skeleton([1.0, 2.0])
assert np.isclose(mu.restricted(1)(1.5), 40.0)

# -- angular dependence -------------------------------------------------------
# A formula of position: a scalar that grows with radius and colatitude.
bump = AnalyticField(lambda r, theta, phi: r * np.cos(theta), sk,
                     character=SCALAR, dimensions=Dimensions.VELOCITY,
                     name="bump")
assert not bump.is_radial
theta, phi = np.array([0.0, np.pi / 3]), np.array([0.0, 1.0])
assert np.allclose(bump.evaluate(r, theta, phi), r * np.cos(theta))
try:
    bump.evaluate(r)                                # angles are required
except (TypeError, ValueError) as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("an angular field must ask for its angles")

# It adds to a radial field of the same character and dimensions.
both = vs + bump
assert not both.is_radial
assert np.allclose(both.evaluate(r, theta, phi),
                   vs.evaluate(r) + r * np.cos(theta))

# Cartesian points are accepted too: the frame is what the coordinates imply.
X = np.array([[0.0, 0.0, 0.5], [1.5, 0.0, 0.0]])          # on the axis; equator
assert np.allclose(bump.evaluate_at(X), [0.5, 0.0])

# -- a vector field, in the local spherical frame ------------------------------
flow = AnalyticField(lambda r, theta, phi: np.stack(
    [r, 0.0 * r, 0.0 * r], axis=-1), sk, character=VECTOR, name="flow")
v_sph = flow.evaluate(np.array([1.0]), np.array([np.pi / 2]), np.array([0.0]))
v_cart = flow.evaluate(np.array([1.0]), np.array([np.pi / 2]), np.array([0.0]),
                       frame="cartesian")
assert np.allclose(v_sph, [[1.0, 0.0, 0.0]])       # radial component
assert np.allclose(v_cart, [[1.0, 0.0, 0.0]])      # e_r is x at the equator, phi=0

print("ok: the algebra is lazy, checked, and closed under itself")
