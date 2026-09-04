# %% [markdown]
# # 2. An analytic mapping
#
# The physical planet is the image of the spherical reference body under a
# **mapping** `m`. A mapping takes reference points to physical points and
# provides its deformation gradient `F`, with `F[i, j] = d m_i / d X_j`,
# and its Jacobian `J = det F`. That is the whole contract: anything with
# those three methods on Cartesian points of shape `(..., 3)` is a mapping.
#
# The one mapping shipped is the **radial stretch**, `m(X) = (r + h) e_r`,
# driven by a scalar radial displacement `h(r, theta, phi)`. Here `h` is
# written down analytically. How a displacement is obtained from data is
# a separate matter and is not part of this tutorial.

# %%
import numpy as np

from planetmodel import (CallableDisplacement, Geometry, RadialStretch,
                         Skeleton, testing, validity_lattice)
from planetmodel.frames import cartesian_points

sk = Skeleton([0.0, 0.19, 0.55, 0.99, 1.0])

# %% [markdown]
# ## A flattened planet
#
# An oblate planet is `h = -f r P2(cos theta)` with `P2` the second
# Legendre polynomial: the poles move in, the equator moves out, and the
# displacement grows linearly with radius so the centre stays put. The
# derivatives are written down too, so the deformation gradient is exact.

# %%
f = 1.0 / 300.0


def p2(theta):
    return 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)


def h(r, theta, phi):
    return -f * r * p2(theta)


def dh_dr(r, theta, phi):
    return -f * p2(theta) + 0.0 * r


def dh_dangles(r, theta, phi):
    return (3.0 * f * r * np.cos(theta) * np.sin(theta),
            np.zeros(np.broadcast(r, theta, phi).shape))


flattening = CallableDisplacement(h, radial_derivative=dh_dr,
                                  angular_gradient=dh_dangles, name="flattening")
m = RadialStretch(flattening, rmax=1.0)
print(m)

# %% [markdown]
# Ask the mapping where a point goes, and for `F` and `J` there. In the
# local frame `(e_r, e_theta, e_phi)` a radial stretch's `F` is sparse and
# easy to read; the Cartesian form is its conjugate by the frame matrix.

# %%
X = cartesian_points(0.8, np.pi / 4, 0.3)
print("X   =", X)
print("m(X)=", m(X), "| moved by", np.linalg.norm(m(X) - X))
np.set_printoptions(precision=5, suppress=True)
print("F (spherical frame):\n", m.deformation_gradient_spherical(X))
print("F (Cartesian):\n", m.deformation_gradient(X))
print("J =", m.jacobian(X))

# %% [markdown]
# `testing.check_mapping` holds any mapping to its contract: `F` against a
# central difference of `m`, `J` against `det F`, the displacement, and
# the inverse where the mapping has one.

# %%
points = cartesian_points(np.linspace(0.1, 0.95, 6)[:, None],
                          np.linspace(0.2, 3.0, 5)[None, :], 0.4).reshape(-1, 3)
testing.check_mapping(m, points)
print("contract passed; inverse round trip error:",
      np.max(np.abs(m.inverse(m(points)) - points)))

# %% [markdown]
# ## Validity
#
# A mapping must preserve orientation, or the physical body folds onto
# itself. For a radial stretch that is two conditions, `1 + dh/dr > 0` and
# `1 + h/r > 0`, checked on a sample of points. `validity_lattice` builds a
# sample covering every layer of a skeleton and both poles. The report says
# which factor failed and where.

# %%
lattice = validity_lattice(sk)
print(m.is_valid(sample=lattice))

# Exaggerate the flattening until the poles move in faster than the
# radius grows.
for amp in (100.0, 250.0, 320.0):
    big = RadialStretch(CallableDisplacement(lambda r, t, p, a=amp: a * h(r, t, p)),
                        rmax=1.0)
    print(f"{amp:5.0f} x:", big.is_valid(sample=lattice))

# %% [markdown]
# ## A geometry with a mapping
#
# A geometry accepts a mapping only if it is valid on the lattice, is
# continuous across every interior boundary, and declares any kink in its
# gradient at a boundary of the skeleton. Those are the invariants that
# let a mesher trust the mapping without checking it again.

# %%
g = Geometry(sk, mapping=m,
             layer_names=["inner_core", "outer_core", "mantle", "crust"],
             interface_names=["icb", "cmb", "moho", "surface"])
print(g)
print("validity:", g.validity())
testing.check_geometry(g)

# %% [markdown]
# A kinked displacement is one whose radial derivative jumps. It is
# allowed, but the kink must sit on a boundary, because that is where a
# mesh will put an element edge. Here relief confined to the crust grows
# linearly from the Moho: the kink is at the Moho, which is a boundary, so
# the geometry accepts it. Move the kink into the mantle and it is refused.


# %%
def crustal_bulge(knot):
    def hk(r, theta, phi):
        return 0.02 * np.maximum(r - knot, 0.0) * np.sin(theta) ** 2 * np.cos(2 * phi)

    return CallableDisplacement(hk, knots=[knot], name="crustal bulge")


ok = Geometry(sk, mapping=RadialStretch(crustal_bulge(0.99), rmax=1.0))
print("kink on the Moho:", ok.knots(), "accepted")
try:
    Geometry(sk, mapping=RadialStretch(crustal_bulge(0.9), rmax=1.0))
except ValueError as err:
    print("kink in the mantle:", err)

# %% [markdown]
# ## A mapping that is not radial
#
# Nothing requires a mapping to move points along rays. Any object with
# the three methods is accepted, and the geometry falls back to the
# generic test `J > 0`. Here a planet is squashed along its axis and
# sheared, with `F` written by hand.


# %%
class Squash:
    """x -> (x, y, c z + b x y)."""

    def __init__(self, c, b):
        self.c, self.b = c, b

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        out = X.copy()
        out[..., 2] = self.c * X[..., 2] + self.b * X[..., 0] * X[..., 1]
        return out

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        F = np.broadcast_to(np.eye(3), X.shape[:-1] + (3, 3)).copy()
        F[..., 2, 2] = self.c
        F[..., 2, 0] = self.b * X[..., 1]
        F[..., 2, 1] = self.b * X[..., 0]
        return F

    def jacobian(self, X):
        return np.full(np.asarray(X).shape[:-1], self.c)


squashed = Geometry(sk, mapping=Squash(0.95, 0.1))
print(squashed, "| validity:", squashed.validity())
testing.check_geometry(squashed)

# %% [markdown]
# Surgery that keeps the mapping continuous is allowed on a geometry with
# a mapping (refining, truncating); surgery that would break it (extending,
# coarsening) is refused until the mapping is rebuilt for the new skeleton.

# %%
print(g.refined([0.9]).nlayers, "layers after refining")
try:
    g.extended([1.2])
except ValueError as err:
    print("refused:", err)

# %% [markdown]
# The next tutorial hands a geometry to the 3D mesher.
