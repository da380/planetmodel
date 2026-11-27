# %% [markdown]
# # 2. An analytic mapping
#
# The physical planet is the image of the spherical reference body under
# a **mapping** `m`. A mapping takes reference points to physical points.
# It also provides its deformation gradient `F`, the matrix of
# derivatives `F[i, j] = d m_i / d X_j`, and its Jacobian `J = det F`,
# which is the local ratio of physical volume to reference volume. That
# is all the library requires of a mapping: any object with those three
# methods, acting on Cartesian points of shape `(..., 3)`, is one.
#
# The library ships the identity, a **radial stretch**, and a scaled
# copy of any mapping. The radial stretch is `m(X) = (r + h) e_r`: every
# point moves along its own radius by a distance `h(r, theta, phi)`
# called the radial displacement. This tutorial writes `h` down
# analytically. Building `h` from data, a topography grid say, is a
# separate matter and not part of this tutorial.

# %%
import numpy as np

from planetmodel import (
    CallableDisplacement,
    Geometry,
    RadialMesh,
    RadialStretch,
    Skeleton,
    testing,
    validity_lattice,
)
from planetmodel.frames import cartesian_points

sk = Skeleton([0.0, 0.19, 0.55, 0.99, 1.0])

# %% [markdown]
# ## A flattened planet
#
# An oblate planet has `h = -a r P2(cos theta)`, where `P2` is the second
# Legendre polynomial. The poles move in, the equator moves out, and the
# displacement grows in proportion to the radius so that the centre stays
# where it is. A sphere of radius `r` gets polar radius `r (1 - a)` and
# equatorial radius `r (1 + a / 2)`, so its flattening, the difference of
# the two over the equatorial radius, is `3a / 2` to first order. To
# give a flattening `f`, the amplitude is `a = 2f / 3`.
#
# A displacement is any function `h(r, theta, phi)`. Wrapping it in a
# `CallableDisplacement` supplies the rest of what a radial stretch needs:
# the derivative of `h` with respect to `r`, the derivatives with respect
# to the two angles, and a list of `knots`, the radii at which `dh/dr` is
# allowed to jump. Derivatives that are passed in are used as given, and
# any that are not are computed by central differences. The three
# arguments arrive broadcast to a common shape, so a derivative that does
# not depend on `r` need not mention it.
#
# The library ships this displacement as `flattening`; it is written out
# here to show the parts.

# %%
f = 1.0 / 300.0
a = 2.0 * f / 3.0


def p2(theta):
    return 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)


def h(r, theta, phi):
    return -a * r * p2(theta)


def dh_dr(r, theta, phi):
    return -a * p2(theta)


def dh_dangles(r, theta, phi):
    return (
        3.0 * a * r * np.cos(theta) * np.sin(theta),
        np.zeros(np.broadcast(r, theta, phi).shape),
    )


oblate = CallableDisplacement(
    h, radial_derivative=dh_dr, angular_gradient=dh_dangles, name="oblate"
)

# %% [markdown]
# A `RadialStretch` also needs the outer radius of the body it is meant
# for. This sets the scale on which a point counts as being at the
# centre, where the radial direction is undefined, and gives the mapping
# a range to search when it inverts itself. The radius can be given as a
# number, or as the skeleton or geometry whose outer boundary it is.

# %%
m = RadialStretch(oblate, rmax=sk)
print(m)

# %% [markdown]
# A mapping can be asked where a point goes, and for `F` and `J` there.
# In the local spherical frame `(e_r, e_theta, e_phi)` the deformation
# gradient of a radial stretch has few non-zero entries and is easy to
# read. Its Cartesian components are `R F R^T`, where `R` is the matrix
# of the frame vectors.

# %%
X = cartesian_points(0.8, np.pi / 4, 0.3)
print("X   =", X)
print("m(X)=", m(X), "| moved by", np.linalg.norm(m(X) - X))
np.set_printoptions(precision=5, suppress=True)
print("F (spherical frame):\n", m.deformation_gradient_spherical(X))
print("F (Cartesian):\n", m.deformation_gradient(X))
print("J =", m.jacobian(X))

# %% [markdown]
# `testing.check_mapping` tests everything a mapping must do: `F` is
# compared with a central difference of `m`, `J` with the determinant of
# `F`, and, where the mapping provides them, the displacement with
# `m(X) - X` and the inverse by a round trip.

# %%
points = cartesian_points(
    np.linspace(0.1, 0.95, 6)[:, None], np.linspace(0.2, 3.0, 5)[None, :], 0.4
).reshape(-1, 3)
testing.check_mapping(m, points)
print(
    "contract passed; inverse round trip error:",
    np.max(np.abs(m.inverse(m(points)) - points)),
)

# %% [markdown]
# ## Validity
#
# A mapping must preserve orientation, meaning `J > 0` everywhere;
# otherwise the physical body folds over on itself. For a radial stretch
# `J = (1 + dh/dr) (1 + h/r)^2`, so this is two conditions, `1 + dh/dr > 0`
# and `1 + h/r > 0`, and `validity` checks them on a sample of points.
# `validity_lattice` builds a sample that covers every layer of a
# skeleton, thin ones as densely as thick ones, and reaches both poles.
# When the check fails, the report says which condition failed and where.

# %%
lattice = validity_lattice(sk)
print(m.validity(sample=lattice))

# Exaggerate the flattening until the poles move inward faster than the
# radius grows, which happens when the amplitude a passes one.
for amp in (100.0, 250.0, 480.0):
    big = RadialStretch(
        CallableDisplacement(lambda r, t, p, a=amp: a * h(r, t, p)), rmax=sk
    )
    print(f"{amp:5.0f} x:", big.validity(sample=lattice))

# %% [markdown]
# ## A geometry with a mapping
#
# A geometry accepts a mapping only if two things hold: it is valid on
# the lattice, and it is continuous across every interior boundary. Its
# gradient may jump, but only at the radii the mapping declares as its
# knots. These are the properties a mesher relies on, so it need not
# check them again.

# %%
g = Geometry(
    sk,
    mapping=m,
    layer_names=["inner_core", "outer_core", "mantle", "crust"],
    interface_names=["icb", "cmb", "moho", "surface"],
)
print(g)
print("validity:", g.validity())
testing.check_geometry(g)

# The same in one step: `stretched` builds the radial stretch with this
# geometry's outer radius and checks it.
same = Geometry(sk).stretched(oblate)
print(same)

# %% [markdown]
# A displacement whose derivative with respect to `r` jumps at some
# radius is allowed, as long as the displacement declares that radius as
# one of its knots. A knot may be a boundary of the skeleton or a radius
# inside a layer. The geometry reports the knots of its mapping, and a
# radial mesh, the chain of elements along the radius that the next
# tutorial builds, puts an element edge on each of them, so that no
# element contains a jump. The example is relief that grows in
# proportion to height above a chosen radius. With that radius on the
# Moho the knot is a boundary, where the mesh has an edge already. With
# it in the mantle the geometry is accepted just the same, and the mesh
# gains an edge at the knot.


# %%
def crustal_bulge(knot):
    def hk(r, theta, phi):
        return 0.02 * np.maximum(r - knot, 0.0) * np.sin(theta) ** 2 * np.cos(2 * phi)

    return CallableDisplacement(hk, knots=[knot], name="crustal bulge")


for where, knot in (("on the Moho", 0.99), ("in the mantle", 0.9)):
    kinked = Geometry(sk).stretched(crustal_bulge(knot))
    mesh = RadialMesh(kinked, drmax=0.25)
    edges = np.append(mesh.left, mesh.right[-1])
    print(f"kink {where}: knots {kinked.knots} | element edges {np.round(edges, 3)}")

# %% [markdown]
# ## A mapping that is not radial
#
# Nothing requires a mapping to move points along their own radius. Any
# object with the three methods is accepted, and for one that provides
# no validity test of its own the geometry checks `J > 0` directly. The
# example squashes the planet along its axis and shears it, with `F`
# written by hand.


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
# A geometry with a mapping can be refined, truncated or hollowed, since
# the mapping stays continuous on the result. It cannot be extended or
# coarsened: the library allows those only while the mapping is the
# identity, and extending would ask the mapping about points it was not
# built for. For those, change the skeleton first and build the mapping
# for the result.

# %%
print(g.refined([0.9]).nlayers, "layers after refining")
try:
    g.extended([1.2])
except ValueError as err:
    print("refused:", err)

# %% [markdown]
# The next tutorial builds a one-dimensional mesh over a skeleton. The
# tutorial after that hands a geometry to the 3D mesher.
