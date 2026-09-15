# %% [markdown]
# # Topography and the mapping
#
# planetmodel works *referentially*. Fields live on a spherically
# symmetric reference body, and the physical planet, with its ellipticity
# and its relief, is the image of that body under a **mapping**. Nothing
# is resampled onto an aspherical grid: a solver evaluates the mapping's
# deformation gradient and Jacobian where it needs them, and a field is
# carried across by the push-forward its tensor character dictates.
#
# This tutorial attaches the Moho of CRUST-1.0 to a reference body, builds
# the mapping, checks that it preserves orientation, pushes a density
# forward, and pulls a physical elastic medium back.

# %%
import warnings
from pathlib import Path

import numpy as np

from planetmodel import GriddedTopography, Surface, read_isotropic_deck

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"

body = read_isotropic_deck(DATA / "prem.nocrust").name_interface(-1, "moho")
moho = body.interface("moho")
print(body)
print(f"outer boundary, the Moho of the deck: {moho.radius / 1e3:.1f} km")

# %% [markdown]
# ## Relief on a boundary
#
# CRUST-1.0 gives the Moho depth on a one-degree grid, in kilometres,
# negative downwards. `GriddedTopography.from_xyz` reads a `lon lat value`
# file onto its grid; `scale` turns kilometres into metres, and degrees
# and latitude never leave the reader: the model layer speaks colatitude
# in radians.
#
# An interface radius **is** the boundary's mean radius, so relief
# attached to it must have zero mean. CRUST-1.0's depths do not: their
# mean is the mean Moho depth. Attaching the raw grid centres it, and a
# warning says by how much.

# %%
depth = GriddedTopography.from_xyz(DATA / "crust-1.0" / "depthtomoho.xyz",
                                   scale=1.0e3)
theta = np.array([0.5, 1.2, 2.0])
phi = np.array([0.3, 2.0, 4.5])
print("Moho depth at three points, km:", np.round(depth(theta, phi) / 1e3, 1))
print(f"area-weighted mean depth: {depth.mean() / 1e3:.2f} km")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with_moho = body.with_surface("moho", depth)
print("attached with a warning:", str(caught[0].message)[:70], "...")
surface = with_moho.surface("moho")
print(surface)
print("relief at the three points, km:",
      np.round(surface.topography(theta, phi) / 1e3, 1),
      "| mean:", f"{surface.topography.mean():.3g} m")

# %% [markdown]
# The explicit route says the same thing in two steps. A `Surface` is a
# reference radius plus relief; `centred()` moves the relief's mean into
# the radius, and `at()` places the relief at another radius. The deck's
# Moho sits 1.5 km below CRUST-1.0's mean Moho, and `with_surface` refuses
# to paper over that: the interface and the data must agree on where the
# boundary is on average.

# %%
crust_moho = Surface(6371.0e3, topography=depth, name="moho").centred()
print(f"CRUST-1.0's mean Moho radius: {crust_moho.reference_radius / 1e3:.1f} km")
try:
    body.with_surface("moho", crust_moho)
except ValueError as err:
    print("refused:", str(err)[:96], "...")
with_moho = body.with_surface("moho", crust_moho.at(moho.radius))
print("placed at the deck's Moho instead:", with_moho.surface("moho"))

# %% [markdown]
# A boundary can be any shape. `ellipsoid_surface` builds a triaxial
# ellipsoid, its reference radius the area-weighted mean radius, its
# relief the departure from it.

# %%
from planetmodel.model import ellipsoid_surface  # noqa: E402

f = 1.0 / 300.0                                      # a flattening
ellipsoid = ellipsoid_surface(6346.0e3, 6346.0e3, 6346.0e3 * (1.0 - f))
print(ellipsoid)
print("relief at the pole and the equator, km:",
      np.round(ellipsoid.topography(np.array([0.0, np.pi / 2]), 0.0) / 1e3, 2))
flattened = body.truncated(ellipsoid.reference_radius, name="surface")
flattened = flattened.with_surface("surface", ellipsoid)
print("a body with an ellipsoidal outer boundary:", flattened.surfaces)

# %% [markdown]
# ## The mapping
#
# `body.mapping(rule=...)` builds the mapping from the attached surfaces.
# The shipped rule, `layer_linear`, spreads each interface's relief
# linearly in radius to the neighbouring interfaces, so the displacement
# vanishes where no surface is attached and its radial derivative jumps
# only where the mesh already has a boundary. Points move along their own
# radius: `m(X) = (r + h) e_r`. Here the relief lives between the Moho and
# the interface 55 km below it, so a point at 6320 km moves and a point at
# 6000 km does not.
#
# The mapping evaluates at Cartesian points, and so do its deformation
# gradient `F` and Jacobian `J = det F`. A mapping is valid where it
# preserves orientation, `1 + dh/dr > 0` and `h > -r`; `is_valid` checks
# that on a lattice and reports the worst point when it fails.

# %%
from planetmodel import layer_linear, validity_lattice  # noqa: E402
from planetmodel.model.frames import cartesian_points  # noqa: E402

mapping = with_moho.mapping(rule=layer_linear())
print(mapping)
X = cartesian_points(np.array([6.32e6]), np.array([1.2]), np.array([2.0]))
x = mapping(X)
F = mapping.deformation_gradient(X)
print("a reference point moves by", np.round(np.linalg.norm(x - X, axis=-1)), "m")
print("F at that point:\n", np.round(F[0], 4))
print("J =", np.round(mapping.jacobian(X), 5))
lattice = validity_lattice(with_moho.skeleton)
print("valid on the lattice:", mapping.is_valid(sample=lattice))

# %% [markdown]
# Exaggerate the relief and the mapping folds: the report names the worst
# point, which is where a mesh would tangle.

# %%
too_much = body.with_surface("moho", crust_moho.at(moho.radius).topography * 3.0)
print(too_much.mapping(rule=layer_linear())
      .is_valid(sample=validity_lattice(too_much.skeleton)))

# %% [markdown]
# ## Pushing a field forward
#
# A density is a scalar of weight one: under the mapping it picks up a
# factor `1 / J`, so that mass is conserved. `push_forward_field` returns
# a lazy field on the reference body whose value at a reference point `X`
# is the physical density at `m(X)`. Nothing is inverted and nothing is
# resampled.

# %%
from planetmodel import push_forward_field  # noqa: E402

rho_phys = push_forward_field(with_moho["rho"], mapping)
r, th, ph = np.array([6.32e6]), np.array([1.2]), np.array([2.0])
print(rho_phys)
print("reference density:", with_moho["rho"].evaluate(r))
print("physical density at m(X):", rho_phys.evaluate(r, th, ph))
print("equals rho / J:",
      np.allclose(rho_phys.evaluate(r, th, ph),
                  with_moho["rho"].evaluate(r) / mapping.jacobian(X)))

# %% [markdown]
# ## Pulling a physical medium back
#
# The other direction matters too: a medium known on the physical planet
# enters the model as a referential field. A physically isotropic medium
# with moduli `kappa(x)` and `mu(x)` pulls back to a referential elasticity
# tensor that is not isotropic, because the mapping stretches it
# anisotropically; that is the tensor a referential solver needs.
# `pulled_back_elastic` builds it in closed form for isotropic and
# transversely isotropic media, and pushing it forward again recovers the
# physical Voigt matrix.

# %%
from planetmodel import Symmetry  # noqa: E402
from planetmodel.model import pulled_back_elastic  # noqa: E402


def kappa_phys(r, theta, phi):
    """A bulk modulus of the physical radius, in Pa."""
    return 1.3e11 + 5.0e10 * (1.0 - r / 6.4e6)


def mu_phys(r, theta, phi):
    return 6.8e10 + 2.0e10 * (1.0 - r / 6.4e6)


referential = pulled_back_elastic(Symmetry.ISOTROPIC,
                                  {"kappa": kappa_phys, "mu": mu_phys}, mapping,
                                  skeleton=with_moho.skeleton, name="elastic_moduli")
C_ref = referential.evaluate(r, th, ph)[0]
np.set_printoptions(suppress=True)
print("referential Voigt matrix at X, GPa (no longer isotropic):")
print(np.round(C_ref / 1e9, 2))
C_phys = push_forward_field(referential, mapping).evaluate(r, th, ph)[0]
print("pushed forward again, GPa:")
print(np.round(C_phys / 1e9, 2))
rp = np.linalg.norm(x)
k, m = kappa_phys(rp, 0, 0), mu_phys(rp, 0, 0)
print("which is the physical isotropic medium at m(X):",
      np.allclose(C_phys[0, 0], k + 4 * m / 3), np.allclose(C_phys[3, 3], m))
