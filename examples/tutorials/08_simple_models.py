# %% [markdown]
# # 8. Simple models
#
# Beside PREM, the catalogue ships a model type built from a handful of
# numbers, for quick experiments and for tests: `LayeredIsotropicElastic`,
# an isotropic model whose layers each have a constant density and
# constant P and S velocities, with a one-layer `homogeneous` sphere as
# a special case. This tutorial builds both, then gives the layered one
# a shape: first an ellipsoid through the `flattening` displacement,
# then a different analytic relief on each boundary through
# `layer_linear`. Both displacements were met in tutorial 2; here they
# are applied to a model.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import (Skeleton, elastic_moduli, flattening,
                         LayeredIsotropicElastic, is_fluid, layer_linear, mass, testing)
from planetmodel.frames import cartesian_points

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## A homogeneous sphere, and a layered one
#
# The numbers here are arbitrary and the models have unit radius. The
# mass of the homogeneous ball is its density times the volume of the
# unit sphere, four pi. In the layered model a layer with zero S
# velocity is fluid, which `is_fluid` reads from the fields; the mixins
# add the five elastic moduli to every layer as they do for PREM.

# %%
ball = LayeredIsotropicElastic.homogeneous(1.0, rho=3.0, vp=2.0, vs=1.0, name="ball")
print(ball, "| mass:", mass(ball), "= 4 pi:", np.isclose(mass(ball), 4 * np.pi))
print("moduli:", elastic_moduli(ball.layer("ball")).symmetry.name)

earthlike = LayeredIsotropicElastic(
    [0.0, 0.55, 1.0], rho=[11.0, 4.5], vp=[9.0, 11.0], vs=[0.0, 6.0],
    layer_names=["core", "mantle"], interface_names=["cmb", "surface"])
print(earthlike)
print("fluid core:", is_fluid(earthlike.layer("core")),
      "| solid mantle:", not is_fluid(earthlike.layer("mantle")))
testing.check_model(earthlike)

# %% [markdown]
# ## An ellipsoidal planet
#
# `flattening(f, rmax=...)` is the displacement `h = -(2f/3) r P2(cos
# theta)`, with `P2` the degree-2 Legendre polynomial, and it carries its
# derivatives in closed form. It moves the pole of a sphere of radius
# `r` inward to `r (1 - 2f/3)` and the equator outward to `r (1 + f/3)`,
# so the flattening of the outer boundary, the difference of the two
# radii over the equatorial one, is `f` to first order. The printed
# value shows this.
#
# `stretched` is the model with its geometry placed in the physical
# world by the radial stretch built from that displacement. It checks
# that the mapping does not fold, and the geometry's validity report
# gives the margin: the smallest factor of the Jacobian found on a
# lattice of test points, which is one for the identity and zero at
# folding.

# %%
oblate = earthlike.stretched(flattening(1 / 300, rmax=1.0))
print(oblate.geometry.validity())
X = cartesian_points([1.0, 1.0], [0.0, np.pi / 2], [0.0, 0.0])
polar, equatorial = np.linalg.norm(oblate.geometry.mapping(X), axis=-1)
print("polar radius:", polar, "| equatorial:", equatorial,
      "| flattening:", (equatorial - polar) / equatorial, "| f:", 1 / 300)

# %% [markdown]
# ## Shapes on the boundaries
#
# `layer_linear` takes one relief per skeleton boundary, a function of
# colatitude and longitude giving the radial displacement of that
# boundary, or `None` where a boundary stays spherical. Within each
# layer it interpolates linearly in radius between the reliefs of the
# two bounding surfaces. The displacement is therefore continuous, and
# its radial derivative can jump only at the boundaries, which is what a
# geometry requires of its mapping. The boundaries are the displacement's
# knots. Here the core-mantle boundary takes a degree-2 shape and the
# surface a degree-2, order-2 one.

# %%
sk = Skeleton([0.0, 0.55, 1.0])


def cmb_relief(theta, phi):
    return 0.03 * (3.0 * np.cos(theta) ** 2 - 1.0) / 2.0


def surface_relief(theta, phi):
    return 0.02 * np.sin(theta) ** 2 * np.cos(2.0 * phi)


h = layer_linear(sk, [None, cmb_relief, surface_relief])
print(h, "| knots:", h.knots)
shaped = earthlike.stretched(h)
print(shaped.geometry.validity())
testing.check_displacement(h, sk)
testing.check_geometry(shaped.geometry)

# %% [markdown]
# ## A picture
#
# A section through the pole of each shaped model, the displacements
# exaggerated by the factor in the title. Grey circles are the
# undeformed boundaries.

# %%
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; no figure")
    raise SystemExit(0)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
theta = np.linspace(0.0, 2.0 * np.pi, 400)
for ax, model, title, gain in ((axes[0], oblate, "flattening(1/300) (x 30)", 30.0),
                               (axes[1], shaped, "layer_linear reliefs (x 5)", 5.0)):
    hh = model.geometry.mapping.h
    for b in sk.boundaries[1:]:
        colat = np.abs(np.mod(theta + np.pi, 2 * np.pi) - np.pi)
        rr = b + gain * hh(b, colat, np.where(theta < np.pi, 0.0, np.pi))
        ax.plot(rr * np.sin(theta), rr * np.cos(theta), lw=1.5)
        ax.plot(b * np.sin(theta), b * np.cos(theta), "0.7", lw=0.6)
    ax.set_aspect("equal"); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
out = FIGURES / "tutorial_08_simple_models.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
