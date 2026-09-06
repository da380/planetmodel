# %% [markdown]
# # 8. Simple models
#
# The catalogue also ships models made from a handful of numbers, for
# quick use and for tests: a homogeneous sphere and a layered one, both
# isotropic with constant exact layers. With the two shipped displacements
# they become ellipsoidal, or take an analytic shape on every boundary.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import (Skeleton, elastic_moduli, flattening, gravity,
                         LayeredIsotropicElastic, is_fluid, layer_linear, mass, testing)
from planetmodel.frames import cartesian_points

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## A homogeneous sphere, and a layered one

# %%
ball = LayeredIsotropicElastic.homogeneous(1.0, rho=3.0, vp=2.0, vs=1.0, name="ball")
print(ball, "| mass:", mass(ball), "| g(1):", gravity(ball, 1.0))
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
# `flattening(f, rmax=...)` is the degree-2 displacement `h = -f r P2`,
# with exact derivatives. `stretched` builds the radial stretch and checks
# it; the geometry's validity report says how far from folding it is.

# %%
oblate = earthlike.stretched(flattening(1 / 300, rmax=1.0))
print(oblate.geometry.validity())
X = cartesian_points([1.0, 1.0], [0.0, np.pi / 2], [0.0, 0.0])
polar, equatorial = np.linalg.norm(oblate.geometry.mapping(X), axis=-1)
print("polar radius:", polar, "| equatorial:", equatorial,
      "| flattening:", (equatorial - polar) / equatorial)

# %% [markdown]
# ## Shapes on the boundaries
#
# `layer_linear` takes one relief per skeleton boundary, `None` where a
# boundary stays spherical, and interpolates linearly in `r` within each
# layer. The boundaries are its knots, so the mapping's gradient may jump
# there and nowhere else, which is what a geometry requires.

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
# A meridional section of the two deformed models, exaggerated.

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
for ax, model, title, gain in ((axes[0], oblate, "flattening 1/300 (x 30)", 30.0),
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
