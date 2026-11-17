# %% [markdown]
# # A flat model, sliced and drawn
#
# A crust over the uppermost mantle in a box 400 km by 300 km and 200 km
# deep (`cartesian.CrustOverMantle`), its shear speed perturbed by a
# Fourier series over the box (a slow blob under a gentle cosine
# pattern, fading with depth), and a mountain range on its surface with
# a root on its Moho as the body's vertical stretch. It is sliced as the
# sphere's models are: a plane at a depth, the topography on the surface,
# a section along a line in the reference body and as the body is, the
# mean profile with its spread, the spectrum by horizontal wavenumber,
# and the box seen from a corner. Depth is drawn downward throughout.
#
# Needs: plot
# Network: none
# Run: poetry run python examples/cartesian/01_a_flat_model.py

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PPoly

from planetmodel import cartesian as flat
from planetmodel import perturbed
from planetmodel.cartesian import slices
from planetmodel.plotting import (block, plane_map, profile, radial_profile,
                                  spectrum, vertical_section)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
KM = 1e-3

# %% [markdown]
# The model and its grid: 64 by 48 nodes over the box, which is the
# period of every Fourier series on it.

# %%
box = flat.Box([(0.0, 400e3), (0.0, 300e3)])
model = flat.CrustOverMantle(box)
grid = flat.Grid(box, (64, 48))
x, y = grid.nodes

# %% [markdown]
# The perturbation of vs, relative, from the surface to the base: a slow
# blob of -4 %, 30 km in its standard deviation and far enough from the
# sides to be periodic to rounding, under a cosine pattern of 1.5 %, the
# series of their values at the grid's nodes, fading linearly in depth to
# half at the base.

# %%
blob = -0.04 * np.exp(-((x - 250e3) ** 2 + (y - 150e3) ** 2) / (2 * 30e3 ** 2))
pattern = 0.015 * np.cos(2 * np.pi * x / 400e3) * np.cos(2 * np.pi * y / 300e3)
c = grid.transform.analyse_grid(blob + pattern)
base = 200e3
fade = PPoly(np.stack([-0.5 / base * c, c])[:, None], [0.0, base])
dlnvs = flat.FourierField((0.0, base), grid, fade)
model = perturbed(model, "vs", dlnvs, kind="relative")

# %% [markdown]
# Topography: a range 3 km high along y at x = 110 km, heights positive
# up, and its Airy root on the Moho, 5.6 times as deep (2800 / (3300 -
# 2800)); the base of the lithosphere and the box's base held. The
# stretched model is the body as it is.

# %%
height = 3e3 * np.exp(-((x - 110e3) / 35e3) ** 2) * (1 + 0.2 * np.cos(
    2 * np.pi * y / 300e3))
topography = flat.FourierFunction(grid, grid.transform.analyse_grid(height))
root = flat.FourierFunction(grid, -5.6 * topography.coefficients)
stretched = model.stretched(flat.layer_linear(model.skeleton,
                                              [topography, root, None, None]))

# %% [markdown]
# A plane at 60 km, vs as a departure from its horizontal mean, beside
# the topography on the surface.

# %%
fine = flat.Grid(box, (128, 96))
at60 = slices.plane(model, "vs", depth=60e3, grid=fine, relative_to="mean")
relief = slices.surface(model, topography, "surface", name="topography", grid=fine)
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
plane_map(left, relief, scale=KM, cmap="YlOrBr", colorbar=True)
left.set_title("the topography on the surface [m]")
plane_map(right, at60, scale=KM, colorbar=True)
right.set_title("vs at 60 km depth")
for ax in (left, right):
    ax.set(xlabel="x [km]", ylabel="y [km]")
fig.savefig(FIGURES / "cartesian_01_plane.png", dpi=120)

# %% [markdown]
# A section along x at y = 150 km, through the blob and across the range:
# in the reference body down to 150 km, and as the body is in its top
# 70 km, the surface raised to the topography and the Moho pushed down
# into its root, sliced at physical points.

# %%
line = dict(end=(400e3, 150e3), n=30, m=201, relative_to="mean", mean_grid=grid)
reference = slices.section(model, "vs", (0.0, 150e3), depths=(0.0, 150e3), **line)
physical = slices.section(stretched, "vs", (0.0, 150e3), depths=(-5e3, 70e3),
                          at="physical", **line)
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), constrained_layout=True,
                         sharex=True)
vertical_section(axes[0], reference, scale=KM, colorbar=True)
axes[0].set(title="the reference body, y = 150 km", ylim=(150, -6))
vertical_section(axes[1], physical, scale=KM, colorbar=True)
axes[1].set(title="the body as it is: the topography and its root", ylim=(70, -6))
for ax in axes:
    ax.set_ylabel("depth [km]")
axes[1].set_xlabel("distance along the section [km]")
fig.savefig(FIGURES / "cartesian_01_section.png", dpi=120)

# %% [markdown]
# The horizontal mean of vs down the depth, with its range over the
# plane at each depth shaded, against the reference model's vs.

# %%
mean = slices.mean_profile(model, "vs", mean_grid=grid, spread="range", n=60)
fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
profile(ax, mean, scale=KM, label="horizontal mean, and range")
radial_profile(ax, flat.CrustOverMantle(box), "vs", scale=KM, label="reference",
               color="k", ls="--", lw=0.8, boundaries=False)
ax.set(xlabel=mean.label, ylabel="depth [km]")
ax.legend(loc="lower left")
fig.savefig(FIGURES / "cartesian_01_profile.png", dpi=120)

# %% [markdown]
# The power of vs's departure by horizontal wavenumber at 60 km, which
# sums to its mean square over the plane, the wavelength along the top.

# %%
departure = slices.plane(model, "vs", depth=60e3, grid=grid, relative_to="mean",
                         departure="difference")
power = slices.spectrum_of(departure)
print(f"the power sums to {power.power.sum():.6g} (m/s)^2, the mean square over "
      f"the plane {grid.mean(departure.values ** 2):.6g}")
fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
spectrum(ax, power, scale=KM, marker="o", ms=3, wavelengths="wavelength [km]")
ax.set(xlabel="|k| [rad/km]", ylabel="power [(m/s)^2]")
ax.set_title(power.label)
fig.savefig(FIGURES / "cartesian_01_spectrum.png", dpi=120)

# %% [markdown]
# The box seen from the southeast, down to 150 km, as the body is: its
# top the topography, the blob's slow column on its east face.

# %%
cube = slices.block(stretched, "vs", depths=(0.0, 150e3), relative_to="mean",
                    mean_grid=grid, n=24, m=81)
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"},
                       constrained_layout=True)
block(ax, cube, scale=KM, physical=True, colorbar=True)
ax.set(xlabel="x [km]", ylabel="y [km]", zlabel="depth [km]")
fig.savefig(FIGURES / "cartesian_01_block.png", dpi=120)
print("figures written to", FIGURES)
