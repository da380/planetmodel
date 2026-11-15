# %% [markdown]
# # A flat model in a model file, its topography in a surface file
#
# The flat model of `01_a_flat_model.py` (a crust over the uppermost
# mantle in a 400 by 300 km box, its shear speed perturbed by a Fourier
# series, a range on its surface with a root on its Moho) written to a
# model file, and its topography to a surface file, then read back and
# compared. The model file is of the flat schema, `planetmodel.model/3`:
# it states the box (its bounds and padding), holds the fields at depth
# nodes on the box's grid, the perturbation as its values there read by
# the Fourier rule, and the range and its root as heights in its
# geometry group; the surface file, `planetmodel.surface/2`, holds the
# topography on the same grid. What comes back is the same model, to
# rounding, which a section through the body as it is shows.
#
# Needs: netcdf, plot
# Network: none
# Run: poetry run python examples/cartesian/10_a_flat_model_file.py

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PPoly

from planetmodel import cartesian as flat
from planetmodel import perturbed
from planetmodel.cartesian import FLAT, slices
from planetmodel.cartesian.modelfile import read_model_file
from planetmodel.cartesian.surfacefile import write_surface
from planetmodel.modelfile import read_model, write_model
from planetmodel.plotting import plane_map, vertical_section
from planetmodel.surfacefile import SurfaceVariable, read_surface
from planetmodel.units import LENGTH

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
KM = 1e-3

# %% [markdown]
# The model of `01`: the box, its grid of 64 by 48 nodes, the slow blob
# under a cosine pattern fading with depth, and the range with its Airy
# root as the body's vertical stretch.

# %%
box = flat.Box([(0.0, 400e3), (0.0, 300e3)])
grid = flat.Grid(box, (64, 48))
x, y = grid.nodes
blob = -0.04 * np.exp(-((x - 250e3) ** 2 + (y - 150e3) ** 2) / (2 * 30e3 ** 2))
pattern = 0.015 * np.cos(2 * np.pi * x / 400e3) * np.cos(2 * np.pi * y / 300e3)
c = grid.transform.analyse_grid(blob + pattern)
base = 200e3
fade = PPoly(np.stack([-0.5 / base * c, c])[:, None], [0.0, base])
model = perturbed(flat.CrustOverMantle(box), "vs",
                  flat.FourierField((0.0, base), grid, fade), kind="relative")
height = 3e3 * np.exp(-((x - 110e3) / 35e3) ** 2) * (1 + 0.2 * np.cos(
    2 * np.pi * y / 300e3))
topography = flat.FourierFunction(grid, grid.transform.analyse_grid(height))
root = flat.FourierFunction(grid, -5.6 * topography.coefficients)
model = model.stretched(flat.layer_linear(model.skeleton,
                                          [topography, root, None, None]))

# %% [markdown]
# Written: the base fields of the model on the box's grid, and the
# topography, a static field on the same grid, read by the Fourier rule.
# Read back: the model as a `cartesian.Model` over the same box, and the
# surface file as a function of time, its topography a gridded function
# on the grid.

# %%
with tempfile.TemporaryDirectory() as tmp:
    path = write_model(Path(tmp) / "model.nc", model, grid=grid,
                       fields=["rho", "vp", "vs"])
    surface = write_surface(
        Path(tmp) / "topography.nc", grid, source="planetmodel example",
        fields={"topography": SurfaceVariable(topography.on_grid(grid),
                                              dimensions=LENGTH,
                                              meaning="height of the free surface",
                                              lateral_interpolation="fourier")})
    data = read_model_file(path)
    back = read_model(path)
    held = read_surface(surface).function("topography")
print(data.schema, data.domain, [part.grid for part in data.parts])
print("the geometry group:", data.geometry.radii, data.geometry.lateral)

# %% [markdown]
# Compared at random points of the body: the shear speed, the body's
# mapped points and the topography read from the surface file.

# %%
rng = np.random.default_rng(0)
d = rng.uniform(0.0, base, 2000)
px, py = rng.uniform(0.0, 400e3, 2000), rng.uniform(0.0, 300e3, 2000)
worst = 0.0
for layer in model.layers:
    inside = (d >= layer.interval[0]) & (d <= layer.interval[1])
    want = layer["vs"].evaluate(d[inside], px[inside], py[inside])
    got = back.layer(layer.index)["vs"].evaluate(d[inside], px[inside], py[inside])
    worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
X = FLAT.points(d, px, py)
moved = np.abs(back.geometry.mapping(X) - model.geometry.mapping(X)).max()
relief = np.abs(held(px, py) - topography(px, py)).max()
print(f"vs read back: largest difference {worst:.1e} of its value")
print(f"the body's points read back: largest difference {moved:.1e} m")
print(f"the topography read back: largest difference {relief:.1e} m")
assert worst < 1e-10 and moved < 1e-9 and relief < 1e-6

# %% [markdown]
# The model read back, sliced: the topography from the surface file on
# the surface, and a section along x at y = 150 km through the body as it
# is, vs as a departure from its horizontal mean, the surface raised to
# the topography and the Moho pushed down into its root.

# %%
fine = flat.Grid(box, (128, 96))
shown = slices.surface(back, held, "surface", name="topography", grid=fine)
section = slices.section(back, "vs", (0.0, 150e3), end=(400e3, 150e3),
                         depths=(-5e3, 70e3), at="physical", n=30, m=201,
                         relative_to="mean", mean_grid=grid)
fig, (top, side) = plt.subplots(2, 1, figsize=(9, 8.4), constrained_layout=True,
                                gridspec_kw={"height_ratios": [1.15, 1.0]})
plane_map(top, shown, scale=KM, cmap="YlOrBr", colorbar=True)
top.axhline(150, color="k", lw=0.8, ls="--")
top.set(title="the topography read from the surface file [m]", xlabel="x [km]",
        ylabel="y [km]")
vertical_section(side, section, scale=KM, colorbar=True)
side.set(title="vs read from the model file, y = 150 km, the body as it is",
         ylim=(70, -6), xlabel="distance along the section [km]",
         ylabel="depth [km]")
fig.savefig(FIGURES / "cartesian_10_files.png", dpi=120)
print("figure written to", FIGURES / "cartesian_10_files.png")
