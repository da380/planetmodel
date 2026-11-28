# %% [markdown]
# # The base of the lithosphere
#
# Going down each radial line, the lithosphere ends where the viscosity
# falls to a threshold, or where its Maxwell time falls to a loading
# time. The base found is a surface: an interface of the model can move
# onto it (geometry), or the viscosity above it can be raised to a lid
# (a field). Here the viscosity is made up, thicker under the northern
# hemisphere.
#
# Needs: maps
# Network: none
# Run: poetry run python examples/conversion/06_lithosphere_base.py

# %%
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import planetmodel as pm
from planetmodel.conversion import lithosphere
from planetmodel.datasets import moved
from planetmodel.plotting import sphere_map

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
TOP = 6368e3

# %% [markdown]
# A viscosity falling with depth over a scale of 15 km at the south pole
# and 25 km at the north, varying with longitude too: 1e25 Pa s at the
# surface.

# %%
model = pm.PREM(ocean=False)
for lay in model.layers:
    lo, hi = lay.interval
    if lo >= 5701e3:
        model = model.with_field(lay.index, "viscosity", pm.AnalyticField(
            (lo, hi), lambda r, t, p: 1e25 * np.exp(
                -(TOP - r) / (20e3 + 5e3 * np.cos(t)) + 0.5 * np.cos(2 * p)),
            name="viscosity"))
grid = pm.equiangular(45, 90)
above = [lay.index for lay in model.layers if lay.interval[0] >= 6151e3]
base = lithosphere.viscosity_threshold(model, 1e22, grid=grid, layers=above)
maxwell = lithosphere.maxwell_threshold(model, 1e4 * 3.156e7, grid=grid, layers=above)
for label, surface in (("1e22 Pa s", base), ("Maxwell time 1e4 years", maxwell)):
    depth = (TOP - surface.function.values) / 1e3
    print(f"{label}: the base from {depth.min():.0f} to {depth.max():.0f} km deep")

# %% [markdown]
# As geometry: PREM with its upper layers merged and an interface at 100
# km moved onto the base. As a field: 1e30 Pa s above it.

# %%
plain, _ = pm.PREM(ocean=False).coarsened(drop=["d80", "d220", "moho", "d15"])
plain = plain.refined([TOP - 100e3], interface_names=["lab"])
placed = moved(plain, {"lab": base})
pm.testing.check_geometry(placed.geometry)
lid = lithosphere.with_lid(model, base, viscosity=1e30)
print("moved, and a lid on layers",
      [lay.name for lay in lid.layers if "viscosity" in lay
       and lay["viscosity"] is not model.layer(lay.index)["viscosity"]])

# %%
fig = plt.figure(figsize=(6, 3.5))
ax = fig.add_subplot(projection=ccrs.Robinson())
depth = pm.slices.surface(model, pm.GriddedFunction(
    grid, (TOP - base.function.values) / 1e3, lateral="linear",
    poles=(TOP - base.function.poles) / 1e3), "surface")
mesh = sphere_map(ax, depth)
fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.7,
             label="depth of the 1e22 Pa s base (km)")
fig.tight_layout()
fig.savefig(FIGURES / "conversion_06.png", dpi=100)
print("figure written to", FIGURES / "conversion_06.png")
