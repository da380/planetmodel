# %% [markdown]
# # sl3d's chain on S20RTS
#
# sl3d, a sea-level code, builds its 3D viscosity from S20RTS without
# its degree 0: the velocity anomaly, a density anomaly by a radial
# scaling, a temperature anomaly by the expansivity, and
# eta = eta_0 exp(-0.01 dT).
# `sl3d_viscosity` is that chain in one call, held in the tests to sl3d's
# own output.
#
# Needs: maps
# Network: first run (S20RTS, 0.1 MB)
# Run: poetry run python examples/conversion/03_sl3d_chain.py

# %%
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import planetmodel as pm
from planetmodel import datasets
from planetmodel.conversion import building
from planetmodel.plotting import sphere_map

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# PREM with sl3d's radial viscosity, 5e20 Pa s in the upper mantle and
# 5e21 below 670 km, and S20RTS put on its vsv and vsh without degree 0.

# %%
prem = pm.PREM()
for lay in prem.layers:
    lo, hi = lay.interval
    if 3480e3 <= lo and hi <= 6291e3:
        prem = prem.with_field(lay.index, "viscosity", pm.constant_field(
            (lo, hi), 5e21 if hi <= 5701e3 else 5e20, name="viscosity"))
mantle = [lay.index for lay in prem.layers if "viscosity" in lay]
model = datasets.transplanted(prem, "S20RTS", remove_mean=True, layers=mantle)
model = building.sl3d_viscosity(model, layers=mantle)

# %%
fig = plt.figure(figsize=(12, 4))
for k, depth in enumerate((300e3, 1000e3, 2500e3)):
    ax = fig.add_subplot(1, 3, k + 1, projection=ccrs.Robinson())
    at = pm.slices.sphere(model, "viscosity", depth=depth, relative_to="reference",
                          departure="log10")
    sphere_map(ax, at, colorbar=True, diverging=True)
    ax.set_title(f"{depth / 1e3:.0f} km deep")
fig.tight_layout()
fig.savefig(FIGURES / "conversion_03.png", dpi=100)
print("figure written to", FIGURES / "conversion_03.png")
print(model.cite())
