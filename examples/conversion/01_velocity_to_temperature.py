# %% [markdown]
# # A velocity anomaly as a temperature anomaly
#
# A tomographic model gives the shear speed's departure from a radial
# reference, dlnVs. Taken as purely thermal, it becomes a temperature
# anomaly in one of two ways: divided by the sensitivity d ln Vs / dT, or
# turned into a density anomaly and divided by the thermal expansivity,
# as the sea-level code sl3d does. Here a made-up anomaly on PREM's
# lower mantle goes both ways.
#
# Needs: plot
# Network: none
# Run: poetry run python examples/conversion/01_velocity_to_temperature.py

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import planetmodel as pm
from planetmodel.conversion import building, parameters

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# A 1 % anomaly, fast at the poles and slow at the equator, on vsv and
# vsh between the core-mantle boundary and 670 km depth.

# %%
cmb, d670 = 3480e3, 5701e3
anomaly = pm.AnalyticField((cmb, d670), lambda r, t, p: 0.01 * np.cos(2 * t) + 0 * r)
model = pm.PREM()
for name in ("vsv", "vsh"):
    model = pm.perturbed(model, name, anomaly, kind="relative")
velocity = building.velocity_anomaly(model)
print("the anomaly lies on layers", velocity.layers)

# %% [markdown]
# The sensitivity is about -1e-4 per kelvin in the lower mantle, so a 1 %
# fast anomaly is about 100 K cold. sl3d's route goes through density
# instead: d ln rho / d ln Vs from a spline, then the expansivity.

# %%
sensitivity = parameters.Sensitivity(-1e-4, name="about -1e-4 per K")
by_sensitivity = building.temperature_anomaly(velocity, sensitivity=sensitivity)
by_density = building.temperature_anomaly(velocity,
                                          scaling=parameters.SL3D_DENSITY_SCALING,
                                          expansivity=parameters.SL3D_EXPANSIVITY)
r = np.linspace(cmb + 1e3, d670 - 1e3, 200)
pole = np.zeros_like(r)


def along(anomaly: building.Anomaly, r: np.ndarray) -> np.ndarray:
    """The anomaly at the north pole, layer by layer."""
    out = np.empty_like(r)
    for i in anomaly.layers:
        lo, hi = model.layer(i).interval
        sel = (r >= lo) & (r < hi)
        out[sel] = anomaly.fields[i](r[sel], pole[sel], pole[sel])
    return out


print(f"at the pole: {along(by_sensitivity, r).min():.0f} K by the sensitivity, "
      f"{along(by_density, r).min():.0f} to {along(by_density, r).max():.0f} K "
      "through density")

# %%
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.plot(parameters.SL3D_DENSITY_SCALING.scaling(r), r / 1e3,
          label="d ln rho / d ln Vs")
left.plot(1e4 * parameters.SL3D_EXPANSIVITY.expansivity(r), r / 1e3,
          label="alpha (1e-4 / K)")
left.set_ylabel("radius (km)")
left.set_title("sl3d's parameters")
left.legend()
right.plot(along(by_sensitivity, r), r / 1e3, label="by the sensitivity")
right.plot(along(by_density, r), r / 1e3, label="through density (sl3d)")
right.set_xlabel("temperature anomaly at the pole (K)")
right.set_title("a 1 % fast anomaly")
right.legend()
fig.tight_layout()
fig.savefig(FIGURES / "conversion_01.png", dpi=100)
print("figure written to", FIGURES / "conversion_01.png")
