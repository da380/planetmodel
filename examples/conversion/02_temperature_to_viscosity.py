# %% [markdown]
# # A temperature anomaly as a viscosity
#
# The exponential law takes a temperature anomaly to a viscosity,
# eta = eta_0 exp(-eps dT), as a log perturbation of the layer's own
# radial viscosity. The sea-level code sl3d takes eps = 0.01 per kelvin,
# Milne et al. (2018) 0.04; Latychev et al. (2005) also rescale each
# depth so that its mean viscosity stays the radial one.
#
# Needs: plot
# Network: none
# Run: poetry run python examples/conversion/02_temperature_to_viscosity.py

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import planetmodel as pm
from planetmodel.conversion import building, parameters

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# PREM with a radial viscosity, 2e21 Pa s below 670 km, and an anomaly
# that varies with longitude there.

# %%
cmb, d670 = 3480e3, 5701e3
model = pm.PREM()
for lay in model.layers:
    lo, hi = lay.interval
    if cmb <= lo and hi <= d670:
        model = model.with_field(lay.index, "viscosity",
                                 pm.constant_field((lo, hi), 2e21, name="viscosity"))
anomaly = pm.AnalyticField(
    (cmb, d670), lambda r, t, p: 0.01 * np.sin(t) ** 2 * np.cos(2 * p) + 0 * r)
for name in ("vsv", "vsh"):
    model = pm.perturbed(model, name, anomaly, kind="relative")
dt = building.temperature_anomaly(building.velocity_anomaly(model),
                                  scaling=parameters.SL3D_DENSITY_SCALING,
                                  expansivity=parameters.SL3D_EXPANSIVITY)

# %% [markdown]
# The same temperature anomaly under each law, and Milne's renormalised.

# %%
laws = {"eps = 0.01 / K (sl3d)": building.with_viscosity(model, dt,
                                                          parameters.SL3D_VISCOSITY),
        "eps = 0.04 / K (Milne et al.)": building.with_viscosity(
            model, dt, parameters.MILNE_2018_VISCOSITY),
        "eps = 0.04 / K, renormalised": building.with_viscosity(
            model, dt, parameters.MILNE_2018_VISCOSITY,
            renormalise=pm.gauss_legendre(16))}
r = 4500e3
i = model.skeleton.locate(r).layer
phi = np.linspace(0.0, 2 * np.pi, 361)
equator = np.full_like(phi, 0.5 * np.pi)

fig, ax = plt.subplots(figsize=(8, 4))
for label, m in laws.items():
    eta = m.layer(i)["viscosity"](np.full_like(phi, r), equator, phi)
    ax.plot(np.degrees(phi), np.log10(eta), label=label)
    print(f"{label}: {eta.min():.2e} to {eta.max():.2e} Pa s at 1871 km depth")
ax.axhline(np.log10(2e21), color="0.6", lw=0.8)
ax.set_xlabel("longitude on the equator (degrees)")
ax.set_ylabel("log10 viscosity (Pa s)")
ax.set_title("viscosity 1871 km deep")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "conversion_02.png", dpi=100)
print("figure written to", FIGURES / "conversion_02.png")
print(laws["eps = 0.01 / K (sl3d)"].cite())
