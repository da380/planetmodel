# %% [markdown]
# # Prony branches
#
# A generalised Maxwell body relaxes its shear modulus in branches, each a
# fraction of the modulus with its own time. One branch is the Maxwell
# body; a Burgers body is exactly two; an Andrade body has no finite
# form, so a fitted set of branches stands in for it. Times are in units
# of the Maxwell time eta / mu, so one set serves every point of a model.
#
# Needs: plot
# Network: none
# Run: poetry run python examples/conversion/04_prony_branches.py

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import planetmodel as pm
from planetmodel.conversion import prony

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

series = [prony.maxwell_series(),
          prony.burgers_series(kelvin_modulus_fraction=0.2, viscosity_ratio=5.0),
          prony.andrade_series(1.0 / 3.0)]
for s in series:
    print(f"{s.name}: {len(s.strengths)} branches, misfit {s.misfit:.2%}")

# %%
omega = np.logspace(-3, 3, 200)
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
for s in series:
    mu = s.modulus(omega)
    left.loglog(omega, np.abs(mu), label=s.name.split(" (")[0])
    right.loglog(omega, mu.imag / mu.real, label=s.name.split(" (")[0])
left.set_xlabel("omega times the Maxwell time")
left.set_ylabel("|mu| / mu_U")
right.set_xlabel("omega times the Maxwell time")
right.set_ylabel("Im mu / Re mu, the loss")
left.legend()
fig.tight_layout()
fig.savefig(FIGURES / "conversion_04.png", dpi=100)
print("figure written to", FIGURES / "conversion_04.png")

# %% [markdown]
# On a model: PREM given a viscosity, the Burgers body put in its place,
# and the model frozen at a period of 100 years. At the radius looked
# at, 5000 km, the lower mantle's Maxwell time is about 300 years, so
# the Maxwell body has barely begun to relax, while the Burgers body's
# faster branch, whose time is 0.146 Maxwell times, about 44 years, has.

# %%
model = pm.PREM()
i = model.layer("lower_mantle").index
model = model.with_field(i, "viscosity", pm.constant_field(model.layer(i).interval,
                                                           2e21, name="viscosity"))
burgers = prony.with_branches(model, series[1])
omega = 2 * np.pi / (100 * 365.25 * 86400)
r = np.array([5000e3])
maxwell_time = 2e21 / model.layer(i)["N"](r)[0] / (365.25 * 86400)
print(f"Maxwell time at {r[0] / 1e3:.0f} km: {maxwell_time:.0f} years; the Burgers "
      f"body's faster branch {min(series[1].times):.3f} of it, "
      f"{min(series[1].times) * maxwell_time:.0f} years")
for label, m in (("Maxwell", model), ("Burgers", burgers)):
    n = m.frozen(omega).layer(i)["N"](r)[0] / model.layer(i)["N"](r)[0]
    print(f"{label}: |N| / N_elastic = {abs(n):.3f}, loss {n.imag / n.real:.3f}")
