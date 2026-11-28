# %% [markdown]
# # Power-law creep beside a viscosity
#
# A code that steps in time can let a relaxation time fall with stress:
# AdGIA, a glacial isostatic adjustment code, divides it by
# 1 + gamma (|dev sigma| / 2 mu_0)^(n - 1), so that creep is linear at
# low stress and power-law at high. A model carries the two parameters
# a code needs, the stress exponent n and the transition stress sigma_t
# at which the two kinds of creep strain equally fast;
# gamma = (2 mu_0 / sigma_t)^(n - 1) follows.
#
# Needs: none
# Network: none
# Run: poetry run python examples/conversion/08_power_law.py

# %%
import numpy as np

import planetmodel as pm
from planetmodel.conversion import building, profiles

model = pm.PREM()
for lay in model.layers:
    lo, hi = lay.interval
    if 3480e3 <= lo and hi <= 6291e3:
        model = model.with_field(lay.index, "viscosity", pm.constant_field(
            (lo, hi), 1e21, name="viscosity"))

# %% [markdown]
# n = 3.5 everywhere, and a transition stress falling from 1 MPa at the
# core-mantle boundary to 0.1 MPa below the lithosphere.

# %%
stress = profiles.tabulated([3480e3, 6291e3], [1e6, 1e5])
model = building.with_power_law(model, exponent=3.5, transition_stress=stress)
lay = model.layer("lower_mantle")
r = np.array([5000e3])
n = lay["stress_exponent"](r)[0]
sigma_t = lay["transition_stress"](r)[0]
mu0 = pm.kappa_mu(lay)[1](r)[0]
print(f"at 1371 km: n = {n}, transition stress {sigma_t / 1e6:.2f} MPa, "
      f"gamma = (2 mu_0 / sigma_t)^(n - 1) = {(2 * mu0 / sigma_t) ** (n - 1):.3g}")
print("held on", [lay.name for lay in model.layers if "stress_exponent" in lay])
