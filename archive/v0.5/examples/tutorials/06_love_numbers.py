# %% [markdown]
# # Love numbers
#
# `planetmodel.loading` solves the quasi-static loading problem of a
# spherically symmetric, self-gravitating elastic planet by radial
# spectral elements, degree by degree, and reports the load Love numbers.
# It is kept in planetmodel for now so that a model can be checked
# end-to-end against a physical prediction, and it is meant to move to
# pyslfp, the surface-loading package, where it belongs. This tutorial
# shows what it does, not how it does it.

# %%
import time
from pathlib import Path

import numpy as np

from planetmodel import RadialMesh, prem
from planetmodel.loading import love_numbers, solve_degree
from planetmodel.mesh1d import G_NEWTON

model = prem(ocean=False)
LMAX = 64
t0 = time.time()
res = love_numbers(model, lmax=LMAX)
print(res["mesh"])
print(f"degrees 1..{LMAX} in {time.time() - t0:.1f} s")

# %% [markdown]
# The conventional dimensionless load Love numbers `h'`, `l'`, `k'`, with
# degree one in the centre-of-mass frame, where `k'_1 = -1` identically.

# %%
print("  l        h'          l'          k'")
for l in (1, 2, 3, 4, 6, 8, 16, 32, 64):
    i = l - 1
    print(f"{l:4d}  {res['hp'][i]:+10.5f}  {res['lp'][i]:+10.5f}  "
          f"{res['kp'][i]:+10.5f}")

# %% [markdown]
# The solver works on a `RadialMesh`, the one-dimensional spectral-element
# mesh that honours the model's skeleton. Refining it is the convergence
# check: the same degree on a finer mesh moves the answer by less than a
# part in a million.

# %%
mesh = res["mesh"]
fine = RadialMesh(model, ngll=7, drmax=0.5 * mesh.drmax)
h1, v1, k1 = solve_degree(mesh, 32)
h2, v2, k2 = solve_degree(fine, 32)
print(f"l = 32, ngll 5 -> 7 and drmax halved: |dh|/|h| = {abs(h1 - h2) / abs(h2):.1e}")

# %% [markdown]
# At high degree the load sees only the crust, and `h'` tends to the
# elastic half-space (Boussinesq) value with the upper crust's moduli.

# %%
rho_c, vp, vs = 2600.0, 5800.0, 3200.0                # PREM's upper crust
mu_c = rho_c * vs ** 2
nu = (vp ** 2 - 2 * vs ** 2) / (2 * (vp ** 2 - vs ** 2))
g = float(mesh.nodal_gravity()[-1, -1])
hp_inf = -g ** 2 * (1.0 - nu) / (2.0 * np.pi * G_NEWTON * mu_c)
print(f"half-space limit h'_inf = {hp_inf:+.3f};  h' at l = {LMAX}: "
      f"{res['hp'][LMAX - 1]:+.3f}")

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def love_figure(path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ls = res["l"].astype(float)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogx(ls, -res["hp"], label="-h'")
    ax.semilogx(ls, -ls * res["kp"], label="-l k'")
    ax.semilogx(ls, ls * res["lp"], label="l l'")
    ax.axhline(-hp_inf, color="0.6", lw=0.8, ls="--", label="half-space -h'")
    ax.set_xlabel("degree l")
    ax.set_ylabel("load Love numbers")
    ax.set_title("PREM (oceanless) elastic load Love numbers")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110)
    print("wrote", path.name)


love_figure(FIGURES / "tutorial_06_love_numbers.png")
