# %% [markdown]
# # 7. PREM
#
# `prem()` is the Preliminary Reference Earth Model built from the Table I
# polynomials of Dziewonski and Anderson (1981): thirteen named layers,
# every field an exact piecewise polynomial in SI, and no file read. This
# tutorial asks it the questions a seismologist asks: values on both sides
# of a discontinuity, which layers are fluid, the elastic moduli, gravity
# and mass, nodal values on a radial mesh, and a sample on an angular grid.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import (RadialMesh, elastic_moduli, gauss_legendre, gravity,
                         is_fluid, kappa_mu, mass, moduli, prem, sample, testing)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## The layers by name

# %%
model = prem()
print(model)
for layer in model.layers:
    lo, hi = layer.interval
    print(f"{layer.index:2d} {layer.name:24s} {lo / 1e3:7.1f} - {hi / 1e3:7.1f} km  "
          f"{'fluid' if is_fluid(layer) else 'solid'}")

# %% [markdown]
# A discontinuity is two layers asked separately. The CMB belongs to both
# the outer core and the lowermost mantle, and each gives its own value
# exactly at the boundary.

# %%
cmb = model.geometry.interface("cmb").radius
oc, dpp = model.layer("outer_core"), model.layer("lowermost_mantle")
for name in ("rho", "vpv", "vsv"):
    print(f"{name:4s} below {oc[name](cmb):10.2f}   above {dpp[name](cmb):10.2f}")

# %% [markdown]
# ## Transverse isotropy, and the moduli
#
# Between 80 and 220 km depth PREM is transversely isotropic; elsewhere
# `vph = vpv`, `vsh = vsv` and `eta = 1` as exact constants. `moduli`
# reads whatever a layer holds and gives `A, C, F, L, N` as fields of
# weight 1, exact polynomials here; `elastic_moduli` is the Voigt matrix
# field, and `kappa_mu` the Voigt averages.

# %%
lid = model.layer("lid")
r80 = 6291e3 + 1.0
print("vph/vpv in the lid:", lid["vph"](r80) / lid["vpv"](r80))
A = moduli(lid)["A"]
print("A in the lid is a polynomial of degree", A.function.degree)
C = elastic_moduli(lid)
print("Voigt matrix at 71 km depth, GPa, spherical frame:")
print(np.round(C(6300e3, 0.3, 0.0) / 1e9, 1))
kappa, mu = kappa_mu(model.layer("lower_mantle"))
print("kappa, mu at the top of the lower mantle (GPa):",
      kappa(5600e3) / 1e9, mu(5600e3) / 1e9)

# %% [markdown]
# ## Gravity and mass
#
# `gravity` integrates `rho r^2` layer by layer through the layer
# functions, so for PREM it is exact. `G` is the model's; after
# `nondimensionalised()` it is one and the numbers are order one.

# %%
print(f"mass: {mass(model):.4e} kg   surface g: {gravity(model, 6371e3):.4f} m/s^2   "
      f"g at the CMB: {gravity(model, cmb):.4f}")
nd = model.nondimensionalised()
print("non-dimensional: G =", nd.G, " mass =", mass(nd), " g(1) =", gravity(nd, 1.0))

# %% [markdown]
# ## On a radial mesh
#
# `RadialMesh.nodal` evaluates the field of each element's own layer at
# its nodes, so a node on a boundary carries both one-sided values; `nu`
# asks for a radial derivative, and `nodal_gravity` gives gravity there.
# A name a layer lacks is refused unless `missing="nan"`.

# %%
mesh = RadialMesh(model, ngll=5, drmax=200e3)
rho_n = mesh.nodal(model, "rho")
print(mesh, "| nodal rho:", rho_n.shape)
e = mesh.element_at(cmb)
print("rho at the CMB from below and above:", rho_n[e - 1, -1], rho_n[e, 0])
qmu = mesh.nodal(model, "qmu", missing="nan")
print("qmu is NaN on", int(np.isnan(qmu).all(axis=1).sum()), "fluid elements")

# %% [markdown]
# ## A sample on an angular grid
#
# `sample` evaluates a model once on the radial nodes times an angular
# grid, the delivery a numerical code wants. A radial field is stored as
# `(nnode,)` plus its components; one that depends on direction gets the
# angular axes. Every array is read-only and `check_sample` holds it to
# the model.

# %%
grid = gauss_legendre(8)
s = sample(model, grid, radial=RadialMesh(model, ngll=4, drmax=500e3))
print(s.radial, "|", grid.ntheta, "x", grid.nphi, "directions")
print("stored shapes:", {n: s.fields[n].shape for n in ("rho", "vpv")})
print("displacement:", s.displacement, "(identity geometry)")
testing.check_sample(s, model)
print("check_sample passes")

# %% [markdown]
# ## A picture

# %%
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; no figure")
    raise SystemExit(0)

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
for layer in model.layers:
    lo, hi = layer.interval
    r = np.linspace(lo, hi, 60)
    km = r / 1e3
    first = layer.index == 0
    left.plot(km, layer["rho"](r) / 1e3, "k", lw=1.2, label="rho" if first else None)
    left.plot(km, layer["vpv"](r) / 1e3, "C0", lw=1.2, label="vpv" if first else None)
    left.plot(km, layer["vsv"](r) / 1e3, "C3", lw=1.2, label="vsv" if first else None)
left.set_xlabel("r (km)"); left.set_ylabel("g/cm^3, km/s"); left.legend()
left.set_title("PREM, one segment per layer")
rr = np.linspace(0.0, 6371e3, 400)
right.plot(rr / 1e3, gravity(model, rr))
right.set_xlabel("r (km)"); right.set_ylabel("g (m/s^2)"); right.set_title("gravity")
fig.tight_layout()
out = FIGURES / "tutorial_07_prem.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
