# %% [markdown]
# # 7. PREM
#
# `PREM` is the Preliminary Reference Earth Model built from the Table I
# polynomials of Dziewonski and Anderson (1981): thirteen named layers,
# every field an exact piecewise polynomial in SI, and no file read. It is
# a model type: a class derived from `Model` with the elastic, gravity and
# rheology behaviours as methods, each of which is one of the library's
# free functions. This tutorial asks it the questions a seismologist asks:
# values on both sides of a discontinuity, which layers are fluid, the
# elastic moduli, gravity and mass, the isotropic and elastic versions,
# nodal values on a radial mesh, and a sample on an angular grid.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import PREM, RadialMesh, gauss_legendre, moduli, sample, testing

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## The layers by name

# %%
model = PREM()
print(model)
for layer in model.layers:
    lo, hi = layer.interval
    print(f"{layer.index:2d} {layer.name:24s} {lo / 1e3:7.1f} - {hi / 1e3:7.1f} km  "
          f"{'fluid' if model.is_fluid(layer.index) else 'solid'}")

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
A = model.moduli("lid")["A"]
print("A in the lid is a polynomial of degree", A.function.degree)
print("the method is the free function:", A(6300e3) == moduli(lid)["A"](6300e3))
C = model.elastic_moduli("lid")
print("Voigt matrix at 71 km depth, GPa, spherical frame:")
print(np.round(C(6300e3, 0.3, 0.0) / 1e9, 1))
kappa, mu = model.kappa_mu("lower_mantle")
print("kappa, mu at the top of the lower mantle (GPa):",
      kappa(5600e3) / 1e9, mu(5600e3) / 1e9)

# %% [markdown]
# ## Isotropic and elastic versions
#
# `isotropic()` replaces every layer's elastic description by its Voigt
# average, exactly, keeping `rho` and the Q fields; `elastic()` drops the
# rheology fields. Both are copies of the same class.

# %%
iso = model.isotropic()
print("the lid now holds:", iso.layer("lid").names)
print("its tensor is", iso.elastic_moduli("lid").symmetry.name,
      "| mass unchanged:", np.isclose(iso.mass(), model.mass()))
print("elastic PREM is viscoelastic nowhere:",
      not any(model.elastic().is_viscoelastic(i) for i in range(model.nlayers)))

# %% [markdown]
# ## Gravity and mass
#
# `gravity` integrates `rho r^2` layer by layer through the layer
# functions, so for PREM it is exact. `G` is the model's; after
# `nondimensionalised()` it is one and the numbers are order one, and the
# copy is still a `PREM`. `with_gravity()` is the model with the same
# gravity attached to every layer as a radial field under the vocabulary
# name `g`, exact for PREM, so it is held, differentiated, sampled and
# drawn like any other field.

# %%
print(f"mass: {model.mass():.4e} kg   surface g: {model.gravity(6371e3):.4f} m/s^2   "
      f"g at the CMB: {model.gravity(cmb):.4f}")
nd = model.nondimensionalised()
print("non-dimensional: G =", nd.G, " mass =", nd.mass(), " g(1) =", nd.gravity(1.0),
      "|", type(nd).__name__)
with_g = model.with_gravity()
g_mantle = with_g.layer("lower_mantle")["g"]
print("g as a field: the lower mantle holds", with_g.layer("lower_mantle").names)
print("dg/dr at 4000 km:", g_mantle.derivative()(4000e3), "1/s^2")

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
#
# `planetmodel.plotting` draws every profile of a spherically symmetric
# model one way: radius on the vertical axis increasing upward, one
# segment per layer, and a line joining the two sides of each
# discontinuity.

# %%
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; no figure")
    raise SystemExit(0)

from planetmodel.plotting import radial_profile  # noqa: E402

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
for name, color in (("rho", "k"), ("vpv", "C0"), ("vsv", "C3")):
    radial_profile(left, with_g, name, scale=1e-3, value_scale=1e-3, color=color,
                   lw=1.2)
left.set_xlabel("g/cm^3, km/s"); left.set_ylabel("r (km)"); left.legend()
left.set_title("PREM, one segment per layer")
radial_profile(right, with_g, "g", scale=1e-3, lw=1.2)
right.set_xlabel("g (m/s^2)"); right.set_title("gravity")
fig.tight_layout()
out = FIGURES / "tutorial_07_prem.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
