# %% [markdown]
# # 7. PREM
#
# `PREM` is the Preliminary Reference Earth Model of Dziewonski and
# Anderson (1981), built from the polynomials of their Table I. It has
# thirteen named layers, every field is an exact piecewise polynomial in
# SI units, and no file is read. It is a model type: a class derived from
# `Model` whose methods for elasticity, gravity and rheology are the
# library's free functions, as tutorial 6 described.
#
# This tutorial asks PREM the questions a seismologist asks: the values
# on either side of a discontinuity, which layers are fluid, the elastic
# moduli, gravity and mass, the isotropic and purely elastic versions of
# the model, the moduli at a frequency, values at the nodes of a radial
# mesh, and a sample on an angular grid.
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
#
# Each layer has a name, an interval of radius, and its own fields. A
# layer is fluid when its shear modulus is zero, which `is_fluid` reads
# from the fields.

# %%
model = PREM()
print(model)
for layer in model.layers:
    lo, hi = layer.interval
    print(f"{layer.index:2d} {layer.name:24s} {lo / 1e3:7.1f} - {hi / 1e3:7.1f} km  "
          f"{'fluid' if model.is_fluid(layer.index) else 'solid'}")

# %% [markdown]
# A discontinuity is not a special object. It is two layers that meet at
# a radius, and each is asked separately. The core-mantle boundary
# belongs to both the outer core and the lowermost mantle, and each gives
# its own value exactly at the boundary.

# %%
cmb = model.geometry.interface("cmb").radius
oc, dpp = model.layer("outer_core"), model.layer("lowermost_mantle")
for name in ("rho", "vpv", "vsv"):
    print(f"{name:4s} below {oc[name](cmb):10.2f}   above {dpp[name](cmb):10.2f}")

# %% [markdown]
# ## Transverse isotropy, and the moduli
#
# PREM is transversely isotropic between the Moho, at 24.4 km depth, and
# 220 km depth: the lid and the low-velocity zone. There the vertical
# and horizontal P and S velocities differ and the fifth parameter `eta`
# is not one. Everywhere else `vph = vpv`, `vsh = vsv` and `eta = 1` are
# held as exact constants.
#
# The table gives velocities. When the model is built, the `Elastic`
# mixin adds the five moduli of a transversely isotropic medium, the
# Love moduli `A, C, F, L, N`, to every layer as fields alongside the
# velocities. Here they are exact polynomials. The methods then read
# those fields: `moduli(which)` returns the five, `elastic_moduli` the
# tensor as a six-by-six Voigt matrix field, and `kappa_mu` the bulk and
# shear moduli of the isotropic average. The free functions of
# `planetmodel.materials` do the same on any layer.

# %%
lid = model.layer("lid")
print("the lid holds:", lid.names)
r80 = 6291e3 + 1.0
print("vph/vpv in the lid:", lid["vph"](r80) / lid["vpv"](r80))
A = lid["A"]
print("A in the lid is a polynomial of degree", A.function.degree,
      "| A = rho vph^2:", A(r80) == lid["rho"](r80) * lid["vph"](r80) ** 2)
print("the method reads the field:", model.moduli("lid")["A"] is A,
      "| the free function too:", moduli(lid)["A"] is A)
L_core = model.layer("outer_core")["L"]
print("the outer core's L is exactly zero:", L_core.function.is_zero())
C = model.elastic_moduli("lid")
print("Voigt matrix at 71 km depth, GPa, spherical frame:")
print(np.round(C(6300e3, 0.3, 0.0) / 1e9, 1))
kappa, mu = model.kappa_mu("lower_mantle")
print("kappa, mu at the top of the lower mantle (GPa):",
      kappa(5600e3) / 1e9, mu(5600e3) / 1e9)

# %% [markdown]
# ## Isotropic and elastic versions
#
# `isotropic()` returns a copy of the model in which every layer's
# elastic description is replaced by its isotropic average, computed
# exactly on the polynomials. The density and the Q fields are kept.
# `elastic()` returns a copy without the rheology fields, here the two
# Q's, so that no layer is viscoelastic. Both copies are still `PREM`.

# %%
iso = model.isotropic()
print("the lid now holds:", iso.layer("lid").names)
print("its tensor is", iso.elastic_moduli("lid").symmetry.name,
      "| mass unchanged:", np.isclose(iso.mass(), model.mass()))
print("elastic PREM is viscoelastic nowhere:",
      not any(model.elastic().is_viscoelastic(i) for i in range(model.nlayers)))

# %% [markdown]
# ## The moduli at a frequency
#
# PREM's elastic values are those at a period of one second, and its Q
# fields say how the moduli change with frequency. Two mixins give that
# frequency dependence.
#
# `ConstantQ` applies the constant-Q absorption band: the bulk and shear
# moduli are dispersed logarithmically about the reference frequency,
# which `reference_omega()` returns and which is one second for PREM.
# `moduli_at(which, omega)` returns a layer's five moduli at angular
# frequency `omega` as complex fields, and `elastic_moduli_at` the
# tensor. The model itself is not changed.
#
# `Viscoelastic` is the general machinery for any linear rheology.
# `frozen(omega)` returns a copy of the model at that frequency in which
# every viscoelastic layer holds complex `A, C, F, L, N`, from constant
# Q where the layer has Q fields and from a Maxwell body where it has a
# viscosity, with the frequency recorded as the model constant `omega`.
# The copy is still a `PREM`, and it is what the Love number solver of
# `planetmodel.loading` takes.
#
# The example looks at the shear modulus `L` in the lower mantle at two
# periods. Its imaginary part over its value at one second is one over
# the shear Q, and its real part falls by about two percent between one
# second and a semidiurnal tide.

# %%
lm = model.layer("lower_mantle")
r0 = 5000e3
omega_tide = 2 * np.pi / 43200.0                    # a semidiurnal tide
print("reference omega:", model.reference_omega(), "rad/s (a period of 1 s)")
for label, T in (("100 s", 100.0), ("12 h", 43200.0)):
    L = model.moduli_at("lower_mantle", 2 * np.pi / T)["L"](r0)
    print(f"L at 1371 km depth, {label:6s}: {L / 1e9:.4f} GPa   "
          f"Im L / L_1s = {L.imag / lm['L'](r0):.2e}",
          f"= 1/Q_mu = {1 / lm['qmu'](r0):.2e}")
change = model.moduli_at("lower_mantle", omega_tide)["L"](r0).real / lm["L"](r0) - 1
print(f"change in Re L from 1 s to 12 h: {100 * change:.2f} %")
tensor_tide = model.elastic_moduli_at("lower_mantle", omega_tide)
print("the model is untouched:", lm["L"].dtype,
      "| the tensor at 12 h:", tensor_tide.dtype)
tide = model.frozen(omega_tide)
print("frozen:", type(tide).__name__, "at omega =", tide.constant("omega"),
      "| its L is the ConstantQ one:",
      np.isclose(tide.layer("lower_mantle")["L"](r0),
                 model.moduli_at("lower_mantle", omega_tide)["L"](r0)))
print("a fluid layer's L stays zero:", tide.layer("outer_core")["L"](2000e3))

# %% [markdown]
# ## Gravity and mass
#
# `gravity` integrates `rho r^2` layer by layer through the layer
# functions, so for PREM it is exact. The gravitational constant is the
# model's own, in the model's units. After `nondimensionalised()` it is
# one, the numbers are of order one, and the copy is still a `PREM`.
#
# `with_gravity()` is a copy of the model with the same gravity attached
# to every layer as a radial field under the vocabulary name `g`. That
# field is exact for PREM, and it can be differentiated, sampled and
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
# A `RadialMesh` divides the radius into spectral elements, each within
# one layer, with Gauss-Lobatto-Legendre nodes in each element.
# `RadialMesh.nodal` evaluates a field at those nodes, element by
# element, taking each element's field from its own layer. A node on a
# layer boundary belongs to two elements and so carries both one-sided
# values. The argument `nu` asks for a radial derivative instead, and
# `nodal_gravity` gives gravity at the nodes. A field that some layer
# lacks is refused unless `missing="nan"` asks for NaN there.

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
# `sample` evaluates a model once on the product of the radial nodes and
# an angular grid, which is the form a numerical code wants its input
# in. A radial field is stored on the radial nodes alone, shape
# `(nnode,)` plus its components. A field that varies with the angles is
# stored with the two angular axes as well. Every array is read-only,
# and `check_sample` verifies the sample against the model.

# %%
grid = gauss_legendre(8)
s = sample(model, grid, radial=RadialMesh(model, ngll=4, drmax=500e3))
print(s.radial, "|", grid.ntheta, "colatitudes x", grid.nphi, "longitudes")
print("stored shapes:", {n: s.fields[n].shape for n in ("rho", "vpv")})
print("displacement:", s.displacement, "(identity geometry)")
testing.check_sample(s, model)
print("check_sample passes")

# %% [markdown]
# ## A picture
#
# `planetmodel.plotting` draws every profile of a spherically symmetric
# model the same way: radius on the vertical axis increasing upward, one
# curve per layer, and a line joining the two sides of each
# discontinuity. The third panel shows the constant-Q dispersion of the
# lower-mantle shear modulus against period.

# %%
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; no figure")
    raise SystemExit(0)

from planetmodel.plotting import radial_profile  # noqa: E402

fig, (left, middle, right) = plt.subplots(1, 3, figsize=(14, 6))
for name, color in (("rho", "k"), ("vpv", "C0"), ("vsv", "C3")):
    radial_profile(left, with_g, name, scale=1e-3, value_scale=1e-3, color=color,
                   lw=1.2)
left.set_xlabel("g/cm^3, km/s"); left.set_ylabel("r (km)"); left.legend()
left.set_title("PREM, one segment per layer")
radial_profile(middle, with_g, "g", scale=1e-3, lw=1.2)
middle.set_xlabel("g (m/s^2)"); middle.set_title("gravity")
# the shear modulus at one radius against period: the constant-Q band
periods = np.logspace(0.0, 6.0, 60)
Ls = np.array([model.moduli_at("lower_mantle", 2 * np.pi / T)["L"](r0)
               for T in periods])
right.semilogx(periods, (Ls.real / lm["L"](r0) - 1) * 100, "C0",
               label="Re L / L_1s - 1 (%)")
right.semilogx(periods, 100 * Ls.imag / Ls.real, "C3", label="100 Im L / Re L")
right.axvline(43200.0, color="0.85", lw=0.8, zorder=0.5)
right.set_xlabel("period (s)"); right.set_title("L at 1371 km depth, constant Q")
right.legend()
fig.tight_layout()
out = FIGURES / "tutorial_07_prem.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
