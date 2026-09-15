# %% [markdown]
# # 10. Love numbers
#
# A surface load, an ice sheet or an ocean say, presses on a planet and
# also pulls on it gravitationally. A tidal potential from another body
# pulls without pressing. A spherically symmetric, self-gravitating
# planet in static equilibrium responds to either with a displacement
# and a change in its own gravitational potential. When the forcing is
# expanded in spherical harmonics, each degree responds independently of
# the others, and the surface response per unit forcing at each degree
# is a small set of numbers, the Love numbers. The response to any load
# or tide is assembled from them. `planetmodel.loading` solves this
# problem on a radial mesh, one degree at a time.
#
# `Material` reads a model onto a mesh once. `love_numbers` solves every
# degree up to a chosen maximum, for a load and for a tide, and returns
# the Love numbers. `solve_degree` returns the radial solution of one
# degree so that it can be looked at. `LoveNumbers.write` writes the
# file that pyslfp, the sea-level code, reads. A model frozen at a
# frequency has complex moduli, and the same solver then gives the
# complex Love numbers of a viscoelastic body at that frequency.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure and the Love-number file are written to `examples/figures/`.

# %%
import time
from pathlib import Path

import numpy as np

from planetmodel import RadialMesh, constant_field, frozen, PREM
from planetmodel.loading import (Material, love_numbers, read_love_numbers,
                                 solve_degree)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## PREM's elastic load Love numbers
#
# A fluid surface cannot carry a load, so PREM is taken without its
# ocean. `RadialMesh` with `lmax` sizes the mesh for degrees up to
# `lmax`: no element is wider than a tenth of the radius divided by
# `lmax + 1`, and here each element has five nodes. The material holds
# the density, the gravity, whether each element is fluid, and the five
# transversely isotropic moduli at every node.

# %%
model = PREM(ocean=False)
LMAX = 64
mesh = RadialMesh(model, ngll=5, lmax=LMAX)
material = Material(mesh, model)
print(mesh)
print(material)

t0 = time.time()
love = love_numbers(material, LMAX)
print(f"degrees 0..{LMAX} in {time.time() - t0:.2f} s:", love)

# %% [markdown]
# `conventional` returns the dimensionless load Love numbers h', l' and
# k': the radial displacement, the tangential displacement and the
# potential perturbation at the surface per unit load at each degree,
# in the usual normalisation. Degree one is given in the centre-of-mass
# frame, where k'_1 is exactly -1. `tidal` returns the tidal Love
# numbers k, h and l, the response to a unit external potential; the
# degree-2 tidal k is the number quoted in geodesy.

# %%
conv = love.conventional()
print("  l        h'          l'          k'")
for l in (1, 2, 3, 4, 6, 8, 16, 32, 64):
    print(f"{l:4d}  {conv['h'][l]:+10.5f}  {conv['l'][l]:+10.5f}  "
          f"{conv['k'][l]:+10.5f}")
tidal = love.tidal()
print(f"tidal degree 2: k = {tidal['k'][2]:.4f}, h = {tidal['h'][2]:.4f}, "
      f"l = {tidal['l'][2]:.4f}")

# %% [markdown]
# Two identities hold to solver precision and are a check on every run.
# A load acts in two ways, by pressing on the surface and by attracting
# the body, and the reciprocity relation g h^phi = k^u links the
# displacement from the second to the potential from the first. At
# degree zero the potential change from a uniform surface load is fixed
# by mass conservation: k_0 = -4 pi G a.

# %%
print("reciprocity residual:", love.reciprocity_residual().max())
k0 = love.k[0] / (-4.0 * np.pi * material.G * material.radius)
print("k_0 / (-4 pi G a) =", k0)

# %% [markdown]
# ## Convergence and the half-space limit
#
# The convergence check is to refine the mesh and solve again. Here
# degree 32 is solved on a mesh with seven nodes per element and elements
# half as wide, and the surface displacement changes in the twelfth
# digit.
#
# At very high degree the load has a short wavelength and deforms only
# the crust, so h' tends to the value for an elastic half-space with the
# upper crust's moduli, the Boussinesq solution. At degree 64 the load
# still reaches well into the mantle and h' is far from that limit. A
# degree-4096 load, with a wavelength of about ten kilometres, is within
# a part in a thousand of it. Each degree below is one solve on a mesh
# sized for that degree, and takes under a second.

# %%
fine = RadialMesh(model, ngll=7, drmax=0.5 * mesh.drmax)
coarse = solve_degree(material, 32).surface
finer = solve_degree(Material(fine, model), 32).surface
print(f"l = 32, ngll 5 -> 7 and drmax halved: |dh|/|h| = "
      f"{abs(coarse[0] - finer[0]) / abs(finer[0]):.1e}")

crust = model.layer("upper_crust")
rho_c = crust["rho"](6368e3)
vp, vs = crust["vpv"](6368e3), crust["vsv"](6368e3)
mu_c = rho_c * vs ** 2
poisson = (vp ** 2 - 2 * vs ** 2) / (2 * (vp ** 2 - vs ** 2))
g = material.surface_gravity
hp_inf = -g ** 2 * (1.0 - poisson) / (2.0 * np.pi * material.G * mu_c)
print(f"half-space limit h'_inf = {hp_inf:+.3f}")
for l in (64, 256, 1024, 4096):
    m = RadialMesh(model, ngll=5, lmax=l)
    h_l = solve_degree(Material(m, model), l).surface[0]
    hp = h_l * g * (2 * l + 1) / (4.0 * np.pi * material.G * material.radius)
    print(f"  l = {l:5d}: h' = {hp:+.3f}")

# %% [markdown]
# ## The radial solution of one degree
#
# `solve_degree` solves one degree for one forcing and returns three
# functions of radius: U, the radial displacement; V, the tangential
# displacement; and phi, the potential perturbation. `surface` gives
# their values at the surface, which are the Love numbers of that
# degree, and `evaluate` gives them at any radii. Here degree 2 under a
# unit load and under a unit tide; the figure at the end draws them.

# %%
load2 = solve_degree(material, 2)
tide2 = solve_degree(material, 2, forcing="tide")
print("surface (U, V, phi) under a unit load:", load2.surface)
radii = np.linspace(0.0, material.radius, 600)
U_load, V_load, phi_load = load2.evaluate(radii)
U_tide, V_tide, phi_tide = tide2.evaluate(radii)

# %% [markdown]
# ## The file for pyslfp
#
# `write` writes a plain-text file with one row per degree from zero and
# seven columns in SI units: the degree, then the displacement and
# potential Love numbers for the pressing part of a load, for its
# attracting part, and for a tide. `read_love_numbers` reads it back.
# The Love numbers are computed in whatever units the model uses. The
# dimensionless numbers do not depend on that choice, and `write`
# converts to SI whatever the units were. Here the same degree-2 number
# from PREM made non-dimensional, where G is 1.

# %%
path = FIGURES / "tutorial_10_love_numbers.dat"
love.write(path)
back = read_love_numbers(path)
print("wrote", path.name, "; read back h_u[2] =", back.h_u[2], "vs", love.h_u[2])

nd = model.nondimensionalised()
love_nd = love_numbers(Material(RadialMesh(nd, ngll=5, lmax=8), nd), 8)
print("non-dimensional model, G =", love_nd.G, "; h'_2 =",
      love_nd.conventional()["h"][2], "vs", conv["h"][2])

# %% [markdown]
# ## A viscoelastic body is a model frozen at a frequency
#
# The solver accepts any model that holds a density and the five moduli,
# real or complex. A linear viscoelastic body forced at one angular
# frequency behaves like an elastic body whose moduli are complex and
# depend on that frequency. `frozen` builds that elastic body. It reads
# the rheology of each layer from the fields the layer holds, a
# `viscosity` for a Maxwell body or `qmu` and `qkappa` for constant-Q
# attenuation, and stores the complex moduli under the names
# `A, C, F, L, N`.
#
# Two examples. PREM with its own Q, frozen at the semidiurnal tidal
# period of 12 hours, gives complex Love numbers a fraction of a per
# cent away from the elastic ones. Then a Maxwell viscosity of 1e21 Pa s
# is given to every layer outside the core, and the degree-2 tidal k is
# computed at periods from minutes to 300 thousand years. It runs from
# the elastic value at short periods to the fluid limit at long ones,
# and the imaginary part, which measures the loss, peaks in between.

# %%
semidiurnal = 2.0 * np.pi / 43200.0
attenuating = love_numbers(Material(mesh, frozen(model, semidiurnal)), 2)
print("PREM with its Q at 12 h:", attenuating)
print("k_2^T =", attenuating.tidal()["k"][2], "vs elastic", tidal["k"][2])

visco = model
for layer in model.layers:
    if layer.interval[0] >= 3480e3:
        visco = visco.with_field(layer.index, "viscosity",
                                 constant_field(1e21, layer.interval, name="viscosity"))
small = RadialMesh(visco, ngll=5, lmax=8)
periods = np.logspace(2, 13, 45)               # seconds: minutes to 300 kyr
k2 = np.empty(periods.size, dtype=complex)
for i, T in enumerate(periods):
    cold = Material(small, frozen(visco, 2.0 * np.pi / T))
    k2[i] = love_numbers(cold, 2).tidal()["k"][2]
print(f"k_2^T at T = 12 h: {np.interp(43200.0, periods, k2.real):.4f}; "
      f"at T = 100 kyr: {np.interp(3.15e12, periods, k2.real):.4f} "
      f"(elastic {tidal['k'][2]:.4f})")

# %% [markdown]
# ## The figure

# %%
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ls = love.degree[1:].astype(float)
    ax.semilogx(ls, -conv["h"][1:], label="-h'")
    ax.semilogx(ls, -ls * conv["k"][1:], label="-l k'")
    ax.semilogx(ls, ls * conv["l"][1:], label="l l'")
    ax.set_xlabel("degree l")
    ax.set_title("PREM load Love numbers")
    ax.legend()

    ax = axes[1]
    km = radii / 1e3
    ax.plot(U_load / abs(load2.surface[0]), km, label="U (load)")
    ax.plot(V_load / abs(load2.surface[0]), km, label="V (load)")
    ax.plot(U_tide / abs(tide2.surface[0]), km, "--", label="U (tide)")
    ax.plot(phi_tide / abs(tide2.surface[2]), km, "--", label="phi (tide)")
    for b in (1221.5, 3480.0):
        ax.axhline(b, color="0.85", lw=0.8, zorder=0.5)
    ax.set_ylabel("radius [km]")
    ax.set_title("degree-2 solutions, scaled by their surface value")
    ax.legend()

    ax = axes[2]
    ax.semilogx(periods / 3.15576e7, k2.real, label="Re k_2^T")
    ax.semilogx(periods / 3.15576e7, -k2.imag, label="-Im k_2^T")
    ax.axhline(tidal["k"][2], color="0.6", lw=0.8, ls="--", label="elastic")
    ax.set_xlabel("period [years]")
    ax.set_title("Maxwell mantle, viscosity 1e21 Pa s")
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "tutorial_10_love_numbers.png"
    fig.savefig(out, dpi=110)
    print("wrote", out.name)
