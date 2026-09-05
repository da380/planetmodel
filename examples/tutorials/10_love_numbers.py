# %% [markdown]
# # 10. Love numbers
#
# `planetmodel.loading` solves the quasi-static loading and tidal problem
# of a spherically symmetric, self-gravitating body on a radial mesh,
# degree by degree, and reports the Love numbers. A `Material` reads a
# model on a mesh once; `love_numbers` assembles and solves every degree
# for the three forcings; `solve_degree` returns the radial solution of
# one degree for interactive use; `LoveNumbers.write` makes the file
# pyslfp reads. A model frozen at a frequency has complex moduli, and the
# same code gives the Love numbers of any linear rheology.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure and the Love-number file are written to `examples/figures/`.

# %%
import time
from pathlib import Path

import numpy as np

from planetmodel import RadialMesh, constant_field, frozen, prem
from planetmodel.loading import (Material, love_numbers, read_love_numbers,
                                 solve_degree)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## PREM's elastic load Love numbers
#
# The loaded surface must be solid, so PREM is taken without its ocean.
# The mesh follows the `lmax` rule, no element wider than a tenth of the
# radius over `lmax + 1`, with five GLL nodes per element; the material
# holds density, gravity, fluidity and the transversely isotropic moduli
# at every node.

# %%
model = prem(ocean=False)
LMAX = 64
mesh = RadialMesh(model, ngll=5, lmax=LMAX)
material = Material(mesh, model)
print(mesh)
print(material)

t0 = time.time()
love = love_numbers(material, LMAX)
print(f"degrees 0..{LMAX} in {time.time() - t0:.2f} s:", love)

# %% [markdown]
# The conventional dimensionless load numbers h', l', k', with degree one
# in the centre-of-mass frame where k'_1 = -1 identically, and the
# geodetic tidal numbers at degree two.

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
# Two identities hold to solver precision: the reciprocity
# g h^phi = k^u of the two load channels, and mass conservation at
# degree zero, k_0 = -4 pi G a.

# %%
print("reciprocity residual:", love.reciprocity_residual().max())
k0 = love.k[0] / (-4.0 * np.pi * material.G * material.radius)
print("k_0 / (-4 pi G a) =", k0)

# %% [markdown]
# ## Convergence and the half-space limit
#
# Refining the mesh is the convergence check. At high degree the load
# sees only the crust and h' tends to the Boussinesq value for an elastic
# half-space with the upper crust's moduli.

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
print(f"half-space limit h'_inf = {hp_inf:+.3f};  h' at l = {LMAX}: "
      f"{conv['h'][LMAX]:+.3f}")

# %% [markdown]
# ## The radial solution of one degree
#
# `solve_degree` returns U, V and phi on the mesh for one forcing and
# evaluates them anywhere; the surface values are the Love numbers.

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
# Seven columns in SI, one row per degree from zero: `l, h_u, k_u, h_phi,
# k_phi, h_t, k_t`. The file round-trips through `read_love_numbers`.
# A non-dimensional model gives the same file, the conversion being by
# dimension.

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
# The solver's model is anything holding density and the five moduli,
# real or complex. A linear rheology enters by freezing the model at an
# angular frequency: `frozen` reads each layer's rheology from the fields
# it holds (`viscosity` for a Maxwell body, `qmu` and `qkappa` for a
# constant-Q band) and stores the complex moduli under `A, C, F, L, N`.
# PREM's own Q gives complex Love numbers a hair off the elastic ones at a
# semidiurnal tide; a Maxwell mantle runs from the elastic value at short
# periods to the fluid limit at long ones, with a loss peak between.

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
    ax.plot(km, U_load / abs(load2.surface[0]), label="U (load)")
    ax.plot(km, V_load / abs(load2.surface[0]), label="V (load)")
    ax.plot(km, U_tide / abs(tide2.surface[0]), "--", label="U (tide)")
    ax.plot(km, phi_tide / abs(tide2.surface[2]), "--", label="phi (tide)")
    for b in (1221.5, 3480.0):
        ax.axvline(b, color="0.8", lw=0.8)
    ax.set_xlabel("radius [km]")
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
