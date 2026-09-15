# %% [markdown]
# # 11. Random fields
#
# A random field is a function whose values are drawn at random with a
# prescribed statistical structure. The fields here are Gaussian with
# zero mean, a chosen standard deviation, and a correlation between
# nearby points that falls off over a length scale. Fields of Matern
# type are the standard family for this: besides the length scale they
# have a smoothness parameter `nu`, and `planetmodel.randomfield` builds
# them on a ball, on a spherical shell, or on the layers of a skeleton.
# The construction is the one through a stochastic partial differential
# equation: the covariance is a power of an elliptic operator, which is
# discretised on a radial spectral-element mesh, one spherical-harmonic
# degree at a time.
#
# Three samplers are provided. `RadialGRF` draws a field of the radius
# alone. `LayeredGRF` draws an independent field on each chosen layer of
# a model and returns fields the model can hold. `SphericalGRF` draws a
# field on a shell, delivered as spherical-harmonic coefficients that
# are functions of radius, and its `to_field` turns one into a field a
# model can hold. The operator family underneath them is what pygeoinf
# will build on.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import SCALAR, PREM, testing
from planetmodel.randomfield import LayeredGRF, RadialGRF, SphericalGRF

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
model = PREM(ocean=False)
a = model.skeleton.boundaries[-1]

# %% [markdown]
# ## A field of radius
#
# `RadialGRF(r1, r2, nu, lam, sigma=...)` is a zero-mean Gaussian field
# on the interval `[r1, r2]` with smoothness `nu`, length scale `lam`
# and standard deviation `sigma`. The last two may be numbers or
# functions of radius. A sample is the vector of the field's values at
# the nodes of its own mesh, and the sampler is scaled so that the
# standard deviation at every node is exactly the `sigma` asked for.
# `to_field` turns a sample into a `RadialField` whose layer function is
# the polynomial through the nodal values on each element. The example
# is a field on the mantle whose standard deviation grows with radius.

# %%
grf = RadialGRF(3480.0e3, a, 1.5, 400.0e3, sigma=lambda r: 0.02 * (r / a))
samples = grf.sample(rng=rng, size=3)
print(grf, "| samples:", samples.shape)
field = grf.to_field(samples[0], name="dv")
testing.check_field(field)
print("as a field at 5000 km:", field(5.0e6))

# %% [markdown]
# ## A field on the layers of a model
#
# `LayeredGRF` draws an independent field on each chosen layer and
# returns one `RadialField` per layer of the model, zero on the layers
# left out. The result is discontinuous across the layer boundaries.
# Here PREM's density is given a random perturbation with a standard
# deviation of two per cent on every layer outside the core, mantle and
# crust alike, and none inside it. The perturbed densities are put back
# into the model with `with_field`.

# %%
mantle = [layer.index for layer in model.layers if layer.interval[0] >= 3480.0e3]
layered = LayeredGRF(model, 1.5, 300.0e3, sigma=0.02, layers=mantle, name="delta")
delta = layered.sample(rng=rng)
perturbed = model
for i in mantle:
    rho = model.layer(i)["rho"]
    perturbed = perturbed.with_field(i, "rho", rho + rho * delta[i], replace=True)
testing.check_model(perturbed)
r = np.array([2.0e6, 4.0e6, 6.0e6])
for x in r:
    i = model.skeleton.locate(x).layer
    print(f"r = {x / 1e3:.0f} km: delta = {delta[i](x):+.4f}, rho "
          f"{model.layer(i)['rho'](x):.1f} -> {perturbed.layer(i)['rho'](x):.1f}")

# %% [markdown]
# ## A field on a spherical shell
#
# `SphericalGRF` builds the same kind of field on a shell. A sample is
# the set of real spherical-harmonic coefficients up to degree `lmax`,
# each a function of radius. As for the radial field, the sampler is
# scaled so that the standard deviation at any point of the truncated
# field is exactly `sigma`. A horizontal length scale `lam_h`, different
# from the radial one, gives structure that is correlated over a
# different distance horizontally than radially. The example is on a
# shell of unit outer radius so that the picture is easy to read. The
# standard deviation of a map at one radius is close to the requested
# value of 1, as one draw from the field should be.

# %%
shell = SphericalGRF(0.55, 1.0, 1.0, 0.06, lam_h=0.25, sigma=1.0, lmax=24)
coeffs = shell.sample(rng=rng)
print(shell, "| coefficients:", coeffs.shape)
random_field = shell.to_field(coeffs, character=SCALAR, name="dv")
testing.check_field(random_field)

# a map at one radius and a meridional slice through the shell
lat = np.linspace(-89.5, 89.5, 90)
lon = np.linspace(0.0, 360.0, 181)
theta = np.deg2rad(90.0 - lat)[:, None]
phi = np.deg2rad(lon)[None, :]
surface_map = random_field.evaluate(0.9, theta, phi)
rr = np.linspace(0.55, 1.0, 60)[:, None]
tt = np.linspace(0.0, np.pi, 181)[None, :]
slice_values = random_field.evaluate(rr, tt, 0.0)
print("map std at r = 0.9:", surface_map.std().round(3), "(sigma = 1)")

# %% [markdown]
# `sample_grid` draws a sample directly on a Gauss-Legendre grid, as an
# array of mesh nodes by grid points, using the fast spherical-harmonic
# transform of pyshtools (the `harmonics` extra, with ducc0 as its
# backend where that is installed). `analyse_grid` is the inverse
# transform for a field band-limited at the grid's degree, and the two
# round-trip to rounding error. The average of the square of one sample
# over the sphere is an estimate of `sigma^2` from a single draw, and
# scatters around 1.

# %%
try:
    from planetmodel import analyse_grid, gauss_legendre, synthesise_grid
    grid = gauss_legendre(shell.lmax)
    on_grid = shell.sample_grid(grid, rng=rng)
    back = analyse_grid(on_grid, grid)
    print("on the grid:", on_grid.shape, "| coefficients back:", back.shape,
          "| round trip error:", np.abs(synthesise_grid(back, grid) - on_grid).max())
    print("sphere average of u^2 at a node:",
          float(np.sum(on_grid[10] ** 2 * grid.weights[:, None])
                * (2 * np.pi / grid.nphi) / (4 * np.pi)).__round__(3), "(sigma^2 = 1)")
except ImportError as exc:
    print("skipped:", str(exc).splitlines()[0])

# %% [markdown]
# ## The operator family underneath
#
# `RadialOperatorFamily` holds the operator A = 1 - div(lambda^2 grad)
# on the mesh as one matrix A_l for each spherical-harmonic degree l,
# together with its mass matrix, its eigenvalues and eigenvectors, its
# powers, its inverse and its white noise. These are the pieces from
# which pygeoinf builds a function space, an operator on it and a
# Gaussian measure. Here the smallest eigenvalues at degree 4, and a
# check that applying A_4 undoes applying its inverse.

# %%
family = shell.family
theta_l, Phi_l = family.eig(4)
print("degree 4:", family.ndof(4), "dofs, smallest eigenvalues",
      theta_l[:3].round(4))
noise = family.white_noise(4, rng=rng)
smooth = family.apply_power(4, -1.0, noise)
print("A_4^-1 white noise, then A_4 back:",
      np.abs(family.apply(4, smooth) - noise).max())

# %% [markdown]
# ## The figure

# %%
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt is not None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for s in samples:
        ax.plot(s, grf.r / 1e3, lw=0.8)
    sig = 0.02 * grf.r / a
    ax.plot(2 * sig, grf.r / 1e3, "k--", lw=0.8)
    ax.plot(-2 * sig, grf.r / 1e3, "k--", lw=0.8)
    ax.set_xlabel("sample value")
    ax.set_ylabel("radius [km]")
    ax.set_title("three radial samples and the 2 sigma envelope")

    ax = axes[0, 1]
    from planetmodel.plotting import radial_profile
    for m, color, label in ((model, "0.6", "PREM"),
                            (perturbed, "C0", "perturbed mantle")):
        radial_profile(ax, m, "rho", scale=1e-3, n=400, color=color, lw=0.9,
                       label=label)
    ax.set_xlabel("density [kg/m^3]")
    ax.set_ylabel("radius [km]")
    ax.legend()

    ax = axes[1, 0]
    im = ax.pcolormesh(lon, lat, surface_map, cmap="RdBu_r", vmin=-3, vmax=3,
                       shading="auto")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("shell field at r = 0.9")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 1]
    x = rr * np.sin(tt)
    z = rr * np.cos(tt)
    im = ax.pcolormesh(x, z, slice_values, cmap="RdBu_r", vmin=-3, vmax=3,
                       shading="gouraud")
    ax.set_aspect("equal")
    ax.set_title("meridional slice, phi = 0")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = FIGURES / "tutorial_11_random_fields.png"
    fig.savefig(out, dpi=110)
    print("wrote", out.name)
