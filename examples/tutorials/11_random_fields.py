# %% [markdown]
# # 11. Random fields
#
# `planetmodel.randomfield` draws Gaussian random fields of Matern type
# on a ball, an annulus, or the layers of a skeleton, with the covariance
# defined through a radial elliptic operator on a spectral-element mesh.
# A `RadialGRF` is a field of radius alone; a `LayeredGRF` puts one on
# each chosen layer and returns model fields; a `SphericalGRF` is a field
# of a shell, delivered as spherical-harmonic coefficient functions and,
# through `to_field`, as a field a model can hold. The operator family
# beneath them is the piece pygeoinf wraps.
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
# `RadialGRF(r1, r2, nu, lam, sigma=...)` is a zero-mean field on
# `[r1, r2]` with smoothness `nu`, length scale `lam` and standard
# deviation `sigma`; the last two may vary with radius. A sample is the
# vector of nodal values at the field's own GLL nodes, exact in
# distribution to the requested `sigma` at every node, and `to_field`
# turns it into an exact `RadialField`.

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
# `LayeredGRF` draws an independent field on each chosen layer and returns
# one `RadialField` per layer: discontinuous at the layer boundaries and
# exactly zero on the layers left out. Here is PREM's density with a two
# per cent perturbation in the mantle and none in the core, put back into
# the model with `with_field`.

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
# `SphericalGRF` extends the construction to a shell: the sample is the
# set of real spherical-harmonic coefficient functions of radius, with the
# degree cut chosen so that the truncated field has the requested
# pointwise standard deviation. A horizontal length scale `lam_h` distinct
# from the radial one gives horizontally correlated structure. On a unit
# shell here, so the picture is easy to read.

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
# On a Gauss-Legendre grid the synthesis is a fast transform through
# pyshtools (the `harmonics` extra, with ducc0 as its backend where it
# is installed): `sample_grid` gives nodes times grid, the layout of a
# sample, and `analyse_grid` is its inverse for a band-limited field.

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
# `RadialOperatorFamily` is the degree-indexed operator A_l of
# A = 1 - div(lambda^2 grad) on a mesh, with its mass, spectrum, powers,
# inverse and white noise: the pieces a space, an operator and a
# Gaussian measure are made of.

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
    ax.set_ylabel("radius [km]")
    ax.set_title("radial samples inside the 2 sigma envelope")

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
