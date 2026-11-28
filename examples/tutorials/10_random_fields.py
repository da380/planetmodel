# %% [markdown]
# # 10. Random fields
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
# alone. `LayeredGRF` draws an independent field of the radius on each
# chosen layer of a model. `SphericalGRF` draws a field on a shell that
# varies with the angles as well as the radius.
#
# All three work in the same way. A sampler has a *basis*: a fixed list
# of functions, and every field the sampler can draw is a sum of those
# functions, each multiplied by a number. The numbers are called
# *coefficients*. The method `sample` draws the coefficients at random
# and returns them as one array, the *coefficient vector*. Everything
# else is worked out from that vector by other methods of the sampler:
# `synthesise` gives the field's values at the nodes of the sampler's
# mesh, `to_field` gives a field a model can hold, and for a shell
# `on_grid` gives the values on a grid of colatitude and longitude. The
# samplers for flat models, in `planetmodel.cartesian`, have the same
# methods; `examples/cartesian/11_a_random_field.py` uses them.
#
# Underneath the samplers are the operator family and its
# eigenfunctions, which are public too, so that other code (pygeoinf,
# for one) can build function spaces and probability measures on them.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import (PREM, SCALAR, Geometry, Model, Skeleton, equiangular,
                         gauss_legendre, slices, testing)
from planetmodel.harmonics import analyse_grid, synthesise_grid
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
# functions of radius. The example is a field on the mantle whose
# standard deviation grows with radius.
#
# `sample` returns the coefficient vector, one number for each function
# of the basis; `grf.size` is how many there are. `synthesise` turns
# the vector into the field's values at the nodes of the sampler's own
# mesh, which are the radii `grf.r`. The sampler is scaled so that the
# standard deviation at every node is exactly the `sigma` asked for.
# `to_field` turns the vector into a `RadialField` whose layer function
# is the polynomial through the nodal values on each element.
#
# With `size=3`, `sample` draws three vectors at once and returns them
# as the rows of one array, called a *batch*. Every method that takes a
# vector also takes a batch. `synthesise` then returns one row of
# values for each vector, and `to_field` returns a tuple of fields, one
# for each vector.

# %%
grf = RadialGRF(3480.0e3, a, 1.5, 400.0e3, sigma=lambda r: 0.02 * (r / a))
vectors = grf.sample(rng=rng, size=3)
samples = grf.synthesise(vectors)
print(grf)
print("coefficients:", vectors.shape, "| values at the nodes:", samples.shape)
field = grf.to_field(vectors[0], name="dv")
testing.check_field(field)
print("as a field at 5000 km:", field(5.0e6))
print("the field at the nodes is the first row of values:",
      np.allclose(field(grf.r), samples[0]))
print("fields from the whole batch:", len(grf.to_field(vectors)))

# %% [markdown]
# ## The basis behind a sampler
#
# The functions of the basis are the eigenfunctions of the sampler's
# operator: functions `phi_j` of radius that the operator merely
# multiplies by a number, the eigenvalue `theta_j`. They are found on
# the sampler's mesh, ordered by eigenvalue, and the smooth ones come
# first.
#
# The coefficients of a sample are independent random numbers. Each has
# zero mean and a standard deviation of its own, held in
# `grf.coefficient_std`, which is smaller for the rougher functions.
# The sum of the basis functions with those coefficients is then
# multiplied by a function of radius, `grf.scale`, and it is this last
# step that makes the standard deviation of the field exactly `sigma`.
#
# The mesh is longer than the interval asked for. The operator needs a
# condition at each end of its mesh, and any condition distorts the
# field near that end, so the mesh is extended by a *pad* on each side
# and the distortion happens there, outside the interval that matters.
# The interval asked for is called the physical interval.
#
# `grf.basis` is the `SpectralBasis` the sampler was built from. It
# holds the eigenvalues `theta` and gives the eigenfunctions on the
# whole mesh (`modes`), or at any radius of the physical interval
# (`evaluate`). Its own `synthesise` turns coefficients into values at
# the nodes, without the scaling, and `analyse` turns values on the
# whole mesh back into coefficients. `white_noise` draws one
# independent number of zero mean and unit standard deviation for each
# function of the basis; multiplied by `coefficient_std`, these are a
# sample made by hand. `weights` are the numbers that turn nodal values
# into an integral over the physical interval, here against `r^2 dr`.

# %%
basis = grf.basis
testing.check_spectral_basis(basis)
print(basis)
print("mesh:", grf.mesh.left[0] / 1e3, "to", grf.mesh.right[-1] / 1e3, "km;",
      "physical interval:", [x / 1e3 for x in basis.restriction.interval], "km")
print("a sample's values are the basis's synthesis times the scale:",
      np.allclose(samples[0], basis.synthesise(vectors[0]) * grf.scale))
coeffs = basis.white_noise(rng=rng) * grf.coefficient_std
values = basis.synthesise(coeffs, physical=False)
print("coefficients recovered from the values:",
      np.allclose(basis.analyse(values), coeffs))
r1, r2 = basis.restriction.interval
print("integral of r^2 over the mantle:", basis.weights().sum(),
      "against", (r2 ** 3 - r1 ** 3) / 3)
print("the first three modes at 5000 km:", basis.evaluate(5.0e6)[:3])

# %% [markdown]
# ## A field on the layers of a model
#
# `LayeredGRF` holds one `RadialGRF` for each chosen layer of a model,
# and the fields of different layers are independent. Its sample is one
# vector: the coefficient vectors of the chosen layers placed end to
# end, in order of layer. `layered.block(i)` says which part of the
# vector belongs to layer `i`. `to_fields` turns the vector into one
# `RadialField` for every layer of the model, zero on the layers left
# out. The result is discontinuous across the layer boundaries.
#
# Here PREM's density is given a random perturbation with a standard
# deviation of two per cent on every layer outside the core, mantle and
# crust alike, and none inside it. The perturbed densities are put back
# into the model with `with_field`.

# %%
mantle = [layer.index for layer in model.layers if layer.interval[0] >= 3480.0e3]
layered = LayeredGRF(model, 1.5, 300.0e3, sigma=0.02, layers=mantle, name="delta")
vector = layered.sample(rng=rng)
print(layered)
print(vector.size, "coefficients; those of layer", mantle[0], "are",
      layered.block(mantle[0]))
delta = layered.to_fields(vector)
perturbed = model
for i in mantle:
    rho = model.layer(i)["rho"]
    perturbed = perturbed.with_field(i, "rho", rho * (1 + delta[i]), replace=True)
testing.check_model(perturbed)
r = np.array([2.0e6, 4.0e6, 6.0e6])
for x in r:
    i = model.skeleton.locate(x).layer
    print(f"r = {x / 1e3:.0f} km: delta = {delta[i](x):+.4f}, rho "
          f"{model.layer(i)['rho'](x):.1f} -> {perturbed.layer(i)['rho'](x):.1f}")

# %% [markdown]
# ## A field on a spherical shell
#
# `SphericalGRF` builds the same kind of field on a shell, now varying
# with the angles as well as the radius. Each function of its basis is
# a product of two parts. One is a real spherical harmonic: a standard
# function of colatitude and longitude, labelled by a degree `l` and an
# order `m` between 0 and `l`, which varies with longitude `phi` as
# either `cos(m phi)` or `sin(m phi)`. The other is an eigenfunction of
# radius, as before. `lmax` is the largest degree used. A horizontal
# length scale `lam_h`, different from the radial one, gives structure
# that is correlated over a different distance horizontally than
# radially. As for the radial field, the sampler is scaled so that the
# standard deviation is exactly `sigma` at every point whose radius is
# a node of the mesh.
#
# `sample` again returns the coefficient vector. Its order is this: by
# degree; within a degree the cosine harmonics of orders 0 to `l` and
# then the sine harmonics of orders 1 to `l`; and for one harmonic the
# eigenfunctions of radius, smoothest first. `shell.basis.block(s, l, m)`
# says which part of the vector belongs to one harmonic, with `s = 0`
# for a cosine and `s = 1` for a sine.
#
# `synthesise` gives, for each harmonic, the function of radius that
# multiplies it, as values at the nodes. The result is an array of
# shape `(2, lmax + 1, lmax + 1, number of nodes)`, indexed by `s`, `l`,
# `m` and the node, which is the layout `planetmodel.harmonics` uses.
# `to_field` turns the vector into a `HarmonicField`: the sum of the
# harmonics, each multiplied by its function of radius, which can be
# evaluated at any point of the shell.
#
# The example is on a shell of unit outer radius so that the picture is
# easy to read. The standard deviation of a map at one radius is close
# to the requested value of 1, as one draw from the field should be.

# %%
shell = SphericalGRF(0.55, 1.0, 1.0, 0.06, lam_h=0.25, sigma=1.0, lmax=24)
vector = shell.sample(rng=rng)
functions = shell.synthesise(vector)
print(shell)
print("coefficients:", vector.shape, "| functions of radius:", functions.shape)
print("the coefficients of the harmonic of degree 3 that varies as cos(2 phi):",
      shell.basis.block(0, 3, 2))
random_field = shell.to_field(vector, character=SCALAR, name="dv")
testing.check_field(random_field)

# a model of the one shell holding the field, sliced on a sphere at one
# radius and through the centre (planetmodel.slices; tutorial 13)
shell_model = Model(Geometry(Skeleton([0.55, 1.0])), [{"dv": random_field}])
surface = slices.sphere(shell_model, "dv", radius=0.9, grid=equiangular(90, 180))
disc = slices.section(shell_model, "dv", (0.5 * np.pi, 0.0), azimuth=0.0,
                      length=2 * np.pi, n=60, m=361)
print("map std at r = 0.9:", surface.values.std().round(3), "(sigma = 1)")

# %% [markdown]
# `on_grid` gives the values of a vector's field on a grid of colatitude
# and longitude at every node of the sampler's mesh, as an array of
# nodes by grid points. It uses the fast spherical-harmonic transforms
# of ducc0. The grid is an argument because the sampler does not hold
# one: any grid of the kinds the library provides that resolves degree
# `lmax` will do, and here it is a Gauss-Legendre grid. A sampler for a
# flat model does hold its grid, so its `on_grid` takes the vector
# alone; that is the one difference between the two families.
# `sample_grid` draws a new sample and returns it on the grid in one
# step.
#
# The transforms themselves are reached through `planetmodel.harmonics`.
# `analyse_grid` goes the other way, from values on the grid to the
# coefficients of the harmonics, and recovers them exactly for a field
# with no harmonic above the grid's degree, so the two round-trip to
# rounding error. The average of the square of one sample over the
# sphere is an estimate of `sigma^2` from a single draw, and scatters
# around 1.

# %%
grid = gauss_legendre(shell.lmax)
on_grid = shell.on_grid(vector, grid)
at_node = random_field.evaluate(shell.r[10], grid.colatitudes[:, None],
                                grid.longitudes[None, :])
print("the sample on the grid:", on_grid.shape, "| against the field at one node:",
      np.abs(on_grid[10] - at_node).max())
drawn = shell.sample_grid(grid, rng=rng)
back = analyse_grid(drawn, grid)
print("a new sample on the grid:", drawn.shape, "| coefficients back:", back.shape,
      "| round trip error:", np.abs(synthesise_grid(back, grid) - drawn).max())
print("sphere average of u^2 at a node:",
      float(np.sum(drawn[10] ** 2 * grid.weights[:, None])
            * (2 * np.pi / grid.nphi) / (4 * np.pi)).__round__(3), "(sigma^2 = 1)")

# %% [markdown]
# `covariance(a, b)` gives the covariance of the field between two
# points, each given as `(r, theta, phi)`. The covariance is the
# average, over many samples, of the product of the field's values at
# the two points; between a point and itself it is the variance,
# `sigma^2`. It is worked out exactly from the basis, with nothing
# drawn. Here it is printed between a point in the middle of the shell,
# at the radius of the node of the mesh nearest to 0.8, and points at
# growing distances from it, first along the radius and then along a
# meridian. It starts at the variance, 1, and falls much faster along
# the radius, where the length scale is 0.06, than horizontally, where
# it is 0.25.

# %%
node = shell.r[np.argmin(np.abs(shell.r - 0.8))]
centre = np.array([node, 0.5 * np.pi, 0.0])
separation = np.linspace(0.0, 0.2, 5)
up = centre + separation[:, None] * np.array([1.0, 0.0, 0.0])
along = centre + separation[:, None] * np.array([0.0, 1.0 / centre[0], 0.0])
print("separation:      ", separation.round(2))
print("along the radius:", shell.covariance(centre, up).round(3))
print("along a meridian:", shell.covariance(centre, along).round(3))

# %% [markdown]
# ## The operator family underneath
#
# `SeparableOperatorFamily` holds the operator A = 1 - div(lambda^2 grad)
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

    from planetmodel.plotting import section, sphere_map
    axes[1, 0].remove()
    axes[1, 1].remove()
    try:
        import cartopy.crs as ccrs
        ax = fig.add_subplot(2, 2, 3, projection=ccrs.Robinson())
        sphere_map(ax, surface, cmap="RdBu_r", vmin=-3, vmax=3, colorbar=True)
        ax.set_title("shell field at r = 0.9")
    except ImportError:
        print("the map needs the maps extra (cartopy); left out")

    ax = fig.add_subplot(2, 2, 4, projection="polar")
    section(ax, disc, cmap="RdBu_r", vmin=-3, vmax=3, colorbar=True)
    ax.set_rticks([0.6, 0.8, 1.0])
    ax.set_title("the plane through the poles")
    fig.tight_layout()
    out = FIGURES / "tutorial_10_random_fields.png"
    fig.savefig(out, dpi=110)
    print("figure written to", out)

    # the first modes of the radial sampler on its padded mesh
    fig, ax = plt.subplots(figsize=(8, 4))
    x = grf.mesh.rglob / 1e3
    for j in range(5):
        ax.plot(x, basis.modes()[:, j], lw=0.9, label=f"mode {j}")
    ax.axvspan(r1 / 1e3, r2 / 1e3, color="0.9", label="physical interval")
    ax.set_xlabel("radius [km]")
    ax.set_ylabel("eigenfunction")
    ax.set_title("the first five modes; the pads lie outside the shaded band")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / "tutorial_10_basis.png"
    fig.savefig(out, dpi=110)
    print("figure written to", out)
