# %% [markdown]
# # Random fields
#
# `planetmodel.randomfield` draws Gaussian random fields of Matern type
# on a ball, an annulus, or the layers of a skeleton, with the covariance
# defined through a radial elliptic operator on the spectral-element mesh
# (`planetmodel.sobolev`). It is kept in planetmodel for now so that a
# model can be perturbed by a field that respects its layers; both modules
# are meant to move to pygeoinf, the inference package, where they belong.
# This tutorial shows what they do.

# %%
from pathlib import Path

import numpy as np

from planetmodel import DENSITY, ComposedField, Dimensions, prem
from planetmodel.randomfield import LayeredGRF, RadialGRF, SphericalGRF

rng = np.random.default_rng(7)
model = prem(ocean=False)
a = float(model.skeleton.boundaries[-1])

# %% [markdown]
# ## A radial field
#
# `RadialGRF(r1, r2, nu, lam, sigma=...)` is a zero-mean field of radius
# on `[r1, r2]` with smoothness `nu`, length scale `lam` and standard
# deviation `sigma`; the last two may vary with radius. A sample is the
# vector of nodal values at the field's own GLL nodes, and `to_ppoly`
# turns it into an exact piecewise polynomial.

# %%
grf = RadialGRF(3480.0e3, a, nu=1.5, lam=400.0e3, sigma=lambda r: 0.02 * (r / a))
samples = grf.sample(rng, size=3)
print("nodes:", grf.r.size, "| sample shape:", samples.shape)
f = grf.to_ppoly(samples[0])
print("as a polynomial at 5000 km:", f(5.0e6))

# %% [markdown]
# ## A field on the layers of a model
#
# `LayeredGRF` draws an independent field on each chosen layer of a
# skeleton and returns a `RadialField` on it: discontinuous at the layer
# boundaries, exactly zero on the layers left out, and ready for field
# arithmetic. Here is PREM's density with a two per cent perturbation in
# the mantle and none in the core: a product of two fields has no
# character in general, so the product is spelled out with
# `ComposedField`, which says what the result is.

# %%
mantle = [lay.index for lay in model.layers
          if lay.state == "solid" and lay.interval[0] >= 3480.0e3]
layered = LayeredGRF(model.skeleton, nu=1.5, lam=300.0e3, sigma=0.02,
                     layers=mantle, name="delta")
delta = layered.sample(rng)
perturbed = ComposedField(lambda rho, d: rho * (1.0 + d), [model["rho"], delta],
                          character=DENSITY, dimensions=Dimensions.DENSITY,
                          name="rho_perturbed")
r = np.array([2.0e6, 4.0e6, 6.0e6])
print("delta at three radii:", np.round(delta.evaluate(r), 4))
print("rho and perturbed rho:", np.round(model["rho"].evaluate(r)),
      np.round(perturbed.evaluate(r)))

# %% [markdown]
# ## A field on a spherical shell
#
# `SphericalGRF` extends the construction to a shell: the sample is the
# set of real spherical-harmonic coefficient functions of radius, with the
# degree cut chosen so that the truncated field has the requested
# pointwise standard deviation. A horizontal length scale `lam_h` distinct
# from the radial one gives horizontally correlated structure.

# %%
shell = SphericalGRF(5.7e6, a, nu=1.0, lam=200.0e3, lam_h=800.0e3, sigma=1.0,
                     lmax=24)
coeffs = shell.sample(rng)
print("coefficients (2, lmax+1, lmax+1, nodes):", coeffs.shape)
print("degree-0 coefficient at the first five nodes:",
      np.round(coeffs[0, 0, 0, :5], 3))

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def figure(path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    for s in samples:
        ax.plot(grf.r / 1e3, s, lw=0.8)
    sig = 0.02 * grf.r / a
    ax.plot(grf.r / 1e3, 2 * sig, "k--", lw=0.8)
    ax.plot(grf.r / 1e3, -2 * sig, "k--", lw=0.8)
    ax.set_xlabel("radius [km]")
    ax.set_title("radial samples inside the 2 sigma envelope")
    ax = axes[1]
    rr = np.linspace(0.0, a, 2000)
    ax.plot(rr / 1e3, model["rho"].evaluate(rr), "0.6", label="PREM")
    ax.plot(rr / 1e3, perturbed.evaluate(rr), lw=0.8, label="perturbed mantle")
    ax.set_xlabel("radius [km]")
    ax.set_ylabel("density [kg/m^3]")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110)
    print("wrote", path.name)


figure(FIGURES / "tutorial_07_random_fields.png")
