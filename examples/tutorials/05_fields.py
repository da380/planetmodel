# %% [markdown]
# # 5. Fields on one interval
#
# A field is data on one layer: an interval, a character that says how it
# transforms, a name, and `evaluate(r, theta, phi, *, frame)`. It knows no
# skeleton, no units and no other layer. This tutorial builds fields from
# polynomials, from callables and from formulas, does arithmetic on them,
# shows where that arithmetic is exact, and pushes a field forward through
# a mapping.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import (DENSITY, ELASTIC, SCALAR, STRESS, AnalyticField,
                         ComposedField, NumericLayer, PolynomialLayer,
                         PushedForwardField, RadialField, RadialStretch,
                         flattening, polynomial_fit, polynomial_layer, testing)
from planetmodel.frames import spherical_frame, voigt_to_tensor

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## Layer functions
#
# Underneath a radial field sits a layer function: a function of one
# radius on one interval that differentiates, integrates, re-states and
# rescales itself. `polynomial_layer` writes one as `sum c_k (r / a)^k`,
# the form reference models are published in; here the density and P
# velocity of PREM's outer core, with `a = 6371 km`.

# %%
A = 6371e3
OC = (1221.5e3, 3480.0e3)
rho_fn = polynomial_layer([12.5815, -1.2638, -3.6426, -5.5281], OC, scale=A)
vp_fn = polynomial_layer([11.0487, -4.0362, 4.8023, -13.5732], OC, scale=A)
print(rho_fn)
print("rho at the CMB:", rho_fn(3480e3), "g/cm^3 in the paper's units")
print("d rho / dr at the CMB:", rho_fn.derivative()(3480e3))
print("integral over the core:", rho_fn.integrate(*OC))

# %% [markdown]
# Arithmetic on polynomial layers is done on the coefficients, so the
# result is again a polynomial and exact: the product `rho vp^2` is a
# degree-9 polynomial, not a refit.

# %%
kappa_fn = rho_fn * vp_fn ** 2
print(kappa_fn, "| exact:", isinstance(kappa_fn, PolynomialLayer))
r = np.linspace(*OC, 5)
print("max error against the pointwise product:",
      np.max(np.abs(kappa_fn(r) - rho_fn(r) * vp_fn(r) ** 2)))

# %% [markdown]
# A bare callable becomes a `NumericLayer`, whose derivative is a central
# difference and whose integral is quadrature. Mixing one into the
# algebra makes the result numeric: still pointwise exact, no longer a
# polynomial.

# %%
width = OC[1] - OC[0]
bump_fn = NumericLayer(lambda r: 0.1 * np.sin(2.0 * np.pi * (r - OC[0]) / width), OC)
mixed = rho_fn + bump_fn
print(type(mixed).__name__, "| derivative at 2000 km:", mixed.derivative()(2000e3))

# %% [markdown]
# ## Fields
#
# A `RadialField` is a layer function with a character and a name. A
# density has character `DENSITY` (rank 0, weight 1: it picks up a factor
# `1/J` under a mapping); a velocity is a plain `SCALAR`. The field algebra
# follows the characters: a density times a velocity squared has weight
# 1, as a modulus should; adding a density to a velocity is refused.

# %%
rho = RadialField(OC, rho_fn, character=DENSITY, name="rho")
vp = RadialField(OC, vp_fn, name="vp")
kappa = rho * vp ** 2
print(kappa, "| function:", kappa.function)
try:
    rho + vp
except ValueError as exc:
    print("refused:", exc)

# %% [markdown]
# A field refuses radii outside its interval. Stepping beyond a layer on
# purpose, as a ray tracer's trial step might, is `on_interval`: the
# same polynomial continued on a wider interval.

# %%
try:
    rho(3500e3)
except ValueError as exc:
    print("refused:", str(exc)[:70], "...")
wider = rho.on_interval(OC[0], OC[1] + 100e3)
print("continued past the CMB:", wider(3500e3))

# %% [markdown]
# ## Composition, and refitting
#
# `ComposedField` is the one escape from the algebra: a pointwise formula
# of other fields, never sampled. It is numeric. `polynomial_fit` is the
# way back into the exact algebra on purpose: a least-squares polynomial
# through the formula whose residual the caller judges.

# %%
bulk_sound = ComposedField(lambda k, d: np.sqrt(k / d), (kappa, rho),
                           character=SCALAR, name="bulk_sound_speed")
print(bulk_sound, "| radial:", bulk_sound.is_radial)
fit = RadialField(OC, polynomial_fit(bulk_sound, OC, degree=3), name="fit")
rr = np.linspace(*OC, 200)
print("max residual of the cubic fit:", np.max(np.abs(fit(rr) - bulk_sound(rr))),
      "m/s on values near", bulk_sound(rr).mean())

# %% [markdown]
# ## Tensor fields and frames
#
# Fields of rank 1 and above give their components in the local spherical
# frame `(e_r, e_theta, e_phi)`; `frame="cartesian"` rotates them. Ranks 2
# and 4 are carried in Voigt form. An `AnalyticField` is a formula of all
# three coordinates; it may return its components in either frame.

# %%
sigma = AnalyticField(OC, lambda r, t, p: np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
                      character=STRESS, name="sigma")
th, ph = 0.6, 1.2
s_sph = sigma(2000e3, th, ph)
s_cart = sigma.evaluate(2000e3, th, ph, frame="cartesian")
R = spherical_frame(th, ph)
print("spherical Voigt:", s_sph)
S_sph = voigt_to_tensor(s_sph, rank=2)
print("cartesian agrees with R S R^T:",
      np.allclose(voigt_to_tensor(s_cart, rank=2), R @ S_sph @ R.T))

# %% [markdown]
# ## Push-forward
#
# A field in a model is referential: reference coordinates in, components
# in the frame at the reference point out. `PushedForwardField` is the
# physical tensor under a mapping: at reference `X` it gives the value at
# `m(X)`, with `rank` factors of `F` and `J^-weight`, in the spherical
# frame at the physical point. A density is divided by `J`.

# %%
m = RadialStretch(flattening(0.05, rmax=OC[1]), rmax=OC[1])
rho_phys = PushedForwardField(rho, m)
X = np.array([[0.0, 0.0, 2000e3]])
print(rho_phys, "| J at the pole:", m.jacobian(X)[0])
print("rho / J:", rho(2000e3) / m.jacobian(X)[0], "=", rho_phys(2000e3, 0.0, 0.0))
C = AnalyticField(OC, lambda r, t, p: np.eye(6) * 1e11, character=ELASTIC)
print("a pushed-forward elastic tensor is still (6, 6):",
      PushedForwardField(C, m)(2000e3, 0.6, 1.2).shape)

# %% [markdown]
# Every shipped field passes `check_field`, and so must a field written
# outside the library.

# %%
for f in (rho, vp, kappa, bulk_sound, sigma, rho_phys):
    testing.check_field(f)
print("all contracts pass")

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

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
km = rr / 1e3
axes[0].plot(km, rho(rr) / 1e3, label="rho")
axes[0].plot(km, (rho + RadialField(OC, bump_fn, character=DENSITY))(rr) / 1e3,
             "--", label="rho + bump (numeric)")
axes[0].set_ylabel("g/cm^3"); axes[0].legend()
axes[1].plot(km, kappa(rr) / 1e9, label="rho vp^2, exact")
axes[1].set_ylabel("GPa"); axes[1].legend()
axes[2].plot(km, fit(rr) - bulk_sound(rr))
axes[2].set_ylabel("m/s"); axes[2].set_title("cubic fit residual of a composed field")
for ax in axes:
    ax.set_xlabel("r (km)")
fig.tight_layout()
out = FIGURES / "tutorial_05_fields.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
