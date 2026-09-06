# %% [markdown]
# # 5. Fields on one layer
#
# A field is a function of position on one layer of the reference body:
# `evaluate(r, theta, phi, *, frame)` takes the spherical coordinates of
# a point and returns the components there, in the local spherical frame
# `(e_r, e_theta, e_phi)` or in Cartesian components. Beside that it has
# an interval, the layer's; a character, which says how it transforms
# under a mapping; and a name. It knows no skeleton, no units and no
# other layer. A *radial* field is the special case with no angular
# dependence, a function of the radius alone: it is what a spherically
# symmetric reference model holds, and it is exact where its polynomials
# are. This tutorial starts with the general field, then the radial one
# and the exact arithmetic it allows, then tensors, frames and the
# push-forward through a mapping.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import (
    DENSITY,
    ELASTIC,
    SCALAR,
    STRESS,
    AnalyticField,
    ComposedField,
    NumericLayer,
    PolynomialLayer,
    PushedForwardField,
    RadialField,
    RadialStretch,
    flattening,
    polynomial_fit,
    polynomial_layer,
    testing,
)
from planetmodel.frames import spherical_frame, voigt_to_tensor

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## A field is three-dimensional
#
# `AnalyticField` is the general field: any formula of `(r, theta, phi)`
# on an interval, given a character and a name. Here a density on the
# interval of PREM's outer core with a degree-2 variation in colatitude,
# the shape a rotating fluid core takes. The coordinates broadcast, the
# result has their shape, and a field that depends on direction must be
# asked all three coordinates. The same constructor wraps any callable
# you already have, an interpolant of a tomographic model say, so that
# the library can carry it.

# %%
A = 6371e3
OC = (1221.5e3, 3480.0e3)


def core_density(r, theta, phi):
    p2 = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    return 12.0e3 - 2.0e3 * (r / OC[1]) ** 2 + 30.0 * (r / OC[1]) ** 2 * p2


rho3 = AnalyticField(OC, core_density, character=DENSITY, name="rho")
print(rho3, "| radial:", rho3.is_radial)
print("at one point:", rho3(2000e3, 0.3, 1.0))
radii, colats = np.linspace(*OC, 3)[:, None], np.linspace(0, np.pi, 4)[None, :]
print("on a grid:", rho3(radii, colats, 0.0).shape)
try:
    rho3(2000e3)
except ValueError as exc:
    print("refused:", str(exc)[:60], "...")

# %% [markdown]
# ## Radial fields, the special case
#
# A `RadialField` depends on the radius alone, so it may be called with
# `r` by itself, and underneath it sits a *layer function*: a function of
# one radius on one interval that differentiates, integrates, re-states
# and rescales itself. `polynomial_layer` writes one as
# `sum c_k (r / a)^k`, the form some reference models are published in;
# here the density and P velocity of PREM's outer core, with
# `a = 6371 km`.

# %%
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
kappa_fn = rho_fn * vp_fn**2
print(kappa_fn, "| exact:", isinstance(kappa_fn, PolynomialLayer))
r = np.linspace(*OC, 5)
print(
    "max error against the pointwise product:",
    np.max(np.abs(kappa_fn(r) - rho_fn(r) * vp_fn(r) ** 2)),
)

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
# A `RadialField` is a layer function with a character and a name. A
# density has character `DENSITY` (rank 0, weight 1: it picks up a factor
# `1/J` under a mapping); a velocity is a plain `SCALAR`. The field algebra
# follows the characters: a density times a velocity squared has weight
# 1, as a modulus should; adding a density to a velocity is refused. When
# every operand is radial the result is radial and, on polynomials,
# exact; a radial field times a field of direction is a general field
# again.

# %%
rho = RadialField(OC, rho_fn, character=DENSITY, name="rho")
vp = RadialField(OC, vp_fn, name="vp")
kappa = rho * vp**2
print(kappa, "| radial:", kappa.is_radial, "| function:", kappa.function)
print(
    "radial field called with r alone:",
    rho(2000e3),
    "| with all three:",
    rho(2000e3, 0.3, 1.0),
)
try:
    rho + vp
except ValueError as exc:
    print("refused:", exc)
shape = AnalyticField(OC, lambda r, t, p: 1.0 + 0.01 * np.cos(t), name="shape")
rho_shaped = rho * shape
print(rho_shaped, "| radial:", rho_shaped.is_radial)

# %% [markdown]
# A field refuses radii outside its interval. Stepping beyond a layer on
# purpose, as a ray tracer's trial step might, is `on_interval`: the
# same function continued on a wider interval.

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
bulk_sound = ComposedField(
    lambda k, d: np.sqrt(k / d), (kappa, rho), character=SCALAR, name="bulk_sound_speed"
)
print(bulk_sound, "| radial:", bulk_sound.is_radial)
fit = RadialField(OC, polynomial_fit(bulk_sound, OC, degree=3), name="fit")
rr = np.linspace(*OC, 200)
print(
    "max residual of the cubic fit:",
    np.max(np.abs(fit(rr) - bulk_sound(rr))),
    "m/s on values near",
    bulk_sound(rr).mean(),
)

# %% [markdown]
# ## Real and complex fields
#
# A field's values are float64, or complex128 where the field is complex,
# and `dtype` says which. A linear viscoelastic body at one frequency is
# an elastic body with complex moduli, so a model frozen at a frequency
# holds complex fields where it holds a rheology, and everything below
# carries the dtype through: complex coefficients make a complex
# polynomial layer, the algebra between a real and a complex field is
# complex, and integrals come back complex. Nothing is cast: a real field
# stays float64.

# %%
mu_r = RadialField(
    OC, polynomial_layer([1.0e11, -2.0e10], OC, scale=A), character=DENSITY, name="mu"
)
mu_c = RadialField(
    OC,
    polynomial_layer([1.0e11 + 2.0e9j, -2.0e10], OC, scale=A),
    character=DENSITY,
    name="mu",
)
print("real:", mu_r.dtype, "| complex:", mu_c.dtype)
print(
    "a real times a complex field:",
    (vp * mu_c).dtype,
    "| a real times a real:",
    (vp * mu_r).dtype,
)
print("complex value:", mu_c(2000e3), "| integral:", mu_c.integrate(*OC))
print(
    "a complex formula is a complex field:",
    AnalyticField(OC, lambda r, t, p: r * np.exp(1j * p), name="phase").dtype,
)

# %% [markdown]
# ## Tensor fields and frames
#
# Fields of rank 1 and above give their components in the local spherical
# frame `(e_r, e_theta, e_phi)`; `frame="cartesian"` rotates them. Ranks 2
# and 4 are carried in Voigt form. A formula may return its components in
# either frame; here a constant stress in the spherical frame, which in
# Cartesian components varies from point to point with the frame.

# %%
sigma = AnalyticField(
    OC,
    lambda r, t, p: np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
    character=STRESS,
    name="sigma",
)
th, ph = 0.6, 1.2
s_sph = sigma(2000e3, th, ph)
s_cart = sigma.evaluate(2000e3, th, ph, frame="cartesian")
R = spherical_frame(th, ph)
print("spherical Voigt:", s_sph)
S_sph = voigt_to_tensor(s_sph, rank=2)
print(
    "cartesian agrees with R S R^T:",
    np.allclose(voigt_to_tensor(s_cart, rank=2), R @ S_sph @ R.T),
)

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
print(
    "a pushed-forward elastic tensor is still (6, 6):",
    PushedForwardField(C, m)(2000e3, 0.6, 1.2).shape,
)

# %% [markdown]
# Every shipped field passes `check_field`, and so must a field written
# outside the library.

# %%
for f in (rho3, rho, vp, kappa, rho_shaped, bulk_sound, mu_c, sigma, rho_phys):
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
axes[0].plot(
    km,
    (rho + RadialField(OC, bump_fn, character=DENSITY))(rr) / 1e3,
    "--",
    label="rho + bump (numeric)",
)
axes[0].set_ylabel("g/cm^3")
axes[0].legend()
axes[1].plot(km, kappa(rr) / 1e9, label="rho vp^2, exact")
axes[1].set_ylabel("GPa")
axes[1].legend()
axes[2].plot(km, fit(rr) - bulk_sound(rr))
axes[2].set_ylabel("m/s")
axes[2].set_title("cubic fit residual of a composed field")
for ax in axes:
    ax.set_xlabel("r (km)")
fig.tight_layout()
out = FIGURES / "tutorial_05_fields.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
