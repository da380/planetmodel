# %% [markdown]
# # 5. Fields on one layer
#
# A field is a function of position on one layer of the reference body.
# Its method `evaluate(r, theta, phi, *, frame)` takes the spherical
# coordinates of a point and returns the value there. For a vector or
# tensor field the components are given in the local spherical frame
# `(e_r, e_theta, e_phi)`, or in Cartesian components if asked. A field
# also has an interval, which is the layer's; a character, which says how
# the field transforms when the body is deformed; and a name. A field
# knows nothing about the skeleton, about units, or about other layers.
#
# A *radial* field is a field whose value depends only on the radius.
# Spherically symmetric reference models are built from radial fields. A
# radial field is a field like any other: it answers the same questions
# in the same way, and code written for a general field works on a
# radial one without change.
#
# This tutorial starts with the general field, then the radial field:
# what it stores, the exact arithmetic that storage makes possible, and
# how a numerical result is fitted back into that form. Then tensor
# fields, frames, and push-forward through a mapping.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

# Fields, and what works on them.
from planetmodel import (
    DENSITY,
    ELASTIC,
    SCALAR,
    STRESS,
    AnalyticField,
    ComposedField,
    PushedForwardField,
    RadialField,
    RadialStretch,
    flattening,
    testing,
)

# Layer functions, which a radial field stores. See the section on them
# below.
from planetmodel import NumericLayer, PolynomialLayer, polynomial_fit, polynomial_layer
from planetmodel.frames import spherical_frame, voigt_to_tensor

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## A field is three-dimensional
#
# `AnalyticField` is the general field. It takes any function of
# `(r, theta, phi)`, an interval, a character and a name. The example is
# a density on PREM's outer core with a degree-2 variation in colatitude,
# roughly the shape a rotating fluid core takes. The coordinates
# broadcast against each other and the result has the broadcast shape.
# The same constructor wraps any callable you already have, an
# interpolant of a tomographic model for instance, so that the library
# can carry it.
#
# The character `DENSITY` says that this scalar is divided by the volume
# change when the body is deformed, as a density must be. Characters are
# explained further below.

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

# %% [markdown]
# This density varies with colatitude, so it has no single value at a
# given radius. If it is called with a radius alone it refuses. A field
# whose value changes with the angles must be given all three
# coordinates.

# %%
try:
    rho3(2000e3)
except ValueError as exc:
    print("refused:", str(exc)[:60], "...")


# %% [markdown]
# ## A radial field is a field
#
# `RadialField` takes a function of the radius alone and makes a field
# from it, on an interval, with a character and a name like any other
# field. The example is the same core density without its angular part.
# It can be asked the same questions as the general field and gives the
# same kind of answers; the angles are accepted and ignored. Because the
# angles do not matter, a scalar radial field can also be called with
# the radius alone.

# %%
rho_simple = RadialField(
    OC, lambda r: 12.0e3 - 2.0e3 * (r / OC[1]) ** 2, character=DENSITY, name="rho"
)
print(rho_simple, "| radial:", rho_simple.is_radial)
print("with all three coordinates:", rho_simple(2000e3, 0.3, 1.0))
print("on the same grid:", rho_simple(radii, colats, 0.0).shape)
print("with the radius alone:", rho_simple(2000e3))

# %% [markdown]
# Code written for a general field therefore works on a radial one
# without change. The function below integrates a density over its shell
# by quadrature. It does not know, or need to know, which kind of field
# it was given.


# %%
def shell_mass(field, n=200):
    r = np.linspace(*field.interval, n)[:, None]
    t = np.linspace(0.0, np.pi, n)[None, :]
    weight = 2.0 * np.pi * r**2 * np.sin(t)
    inner = np.trapezoid(field(r, t, 0.0) * weight, t.ravel(), axis=1)
    return np.trapezoid(inner, r.ravel())


print("mass of the shaped core: ", shell_mass(rho3), "kg")
print("mass of the radial core: ", shell_mass(rho_simple), "kg")

# %% [markdown]
# ## What a radial field stores
#
# Inside, a radial field holds one *layer function* for each component.
# A layer function is a function of one radius on one interval that can
# also differentiate, integrate and rescale itself, and continue itself
# onto a wider interval. It knows nothing about angles, frames or
# characters. The field's `function` property returns it. If the field
# was given a bare callable, as above, the layer function is a
# `NumericLayer`: its derivative is a central difference and its integral
# is computed by quadrature. If the field is given a polynomial, the
# layer function is a `PolynomialLayer`, which works on the coefficients
# and is therefore exact.
#
# `polynomial_layer` builds a polynomial of the form `sum c_k (r / a)^k`,
# which is how many reference models are published. The example is the
# density and P velocity of PREM's outer core with `a = 6371 km`. The
# published coefficients are in g/cm^3 and km/s, so they are multiplied
# by 1000 to give SI values; everything else in this tutorial is in
# metres, kilograms and seconds.

# %%
print("stored by the field above:", rho_simple.function)

rho_poly = polynomial_layer(
    np.array([12.5815, -1.2638, -3.6426, -5.5281]) * 1e3, OC, scale=A
)
vp_poly = polynomial_layer(
    np.array([11.0487, -4.0362, 4.8023, -13.5732]) * 1e3, OC, scale=A
)
rho = RadialField(OC, rho_poly, character=DENSITY, name="rho")
vp = RadialField(OC, vp_poly, name="vp")
print("stored by a polynomial field:", rho.function)
print("rho at the CMB:", rho(3480e3), "kg/m^3")

# %% [markdown]
# The field exposes the layer function's operations as its own methods.
# `derivative` returns another radial field and `integrate` returns a
# number. On a polynomial both are exact; on a numeric layer they are
# approximations.

# %%
print("d rho / dr at the CMB:", rho.derivative()(3480e3), "kg/m^4")
print("integral of rho over the core radius:", rho.integrate(*OC), "kg/m^2")
print(
    "the same for the numeric field:",
    rho_simple.derivative()(3480e3),
    rho_simple.integrate(*OC),
)

# %% [markdown]
# ## The exact arithmetic
#
# Fields can be added and subtracted when they have the same character,
# multiplied and divided when they are scalars, raised to integer powers,
# and scaled by a number. When every operand is radial the result is
# radial. When every layer function involved is a polynomial, the
# arithmetic is done on the coefficients: the product `rho vp^2` below is
# a polynomial of degree 9, not a fit through sample values. Its values
# agree with the product computed point by point to rounding error.

# %%
kappa = rho * vp**2
kappa = kappa.renamed("kappa")
print(kappa, "| radial:", kappa.is_radial)
print("stored:", kappa.function)
print("exact:", isinstance(kappa.function, PolynomialLayer))
r = np.linspace(*OC, 5)
print(
    "max error against the pointwise product:",
    np.max(np.abs(kappa(r) - rho(r) * vp(r) ** 2)),
    "Pa on values near",
    kappa(r).mean(),
)

# %% [markdown]
# If one operand holds a numeric layer, the result holds one too. It is
# still correct at every point, but it is no longer a polynomial. Here a
# small bump is added to the density.

# %%
width = OC[1] - OC[0]
bump = RadialField(
    OC, lambda r: 100.0 * np.sin(2.0 * np.pi * (r - OC[0]) / width), character=DENSITY
)
rho_bumped = rho + bump
print(
    "numeric:",
    isinstance(rho_bumped.function, NumericLayer),
    "| derivative at 2000 km:",
    rho_bumped.derivative()(2000e3),
)

# %% [markdown]
# The arithmetic follows the characters. A density has character
# `DENSITY`: rank 0 and weight 1, where the weight is the power of the
# volume change that divides the field when the body is deformed. A
# velocity is a plain `SCALAR` with weight 0. A density times a velocity
# squared has weight 1, as a modulus should; adding a density to a
# velocity is refused. A radial field times a field that varies with the
# angles is no longer radial.

# %%
try:
    rho + vp
except ValueError as exc:
    print("refused:", exc)
shape = AnalyticField(OC, lambda r, t, p: 1.0 + 0.01 * np.cos(t), name="shape")
rho_shaped = rho * shape
print(rho_shaped, "| radial:", rho_shaped.is_radial)

# %% [markdown]
# A field refuses a radius outside its interval. To step beyond a layer
# on purpose, as a ray tracer's trial step might, use `on_interval`,
# which returns the same function continued onto a wider interval.

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
# `ComposedField` is the way out of the arithmetic above. It applies any
# function to the values of other fields, point by point, so it is not a
# polynomial and its derivatives and integrals are not exact. It is
# radial when all of its sources are. The example is the P-wave slowness,
# one over the velocity, which no polynomial can give. `polynomial_fit`
# is the way back: it fits a least-squares polynomial to a function on
# an interval and returns a polynomial layer. The caller should check the
# residual before wrapping the result in a field.

# %%
slowness = ComposedField(lambda v: 1.0 / v, (vp,), character=SCALAR, name="slowness")
print(slowness, "| radial:", slowness.is_radial)
fit = RadialField(OC, polynomial_fit(slowness, OC, degree=3), name="fit")
rr = np.linspace(*OC, 200)
print(
    "max residual of the cubic fit:",
    np.max(np.abs(fit(rr) - slowness(rr))),
    "s/m on values near",
    slowness(rr).mean(),
)

# %% [markdown]
# ## Real and complex fields
#
# A field's values are float64, or complex128 if the field is complex;
# `dtype` says which. A linear viscoelastic body at one frequency behaves
# like an elastic body with complex moduli, so a model frozen at a
# frequency holds complex fields wherever it has a rheology. Everything
# underneath handles this: complex coefficients give a complex polynomial
# layer, the product of a real and a complex field is complex, and the
# integral of a complex field is complex. Nothing is cast: a real field
# stays float64. A shear modulus has weight 1 like a density, so it is
# given the character `DENSITY` here.

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
# A field of rank 1 or higher gives its components in the local
# spherical frame `(e_r, e_theta, e_phi)` at the point. Passing
# `frame="cartesian"` rotates them into Cartesian components. Fields of
# rank 2 and 4 are stored in Voigt form: a symmetric rank-2 tensor as a
# vector of its six independent components in the order
# `(11, 22, 33, 23, 13, 12)`, and a rank-4 tensor as a six-by-six matrix.
# The function given to `AnalyticField` may return components in either
# frame. The example is a stress that is constant in the spherical frame.
# Its Cartesian components change from point to point because the frame
# does.

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
# A field in a model is defined on the reference body: it takes
# reference coordinates and returns components in the frame at the
# reference point. A mapping `m` takes the reference body to the physical
# one; `F` is its deformation gradient and `J = det F` its volume change.
# `PushedForwardField` is the physical tensor. Asked at a reference point
# `X`, it returns the value at `m(X)`, with one factor of `F` applied to
# each tensor index and the whole divided by `J` raised to the weight,
# expressed in the spherical frame at the physical point. A density, with
# weight 1, is divided by `J`.

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
# Every field shipped with the library passes `check_field`, which tests
# everything a field must do. A field written outside the library must
# pass it too.

# %%
for f in (
    rho3,
    rho_simple,
    rho,
    vp,
    kappa,
    rho_bumped,
    rho_shaped,
    slowness,
    mu_c,
    sigma,
    rho_phys,
):
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
axes[0].plot(km, rho_bumped(rr) / 1e3, "--", label="rho + bump (numeric)")
axes[0].set_ylabel("g/cm^3")
axes[0].legend()
axes[1].plot(km, kappa(rr) / 1e9, label="rho vp^2, exact")
axes[1].set_ylabel("GPa")
axes[1].legend()
axes[2].plot(km, fit(rr) - slowness(rr))
axes[2].set_ylabel("s/m")
axes[2].set_title("cubic fit residual of a composed field")
for ax in axes:
    ax.set_xlabel("r (km)")
fig.tight_layout()
out = FIGURES / "tutorial_05_fields.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
