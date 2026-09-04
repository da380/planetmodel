# %% [markdown]
# # Layered rheology
#
# A body for glacial isostatic adjustment: an elastic lithosphere over a
# Maxwell mantle over a fluid core. In planetmodel a rheological **law**
# is a function that takes a layer's static fields and returns a
# frequency-dependent field; which law applies where is decided by
# whoever builds the layers, one `with_field` at a time, and the model
# class checks the result. A layer with no frequency-dependent moduli is
# elastic: its static tensor is lifted when the body-wide view is asked
# for, and nothing is stored for it.
#
# The frequency-dependent moduli are complex-valued, entire in the
# Laplace variable `s = i omega`, and evaluated off the real axis as a
# time-domain code needs.

# %%
from pathlib import Path

import numpy as np

from planetmodel import (Dimensions, RadialField, ReferenceBody, Skeleton,
                         ViscoelasticModel, maxwell, prem)

base = prem(ocean=False)
for lay in base.layers:
    lo, hi = lay.interval
    print(f"{lay.index:2d} [{lo / 1e3:6.0f}, {hi / 1e3:6.0f}] km {lay.state:6s}")

# %% [markdown]
# ## Building the layers
#
# PREM's layers arrive with `viscoelastic_moduli` under constant Q. We
# take the static fields only, then attach a viscosity and a Maxwell law to
# the mantle layers below 80 km depth. `maxwell(moduli, viscosity)` relaxes
# everything but the bulk part: the shear modulus follows
# `mu(s) = mu_0 s tau / (1 + s tau)` with `tau = eta / mu_0`, so a stress
# left alone loses its deviatoric part entirely and the bulk modulus never
# moves. For PREM's transversely isotropic zone the same rule acts on the
# tensor's deviatoric part with the one Maxwell time of the isotropic
# equivalent.

# %%
def constant(layer, value, *, name, dimensions):
    """A uniform single-layer field on the layer's own interval."""
    return RadialField(Skeleton(layer.interval), [lambda r: np.full_like(r, value)],
                       name=name, dimensions=dimensions)


static = ReferenceBody(base.without_field("viscoelastic_moduli").layers,
                       meta={"name": "PREM, layered rheology"})
lithosphere_base = 6291.0e3                      # 80 km depth
layers = []
for lay in static.layers:
    lo, hi = lay.interval
    if lay.state == "solid" and hi <= lithosphere_base and lo >= 3480.0e3:
        eta = constant(lay, 1.0e21, name="viscosity",
                       dimensions=Dimensions.VISCOSITY)
        lay = lay.with_field("viscosity", eta)
        lay = lay.with_field("viscoelastic_moduli",
                             maxwell(lay["elastic_moduli"], eta))
    layers.append(lay)
model = ReferenceBody(layers, meta=static.meta).as_class(ViscoelasticModel)
print(model)
for lay in model.layers:
    kind = ("fluid" if lay.state == "fluid"
            else "Maxwell" if "viscoelastic_moduli" in lay.fields else "elastic")
    print(f"  layer {lay.index:2d}: {kind}")

# %% [markdown]
# ## The moduli off the real axis
#
# `viscoelastic_moduli` is one view across the body. It takes a complex
# `omega`; a Laplace-domain code substitutes `omega = -i s` for real
# `s > 0`. On the Maxwell mantle the shear modulus relaxes from `mu_0` at
# short times (`s -> inf`) to zero at long times (`s -> 0`), while the
# bulk modulus stays put; on the lithosphere and in the core the view
# returns the static tensor at every `s`.

# %%
tau = 1.0e21 / model["elastic_moduli"].evaluate(5.0e6)[4, 4]
print(f"Maxwell time in the mid mantle: {tau:.3g} s = {tau / 3.156e7:.0f} yr")
r = np.array([5.0e6, 6.32e6])                    # mid mantle, lithosphere
for s in (1e-13, 1.0 / tau, 1e-6):
    C = model.viscoelastic_moduli.evaluate(r, omega=-1j * s)
    ratio = C[:, 4, 4].real / model.elastic_moduli.evaluate(r)[:, 4, 4]
    print(f"s = {s:8.1e} 1/s: L / L_0 = {np.round(ratio, 4)}")
print("bulk part unrelaxed at s -> 0:",
      np.allclose(np.linalg.eigvalsh(model.viscoelastic_moduli.evaluate(
          r[:1], omega=-1e-13j)[0].real).max(),
          np.linalg.eigvalsh(model.elastic_moduli.evaluate(r[:1])[0]).max(), rtol=1e-3))

# %% [markdown]
# At real frequencies the same field gives the complex modulus: the real
# part is the storage modulus and the imaginary part the loss, with the
# convention `exp(+i omega t)` so that the loss is positive. A Maxwell
# body's loss peaks at `omega tau = 1`.

# %%
omegas = np.logspace(-13, -8, 41)
L = np.array([model.viscoelastic_moduli.evaluate(r[:1], omega=w)[0, 4, 4]
              for w in omegas])
L0 = model.elastic_moduli.evaluate(r[:1])[0, 4, 4]
k = np.argmax(L.imag)
print(f"loss peaks at omega tau = {omegas[k] * tau:.2f}; there Re L / L_0 = "
      f"{L[k].real / L0:.3f}, Im L / L_0 = {L[k].imag / L0:.3f}")

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def relaxation_figure(path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(omegas * tau, L.real / L0, label="Re L / L0")
    ax.semilogx(omegas * tau, L.imag / L0, label="Im L / L0")
    ax.set_xlabel("omega tau")
    ax.set_title("Maxwell shear modulus in the mid mantle")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110)
    print("wrote", path.name)


relaxation_figure(FIGURES / "tutorial_04_maxwell.png")

# %% [markdown]
# ## A Prony series
#
# A generalised Maxwell body is a sum of Maxwell elements: `prony` takes
# the relaxation times and strengths as sequences of scalar fields, one
# per term, and a long-time modulus. One term with no long-time modulus is
# Maxwell, which is the check below; two terms give a body with two
# relaxation times.

# %%
from planetmodel import prony  # noqa: E402

mantle = model.layers[3]                          # the lower mantle
tau_field = constant(mantle, tau, name="tau", dimensions=Dimensions.TIME)
mu_field = constant(mantle, float(L0), name="mu_1", dimensions=Dimensions.MODULUS)
one_term = prony(mantle["elastic_moduli"], [tau_field], [mu_field])
s = 3.0 / tau
same = np.allclose(one_term.evaluate(np.array([5.0e6]), omega=-1j * s)[0, 4, 4],
                   mantle["viscoelastic_moduli"].evaluate(np.array([5.0e6]),
                                                          omega=-1j * s)[0, 4, 4],
                   rtol=1e-12)
print("one-term Prony equals Maxwell:", same)
two_terms = prony(mantle["elastic_moduli"],
                  [tau_field, constant(mantle, 0.1 * tau, name="tau_2",
                                       dimensions=Dimensions.TIME)],
                  [0.5 * mu_field, 0.5 * mu_field])
print("two terms at s tau = 3: L / L_0 =",
      (two_terms.evaluate(np.array([5.0e6]), omega=-1j * s)[0, 4, 4] / L0).real)

# %% [markdown]
# ## What the file records
#
# The field a law returns carries a `LawRecord`: the law's registered
# name, the fields it read, its constants and its convention. That is
# provenance, not data. The netCDF writer copies it into a `/rheology`
# group so that a reader, ours or somebody else's, can rebuild the field
# from the static ones by calling the law by name. Elastic layers are
# recorded as `static`.

# %%
print("mantle record:", mantle["viscoelastic_moduli"].law)
print("lithosphere holds", model.layers[-1].field_names)

try:
    import netCDF4  # noqa: F401
except ImportError:
    print("netCDF4 not installed (pip install 'planetmodel[netcdf]'); "
          "skipping the file")
else:
    import tempfile

    from planetmodel import AngularGrid, read_model, write_model

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gia.nc"
        write_model(model, model.sample(AngularGrid.gauss_legendre(4)), path)
        back, _ = read_model(path)
        print("read back:", type(back).__name__)
        for i in (1, 4, 10):
            lay = back.layers[i]
            law = (lay["viscoelastic_moduli"].law.law
                   if "viscoelastic_moduli" in lay.fields else "static")
            print(f"  layer {i:2d}: {law}")
        w = -1j / tau
        print("the rebuilt mantle agrees:",
              np.allclose(back.viscoelastic_moduli.evaluate(r, omega=w),
                          model.viscoelastic_moduli.evaluate(r, omega=w), rtol=1e-10))

# %% [markdown]
# ## A law of your own
#
# A law is a function returning a `ComposedFrequencyField` with a
# `LawRecord`, registered under the `"rheology"` kind, and carrying its own
# inverse, `from_record`, so that a file can name it. Here is a standard
# linear solid: Maxwell in parallel with a spring of relative strength
# `alpha`, so that the shear modulus relaxes to `alpha mu_0` rather than
# to zero.

# %%
from planetmodel import LawRecord, register  # noqa: E402
from planetmodel.model import ComposedFrequencyField, ELASTIC  # noqa: E402
from planetmodel.model.rheology import rebuild  # noqa: E402


def standard_linear_solid(moduli, viscosity, *, alpha, name=None):
    """Shear relaxing to alpha * mu_0; the bulk part unrelaxed."""
    kappa, mu = moduli.components["kappa"], moduli.components["mu"]

    def fn(omega, kappa, mu, eta):
        s = 1j * omega
        tau = np.where(mu > 0.0, eta / np.where(mu > 0.0, mu, 1.0), 0.0)
        mu_s = mu * (alpha + (1.0 - alpha) * s * tau / (1.0 + s * tau))
        return isotropic_voigt(kappa, mu_s)

    record = LawRecord("standard_linear_solid",
                       parameters=(moduli.name, viscosity.name),
                       constants={"alpha": alpha})
    return ComposedFrequencyField(fn, [kappa, mu, viscosity], character=ELASTIC,
                                  dimensions=moduli.dimensions,
                                  name=name or "viscoelastic_moduli", law=record)


def isotropic_voigt(kappa, mu):
    """The 6x6 Voigt matrix of an isotropic medium, broadcasting."""
    lam = kappa - 2.0 * mu / 3.0
    C = np.zeros(np.shape(mu) + (6, 6), dtype=np.result_type(kappa, mu))
    for a in range(3):
        for b in range(3):
            C[..., a, b] = lam
        C[..., a, a] = lam + 2.0 * mu
        C[..., 3 + a, 3 + a] = mu
    return C


standard_linear_solid.from_record = lambda record, fields: standard_linear_solid(
    *(fields[n] for n in record.parameters), alpha=record.constants["alpha"])
standard_linear_solid.constant_dimensions = {"alpha": Dimensions.DIMENSIONLESS}
register("rheology", "standard_linear_solid", standard_linear_solid)

# Try it on an isotropic layer: prem.nocrust's lower mantle.
from planetmodel import read_isotropic_deck  # noqa: E402

iso = read_isotropic_deck(Path(__file__).resolve().parents[2] / "tests" / "data"
                          / "prem.nocrust")
lower_mantle = iso.layers[3]
eta = constant(lower_mantle, 3.0e21, name="viscosity",
               dimensions=Dimensions.VISCOSITY)
sls = standard_linear_solid(lower_mantle["elastic_moduli"], eta, alpha=0.3)
lower_mantle = lower_mantle.with_field("viscosity", eta).with_field(
    "viscoelastic_moduli", sls)
rr = np.array([4.5e6])
mu0 = lower_mantle["elastic_moduli"].evaluate(rr)[0, 3, 3]
print("mu(s -> 0) / mu_0 =", sls.evaluate(rr, omega=-1e-18j)[0, 3, 3].real / mu0)
print("mu(s -> inf) / mu_0 =", sls.evaluate(rr, omega=-1e-3j)[0, 3, 3].real / mu0)
again = rebuild(sls.law, {n: lower_mantle[n] for n in ("elastic_moduli", "viscosity")})
print("rebuilt from its record:", np.allclose(again.evaluate(rr, omega=1e-12),
                                              sls.evaluate(rr, omega=1e-12)))
