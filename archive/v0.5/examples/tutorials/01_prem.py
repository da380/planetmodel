# %% [markdown]
# # PREM
#
# planetmodel is a library for spherically layered planets. This tutorial
# reads the one model everybody knows, PREM, and does the things you would
# first want to do with it: look at what it is made of, evaluate it, plot
# it, ask for its moduli at a period, and write it to a file that a
# numerical code can read back as the same model.
#
# Run it as a script from anywhere, or open it as a notebook: the `# %%`
# markers are cells.

# %%
from pathlib import Path

import numpy as np

from planetmodel import prem

model = prem()
print(model)

# %% [markdown]
# ## What PREM is, to planetmodel
#
# `prem()` is not a table. It returns a **model class**, a
# `ViscoelasticModel`, which guarantees on every layer that has material a
# density `rho` and the elastic tensor `elastic_moduli`, and offers
# `viscoelastic_moduli`, the same tensor as a function of frequency. PREM
# quotes its moduli at a period of 1 s with quality factors `qkappa` and
# `qmu`; that is an anelastic model, and the frequency dependence is built
# for you from the published constant-Q law.
#
# A model is a list of **layers**. Each layer is an interval of radius, the
# fields it holds by name, and its state: solid, fluid or vacuum. Nothing
# is defined "everywhere" by default; a field belongs to the layers that
# hold it.

# %%
for lay in model.layers:
    lo, hi = lay.interval
    print(
        f"layer {lay.index:2d}  [{lo / 1e3:7.1f}, {hi / 1e3:7.1f}] km  "
        f"{lay.state:6s} holds {len(lay.fields)} fields"
    )
print("\nfield names:", model.field_names)

# %% [markdown]
# The outer core and the ocean are fluid: PREM's shear velocity is zero
# there, and the model was classified on construction. A layer's state is
# the one thing the fields cannot say by themselves and a mesh needs.
#
# ## Views and domains
#
# `model["rho"]` (or `model.rho`) is a **view**: one field across the body,
# assembled from the pieces the layers hold. Its `domain` lists the layers
# that hold a piece, and it refuses a radius outside them by name rather
# than inventing a value. At an interface a field has two values, one from
# each side, and both are kept: ask for the lower side explicitly.

# %%
rho = model["rho"]
print("\nrho is defined on layers", rho.domain)
cmb = model.interface(1).radius  # interfaces are numbered from the centre
print(f"CMB at {cmb / 1e3:.0f} km")
print(f"rho just below: {rho.evaluate(cmb, side='lower'):8.1f} kg/m^3")
print(f"rho just above: {rho.evaluate(cmb):8.1f} kg/m^3")

r = np.linspace(0.0, model.skeleton.boundaries[-1], 7)
print("\nvsv along the radius (m/s):", np.round(model["vsv"].evaluate(r)))

# %% [markdown]
# Every field carries a tensor **character** (rank and weight, which fix
# how it transforms under a mapping) and physical **dimensions**. The
# elastic tensor evaluates to a Voigt 6x6 at each point in the local
# spherical frame `(e_r, e_theta, e_phi)`. PREM is transversely isotropic
# in the upper mantle, so the tensor is built from five moduli, A, C, F,
# L, N, which are fields in their own right.

# %%
print("rho:", rho.character, rho.dimensions)
C = model.elastic_moduli.evaluate(np.array([6.0e6]))
print("Voigt matrix at 6000 km, in GPa, rounded:")
print(np.round(C[0] / 1e9, 1))
print(
    "symmetry:",
    model.symmetry.name,
    "| L there:",
    f"{model['L'].evaluate(6.0e6) / 1e9:.1f} GPa",
)

# %% [markdown]
# ## A profile plot
#
# Every field has a `plot` that draws one segment per layer and links the
# two-sided values at each discontinuity, so the jumps are visible as
# jumps.

# %%
FIGURES = Path(__file__).resolve().parents[1] / "figures"


def profile_figure(path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; no figure written")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, name, unit in zip(axes, ("rho", "vpv", "vsv"), ("kg/m^3", "m/s", "m/s")):
        model[name].plot(ax=ax, show_boundaries=True)
        ax.set_xlabel(f"{name} [{unit}]")
    axes[0].set_ylabel("radius [m]")
    fig.suptitle("PREM")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110)
    print("wrote", path.name)


profile_figure(FIGURES / "tutorial_01_prem_profiles.png")

# %% [markdown]
# ## The moduli at a period
#
# `viscoelastic_moduli` is a frequency-dependent field: it takes an angular
# frequency `omega` and returns the complex tensor, the imaginary part
# being the loss. The complex value is the object; take `.real` or `.imag`
# yourself when you want one part. At the reference period the real part
# is the static tensor. `moduli_at(omega)` freezes the field at one
# frequency as an ordinary static field, complex-valued by default, real
# on request.

# %%
w_ref = 2.0 * np.pi / 1.0  # PREM's reference period is 1 s
w_100 = 2.0 * np.pi / 100.0
r = np.array([6.0e6])
at_ref = model.viscoelastic_moduli.evaluate(r, omega=w_ref)
print(
    "at 1 s the real part equals the static tensor:",
    np.allclose(at_ref.real, model.elastic_moduli.evaluate(r)),
)
loss = at_ref[0, 4, 4].imag / at_ref[0, 4, 4].real
print(
    f"loss at 1 s, Im L / Re L: {loss:.4f}"
    f"  (1 / Q_mu = {1.0 / model['qmu'].evaluate(6.0e6):.4f})"
)

static_100 = model.moduli_at(w_100, part="real")
ratio = static_100.evaluate(r)[0, 4, 4] / model.elastic_moduli.evaluate(r)[0, 4, 4]
print(f"L at 100 s / L at 1 s: {ratio:.4f}  -- softer at long period")
print("moduli_at returns a static field of kind", repr(static_100.kind))

# %% [markdown]
# The dispersion across the seismic band, layer by layer: the shear
# modulus L relative to its 1 s value, from 1 s to 1000 s.

# %%
periods = np.logspace(0.0, 3.0, 25)
radii = np.array([1.0e6, 4.0e6, 5.8e6, 6.3e6])
L_ref = model.elastic_moduli.evaluate(radii)[:, 4, 4]
dispersion = (
    np.array(
        [
            model.viscoelastic_moduli.evaluate(radii, omega=2.0 * np.pi / T)[
                :, 4, 4
            ].real
            for T in periods
        ]
    )
    / L_ref
)
for k, rr in enumerate(radii):
    print(
        f"r = {rr / 1e3:6.0f} km: L(1000 s) / L(1 s) = {dispersion[-1, k]:.4f}"
        f"   Q_mu = {model['qmu'].evaluate(rr):.0f}"
    )


def dispersion_figure(path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, rr in enumerate(radii):
        ax.semilogx(periods, dispersion[:, k], label=f"r = {rr / 1e3:.0f} km")
    ax.set_xlabel("period [s]")
    ax.set_ylabel("L(T) / L(1 s)")
    ax.set_title("PREM under constant Q: dispersion of the shear modulus")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("wrote", path.name)


dispersion_figure(FIGURES / "tutorial_01_dispersion.png")

# %% [markdown]
# ## Writing a file, and reading it back
#
# A model is delivered to a numerical code as a **sample** on that code's
# nodes: a radial spectral-element mesh times an angular grid. The netCDF
# file stores the sample, the layers, and a record of how each layer's
# frequency-dependent moduli were built, so that the reader can rebuild
# the model class on the other side. Reading the file back gives a
# `ViscoelasticModel` again, not a bag of arrays.
#
# This needs the `netcdf` extra; the cell says so and moves on if it is
# missing.

# %%
try:
    import netCDF4  # noqa: F401
except ImportError:
    print(
        "netCDF4 not installed (pip install 'planetmodel[netcdf]'); "
        "skipping the file"
    )
else:
    import tempfile

    from planetmodel import AngularGrid, read_model, write_model

    grid = AngularGrid.gauss_legendre(8)  # a Gauss-Legendre grid for lmax = 8
    sample = model.sample(grid)  # every static field, on GLL nodes
    print("sampled:", sorted(sample.fields), "on", sample.radius.size, "nodes")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prem.nc"
        write_model(model, sample, path)
        back, sample_back = read_model(path)
        print(
            "read back:",
            type(back).__name__,
            "|",
            back.skeleton.nlayers,
            "layers | states",
            {lay.state for lay in back.layers},
        )
        r = np.array([4.0e6, 6.0e6])
        print("rho agrees:", np.allclose(back.rho.evaluate(r), model.rho.evaluate(r)))
        print(
            "viscoelastic moduli agree at 100 s:",
            np.allclose(
                back.viscoelastic_moduli.evaluate(r, omega=w_100),
                model.viscoelastic_moduli.evaluate(r, omega=w_100),
                rtol=1e-9,
            ),
        )
        law = back.layers[5]["viscoelastic_moduli"].law
        print("layer 5 rebuilt from the record:", law.law, law.constants)

# %% [markdown]
# ## Where next
#
# - `02_a_body_of_your_own.py` builds a model from your own tables and
#   layers, and shows the surgery.
# - `03_topography_and_mapping.py` puts relief on a boundary and maps the
#   spherical reference body to the physical one.
# - `04_layered_rheology.py` builds an elastic lithosphere over a Maxwell
#   mantle.
# - `05_a_mesh_for_mfem.py` meshes a layered planet for a finite-element
#   code.
