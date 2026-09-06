# %% [markdown]
# # 6. A model of your own
#
# A model is a geometry with a collection of fields on every layer, the scales
# its numbers are in, and the specs that say what its names mean. `Model`
# is the one base and there is no hierarchy beneath it: a model type is a
# class derived from `Model` alone, and the behaviour it exposes it gets
# by wrapping the library's free functions of a layer or a model. This
# tutorial builds a viscous planet, validates it, converts its units, cuts
# it, and then makes it a model type of its own.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import (
    DENSITY,
    SCALAR,
    Dimensions,
    FieldSpec,
    Geometry,
    Model,
    RadialField,
    SelfGravitating,
    Skeleton,
    constant_field,
    gravity,
    layer_method,
    polynomial_layer,
    testing,
)
from planetmodel.units import unit_string

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## The geometry, then the fields
#
# Names come from the geometry. Fields are attached one layer at a time,
# each on exactly its layer's interval; here a fluid core with a density
# alone, and a mantle with a density, a viscosity and a yield stress.

# %%
sk = Skeleton([0.0, 3480e3, 6371e3])
g = Geometry(sk, layer_names=["core", "mantle"], interface_names=["cmb", "surface"])
core, mantle = sk.interval(0), sk.interval(1)

layers = [
    {"rho": constant_field(11e3, core, character=DENSITY, name="rho")},
    {
        "rho": RadialField(
            mantle,
            polynomial_layer([7.9565, -6.4761, 5.5283, -3.0807], mantle, scale=6371e3)
            * 1e3,
            character=DENSITY,
            name="rho",
        ),
        "viscosity": constant_field(1e21, mantle, name="viscosity"),
        "yield_stress": constant_field(1e8, mantle, name="yield_stress"),
    },
]

# %% [markdown]
# `rho` and `viscosity` are in the vocabulary, which fixes their character
# and dimensions. `yield_stress` is not: a name with no spec is accepted,
# but it has no dimensions and would be refused on conversion. A spec
# gives it both.

# %%
PRESSURE = Dimensions(mass=1, length=-1, time=-2)
model = Model(
    g,
    layers,
    specs={"yield_stress": FieldSpec(SCALAR, PRESSURE, meaning="yield stress")},
)
print(model)
print("mantle holds:", model.layer("mantle").names)
print("yield stress unit:", unit_string(model.spec("yield_stress").dimensions))
print("common to every layer:", model.common_names())

# %% [markdown]
# Validation is by name and character: a field attached under a name with
# a spec must have the spec's character. A plain scalar under `rho` is
# refused, as is a field on the wrong interval.

# %%
try:
    model.with_field("core", "rho", constant_field(11e3, core), replace=True)
except ValueError as exc:
    print("refused:", exc)
try:
    model.with_field("core", "viscosity", constant_field(1e20, mantle))
except ValueError as exc:
    print("refused:", exc)

# %% [markdown]
# ## Units
#
# The model alone knows units. `Scales` say what one stored unit of
# length, mass and time is in SI; `nondimensionalised` picks the
# geophysical scales in which `G = 1`, and every field is converted by
# name through the dimensions its spec declares. Polynomial layers convert
# exactly, so the round trip is exact.

# %%
nd = model.nondimensionalised()
print("scales:", nd.scales)
print("G:", nd.G, "| outer radius:", nd.skeleton.boundaries[-1])
print("mantle viscosity, non-dimensional:", nd.layer("mantle")["viscosity"](0.8))
back = nd.in_si()
r = np.linspace(*mantle, 5)
print(
    "round trip error in rho:",
    np.max(np.abs(back.layer("mantle")["rho"](r) - model.layer("mantle")["rho"](r))),
)

# %% [markdown]
# ## Surgery carries the fields
#
# Refining, truncating and hollowing go through the geometry and re-state
# each affected field on its new interval by the field's own rule, exactly
# for polynomials. Extending appends shells holding what they are given.

# %%
split = model.refined([5701e3], names=["d670"])
print(split.nlayers, "layers;", [lay.name for lay in split.layers])
print(
    "rho continuous across the new boundary:",
    split.layer(1)["rho"](5701e3),
    split.layer(2)["rho"](5701e3),
)
cut = model.truncated(6000e3, name="top")
print("cut at 6000 km:", cut.skeleton.boundaries[-1], cut.geometry.interfaces[-1].name)
with_air = model.extended([6500e3], names=["atmosphere"])
print("an empty shell holds:", with_air.layer("atmosphere").names)

# %% [markdown]
# ## Free functions, and a model type that wraps them
#
# Gravity is a function of a model that asks each layer for `rho`; the
# elastic functions are functions of a layer. Both work on a bare `Model`.
# A model type of your own is a class derived from `Model` alone that
# exposes them as methods: a function of a model is a method as soon as
# it is assigned in the class body, a function of a layer goes through
# `layer_method`, which resolves an index or a name through `model.layer`,
# and the shipped mixins (`Elastic`, `SelfGravitating`, `Viscoelastic`)
# bundle the wrapped methods a kind of model exposes. Each mixin is a
# class body of such assignments and nothing else, so the hierarchy stays
# flat. Every copy goes through `Model.replaced`, a shallow copy, so the
# constructor is yours to design and surgery keeps the class.

# %%
print("surface gravity:", gravity(model, 6371e3), "m/s^2")


def viscosity_contrast(layer):
    """The viscosity at the top of a layer over that at its bottom."""
    eta = layer["viscosity"]
    lo, hi = layer.interval
    return eta(hi) / eta(lo)


class ViscousPlanet(SelfGravitating, Model):
    """A model type with the questions a convection code asks.

    A fluid core of constant density under a mantle whose density is a
    cubic in r / a, with a constant viscosity and yield stress.  The
    constructor builds the geometry and the fields from these numbers
    and declares the one name outside the vocabulary; nothing is taken
    from the surrounding script.
    """

    PRESSURE = Dimensions(mass=1, length=-1, time=-2)
    SPECS = {"yield_stress": FieldSpec(SCALAR, PRESSURE, meaning="yield stress")}

    viscosity_contrast = layer_method(viscosity_contrast)

    def __init__(
        self,
        viscosity,
        *,
        yield_stress=1e8,
        core_radius=3480e3,
        radius=6371e3,
        core_density=11e3,
        mantle_density=(7.9565, -6.4761, 5.5283, -3.0807),
    ):
        sk = Skeleton([0.0, core_radius, radius])
        geometry = Geometry(
            sk, layer_names=["core", "mantle"], interface_names=["cmb", "surface"]
        )
        core, mantle = sk.interval(0), sk.interval(1)
        rho_mantle = polynomial_layer(mantle_density, mantle, scale=radius) * 1e3
        layers = [
            {"rho": constant_field(core_density, core, character=DENSITY, name="rho")},
            {
                "rho": RadialField(mantle, rho_mantle, character=DENSITY, name="rho"),
                "viscosity": constant_field(viscosity, mantle, name="viscosity"),
                "yield_stress": constant_field(
                    yield_stress, mantle, name="yield_stress"
                ),
            },
        ]
        super().__init__(geometry, layers, specs=self.SPECS)


planet = ViscousPlanet(1e22)
print("surface gravity as a method:", planet.gravity(6371e3))
print("mantle viscosity contrast:", planet.viscosity_contrast("mantle"))
print("mass:", planet.mass(), "kg")
cut = planet.refined([5701e3])
print(
    type(cut).__name__, "keeps its class through surgery, with", cut.nlayers, "layers"
)
testing.check_model(planet)
print("check_model passes")

# %% [markdown]
# ## A picture
#
# Density and gravity of the planet against radius, drawn the way
# `planetmodel.plotting` draws every profile of a spherically symmetric
# model: radius upward, one segment per layer, and a line joining the
# two sides of each discontinuity. `with_gravity()` is the planet with
# its gravity attached to every layer as a field under the vocabulary
# name `g`, so both panels are drawn the same way, by the name of a
# field the model holds.

# %%
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; no figure")
    raise SystemExit(0)

from planetmodel.plotting import radial_profile  # noqa: E402

with_g = planet.with_gravity()
print("the planet now holds", with_g.layer("mantle").names)

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
radial_profile(left, with_g, "rho", scale=1e-3, value_scale=1e-3, lw=1.4)
radial_profile(right, with_g, "g", scale=1e-3, lw=1.4)
left.set_xlabel("rho (g/cm^3)")
left.set_ylabel("r (km)")
left.set_title("density")
right.set_xlabel("g (m/s^2)")
right.set_title("gravity")
fig.tight_layout()
out = FIGURES / "tutorial_06_a_model_of_your_own.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
