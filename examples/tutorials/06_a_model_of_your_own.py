# %% [markdown]
# # 6. A model of your own
#
# A model is a geometry with a set of fields on every layer, together
# with the scales its numbers are measured in and the specs that say what
# each field name means. `Model` is the one base class. A model type, such
# as `PREM` or one you write yourself, is a class derived from `Model`
# alone, and any behaviour it offers as methods comes from wrapping the
# library's free functions. There is no hierarchy of model types.
#
# This tutorial builds a two-layer viscous planet from scratch, checks
# it, converts its units, cuts it into more layers, and then turns the
# same construction into a model type of its own.
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
# The geometry gives the layers and interfaces their names. Fields are
# then attached one layer at a time, each defined on exactly its layer's
# interval. The example is a fluid core with a density alone, and a
# mantle with a density, a viscosity and a yield stress. The mantle
# density is PREM's lower-mantle polynomial, published in g/cm^3 and
# multiplied by 1000 here to give kg/m^3.

# %%
sk = Skeleton([0.0, 3480e3, 6371e3])
g = Geometry(sk, layer_names=["core", "mantle"], interface_names=["cmb", "surface"])
core, mantle = sk.interval(0), sk.interval(1)

layers = [
    {"rho": constant_field(core, 11e3, character=DENSITY, name="rho")},
    {
        "rho": RadialField(
            mantle,
            polynomial_layer(mantle, [7.9565, -6.4761, 5.5283, -3.0807], scale=6371e3)
            * 1e3,
            character=DENSITY,
            name="rho",
        ),
        "viscosity": constant_field(mantle, 1e21, name="viscosity"),
        "yield_stress": constant_field(mantle, 1e8, name="yield_stress"),
    },
]

# %% [markdown]
# ## What the names mean
#
# A *spec* says what a field name means: its character and its physical
# dimensions. The library ships a vocabulary of specs for the common
# names, and `rho` and `viscosity` are in it. `yield_stress` is not. A
# name with no spec is accepted, but the model then knows nothing about
# its dimensions and refuses to convert it between units. Passing a spec
# to the model gives the name a character and dimensions.
#
# `Dimensions` records powers of mass, length and time. A stress is
# mass per length per time squared.

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
# The model checks its fields when it is built. A field attached under a
# name that has a spec must have the spec's character, and every field
# must be defined on its own layer's interval. Below, a plain scalar is
# refused under the name `rho`, and a field defined on the mantle's
# interval is refused on the core.

# %%
try:
    model.with_field("core", "rho", constant_field(core, 11e3), replace=True)
except ValueError as exc:
    print("refused:", exc)
try:
    model.with_field("core", "viscosity", constant_field(mantle, 1e20))
except ValueError as exc:
    print("refused:", exc)

# %% [markdown]
# ## Units
#
# Only the model knows about units. Its `Scales` record what one stored
# unit of length, mass and time is in SI; the default is SI itself.
# `nondimensionalised` converts the model to scales in which the outer
# radius, the Earth's mean density and the gravitational constant `G`
# are all one. Each field is converted through the dimensions its spec
# declares, which is why every name needs a spec. Polynomial layers are
# converted by rescaling their coefficients, so converting and
# converting back reproduces the original exactly.

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
# ## Changing the layering
#
# A model can be refined by inserting boundaries, truncated at a radius,
# or extended by shells outside its surface. Each operation goes through
# the geometry and then restates every affected field on its new
# interval. For a polynomial this is exact. Refining splits a layer into
# two unnamed parts that hold the same fields, so the density is
# continuous across the new boundary. Extending appends shells that hold
# only the fields they are given, none here.

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
# ## The free functions
#
# The library's shared behaviour is written as free functions. Some take
# a model, such as `gravity(model, r)`, which asks each layer for `rho`
# and integrates. Others take a single layer, such as the elastic
# functions of `planetmodel.materials`. All of them work on a bare
# `Model` with the right fields, and nothing more is needed to use them.

# %%
print("surface gravity:", gravity(model, 6371e3), "m/s^2")

# %% [markdown]
# ## A model type of your own
#
# A model type is a class derived from `Model` that builds its fields in
# its constructor and exposes the free functions as methods. The rules
# are simple. A function of a model becomes a method as soon as it is
# assigned in the class body. A function of a layer is wrapped by
# `layer_method`, which turns `fn(layer)` into `model.fn(which)` and
# accepts a layer index or a layer name. The library's *mixins*,
# `Elastic`, `SelfGravitating` and `Viscoelastic`, are classes that hold
# nothing but such assignments, one bundle per kind of behaviour, so a
# model type lists them in its bases instead of repeating the
# assignments. Every copy of a model, whether from a unit conversion or
# a change of layering, is a shallow copy made by `Model.replaced`, so
# the constructor is entirely yours to design and a copy keeps your
# class.
#
# The example is a function of a layer that a convection code might
# want: the strain rate at which the viscous stress reaches the yield
# stress, a field on the layer. The model type wraps it, mixes in
# gravity, and builds the planet above from a handful of numbers.


# %%
def yield_strain_rate(layer):
    """The strain rate at which viscous stress reaches the yield stress."""
    return (layer["yield_stress"] / layer["viscosity"]).renamed("yield_strain_rate")


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

    yield_strain_rate = layer_method(yield_strain_rate)

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
        rho_mantle = polynomial_layer(mantle, mantle_density, scale=radius) * 1e3
        layers = [
            {"rho": constant_field(core, core_density, character=DENSITY, name="rho")},
            {
                "rho": RadialField(mantle, rho_mantle, character=DENSITY, name="rho"),
                "viscosity": constant_field(mantle, viscosity, name="viscosity"),
                "yield_stress": constant_field(
                    mantle, yield_stress, name="yield_stress"
                ),
            },
        ]
        super().__init__(geometry, layers, specs=self.SPECS)


planet = ViscousPlanet(1e22)
print("surface gravity as a method:", planet.gravity(6371e3), "m/s^2")
print("mass:", planet.mass(), "kg")
rate = planet.yield_strain_rate("mantle")
print(rate, "| at 5000 km:", rate(5000e3), "1/s")
cut = planet.refined([5701e3])
print(type(cut).__name__, "keeps its class when refined, with", cut.nlayers, "layers")
testing.check_model(planet)
print("check_model passes")

# %% [markdown]
# ## A picture
#
# `planetmodel.plotting` draws every profile of a spherically symmetric
# model the same way: radius on the vertical axis increasing upward, one
# curve per layer, and a line joining the two sides of each
# discontinuity. It draws a field by name, so to draw gravity the model
# must hold it as a field. `with_gravity()` is a copy of the planet with
# its gravity attached to every layer under the vocabulary name `g`.

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
