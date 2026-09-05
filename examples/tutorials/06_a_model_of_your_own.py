# %% [markdown]
# # 6. A model of your own
#
# A model is a geometry with a bag of fields on every layer, the scales
# its numbers are in, and the specs that say what its names mean. There
# is one class, `Model`, and no hierarchy: a named model is an instance,
# and behaviour shared by groups of models is a free function of a layer
# or a model. This tutorial builds one for a viscous planet, validates it,
# converts its units, cuts it, and wraps a free function as a method.

# %%
import numpy as np

from planetmodel import (DENSITY, SCALAR, Dimensions, FieldSpec, Geometry, Model,
                         RadialField, Skeleton, constant_field, gravity,
                         polynomial_layer, testing)
from planetmodel.units import unit_string

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
    {"rho": RadialField(mantle, polynomial_layer([7.9565, -6.4761, 5.5283, -3.0807],
                                                 mantle, scale=6371e3) * 1e3,
                        character=DENSITY, name="rho"),
     "viscosity": constant_field(1e21, mantle, name="viscosity"),
     "yield_stress": constant_field(1e8, mantle, name="yield_stress")},
]

# %% [markdown]
# `rho` and `viscosity` are in the vocabulary, which fixes their character
# and dimensions. `yield_stress` is not: a name with no spec is accepted,
# but it has no dimensions and would be refused on conversion. A spec
# gives it both.

# %%
PRESSURE = Dimensions(mass=1, length=-1, time=-2)
model = Model(g, layers, specs={"yield_stress": FieldSpec(SCALAR, PRESSURE,
                                                          meaning="yield stress")})
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
print("round trip error in rho:",
      np.max(np.abs(back.layer("mantle")["rho"](r) - model.layer("mantle")["rho"](r))))

# %% [markdown]
# ## Surgery carries the fields
#
# Refining, truncating and hollowing go through the geometry and re-state
# each affected field on its new interval by the field's own rule, exactly
# for polynomials. Extending appends shells holding what they are given.

# %%
split = model.refined([5701e3], names=["d670"])
print(split.nlayers, "layers;", [lay.name for lay in split.layers])
print("rho continuous across the new boundary:",
      split.layer(1)["rho"](5701e3), split.layer(2)["rho"](5701e3))
cut = model.truncated(6000e3, name="top")
print("cut at 6000 km:", cut.skeleton.boundaries[-1], cut.geometry.interfaces[-1].name)
with_air = model.extended([6500e3], names=["atmosphere"])
print("an empty shell holds:", with_air.layer("atmosphere").names)

# %% [markdown]
# ## Free functions, and a subclass that wraps them
#
# Gravity is a function of a model that asks each layer for `rho`. A
# subclass may wrap such functions as methods; the library does not.

# %%
print("surface gravity:", gravity(model, 6371e3), "m/s^2")


class ViscousPlanet(Model):
    """A Model with the questions a convection code asks."""

    def gravity(self, radii):
        return gravity(self, radii)

    def viscosity_contrast(self):
        eta = self.layer("mantle")["viscosity"]
        lo, hi = self.layer("mantle").interval
        return eta(hi) / eta(lo)


planet = ViscousPlanet(g, layers, specs=model.specs)
print(type(planet.refined([5701e3])).__name__, "keeps its class through surgery")
print("viscosity contrast:", planet.viscosity_contrast())
testing.check_model(planet)
print("check_model passes")
