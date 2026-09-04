# %% [markdown]
# # 1. Skeletons and geometries
#
# planetmodel describes a planet in three levels. The lowest is the
# **skeleton**: nothing but the radii of the boundaries between concentric
# layers. The next is the **geometry**: the skeleton placed in the physical
# world by one continuous mapping, together with the names of its layers
# and interfaces. Fields, and with them the physics, come later and hang
# on the geometry's layers.
#
# Everything at these two levels is a number. A skeleton does not know
# whether its radii are metres or Earth radii; that meaning is supplied by
# whoever builds a concrete model on top. Every tolerance is relative to
# the skeleton's span.
#
# Run this file as a script, or cell by cell in an editor that understands
# `# %%` markers.

# %%
import numpy as np

from planetmodel import Geometry, Skeleton

# %% [markdown]
# ## A skeleton
#
# A skeleton is a strictly increasing list of boundary radii. Here is a
# four-layer planet in units of its own radius: an inner core, an outer
# core, a mantle and a thin crust.

# %%
sk = Skeleton([0.0, 0.19, 0.55, 0.99, 1.0])
print(sk)
print("layers:", sk.nlayers, "| span:", sk.span, "| hollow:", sk.is_hollow)
for i in range(sk.nlayers):
    print(f"  layer {i}: interval {sk.interval(i)}")

# %% [markdown]
# Asking where a radius lies returns a `Location`. At an interior boundary
# both neighbouring layers are candidates and the skeleton does not choose
# for you: a layered model is two-valued there, and which side you mean is
# your decision.

# %%
print(sk.locate(0.3))
print(sk.locate(0.55))
try:
    sk.locate(0.55).layer
except ValueError as err:
    print("refused:", err)
print("with a side chosen:", sk.locate(0.55).layers[1])

# A radius within rtol * span of a boundary counts as being on it.
print(sk.locate(0.55 + 1e-12).boundary, sk.locate(0.55 + 1e-6).boundary)

# %% [markdown]
# ## Surgery on a skeleton
#
# A skeleton can be refined (boundaries inserted), truncated (cut from
# above), hollowed (cut from below), extended (layers appended outside) or
# coarsened (interior boundaries removed). Each returns a new skeleton;
# the original is untouched. Coarsening also returns a map recording which
# fine layers each coarse layer merged.

# %%
fine = sk.refined([0.9])
cut = sk.truncated(0.99)
cored = sk.hollowed(0.55)
grown = sk.extended([1.2])
coarse, cmap = sk.coarsen(drop=[0])
print("refined:  ", fine.boundaries)
print("truncated:", cut.boundaries)
print("hollowed: ", cored.boundaries)
print("extended: ", grown.boundaries)
print("coarsened:", coarse.boundaries, "|", cmap)

# %% [markdown]
# A skeleton may be hollow: an innermost radius above zero describes a
# spherical shell, as a mantle-convection model needs.

# %%
shell = Skeleton([0.55, 0.99, 1.0])
print(shell, "| hollow:", shell.is_hollow)

# %% [markdown]
# ## A geometry
#
# A geometry is a skeleton with a mapping and names. With no mapping given
# it is the identity: the physical planet *is* the spherical reference
# body. Names are optional and are what the meshers pass on to a solver as
# attribute names; layers and interfaces can be reached by index or by
# name.

# %%
g = Geometry(sk,
             layer_names=["inner_core", "outer_core", "mantle", "crust"],
             interface_names=["icb", "cmb", "moho", "surface"])
print(g)
print(g.layer("mantle"))
print(g.interface("cmb"))
print("outer interface:", g.interface(-1))

# %% [markdown]
# Interfaces are numbered from the centre. For a full geometry interface
# `k` separates layers `k` and `k + 1`, and the outer interface has `-1`
# for the layer above it, meaning the outside. For a hollow geometry the
# inner boundary is interface 0 with `-1` below it.

# %%
def show(geometry):
    for f in geometry.interfaces:
        print(f"  interface {f.index} {f.name!r:10s} r = {f.radius:.2f}  "
              f"between {f.between}")


show(g)
hollow = Geometry(shell, interface_names=["cmb", "moho", "surface"])
show(hollow)

# %% [markdown]
# Surgery on a geometry carries the names along. A split layer loses its
# name, a merged layer's name is `None`, and an interface keeps its name
# wherever its radius survives.

# %%
refined = g.refined([0.9], names=["floor"])
print([lay.name for lay in refined.layers])
print([f.name for f in refined.interfaces])
merged, _ = g.coarsened(drop=[0])
print([lay.name for lay in merged.layers], [f.name for f in merged.interfaces])

# %% [markdown]
# ## Scaling
#
# Because nothing inside knows about units, changing them is a pure
# rescaling: `scaled(k)` multiplies every length by `k`. A concrete model
# built on top of a geometry decides once what one unit means, and hands
# out scaled geometries when a consumer wants other numbers.

# %%
earth = g.scaled(6.371e6)
print(earth.skeleton.boundaries)
print("moho radius:", earth.interface("moho").radius)
assert np.allclose(earth.scaled(1.0 / 6.371e6).skeleton.boundaries, sk.boundaries)

# %% [markdown]
# The next tutorial replaces the identity mapping with an analytic one and
# looks at what a geometry checks before accepting it.
