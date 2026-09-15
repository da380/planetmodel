# %% [markdown]
# # 1. Skeletons and geometries
#
# planetmodel describes a planet in three levels. The lowest is the
# **skeleton**: the radii of the boundaries between concentric layers,
# and nothing else. The next is the **geometry**: the skeleton placed in
# the physical world by one continuous mapping, together with names for
# its layers and interfaces. The third level is the model, which attaches
# fields, and with them the physics, to the layers of a geometry. This
# tutorial covers the first two levels.
#
# Everything at these two levels is a number. A skeleton does not know
# whether its radii are metres or Earth radii; that meaning is supplied
# by whoever builds a model on top. Every tolerance is a fraction of the
# skeleton's span, the distance from its innermost radius to its
# outermost.
#
# Run this file as a script, or cell by cell in an editor that understands
# `# %%` markers.

# %%
import numpy as np

from planetmodel import Geometry, Skeleton

# %% [markdown]
# ## A skeleton
#
# A skeleton is a strictly increasing list of boundary radii. The example
# is a four-layer planet in units of its own radius: an inner core, an
# outer core, a mantle and a thin crust. Layers are numbered from the
# centre, starting at zero, and each has an interval of radius.

# %%
sk = Skeleton([0.0, 0.19, 0.55, 0.99, 1.0])
print(sk)
print("layers:", sk.nlayers, "| span:", sk.span, "| hollow:", sk.is_hollow)
for i in range(sk.nlayers):
    print(f"  layer {i}: interval {sk.interval(i)}")

# %% [markdown]
# `locate` says which layer a radius lies in and returns a `Location`. At
# a boundary between two layers both are candidates, and the skeleton
# does not choose between them. A layered model can have two different
# values at a boundary, one from each side, so which side you mean is
# your decision. A radius within a small fraction of the span of a
# boundary counts as being on it.

# %%
print(sk.locate(0.3))
print(sk.locate(0.55))
try:
    sk.locate(0.55).layer
except ValueError as err:
    print("refused:", err)
print("with a side chosen:", sk.locate(0.55).layers[1])

print(sk.locate(0.55 + 1e-12).boundary, sk.locate(0.55 + 1e-6).boundary)

# %% [markdown]
# ## Changing a skeleton
#
# A skeleton can be refined, which inserts boundaries; truncated, which
# cuts it from above; hollowed, which cuts it from below; extended, which
# appends layers outside; or coarsened, which removes interior
# boundaries. Each of these returns a new skeleton and leaves the
# original as it was. Coarsening also returns a map that records which
# of the old layers each new layer was made from.

# %%
fine = sk.refined([0.9])
cut = sk.truncated(0.99)
cored = sk.hollowed(0.55)
grown = sk.extended([1.2])
coarse, cmap = sk.coarsened(drop=[0])
print("refined:  ", fine.boundaries)
print("truncated:", cut.boundaries)
print("hollowed: ", cored.boundaries)
print("extended: ", grown.boundaries)
print("coarsened:", coarse.boundaries, "|", cmap)

# %% [markdown]
# A skeleton may be hollow. An innermost radius above zero describes a
# spherical shell, which is what a model of mantle convection needs.

# %%
shell = Skeleton([0.55, 0.99, 1.0])
print(shell, "| hollow:", shell.is_hollow)

# %% [markdown]
# ## A geometry
#
# A geometry is a skeleton with a mapping and names. The mapping takes
# the spherical reference body to the physical planet. When no mapping is
# given it is the identity, and the physical planet is the sphere
# itself. Names are optional. The meshers pass them to a solver as the
# names of the mesh's regions and boundaries, and layers and interfaces
# can be reached by name as well as by index.

# %%
g = Geometry(sk,
             layer_names=["inner_core", "outer_core", "mantle", "crust"],
             interface_names=["icb", "cmb", "moho", "surface"])
print(g)
print(g.layer("mantle"))
print(g.interface("cmb"))
print("outer interface:", g.interface(-1))

# %% [markdown]
# An interface is a boundary between two layers, or the outer boundary.
# Interfaces are numbered from the centre, starting at zero. In a full
# geometry, one whose innermost radius is zero, interface `k` separates
# layers `k` and `k + 1`; the outer interface has `-1` as the layer above
# it, meaning the outside. In a hollow geometry the inner boundary is
# also an interface: it is interface 0, with `-1` below it.

# %%
def show(geometry):
    for f in geometry.interfaces:
        print(f"  interface {f.index} {f.name!r:10s} r = {f.radius:.2f}  "
              f"between {f.between}")


show(g)
hollow = Geometry(shell, interface_names=["cmb", "moho", "surface"])
show(hollow)

# %% [markdown]
# The same operations that change a skeleton also change a geometry, and
# the names come along. A layer that is split loses its name; a layer
# made by merging others has no name; an interface keeps its name
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
# Because nothing at this level knows about units, changing them is a
# matter of multiplying every length by a number, which `scaled` does. A
# model built on top of a geometry decides once what one unit means, and
# hands out scaled geometries when a consumer wants other numbers.

# %%
earth = g.scaled(6.371e6)
print(earth.skeleton.boundaries)
print("moho radius:", earth.interface("moho").radius)
assert np.allclose(earth.scaled(1.0 / 6.371e6).skeleton.boundaries, sk.boundaries)

# %% [markdown]
# The next tutorial replaces the identity mapping with an analytic one
# and shows what a geometry checks before accepting it.
