# %% [markdown]
# # A body of your own
#
# PREM comes ready-made. This tutorial builds models from scratch: a
# density profile from a table, a layered body assembled from `Layer`
# values, a vacuum shell, the surgery that cuts and merges layers while
# the fields follow, a field type of your own checked against the
# library's contract, and finally a model class that says what the body
# guarantees.

# %%
import numpy as np

from planetmodel import (ElasticModel, Layer, RadialField, ReferenceBody,
                         Skeleton)
from planetmodel.model import make_fitter

# %% [markdown]
# ## A density from a table
#
# The geometry of a model is its **skeleton**: the ordered boundary radii.
# Everything else is defined against it. A `RadialField` binds one
# function of radius per layer; here the functions are fitted to
# tabulated knots, one fit per layer so that a discontinuity at a
# boundary stays a discontinuity.
#
# The table below is a three-layer toy planet: a dense core, a mantle,
# and a thin crust, with a jump at each boundary.

# %%
skeleton = Skeleton([0.0, 2.0e6, 5.5e6, 6.0e6])
knots = {
    0: (np.linspace(0.0, 2.0e6, 6), 12000.0 - 1500.0 * np.linspace(0, 1, 6) ** 2),
    1: (np.linspace(2.0e6, 5.5e6, 8), 5500.0 - 2000.0 * np.linspace(0, 1, 8)),
    2: (np.linspace(5.5e6, 6.0e6, 3), np.full(3, 2800.0)),
}
fit = make_fitter(kind="cubic")
rho = RadialField(skeleton, [fit(*knots[i]) for i in range(3)], name="rho")
print(rho)
print("rho at the core-mantle boundary, below and above:",
      rho.evaluate(2.0e6, side="lower"), rho.evaluate(2.0e6))
print("integrate over the mantle:", rho.integrate(2.0e6, 5.5e6))

# %% [markdown]
# A field carries a **character** (how it transforms under a mapping) and
# **dimensions**. Density is a weight-one scalar; the library's vocabulary
# knows the canonical names, and you can say so explicitly too.

# %%
from planetmodel import DENSITY, Dimensions  # noqa: E402

rho = RadialField(skeleton, [fit(*knots[i]) for i in range(3)], name="rho",
                  character=DENSITY, dimensions=Dimensions.DENSITY)
print(rho.character, rho.dimensions)

# %% [markdown]
# ## A body from layers
#
# A body is a list of **layers**. Each `Layer` is a value: its interval,
# the fields it holds by name, a state, a name. The fields a layer holds
# are single-layer fields on exactly its interval; `rho[i]` is the piece
# of `rho` on layer `i`. The body's skeleton is the layers' intervals laid
# end to end.

# %%
layers = [
    Layer(index=0, interval=(0.0, 2.0e6), name="core", state="fluid",
          fields={"rho": rho[0]}),
    Layer(index=1, interval=(2.0e6, 5.5e6), name="mantle",
          fields={"rho": rho[1]}),
    Layer(index=2, interval=(5.5e6, 6.0e6), name="crust",
          fields={"rho": rho[2]}),
]
body = ReferenceBody(layers, meta={"name": "toy planet"})
print(body)
for lay in body.layers:
    print(f"  {lay.index} {lay.name:7s} {lay.state:6s} holds {lay.field_names}")
print("rho view domain:", body["rho"].domain, "| the mantle piece integrates to",
      body.layer("mantle")["rho"].integrate(2.0e6, 5.5e6))

# %% [markdown]
# The same body can be built the other way round, from body-wide fields
# on a skeleton, which splits each field into the pieces its layers hold.

# %%
same = ReferenceBody.from_fields(
    skeleton, {"rho": rho},
    layers=[Layer(index=0, state="fluid"), Layer(index=1), Layer(index=2)])
print("from_fields gives the same view:",
      np.allclose(same["rho"].evaluate(np.linspace(0, 6e6, 9)),
                  body["rho"].evaluate(np.linspace(0, 6e6, 9))))

# %% [markdown]
# ## A vacuum shell, and a field with a hole
#
# `extended` appends shells beyond the outer boundary. With no fields
# given they are empty; with `state="vacuum"` they are voids, the buffer
# region a mesh needs beyond the planet. `names` names the new shells and
# `interface_names` their outer boundaries. A field is defined where it is
# defined: the density view now has a **domain** that stops at the crust,
# and it refuses the vacuum by name rather than filling it with zeros.

# %%
with_shell = body.extended([7.0e6], state="vacuum", names=["space"],
                           interface_names=["outer"])
print("layers:", [(lay.name, lay.state, lay.field_names) for lay in with_shell.layers])
print("outer boundary:", with_shell.interface("outer").radius / 1e3, "km")
print("rho domain:", with_shell["rho"].domain)
try:
    with_shell["rho"].evaluate(6.5e6)
except ValueError as err:
    print("refused:", str(err)[:90], "...")

# %% [markdown]
# ## Surgery
#
# Every operation returns a new body: `truncated` cuts at a radius,
# `refined` splits a layer at new radii, `coarsened` merges neighbours.
# The fields follow: a cut restricts them, a split gives each part its
# piece, and a merge keeps the fine pieces underneath. Merging a fluid
# layer with a solid one is refused unless you say which state the result
# has.

# %%
cut = body.truncated(4.0e6)
print("truncated at 4000 km:", cut.skeleton.boundaries / 1e3,
      "| rho at the new top:", cut["rho"].evaluate(4.0e6))

split = body.refined([3.0e6, 4.0e6])
split = split.annotate(1, name="lower_mantle").annotate(3, name="upper_mantle")
print("refined:", [lay.name for lay in split.layers])

# Interior boundaries are numbered from the centre; keeping the original
# two merges the mantle back into one layer.
merged, cmap = split.coarsened(keep=[0, 3])
print("coarsened back:", [lay.name for lay in merged.layers], "|", cmap)
r = np.linspace(2.0e6, 5.5e6, 7)
print("and the density is unchanged:",
      np.allclose(merged["rho"].evaluate(r), body["rho"].evaluate(r)))

# Merging the core into the mantle keeps the fine pieces underneath, so the
# merged layer's density still jumps where the boundary was.
one_layer, _ = body.coarsened(drop=[0], state="solid")
print("core merged into the mantle:", [lay.name for lay in one_layer.layers],
      "| rho a metre below / above the old boundary:",
      one_layer["rho"].evaluate(2.0e6 - 1.0), one_layer["rho"].evaluate(2.0e6 + 1.0))

# %% [markdown]
# ## A field type of your own
#
# A field is anything with a `skeleton`, a `character`, a `name` and an
# `evaluate(r, theta, phi, *, layer, side, frame)` whose result has the
# broadcast shape of its three arguments. Subclassing `FieldBase` adds the
# algebra and the conveniences (`__call__`, `restricted`, `evaluate_at`).
# Here is a field tabulated at a few radii on one layer,
# attached to a body, and checked against the library's executable
# contract, `check_field`, which is what every shipped field passes.

# %%
from planetmodel import SCALAR, FieldBase  # noqa: E402
from planetmodel.testing import check_field  # noqa: E402


class TabulatedField(FieldBase):
    """Linear interpolation of tabulated values on one layer."""

    def __init__(self, skeleton, radii, values, *, name=None):
        self.skeleton = skeleton
        self.character = SCALAR
        self.dimensions = Dimensions.DIMENSIONLESS
        self.name = name
        self._r = np.asarray(radii, float)
        self._v = np.asarray(values, float)

    @property
    def is_radial(self):
        return True

    def evaluate(self, r, theta=None, phi=None, *, layer=None, side="upper",
                 frame="spherical"):
        r = np.asarray(r, float)
        lo, hi = self.skeleton.boundaries[[0, -1]]
        if np.any(r < lo - 1e-9 * hi) or np.any(r > hi + 1e-9 * hi):
            raise ValueError(f"{self.name}: radius outside [{lo}, {hi}]")
        values = np.interp(r, self._r, self._v)
        if theta is not None:                     # broadcast with the angles
            values = np.broadcast_arrays(values, theta, phi)[0]
        return values


mantle = body.layer("mantle")
porosity = TabulatedField(Skeleton(mantle.interval), [2.0e6, 4.0e6, 5.5e6],
                          [0.0, 0.05, 0.12], name="porosity")
check_field(porosity)
print("check_field passed")
porous = body.with_layer(1, mantle.with_field("porosity", porosity))
print("porosity is a view on layers", porous["porosity"].domain,
      "| 2 * porosity at 4000 km:", (2 * porous["porosity"]).evaluate(4.0e6))

# %% [markdown]
# ## What the body guarantees
#
# A plain `ReferenceBody` promises nothing about its fields. A **model
# class** does: `ElasticModel` guarantees `rho` and `elastic_moduli` on
# every layer that holds either, and `validate()` refuses a body where a
# layer has one without the other. `as_class` promotes a body, keeping its
# layers and annotations; the promotion fails here because the toy planet
# has no moduli yet.

# %%
try:
    body.as_class(ElasticModel)
except ValueError as err:
    print("refused:", err)

# %% [markdown]
# Give every layer a modulus and the promotion goes through. The elastic
# tensor of an isotropic medium is an `ElasticField` built from `kappa`
# and `mu`; the fluid core carries one too, with zero shear modulus, as
# PREM's does. A class checks the layers that hold any of its fields, so
# an empty shell or a vacuum passes, but a layer with density and no
# tensor does not: the guarantee is per layer, all or nothing.

# %%
from planetmodel import ElasticField, Symmetry  # noqa: E402

def constant(layer, value):
    """A uniform single-layer field: one function on the layer's interval."""
    return RadialField(Skeleton(layer.interval), [lambda r: np.full_like(r, value)])


elastic = {}
for i, mu in ((0, 0.0), (1, 6.0e10), (2, 3.0e10)):
    lay = body.layers[i]
    elastic[i] = ElasticField(Symmetry.ISOTROPIC,
                              {"kappa": constant(lay, 1.5e11), "mu": constant(lay, mu)},
                              name="elastic_moduli")
solid = body
for i, field in elastic.items():
    solid = solid.with_field(i, "elastic_moduli", field)
model = solid.as_class(ElasticModel)
print(type(model).__name__, "| guaranteed on layers", model.guaranteed_layers)
print("moduli at 4000 km, GPa:")
print(np.round(model.elastic_moduli.evaluate(4.0e6) / 1e9, 1))
print("surgery keeps the class:", type(model.truncated(5.0e6)).__name__)
