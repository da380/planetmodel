"""Skeleton, layers and the reference body.

A Skeleton is an ordered list of boundary radii: the geometry every
field and mesh is stated against.  A Layer is one interval of it, the
fields the layer holds by name, a name, and whether it is solid, fluid
or vacuum.  A ReferenceBody is a list of layers whose intervals abut,
centre outward; its interfaces are the boundaries between them.  All
surgery on a body is a list operation on its layers and returns a new
body: the original is never changed.

This script builds a three-layer body by hand and exercises the
skeleton, the annotations, the interfaces and each surgery.
"""
import numpy as np

from planetmodel import (DENSITY, Dimensions, Layer, RadialField, ReferenceBody,
                         Skeleton)

# -- the skeleton ----------------------------------------------------------
sk = Skeleton([0.0, 1221.5e3, 3480.0e3, 6371.0e3])       # metres, centre out
assert sk.nlayers == 3
assert sk.interval(1) == (1221.5e3, 3480.0e3)
assert sk.layer_index(-1) == 2                            # negative indices wrap

where = sk.locate(3480.0e3)                               # a radius on a boundary
assert where.layers == (1, 2) and where.boundary == 2     # both sides named
assert sk.locate(5000.0e3).layers == (2,)                 # an interior point
assert sk.spans(1221.5e3, 3480.0e3, layer=1)              # to a relative tolerance
print("skeleton:", sk.boundaries / 1e3, "km;", sk.nlayers, "layers")


# -- single-layer fields, one per layer -------------------------------------
def density_on(i, value):
    """A constant density on layer i, stated on that layer's own skeleton."""
    return RadialField(Skeleton(sk.interval(i)), [lambda r: 0.0 * r + value],
                       name="rho", character=DENSITY,
                       dimensions=Dimensions.DENSITY)


layers = [
    Layer(0, interval=sk.interval(0), name="inner core",
          fields={"rho": density_on(0, 13.0e3)}),
    Layer(1, interval=sk.interval(1), name="outer core", state="fluid",
          fields={"rho": density_on(1, 11.0e3)}),
    Layer(2, interval=sk.interval(2), name="mantle",
          fields={"rho": density_on(2, 4.5e3)}),
]
body = ReferenceBody(layers)
assert body.skeleton == sk                                # the intervals abut
assert body.field_names == ("rho",)
assert body.layers[1].is_fluid and not body.layers[0].is_fluid
assert [lay.name for lay in body.layers] == ["inner core", "outer core", "mantle"]

# A layer is a value: with_field returns a new layer, and the body is
# immutable except for add_field, which attaches a body-wide field.
extra = body.layers[2].with_field("porosity", density_on(2, 0.1))
assert "porosity" in extra and "porosity" not in body.layers[2]

# -- interfaces --------------------------------------------------------------
body = body.name_interface(0, "icb").name_interface(1, "cmb")
body = body.name_interface(-1, "surface")
cmb = body.interface("cmb")
assert cmb.radius == 3480.0e3 and cmb.between == (1, 2)
print("interfaces:", [(f.name, f.radius / 1e3) for f in body.interfaces])

# -- surgery: every method returns a new body ---------------------------------
cut = body.truncated(5000.0e3, name="top")               # cuts the mantle
assert cut.skeleton.boundaries[-1] == 5000.0e3 and cut.skeleton.nlayers == 3
assert cut.layers[2].interval == (3480.0e3, 5000.0e3)
assert np.isclose(cut["rho"].evaluate(4000.0e3), 4.5e3)   # fields follow the cut

split = body.refined([5701.0e3], names=["670"])     # names the new interface
assert split.skeleton.nlayers == 4
assert split.interface("670").radius == 5701.0e3
assert np.isclose(split["rho"].evaluate(6000.0e3), 4.5e3)  # the field, re-stated

grown = body.extended([6500.0e3])                        # no fields: empty
assert grown.layers[3].fields == {} and grown.layers[3].state == "solid"
assert grown["rho"].domain == (0, 1, 2)                   # the view knows

stretched = body.extended([6500.0e3], fields="extrapolate")
assert "rho" in stretched.layers[3]                       # the mantle's rule continues
assert np.isclose(stretched["rho"].evaluate(6400.0e3), 4.5e3)

buffered = body.with_buffer(ratio=0.2)                    # a vacuum shell
assert buffered.layers[-1].is_vacuum and buffered.layers[-1].fields == {}
assert np.isclose(buffered.skeleton.boundaries[-1], 1.2 * 6371.0e3)

# Coarsening drops interior boundaries (here the ICB) and merges the layers
# either side; the fields keep their resolution.  A fluid merged with a
# solid is neither, so the caller says what the merged layer is.
merged, mapping = body.coarsened(drop=[0], state="solid")
assert merged.skeleton.nlayers == 2 and merged.layers[0].state == "solid"
assert np.isclose(merged["rho"].evaluate(500.0e3), 13.0e3)
assert np.isclose(merged["rho"].evaluate(2000.0e3), 11.0e3)
print("merged:", [lay.interval for lay in merged.layers], "map:", mapping)

# The original is unchanged by all of the above.
assert body.skeleton == sk and body.skeleton.nlayers == 3
print("ok: skeleton, layers, interfaces and surgery behave as documented")
