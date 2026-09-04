"""The mesh manifest, `planetmodel.mesh.manifest/1`.

A mesh file says where the nodes are and which attribute each element
carries; the manifest beside it says what the numbers mean.  It is a
JSON document with these blocks: `model` (what was meshed and its
units), `mesh` (dimension, order, counts, how it was curved),
`delivery` ("physical" nodes or "referential" nodes plus a mapping),
`layers` (one entry per element attribute, centre outward: name,
radii in mesh units, state, the fields the layer holds, whether it is
vacuum, and the law behind its moduli), `interfaces` (one per boundary
attribute), `sizing`, `validation`, `provenance`, an optional
`coarsening` and `mapping`, and `files` with digests.  The mesher
builds it from typed entries in one place; the same structure is what
`validate_structure` checks on read.

This script builds a manifest for a small hand-made body without
running gmsh, writes and reads it, and validates it against the counts
a mesh would carry.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from planetmodel import DENSITY, Dimensions, RadialField, ReferenceBody, Skeleton
from planetmodel.io import manifest as mf

# -- a body: solid core, fluid ocean, vacuum buffer outside -------------------------
sk = Skeleton([0.0, 0.5e6, 1.0e6])
rho = RadialField(sk, [lambda r: 0.0 * r + 5.0e3, lambda r: 0.0 * r + 1.0e3],
                  name="rho", character=DENSITY, dimensions=Dimensions.DENSITY)
body = (ReferenceBody.from_fields(sk, {"rho": rho})
        .annotate(0, name="core").annotate(1, name="ocean", state="fluid")
        .name_interface(0, "cmb").name_interface(1, "surface")
        .with_buffer(ratio=0.5))
rref = 1.0e6                                              # one mesh unit, in metres

# -- the entries, one per attribute, centre outward --------------------------------
edges = np.asarray(body.skeleton.boundaries) / rref
layers = [mf.LayerEntry.from_layer(lay, attribute=i + 1, r_inner_nd=edges[i],
                                   r_outer_nd=edges[i + 1])
          for i, lay in enumerate(body.layers)]
interfaces = [mf.InterfaceEntry.from_interface(face, attribute=i + 1,
                                               mean_radius_nd=face.radius / rref)
              for i, face in enumerate(body.interfaces)]
assert [e.name for e in layers] == ["core", "ocean", "buffer"]
assert [e.state for e in layers] == ["solid", "fluid", "vacuum"]
assert [e.fields for e in layers] == [["rho"], ["rho"], []]
assert [e.law for e in layers] == [None, None, None]      # no moduli anywhere
assert [e.name for e in interfaces] == ["cmb", "surface", "buffer"]

# -- what a build would report ---------------------------------------------------------
# These come from gmsh and the validator in a real build; here they are
# stated so the blocks can be shown.
report = SimpleNamespace(negative_jacobians=0, min_sicn=0.4, negative_cells=0,
                         inward_faces=0, max_interface_radius_error=1e-9,
                         knots_aligned=True, warnings=[])
orientation = SimpleNamespace(faces_flipped=0)
sizing = SimpleNamespace(size=0.1, far_size=0.3, decay_width=0.2)

card = mf.MeshManifest.from_build(
    model={"name": "two shells", "source": None, "sha256": None, "rref_m": rref,
           "units": mf.units_block(body.scales, rref, rref)},
    mesh=mf.mesh_block(dimension=3, order=2, gmsh_version="4.15",
                       algorithm_2d=6, algorithm_3d=1,
                       counts={"nodes": 100, "elements": 400},
                       curving={"optimized": False}),
    delivery="physical", layers=layers, interfaces=interfaces,
    sizing=mf.sizing_block(policy="uniform_interfaces",
                           sizes={i: sizing for i in range(3)}),
    validation=mf.validation_block(report, orientation),
    provenance=mf.provenance_block(mesh_file="two_shells.msh"))

assert card.schema == mf.SCHEMA
assert card.vacuum_attributes == (3,)                    # a solver excludes these
assert card.layer_attribute("ocean") == 2 and card.interface_attribute("cmb") == 1
assert card.layers[1]["is_vacuum"] is False and card.layers[2]["is_vacuum"] is True
assert card.model["units"]["geometry_divisor"] == rref    # what one mesh unit is
assert card.provenance["planetmodel_version"] == mf.planetmodel_version()
print("blocks:", [k for k in vars(card) if not k.startswith("_")])
print("layers:", [(e["attribute"], e["name"], e["state"]) for e in card.layers])

# -- write, read, validate ------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    path = mf.write(Path(tmp) / "two_shells", card)      # the suffix is chosen for you
    print("written:", path.name)
    with open(path) as fh:
        raw = json.load(fh)
    assert raw["schema"] == mf.SCHEMA and raw["delivery"] == "physical"
    assert raw["interfaces"][1]["between_layers"] == [1, 2]

    back = mf.read(path)
    assert back.layers == card.layers and back.interfaces == card.interfaces
    mf.validate_structure(back)                             # every key of every block
    mf.validate_against(back, layer_count=3, interface_count=3)   # what the mesh has
    try:
        mf.validate_against(back, layer_count=2, interface_count=3)
    except ValueError as exc:
        print("refused as expected:", exc)
    else:
        raise AssertionError("a manifest must match the mesh it describes")

print("ok: the manifest is built once from typed entries and validated the same way")
