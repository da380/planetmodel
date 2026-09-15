# %% [markdown]
# # A mesh for MFEM
#
# A reference body, surgery on its geometry, relief on the Moho, and a
# three-dimensional mesh with a **manifest** that tells a finite-element
# code what each region and boundary is. Then the same build written as a
# recipe file, and the export to MFEM's own formats. Everything runs at a
# coarse resolution in a few seconds; the last cell says what changes at
# Earth resolution.
#
# You need the meshing extra, `pip install 'planetmodel[meshing]'`, which
# brings gmsh; the export at the end needs `planetmodel[mfem]` too.

# %%
import tempfile
from pathlib import Path

import numpy as np

from planetmodel import GriddedTopography, Surface, read_isotropic_deck

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"
deck = read_isotropic_deck(DATA / "prem.nocrust")
print(deck)

# %% [markdown]
# ## The body
#
# `prem.nocrust` is an isotropic deck that stops at the base of the crust.
# Read without a reference period it is an `ElasticModel`: it has Q
# columns but no calibration, and the type does not pretend otherwise.
#
# A mesh wants named interfaces, and it does not want PREM's ten layers:
# the thin ones near the top would force tiny elements everywhere. So we
# **coarsen** the geometry, keeping the two boundaries that matter to a
# solver, and add a vacuum buffer above the surface. Surgery is
# copy-on-write: each call returns a new body.

# %%
body = (deck.name_interface(0, "icb").name_interface(1, "cmb")
            .name_interface(-1, "moho"))
keep = [body.interface(n).index for n in ("icb", "cmb")]
body, merged = body.coarsened(keep=keep, state="solid")
body = (body.annotate(0, name="inner_core").annotate(1, name="outer_core")
            .annotate(2, name="mantle").with_buffer(ratio=0.2))
print("layers after coarsening:")
for lay in body.layers:
    print(f"  {lay.index}: {lay.name or '-':10s} {lay.state:7s} "
          f"[{lay.interval[0] / 1e3:.0f}, {lay.interval[1] / 1e3:.0f}] km  "
          f"{len(lay.fields)} fields")
print("interfaces:", [f.name for f in body.interfaces])

# %% [markdown]
# Coarsening the geometry did not coarsen the model. The mantle is one
# layer now, but its density still jumps at 670 km, because the merged
# layer's field answers from whichever fine piece contains the radius:

# %%
b = deck.skeleton.boundaries
r670 = float(b[np.argmin(np.abs(b - 5701.0e3))])   # the 670 km discontinuity
print(f"boundary merged away at {r670 / 1e3:.0f} km; rho below / above it: "
      f"{body.rho.evaluate(r670 - 1.0):.0f} / {body.rho.evaluate(r670 + 1.0):.0f}")

# %% [markdown]
# The buffer is a vacuum shell holding no fields, and the manifest will
# say so with an empty field list; it is the reference domain a mapping
# leaves fixed at its outer edge.
#
# ## Relief on the Moho
#
# The Moho of CRUST-1.0, centred and placed at the deck's Moho radius, as
# in the topography tutorial. The **mapping** spreads the relief linearly
# down to the core-mantle boundary, which is the next interface, and a
# mesh can be delivered either way: physical nodes, or reference nodes
# with the mapping beside them for the solver to apply.

# %%
from planetmodel import layer_linear, validity_lattice  # noqa: E402

depth = GriddedTopography.from_xyz(DATA / "crust-1.0" / "depthtomoho.xyz",
                                   scale=1.0e3)
moho = Surface(6371.0e3, topography=depth, name="moho").centred()
body = body.with_surface("moho", moho.at(body.interface("moho").radius))
mapping = body.mapping(rule=layer_linear())
print(mapping)
print("the mapping is orientation-preserving:",
      mapping.is_valid(sample=validity_lattice(body.skeleton)))

# %% [markdown]
# ## The mesh
#
# A `MeshSpec` says which body, how to size the elements, and how to
# deliver. The sizing here is deliberately coarse. Everything inside the
# mesher is non-dimensional by the reference radius `rref`; the manifest
# records the scale so a consumer can put the metres back.

# %%
try:
    import gmsh  # noqa: F401
except ImportError:
    gmsh = None
    print("gmsh not installed (pip install 'planetmodel[meshing]'); stopping here")
else:
    from planetmodel.io import manifest as mf
    from planetmodel.mesh3d import MeshSpec, UniformInterfaces, build_layered_mesh

    rref = 6371.0e3                                  # one mesh unit, in metres
    spec = MeshSpec(body=body, rref=rref, order=2,
                    sizing=UniformInterfaces(0.15 * rref, 0.30 * rref, 0.30 * rref),
                    mapping_rule=layer_linear(), delivery="referential")
    workdir = Path(tempfile.mkdtemp())
    result = build_layered_mesh(spec, workdir / "planet")
    print(result)
    card = mf.read(result.manifest_path)
    print("\nlayers in the manifest (attribute, name, state, fields):")
    for lay in card.layers:
        print(f"  {lay['attribute']}  {lay['name']:10s} {lay['state']:7s} "
              f"{lay['fields']}")
    print("interfaces:", [f["name"] for f in card.interfaces])
    print("validation:", {k: v for k, v in card.validation.items()
                          if k in ("negative_jacobians", "wrong_orientation",
                                   "min_sicn")})
    print("vacuum attributes, for a solver to exclude:", card.vacuum_attributes)

# %% [markdown]
# The **manifest** is a JSON file beside the mesh. Attributes are numbered
# from the centre outward; names are advisory; the field list per layer
# says what material the model supplies, and an empty list means the
# consumer does. A finite-element code reads the manifest and never
# hard-codes which attribute the mantle is.
#
# ## The same build as a recipe
#
# A **recipe** is a TOML file that says everything the build above said,
# so that a mesh can be reproduced from one file and one command:
# `python -m planetmodel.mesh3d recipe.toml`. Lengths carry a unit suffix,
# surfaces name their data files, and the outer boundary is placed at a
# radius you compute and write down. The recipe holds you to the data: a
# boundary named in `[[surfaces]]` must sit at the mean radius its files
# give, so here the mantle is grown by 3 km to CRUST-1.0's mean Moho
# rather than the relief being moved to the deck's. The manifest records
# the recipe's digest.

# %%
if gmsh is not None:
    from planetmodel.io import recipe as rp

    text = f"""
[model]
source = "{DATA / 'prem.nocrust'}"
reader = "isotropic_deck"
rref_m = {rref}

[geometry]
drop_outermost_interfaces = 7
truncate_at = {moho.reference_radius}
truncate_name = "moho"
buffer = {{ ratio = 0.2, name = "buffer" }}

[mesh]
dimension = 3
order = 2

[sizing]
policy = "uniform_interfaces"
h_min_km = {0.15 * rref / 1e3:.0f}
h_max_km = {0.30 * rref / 1e3:.0f}
decay_width_km = {0.30 * rref / 1e3:.0f}

[[surfaces]]
name = "moho"
files = ["{DATA / 'crust-1.0' / 'depthtomoho.xyz'}"]
units = "km"

[mapping]
mode = "referential"
rule = "layer_linear"
exaggeration = 1

[output]
path = "from_recipe"
"""
    recipe_path = workdir / "planet.toml"
    recipe_path.write_text(text)
    recipe = rp.read(recipe_path)
    print(recipe, "|", recipe.command)
    print("outer boundary:", recipe.spec.outer_name, "at",
          f"{recipe.spec.outer_radius / 1e3:.1f} km")
    again = recipe.build()
    card2 = mf.read(again.manifest_path)
    print("layers:", [(lay["name"], lay["state"]) for lay in card2.layers])
    print("same attributes as the build above:",
          [lay["attribute"] for lay in card2.layers]
          == [lay["attribute"] for lay in card.layers],
          "| recipe digest recorded:", "recipe" in card2.provenance)

# %% [markdown]
# ## Export for MFEM
#
# The exporter writes MFEM's own formats: the mesh with curved nodes, one
# GridFunction per field, and the displacement of the mapping. It reads a
# referential build and produces either delivery; here the physical one,
# with the displacement added to the nodes and the fields pushed forward.

# %%
if gmsh is not None:
    try:
        import mfem.ser  # noqa: F401
    except ImportError:
        print("PyMFEM not installed (pip install 'planetmodel[mfem]'); no export")
    else:
        from planetmodel.mesh3d import export_mfem

        out = export_mfem(result, workdir / "planet", fields=["rho"],
                          delivery="physical")
        print(out)
        card = mf.read(out.manifest_path)
        gf = card.files["grid_functions"][0]
        print("GridFunction:", gf["name"], "in", gf["fe_space"], "on layers",
              gf["layers"], "| units", gf["units"])
        print("read the mesh as Mesh(path, 1, 0, false):",
              card.files["mesh_read_options"])

# %% [markdown]
# ## At Earth resolution
#
# Two things change. The sizing: replace the three sizes with the element
# sizes you want at the interfaces, in metres, and the build takes minutes
# and gives a mesh of a hundred thousand or more tetrahedra. And the
# crust: `extended([6371.0e3], interface_names=["surface"])` before the buffer
# adds a 24 km shell with no fields, whose material the finite-element code
# supplies by attribute, and which needs elements a few tens of kilometres
# across to mesh. Both are manual runs, not tests; `scripts/` holds one.
