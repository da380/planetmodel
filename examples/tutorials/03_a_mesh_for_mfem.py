# %% [markdown]
# # 3. A mesh for MFEM
#
# A geometry is what the 3D mesher takes: its skeleton gives the
# concentric boundaries, its names become the attribute names, and its
# mapping either moves the nodes or travels beside the mesh for the solver
# to apply. The mesher itself knows nothing about units. It divides every
# length by a number of its choosing, the outer radius by default, so that
# gmsh works near unity, and records that number in the manifest.
#
# This tutorial needs the `meshing` extra (gmsh) and, for the last
# section, the `mfem` extra (PyMFEM). Everything is coarse and runs in a
# few seconds; the files go to a temporary directory.

# %%
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton

try:
    from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                    build_layered_mesh, export_mfem_mesh, manifest)
except ImportError as err:
    raise SystemExit(f"this tutorial needs gmsh: {err}")

workdir = Path(tempfile.mkdtemp(prefix="planetmodel_"))

# %% [markdown]
# ## The geometry and the spec
#
# A three-layer unit planet with a mapping confined to its outer part: the
# flattening is tapered to zero at the surface, with the two kinks it
# introduces declared as knots on skeleton boundaries. Confining it matters
# below, where a buffer shell is added outside: the mesher requires the
# mapping to be the identity on the outer boundary of the whole domain.

# %%
sk = Skeleton([0.0, 0.4, 0.8, 1.0])


def confined(r, theta, phi):
    taper = np.clip((1.0 - r) / 0.2, 0.0, 1.0)
    return -0.05 * r * taper * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)


mapping = RadialStretch(CallableDisplacement(confined, knots=[0.8, 1.0]), rmax=1.2)
g = Geometry(sk, mapping=mapping,
             layer_names=["core", "mantle", "crust"],
             interface_names=["cmb", "moho", "surface"])
print(g)

# %% [markdown]
# A `MeshSpec` says everything else: the sizing rule (here the same target
# size at every interface, in the geometry's own lengths), the dimension
# and element order, shells to append outside the geometry, and the
# delivery. `physical` moves the nodes by the mapping; `referential`
# leaves the mesh spherical and records the mapping for the consumer.

# %%
spec = MeshSpec(g, UniformInterfaces(0.15, 0.3, 0.3), dimension=3, order=2,
                shells=[Shell(ratio=0.2, name="buffer")], delivery="referential")
print("computational domain:", [lay.name for lay in spec.layers])
print("divisor:", spec.effective_divisor)

# %% [markdown]
# ## Building
#
# `build_layered_mesh` writes an MSH 2.2 file and a JSON manifest beside
# it, and returns what it built: counts, timings, the validation report and
# the mapping in mesh units.

# %%
result = build_layered_mesh(spec, workdir / "planet")
print(result)
print("validation ok:", result.validation.ok, "| warnings:", result.validation.warnings)
print("timings (s):", {k: round(v, 2) for k, v in result.timings.items()})

# %% [markdown]
# ## The manifest
#
# The manifest is the contract with the consumer: which attribute is which
# layer or boundary, in mesh units, and what was done to the mesh. Layers
# and interfaces are numbered from the centre; a shell is a layer outside
# the geometry, marked `in_geometry: false`.

# %%
card = manifest.read(result.manifest_path)
print("schema:", card.schema)
print("geometry block:", card.geometry)
for lay in card.layers:
    print(f"  layer {lay['attribute']}: {lay['name']:8s} "
          f"[{lay['r_inner_nd']:.3f}, {lay['r_outer_nd']:.3f}]  "
          f"in geometry: {lay['in_geometry']}")
for face in card.interfaces:
    print(f"  interface {face['attribute']}: {face['name']:12s} "
          f"r = {face['mean_radius_nd']:.3f}  between {face['between_layers']}")
print("mapping:", card.mapping["kind"], "| applied to nodes:",
      card.mapping["applied_to_nodes"])
print("shell attributes:", card.shell_attributes)
print("attribute of 'moho':", card.interface_attribute("moho"))

# %% [markdown]
# ## Two dimensions, and a hollow geometry
#
# The same spec meshes a disc when `dimension=2`, which is the cheap way
# to try things. A hollow geometry, one whose skeleton starts above zero,
# meshes as a shell with an inner boundary that is an interface in its own
# right.

# %%
disc = build_layered_mesh(MeshSpec(g, UniformInterfaces(0.15, 0.3, 0.3), dimension=2,
                                   shells=[Shell(ratio=0.2)], delivery="referential"),
                          workdir / "disc")
print(disc)

hollow = Geometry(Skeleton([0.5, 0.8, 1.0]), layer_names=["lower", "upper"],
                  interface_names=["cmb", "moho", "surface"])
shell = build_layered_mesh(MeshSpec(hollow, UniformInterfaces(0.15, 0.3, 0.3)),
                           workdir / "shell")
print(shell)
print([(f["name"], f["between_layers"]) for f in
       manifest.read(shell.manifest_path).interfaces])

# %% [markdown]
# ## Export to MFEM
#
# `export_mfem_mesh` writes the mesh in MFEM's own format and, in
# referential delivery, the displacement `m(X) - X` as a GridFunction on
# the mesh's nodes, so the solver can apply the mapping itself. The
# manifest gains a `files` block saying what was written and how to read
# it back: the same `mfem::Mesh` arguments must be used for the
# GridFunction to line up.

# %%
try:
    import mfem.ser as mfem
except ImportError:
    print("PyMFEM is not installed; the tutorial stops here")
    shutil.rmtree(workdir)
    raise SystemExit(0)

ref = export_mfem_mesh(result, workdir / "planet_ref", delivery="referential")
phys = export_mfem_mesh(result, workdir / "planet_phys", delivery="physical")
print(ref)
print(phys)
print(json.dumps(ref.files, indent=1)[:600], "...")

opts = ref.files["mesh_read_options"]
mesh = mfem.Mesh(str(ref.mesh_path), opts["generate_edges"], opts["refine"],
                 opts["fix_orientation"])
print("MFEM reads", mesh.GetNE(), "elements,", mesh.GetNBE(), "boundary elements,",
      "attributes", list(mesh.attributes.ToList()))
u = mfem.GridFunction(mesh, str(ref.displacement_path))
print("displacement GridFunction has", u.Size(), "values")

# %% [markdown]
# Fields come in the next stage; this stage delivers the geometry.

# %%
total = sum(p.stat().st_size for p in workdir.iterdir()) / 1e6
print(f"{len(list(workdir.iterdir()))} files, {total:.1f} MB, removed from {workdir}")
shutil.rmtree(workdir)
