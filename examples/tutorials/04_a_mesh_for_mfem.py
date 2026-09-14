# %% [markdown]
# # 4. A mesh for MFEM
#
# A geometry is what the 3D mesher takes: its skeleton gives the
# concentric boundaries and its names become the attribute names; gmsh
# meshes the skeleton, and the geometry's mapping, if it has one, is
# applied in MFEM. The mesher hands the geometry's numbers to gmsh exactly
# as they are; it knows nothing about units and changes no lengths.
#
# The tutorial goes in three steps. First a skeleton alone is meshed: a
# spherical mesh, its manifest, and its delivery to MFEM. Then topography
# is added through a mapping and the physical mesh is made in MFEM.
# Finally a buffer shell is added outside the planet and the topography is
# tapered away across it, so that the outer boundary of the mesh is a
# sphere.
#
# This tutorial needs the `meshing` extra (gmsh) and, for the MFEM
# sections, the `mfem` extra (PyMFEM). Everything is coarse and runs in a
# few seconds. The files it writes are kept under
# `examples/figures/tutorial_04_meshes/` for inspection with gmsh and
# glvis; run it with `--temp` to use a temporary directory that is removed
# at the end instead.

# %%
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from planetmodel import CallableDisplacement, Geometry, Skeleton, flattening

try:
    from planetmodel.mesh3d import (
        MeshSpec,
        Shell,
        UniformInterfaces,
        build_layered_mesh,
        export_mfem_mesh,
        manifest,
    )
except ImportError as err:
    raise SystemExit(f"this tutorial needs gmsh: {err}")

try:
    import mfem.ser as mfem
except ImportError:
    mfem = None

parser = argparse.ArgumentParser(description="a mesh for MFEM")
parser.add_argument(
    "--temp",
    action="store_true",
    help="write to a temporary directory and remove it at the end",
)
args, _ = parser.parse_known_args(sys.argv[1:] if __name__ == "__main__" else [])

if args.temp:
    workdir = Path(tempfile.mkdtemp(prefix="planetmodel_"))
else:
    workdir = Path(__file__).resolve().parent.parent / "figures" / "tutorial_04_meshes"
    workdir.mkdir(parents=True, exist_ok=True)


def read_with_mfem(export):
    """Open an MFEM export the way its manifest says, and report it."""
    opts = export.read_options
    mesh = mfem.Mesh(
        str(export.mesh_path),
        opts["generate_edges"],
        opts["refine"],
        opts["fix_orientation"],
    )
    x = np.array(mesh.GetNodes().GetDataArray()).reshape(-1, mesh.SpaceDimension())
    r = np.linalg.norm(x, axis=1)
    print(
        f"MFEM reads {export.mesh_path.name}: {mesh.GetNE()} elements, "
        f"{mesh.GetNBE()} boundary elements, attributes "
        f"{list(mesh.attributes.ToList())}, node radii up to {r.max():.4f}"
    )
    return mesh


# %% [markdown]
# ## 1. A mesh of a skeleton
#
# A skeleton and names make a geometry; with no mapping given, the
# geometry is the reference ball itself and the mesh is spherical. A
# `MeshSpec` is the complete description of the mesh to build. Beyond the
# geometry it says:
#
# - the **sizing**: a rule giving every interface a target element size, a
#   size far from it and the distance over which one relaxes to the other,
#   all in the geometry's own lengths. `UniformInterfaces` uses the same
#   three numbers everywhere; `AngularResolution` scales the size with the
#   interface radius; `PerInterface` takes a dictionary by name;
# - the **dimension** (3 for balls, 2 for discs) and the element **order**.
#
# Shells, the outer boundary and the delivery come later in the tutorial.

# %%
sk = Skeleton([0.0, 0.4, 0.8, 1.0])
sphere = Geometry(
    sk,
    layer_names=["core", "mantle", "crust"],
    interface_names=["cmb", "moho", "surface"],
)
print(sphere)

sizing = UniformInterfaces(0.15, 0.3, 0.3)
spec = MeshSpec(sphere, sizing, dimension=3, order=2)

# %% [markdown]
# `build_layered_mesh` writes an MSH 2.2 file and a JSON manifest beside
# it, and returns what it built: counts, timings and the validation report.

# %%
result = build_layered_mesh(spec, workdir / "sphere")
print(result)
print("validation:", result.validation)
print("timings (s):", {k: round(v, 2) for k, v in result.timings.items()})

# %% [markdown]
# ### The manifest
#
# The mesh file carries numbered attributes and nothing else. The manifest
# beside it is the contract with the consumer: the mesh file and how to
# open it, whether its nodes are reference or physical coordinates, which
# attribute is which layer or boundary in the geometry's numbers, and,
# once fields are exported, where each field is and what it means. Layers
# and interfaces are numbered from the centre. The validation report and
# the build's counts and timings stay on the `MeshResult`; the manifest
# holds only what a consumer needs. `describe()` prints it; the JSON file
# itself reads the same way.

# %%
card = manifest.read(result.manifest_path)
print(card.describe())
print("attribute of 'moho':", card.interface_attribute("moho"))

# %% [markdown]
# ### Two dimensions, and a hollow geometry
#
# The same spec meshes a disc when `dimension=2`, which is the cheap way
# to try things. A hollow geometry, one whose skeleton starts above zero,
# meshes as a shell with an inner boundary that is an interface in its own
# right.

# %%
disc = build_layered_mesh(MeshSpec(sphere, sizing, dimension=2), workdir / "disc")
print(disc)

hollow = Geometry(
    Skeleton([0.5, 0.8, 1.0]),
    layer_names=["lower", "upper"],
    interface_names=["cmb", "moho", "surface"],
)
shell = build_layered_mesh(MeshSpec(hollow, sizing), workdir / "shell")
print(shell)
print(
    [
        (f["name"], f["between_layers"])
        for f in manifest.read(shell.manifest_path).interfaces
    ]
)

# %% [markdown]
# ### Delivery to MFEM
#
# `export_mfem_mesh` writes the mesh in MFEM's own format with a manifest
# of the same shape, whose `mesh.read_options` are the `mfem::Mesh`
# arguments the file was written under. A separate basename keeps the
# mesher's own manifest beside the `.msh`.

# %%
if mfem is None:
    print("PyMFEM is not installed; the MFEM sections are skipped")
else:
    sphere_mfem = export_mfem_mesh(result, workdir / "sphere_mfem")
    print(sphere_mfem)
    print(manifest.read(sphere_mfem.manifest_path).describe())
    read_with_mfem(sphere_mfem)

# %% [markdown]
# ## 2. Topography through a mapping
#
# Topography enters as a mapping of the geometry: a displacement `h` of
# every radius, here a degree-two flattening that grows linearly with
# radius and does not vanish at the surface, so the planet is oblate
# through and through. `Geometry.stretched(h)` builds the radial stretch
# and checks it: orientation preserved everywhere, and any kink in `h`
# declared as a knot lying on a skeleton boundary, where the mesh will put
# element edges. A flattening has no kinks.
#
# gmsh only ever meshes the skeleton: the `.msh` is the reference mesh
# whatever the mapping, and the mapping travels on the `MeshResult`. It is
# applied in MFEM, at export, where a displacement is a vector in the
# mesh's own nodal space; that is what lets a mapping be anything from a
# formula to a general vector field. The **delivery** says what the export
# does with it. The default, `physical`, moves the nodes: the mesh file is
# the deformed planet, its manifest says its nodes are physical
# coordinates, and the mapping is forgotten, since the mesh now is the
# shape. The other delivery keeps the mesh spherical and hands the
# displacement to the solver as a field beside it; that is tutorial 9's,
# where a model's fields travel the same way.

# %%
oblate = sphere.stretched(flattening(0.05, rmax=1.0))
print(oblate)
print("kinks at:", oblate.knots(), "| validity:", oblate.validity())

oblate_result = build_layered_mesh(
    MeshSpec(oblate, sizing, dimension=3, order=2), workdir / "oblate"
)
print(oblate_result)
print(manifest.read(oblate_result.manifest_path).mesh)

# %% [markdown]
# The export moves the nodes and checks the moved mesh for folding: the
# Jacobian of every element at its quadrature and nodal points must stay
# positive, and `quality` reports the worst ratio of least to greatest
# within an element. No displacement is written, and the manifest says
# so. The node radii run from 0.967 at the poles to 1.017 at the equator
# (for a flattening f the polar radius is `1 - 2f/3` and the equatorial
# `1 + f/3`).

# %%
if mfem is not None:
    oblate_mfem = export_mfem_mesh(oblate_result, workdir / "oblate_mfem")
    print(oblate_mfem)
    print("quality:", oblate_mfem.quality)
    print(manifest.read(oblate_mfem.manifest_path).mesh)
    read_with_mfem(oblate_mfem)

# %% [markdown]
# ## 3. A buffer shell, and the topography tapered across it
#
# A solver may want the mesh to extend beyond the planet, a buffer in
# which a far-field condition is applied on a sphere. A `Shell` appends a
# layer outside the geometry, numbered after its layers; the computational
# domain is the geometry followed by its shells. The mapping must then be
# defined and orientation-preserving on the shells too. What it does to
# the outer boundary of the domain is the spec's choice: by default that
# boundary may carry topography like any other, and
# `outer_boundary="spherical"` requires the mapping to be the identity
# there, which is what the buffer is for here.
#
# So the flattening is tapered to zero across the shell, between `r = 1`
# and `r = 1.2`. The displacement is now a plain function of
# `(r, theta, phi)` with a kink where the taper starts, at the surface;
# a `CallableDisplacement` declares that radius as a knot, and the
# geometry accepts it because it lies on a skeleton boundary. The taper is
# quadratic so that it joins the identity smoothly at `r = 1.2` and
# declares no knot there.

# %%


def tapered(r, theta, phi):
    taper = np.clip((1.2 - r) / 0.2, 0.0, 1.0) ** 2
    return -0.05 * np.minimum(r, 1.0) * taper * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)


buffered = sphere.stretched(CallableDisplacement(tapered, knots=[1.0], name="tapered"))
print(buffered)
print("kinks at:", buffered.knots(), "| validity:", buffered.validity())

buffered_spec = MeshSpec(
    buffered,
    sizing,
    dimension=3,
    order=2,
    shells=[Shell(ratio=0.2, name="buffer")],
    outer_boundary="spherical",
)
print("computational domain:", [lay.name for lay in buffered_spec.layers])
print("outer radius:", buffered_spec.outer_radius)

# %% [markdown]
# A mapping that still moves the buffer's outer edge is refused by this
# spec before anything is meshed: the untapered flattening of section 2,
# say. The same spec with `outer_boundary="free"` would mesh it with an
# oblate outer boundary.

# %%
try:
    build_layered_mesh(
        MeshSpec(
            oblate,
            sizing,
            shells=[Shell(ratio=0.2, name="buffer")],
            outer_boundary="spherical",
        ),
        workdir / "refused",
    )
except ValueError as e:
    print("refused:", str(e).split(";")[0])

# %% [markdown]
# The tapered geometry meshes, and its export moves the nodes as before.
# In the manifest the buffer is a layer outside the geometry, marked
# `in_geometry: false`, with its own attribute; its outer boundary takes
# the default interface name.

# %%
buffered_result = build_layered_mesh(buffered_spec, workdir / "buffered")
print(buffered_result)
print("validation:", buffered_result.validation)
card = manifest.read(buffered_result.manifest_path)
print(card.describe())
print("shell attributes:", card.shell_attributes)

if mfem is not None:
    buffered_mfem = export_mfem_mesh(buffered_result, workdir / "buffered_mfem")
    print("quality:", buffered_mfem.quality)
    read_with_mfem(buffered_mfem)

# %% [markdown]
# ## Looking at a mesh
#
# The `.msh` files open in gmsh's own viewer, from a shell with
# `gmsh oblate.msh`, or from Python:
#
# ```python
# import gmsh
# gmsh.initialize()
# gmsh.open(str(workdir / "oblate.msh"))
# gmsh.fltk.run()
# gmsh.finalize()
# ```
#
# The MFEM files open in glvis:
#
# ```
# glvis -m oblate_mfem.mesh
# glvis -m buffered_mfem.mesh
# ```

# %%
files = sorted(p.name for p in workdir.iterdir())
total = sum(p.stat().st_size for p in workdir.iterdir()) / 1e6
if args.temp:
    print(f"{len(files)} files, {total:.1f} MB, removed from {workdir}")
    shutil.rmtree(workdir)
else:
    print(f"{len(files)} files, {total:.1f} MB, kept in {workdir}:")
    print("  " + "\n  ".join(files))
