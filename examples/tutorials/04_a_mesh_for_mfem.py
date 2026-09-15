# %% [markdown]
# # 4. A mesh for MFEM
#
# The 3D mesher takes a geometry. The geometry's skeleton gives the
# concentric boundaries to mesh, and its layer and interface names become
# the names of the numbered regions in the mesh. gmsh meshes the
# skeleton, and nothing else: if the geometry has a mapping, the mapping
# is applied later, when the mesh is written for MFEM. The mesher hands
# the geometry's numbers to gmsh exactly as they are. It knows nothing
# about units and changes no lengths.
#
# The tutorial has three steps. First a skeleton alone is meshed: a
# spherical mesh, the manifest that describes it, and the export to
# MFEM. Then topography is added through a mapping, and the mesh of the
# deformed planet is made at the export. Finally a buffer shell is added
# outside the planet and the topography is tapered away across it, so
# that the outer boundary of the mesh is a sphere.
#
# This tutorial needs the `meshing` extra (gmsh) and, for the MFEM
# sections, the `mfem` extra (PyMFEM). Everything is coarse and runs in a
# few seconds. The files it writes are kept under
# `examples/figures/tutorial_04_meshes/`, where they can be opened in gmsh
# and glvis. Run it with `--temp` to write to a temporary directory that
# is removed at the end instead.

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
# A skeleton and a set of names make a geometry. With no mapping given,
# the geometry is the reference ball itself and the mesh is spherical.
#
# A `MeshSpec` is the complete description of the mesh to build. Besides
# the geometry it holds:
#
# - the **sizing**, a rule that gives every interface a target element
#   size, an element size far from the interface, and the distance over
#   which the size grows from the first to the second, all in the
#   geometry's own lengths. `UniformInterfaces` uses the same three
#   numbers at every interface. `AngularResolution` scales the size with
#   the radius of the interface, so that deep and shallow interfaces are
#   resolved to the same angle. `PerInterface` takes the sizes from a
#   dictionary keyed by interface name;
# - the **dimension**, 3 for a ball or 2 for a disc, and the polynomial
#   **order** of the elements.
#
# A spec may also add shells outside the geometry and say what the
# mapping may do to the outer boundary; both come in section 3.

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
# `build_layered_mesh` builds the mesh, writes it as a file in gmsh's
# MSH 2.2 format with a JSON manifest beside it, and returns a
# `MeshResult`: the paths, the element counts, the timings and the
# validation report.

# %%
result = build_layered_mesh(spec, workdir / "sphere")
print(result)
print("validation:", result.validation)
print("timings (s):", {k: round(v, 2) for k, v in result.timings.items()})


# %% [markdown]
# ### The manifest
#
# A mesh file labels each element and each boundary face with an
# integer, its *attribute*, and records nothing else about them. The
# manifest is the file beside the mesh that says what the attributes
# mean. It names the mesh file and says how to open it; it says whether
# the node coordinates are those of the reference body or of the
# deformed planet; it lists the layers and interfaces with their
# attributes and radii, in the geometry's own lengths; and, once fields
# have been exported, it lists where each field is and what it means.
# Layers and interfaces are numbered from the centre outwards. The
# validation report, the counts and the timings are not in the manifest,
# since a consumer of the mesh has no use for them; they stay on the
# `MeshResult`. `describe()` prints the manifest, and the JSON file reads
# the same way.

# %%
card = manifest.read(result.manifest_path)
print(card.describe())
print("attribute of 'moho':", card.interface_attribute("moho"))


# %% [markdown]
# ### Two dimensions, and a hollow geometry
#
# The same spec with `dimension=2` meshes a disc, which is a cheap way to
# try things out. A hollow geometry, one whose skeleton starts above zero,
# meshes as a shell. Its inner boundary is an interface like any other,
# with the outside on its inner side, which the manifest records as
# layer index -1.

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
# ### Export to MFEM
#
# `export_mfem_mesh` writes the mesh again in MFEM's own format, with a
# manifest of the same shape beside it. The manifest's `read_options`
# are the arguments to give MFEM's mesh constructor so that it reads the
# file as written, without renumbering anything. The export is given its
# own base name, so that the mesher's manifest stays beside the `.msh`
# file.

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
# Topography is given to a geometry as a mapping from the reference body
# to the deformed planet. The simplest mapping is a radial stretch: a
# displacement `h(r, theta, phi)` that moves every point along its own
# radius. `Geometry.stretched(h)` builds that mapping and checks it: the
# mapping must preserve orientation everywhere, and any radius at which
# `h` has a kink must be declared as a knot and must lie on a skeleton
# boundary, where the mesh has element faces. The example is the
# degree-2 shape of a rotating planet with parameter 0.05, which grows in
# proportion to the radius and does not vanish at the surface, so the
# planet is oblate all the way through. This displacement has no kinks.
#
# gmsh only ever meshes the skeleton. The `.msh` file is the mesh of the
# reference body whatever the mapping is, and the mapping travels with
# the `MeshResult`. The mapping is applied at the MFEM export, where the
# displacement of every node is a vector that MFEM can hold in the mesh's
# own nodal space. This is what lets a mapping be anything from a formula
# to an interpolated vector field. The export has two *deliveries*, two
# ways of handing the mapping to the solver. The default, `physical`,
# moves the nodes: the mesh file is the deformed planet, its manifest
# says that the node coordinates are physical, and the mapping is not
# recorded, since the mesh itself is now the shape. The other delivery,
# `referential`, keeps the mesh spherical and writes the displacement
# beside it as a field for the solver to apply. Tutorial 9 uses it, since
# that is how a model's fields travel too.

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
# The export moves the nodes and then checks the moved mesh for folding.
# The Jacobian of every element, evaluated at its quadrature points and
# at its nodes, must stay positive, and `quality` reports the smallest
# ratio of least to greatest Jacobian within one element. No
# displacement field is written, and the manifest says so. The
# displacement gives a flattening of 0.05, so the node radii run from
# 0.967 at the poles to 1.017 at the equator.

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
# A solver may need the mesh to extend beyond the planet, into a buffer
# region on whose outer boundary a far-field condition is applied. A
# `Shell` in the spec appends a layer outside the geometry, numbered
# after the geometry's layers. The computational domain is then the
# geometry followed by its shells, and the mapping must be defined, and
# orientation-preserving, on the shells as well. What the mapping may do
# to the outer boundary of the domain is the spec's choice. By default
# that boundary may carry topography like any other. With
# `outer_boundary="spherical"` the mapping must be the identity there,
# which is what a far-field condition on a sphere needs, and what this
# example asks for.
#
# So the displacement of section 2 is tapered to zero across the shell,
# between `r = 1` and `r = 1.2`. It is now a plain function of
# `(r, theta, phi)` with a kink at the surface, where the taper starts.
# `CallableDisplacement` declares that radius as a knot, and the geometry
# accepts it because the surface is a skeleton boundary. The taper is
# quadratic, so that the displacement joins the identity smoothly at
# `r = 1.2` and no knot is needed there.

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
# A spec that asks for a spherical outer boundary refuses a mapping that
# moves that boundary, before anything is meshed. The untapered
# displacement of section 2 is refused here. The same spec with
# `outer_boundary="free"` would mesh it, with an oblate outer boundary.

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
# The tapered geometry meshes, and the export moves the nodes as before.
# In the manifest the buffer is a layer outside the geometry, marked as a
# shell, with its own attribute. Its outer boundary was not named, so it
# takes the default interface name.

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
