# %% [markdown]
# # 4. A mesh for MFEM
#
# A geometry is what the 3D mesher takes: its skeleton gives the
# concentric boundaries, its names become the attribute names, and its
# mapping either moves the nodes or travels beside the mesh for the solver
# to apply. The mesher hands the geometry's numbers to gmsh exactly as they
# are; it knows nothing about units and changes no lengths.
#
# This tutorial needs the `meshing` extra (gmsh) and, for the last
# section, the `mfem` extra (PyMFEM). Everything is coarse and runs in a
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

from planetmodel import CallableDisplacement, Geometry, Skeleton

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

# %% [markdown]
# ## The geometry
#
# A three-layer unit planet, oblate by a degree-two flattening confined to
# its outer part. The displacement `h` is written as a plain function of
# `(r, theta, phi)`: a flattening that grows linearly with radius, times a
# taper that takes it smoothly to zero at the surface over the crust. Two
# things about it matter to the mesher.
#
# First, `h` has kinks: its radial derivative jumps where the taper starts
# (the Moho, `r = 0.8`) and where it ends (the surface, `r = 1`). A
# `CallableDisplacement` declares those radii as `knots`, and a geometry
# accepts the mapping only because both lie on skeleton boundaries, which
# is where the mesh will put element edges.
#
# Second, `h` vanishes at the surface and beyond. Below, a buffer shell is
# added outside the planet, and the mesher requires the mapping to be
# defined on the shell and to be the identity on the outer boundary of the
# whole computational domain. A flattening that still moved the surface
# would be refused.

# %%
sk = Skeleton([0.0, 0.4, 0.8, 1.0])


def confined(r, theta, phi):
    taper = np.clip((1.0 - r) / 0.2, 0.0, 1.0)
    return -0.2 * r * taper * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)


g = Geometry(
    sk,
    layer_names=["core", "mantle", "crust"],
    interface_names=["cmb", "moho", "surface"],
).stretched(CallableDisplacement(confined, knots=[0.8, 1.0], name="confined"))
print(g)
print("kinks at:", g.knots(), "| validity:", g.validity())

# %% [markdown]
# ## The spec
#
# A `MeshSpec` is the complete description of the mesh to build. Beyond
# the geometry it says:
#
# - the **sizing**: a rule giving every interface a target element size, a
#   size far from it and the distance over which one relaxes to the other,
#   all in the geometry's own lengths. `UniformInterfaces` uses the same
#   three numbers everywhere; `AngularResolution` scales the size with the
#   interface radius; `PerInterface` takes a dictionary by name;
# - the **dimension** (3 for balls, 2 for discs) and the element **order**;
# - **shells** appended outside the geometry, here one buffer of 20 % of
#   the radius, numbered after the geometry's layers;
# - the **delivery**: `physical` moves the nodes by the mapping so the mesh
#   file is the deformed planet; `referential` leaves the mesh spherical
#   and records the mapping for the solver to apply.
#
# The computational domain is the geometry followed by its shells.

# %%
spec = MeshSpec(
    g,
    UniformInterfaces(0.15, 0.3, 0.3),
    dimension=3,
    order=2,
    shells=[Shell(ratio=0.2, name="buffer")],
    delivery="referential",
)
print("computational domain:", [lay.name for lay in spec.layers])
print("outer radius:", spec.outer_radius)

# %% [markdown]
# ## Building
#
# `build_layered_mesh` writes an MSH 2.2 file and a JSON manifest beside
# it, and returns what it built: counts, timings, the validation report and
# the mapping.

# %%
result = build_layered_mesh(spec, workdir / "planet")
print(result)
print("validation:", result.validation)
print("timings (s):", {k: round(v, 2) for k, v in result.timings.items()})

# %% [markdown]
# ## The manifest
#
# The manifest is the contract with the consumer: which attribute is which
# layer or boundary, in the geometry's numbers, what mapping the mesh
# carries and whether it has already been applied to the nodes, and what
# the validation found. Layers and interfaces are numbered from the centre;
# a shell is a layer outside the geometry, marked `in_geometry: false`.
# `describe()` prints it; the JSON file itself reads the same way.

# %%
card = manifest.read(result.manifest_path)
print(card.describe())
print("attribute of 'moho':", card.interface_attribute("moho"))
print("shell attributes:", card.shell_attributes)

# %% [markdown]
# ## Two dimensions, and a hollow geometry
#
# The same spec meshes a disc when `dimension=2`, which is the cheap way
# to try things. A hollow geometry, one whose skeleton starts above zero,
# meshes as a shell with an inner boundary that is an interface in its own
# right.

# %%
disc = build_layered_mesh(
    MeshSpec(
        g,
        UniformInterfaces(0.15, 0.3, 0.3),
        dimension=2,
        shells=[Shell(ratio=0.2)],
        delivery="referential",
    ),
    workdir / "disc",
)
print(disc)

hollow = Geometry(
    Skeleton([0.5, 0.8, 1.0]),
    layer_names=["lower", "upper"],
    interface_names=["cmb", "moho", "surface"],
)
shell = build_layered_mesh(
    MeshSpec(hollow, UniformInterfaces(0.15, 0.3, 0.3)), workdir / "shell"
)
print(shell)
print(
    [
        (f["name"], f["between_layers"])
        for f in manifest.read(shell.manifest_path).interfaces
    ]
)

# %% [markdown]
# ## Looking at a mesh
#
# The `.msh` files open in gmsh's own viewer, from a shell with
# `gmsh planet.msh`, or from Python:
#
# ```python
# import gmsh
# gmsh.initialize()
# gmsh.open(str(workdir / "planet.msh"))
# gmsh.fltk.run()
# gmsh.finalize()
# ```
#
# The MFEM files written below open in glvis, mesh alone or with the
# displacement as a vector field:
#
# ```
# glvis -m planet_ref.mesh
# glvis -m planet_ref.mesh -g planet_ref.displacement.gf
# ```

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
    mfem = None

if mfem is None:
    print("PyMFEM is not installed; the export section is skipped")
else:
    ref = export_mfem_mesh(result, workdir / "planet_ref", delivery="referential")
    phys = export_mfem_mesh(result, workdir / "planet_phys", delivery="physical")
    print(ref)
    print(phys)
    print(manifest.read(ref.manifest_path).describe())

    opts = ref.files["mesh_read_options"]
    mesh = mfem.Mesh(
        str(ref.mesh_path),
        opts["generate_edges"],
        opts["refine"],
        opts["fix_orientation"],
    )
    print(
        "MFEM reads",
        mesh.GetNE(),
        "elements,",
        mesh.GetNBE(),
        "boundary elements, attributes",
        list(mesh.attributes.ToList()),
    )
    u = mfem.GridFunction(mesh, str(ref.displacement_path))
    print("displacement GridFunction has", u.Size(), "values")

# %% [markdown]
# Fields come in the next stage; this stage delivers the geometry.

# %%
files = sorted(p.name for p in workdir.iterdir())
total = sum(p.stat().st_size for p in workdir.iterdir()) / 1e6
if args.temp:
    print(f"{len(files)} files, {total:.1f} MB, removed from {workdir}")
    shutil.rmtree(workdir)
else:
    print(f"{len(files)} files, {total:.1f} MB, kept in {workdir}:")
    print("  " + "\n  ".join(files))
