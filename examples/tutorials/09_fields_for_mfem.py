# %% [markdown]
# # 9. Fields for MFEM
#
# Tutorial 4 wrote a mesh for MFEM and applied the geometry's mapping at
# the export. This tutorial adds a model's fields beside the mesh.
# `export_mfem` writes the mesh as before and one file per field, holding
# an MFEM `GridFunction`: the values of a finite-element function on the
# mesh. The space used is an L2 space, in which each element has its own
# values and nothing is shared across element faces, so that a jump in a
# field across an interface is carried exactly. Each value is the field
# at the reference coordinates of a degree of freedom, as Cartesian
# components in the model's units, evaluated layer by layer. The manifest
# lists each field beside the displacement, with its character, its unit
# and the layers on which it is defined, and records the model's scales
# and constants, so that every number in the files has known units.
#
# The mesh here is a coarse mesh of a small three-layer model. PREM's
# thin crust needs an Earth-scale mesh, which `scripts/mfem_cross_check.py`
# builds. Files go to `examples/figures/tutorial_09_fields/` unless
# `--temp` is passed. This tutorial needs gmsh and PyMFEM.

# %%
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from planetmodel import elastic_moduli, flattening, LayeredIsotropicElastic, mass
from planetmodel.mesh3d import (MeshSpec, UniformInterfaces, build_layered_mesh,
                                export_mfem, manifest)

parser = argparse.ArgumentParser(description="fields for MFEM")
parser.add_argument("--temp", action="store_true",
                    help="write to a temporary directory and remove it at the end")
args, _ = parser.parse_known_args(sys.argv[1:] if __name__ == "__main__" else [])
if args.temp:
    workdir = Path(tempfile.mkdtemp(prefix="planetmodel_"))
else:
    workdir = Path(__file__).resolve().parent.parent / "figures" / "tutorial_09_fields"
    workdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## A small model on a deformed geometry
#
# The model has three layers with constant properties and Earth-like
# contrasts, in a ball of unit radius, made oblate by the degree-2
# displacement of tutorial 4. The mantle's elastic moduli are computed
# from its velocities and attached as a rank-4 field under the
# vocabulary name `elastic_moduli`. The core layers hold no such field,
# so the exported values are zero there and the manifest says on which
# layers the field is defined.

# %%
model = LayeredIsotropicElastic(
    [0.0, 0.19, 0.55, 1.0], rho=[13.0, 11.0, 4.5], vp=[11.0, 9.0, 11.0],
    vs=[3.5, 0.0, 6.0], layer_names=["inner_core", "outer_core", "mantle"],
    interface_names=["icb", "cmb", "surface"])
moduli = elastic_moduli(model.layer("mantle"))
model = model.with_field("mantle", "elastic_moduli", moduli)
model = model.stretched(flattening(0.05, rmax=1.0))
print(model)
print("mass:", mass(model))

# %% [markdown]
# ## Mesh, then export with fields
#
# The mesh is built on the model's geometry as in tutorial 4, without a
# buffer this time. The export uses the `referential` delivery: the mesh
# stays spherical and the displacement `m(X) - X` is written beside it as
# a vector field on the nodes, for the solver to apply itself. This is
# the natural delivery once fields travel with the mesh, because the
# field values are referential in any case: each is the model's own
# field at the reference point, whatever shape the mesh is drawn in.
#
# `export_mfem` takes the mesh result and the model, which must sit on the
# skeleton the mesh was built from. `fields=None` would write every field
# the model holds; here three names are chosen. `rho` and `vs` are
# scalars, and `elastic_moduli` is a rank-4 tensor stored in Voigt form
# as a six-by-six matrix, so its GridFunction has 36 components per
# degree of freedom.

# %%
spec = MeshSpec(model.geometry, UniformInterfaces(0.15, 0.3, 0.3), dimension=3,
                order=2)
result = build_layered_mesh(spec, workdir / "planet")
print(result)
exported = export_mfem(result, workdir / "planet_ref", model=model,
                       fields=["rho", "vs", "elastic_moduli"],
                       delivery="referential")
print(exported)
for name, path in exported.field_paths.items():
    print(f"  {name:16s} -> {path.name}")

# %% [markdown]
# ## The manifest's fields
#
# The manifest now has a record per field: the file, the finite-element
# space to read it into and the number of components, the character, the
# unit, and the attributes of the layers on which the field is defined.
# The displacement is listed as a field like the others. The scales and
# constants of the model follow.

# %%
card = manifest.read(exported.manifest_path)
print(card.describe())

# %% [markdown]
# ## Reading a field back
#
# A consumer opens the mesh with the manifest's read options and then
# reads each `.gf` file against it. Here the density is read back and
# integrated over the reference mesh. The result is the model's mass, up
# to the error of the curved mesh's approximation of the sphere. Mass is
# preserved by the mapping, so the referential density integrated over
# the reference volume is the physical mass whatever the shape of the
# planet.

# %%
try:
    import mfem.ser as mfem
except ImportError:
    print("PyMFEM is not installed; stopping here")
    raise SystemExit(0)
opts = card.mesh["read_options"]
mesh = mfem.Mesh(str(exported.mesh_path), opts["generate_edges"], opts["refine"],
                 opts["fix_orientation"])
rho_gf = mfem.GridFunction(mesh, str(exported.field_paths["rho"]))
one = mfem.LinearForm(rho_gf.FESpace())
unity = mfem.ConstantCoefficient(1.0)      # kept alive while the form is
one.AddDomainIntegrator(mfem.DomainLFIntegrator(unity))
one.Assemble()
integrated = one * rho_gf
volume = sum(mesh.GetElementVolume(e) for e in range(mesh.GetNE()))
print(f"mesh volume of the model: {volume:.4f}  vs  4/3 pi = {4 * np.pi / 3:.4f}")
print(f"integrated density:       {integrated:.4f}  vs  mass {mass(model):.4f}")

if args.temp:
    shutil.rmtree(workdir, ignore_errors=True)
    print("temporary files removed")
else:
    print("files kept under", workdir)
