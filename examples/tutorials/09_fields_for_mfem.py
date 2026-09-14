# %% [markdown]
# # 9. Fields for MFEM
#
# Stage 1 handed a mesh and a displacement to MFEM; now a model's fields
# go beside them. `export_mfem` writes the mesh as before and one
# GridFunction per field: an L2 (discontinuous) space whose values at the
# reference mesh's degrees of freedom are the field's Cartesian components
# in the model's units, evaluated layer by layer, so a discontinuity across
# an interface is carried exactly. The manifest lists each field beside
# the displacement, with its character, unit and layers, and records the
# model's scales and constants so that every value has its units.
#
# A coarse mesh of a small layered model is built here; PREM's thin crust
# needs an Earth-scale mesh, which `scripts/mfem_cross_check.py` does.
# Files go to `examples/figures/tutorial_09_fields/` unless `--temp` is
# passed. This tutorial needs gmsh and PyMFEM.

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
# Three constant layers with Earth-like contrasts in a unit ball, made
# oblate by a flattening. The moduli of the mantle are attached as a
# rank-4 field under the vocabulary name `elastic_moduli`; the core layers
# have none, so their export is zero there.

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
# buffer this time, and exported in the other delivery: `referential`
# leaves the mesh spherical and hands the displacement `m(X) - X` to the
# solver as a vector field on the mesh's nodes, so that the solver applies
# the mapping itself. That is the natural delivery once fields travel too, since a
# field's values are referential, the model's own field at the reference
# point, whichever shape the mesh is drawn in. `export_mfem` takes the
# build's result and the model, which must sit on the same skeleton.
# `fields=None` writes every name the model holds; `rho` is a scalar,
# `elastic_moduli` a Voigt (6, 6) matrix stored with `vdim = 36`.

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

# %%
card = manifest.read(exported.manifest_path)
print("schema:", card.schema)
print("scales:", card.scales, "| constants:", card.constants)
for entry in card.fields:
    print(f"  {entry['name']:16s} {entry['file']:28s} rank {entry['rank']} "
          f"weight {entry['weight']} unit {entry['unit']:10s} "
          f"on attributes {entry['layers']}")

# %% [markdown]
# ## Reading a field back
#
# A consumer reads the mesh with the manifest's options and each `.gf`
# against it. Here the density is read back and integrated over the
# reference mesh, which is the model's mass up to the mesh's approximation
# of the sphere: the referential density over the reference volume is the
# physical mass, whatever the mapping.

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
