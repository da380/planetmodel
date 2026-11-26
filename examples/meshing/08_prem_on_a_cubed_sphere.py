# %% [markdown]
# # PREM on a cubed sphere, for MFEM
#
# A cubed sphere of a real model, taken all the way to the files a
# solver reads. The model is PREM with every one of its thirteen layers
# kept. A mesh of tetrahedra has to merge the thin ones away, because
# the mesher cannot fill a shell far thinner than its elements are
# wide; hexahedra are made by construction, so the 3 km ocean is simply
# one cell thick.
#
# The radial sizes are given by a rule written here, to show that a
# rule is any function from the interfaces to their sizings: 60 km at
# the surface growing to 800 km at the centre, which is what a loading
# calculation wants. The lateral size is set by `edge` alone and goes
# the other way, narrowing with depth. The body is flattened by an
# exaggerated ellipticity so that the mapping shows in a picture, and a
# buffer shell is added outside, as a calculation of gravity uses one.
# The mapping is applied at the export, where the density is also
# written, and everything is then read back as a consumer reads it:
# the manifest first, and the mesh and the fields the way it says.
#
# Needs: mfem, plot
# Network: none
# Run: poetry run python examples/meshing/08_prem_on_a_cubed_sphere.py

# %%
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import mfem.ser as mfem
import numpy as np

from planetmodel import PREM, CallableDisplacement, RadialStretch, mass
from planetmodel.geometry import InterfaceInfo
from planetmodel.mesh3d import (CubedSphereSpec, InterfaceSizing, Shell,
                                build_cubed_sphere, export_mfem, manifest, read_mesh)
from planetmodel.plotting import mesh_section, plane_cut
from planetmodel.units import LENGTH

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
INK, EDGES, BUFFER = "#52514e", "#a3a29b", "#f4f3f0"
EDGE, BUFFER_RATIO, FLATTENING = 8, 0.3, 0.08

model = PREM().nondimensionalised()        # radii in units of the outer radius
KM = model.scales.factor(LENGTH) / 1e3     # kilometres to a unit of length
a = float(model.skeleton.boundaries[-1])
b = a * (1.0 + BUFFER_RATIO)

# %% [markdown]
# The rule and the mapping. The flattening grows with radius inside the
# body and falls back to zero across the buffer, so the outer boundary
# of the mesh stays a sphere; it has a corner at the surface, which the
# mapping declares as a knot.

# %%
def growing_with_depth(interfaces: Sequence[InterfaceInfo], _: float
                       ) -> dict[int, InterfaceSizing]:
    """60 km at the surface and above, 800 km at the centre, linear in
    radius between; each interface's size relaxes over half its radius."""
    surface, deep = 60.0 / KM, 800.0 / KM
    sizes = {}
    for face in interfaces:
        size = surface + (deep - surface) * max(0.0, 1.0 - face.radius / a)
        sizes[face.index] = InterfaceSizing(size, max(deep, size), 0.5 * face.radius)
    return sizes


def flattening(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    taper = np.clip((b - r) / (b - a), 0.0, 1.0)
    return -FLATTENING * r * taper * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)


geometry = model.geometry.with_mapping(RadialStretch(
    CallableDisplacement(flattening, knots=[a], name="flattening"), rmax=b))
spec = CubedSphereSpec(geometry, EDGE, sizing=growing_with_depth, order=2,
                       shells=(Shell(ratio=BUFFER_RATIO),))
steps = np.diff(spec.spheres[spec.spheres <= a]) * KM
print(f"{spec.cells} cells: {spec.blend.size - 1} levels across the blend and "
      f"{spec.spheres.size - 1} between {spec.spheres.size} spheres; radial sizes "
      f"in the body from {steps.min():.0f} to {steps.max():.0f} km")

# %% [markdown]
# The build, and the export in the physical delivery: the nodes moved
# by the mapping, the density in an L2 space, each layer's cells
# holding that layer's own values. The report warns of the flattest
# cell, the ocean's, which is what was asked for.

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_prem_cubed_"))
result = build_cubed_sphere(spec, work / "prem")
print(result.summary())
for warning in result.validation.warnings:
    print(f"  warning: {warning}")
export = export_mfem(result, work / "delivery", model=model, fields=["rho"])
print(export)

# %% [markdown]
# A consumer starts from the manifest: the layers with their element
# attributes, the interfaces with their boundary attributes and kinds,
# and how to open the mesh. Nothing in it says the mesh is a cubed
# sphere; the mesh file holds what its elements are.

# %%
card = manifest.read(export.manifest_path)
print(f"{len(card.layers)} layers, attribute and name: "
      + ", ".join(f"{e['attribute']} {e['name']}" for e in card.layers))
print("fluid layers: "
      + ", ".join(e["name"] for e in card.layers if e["fluid"]))
print("interfaces by kind: " + ", ".join(
    f"{kind} x{sum(e['kind'] == kind for e in card.interfaces)}"
    for kind in ("fluid-solid", "solid-solid", "free", "outer")))
options = card.mesh["read_options"]
delivered = mfem.Mesh(str(export.mesh_path), options["generate_edges"],
                      options["refine"], options["fix_orientation"])
rho = mfem.GridFunction(delivered, str(export.field_paths["rho"]))
print(f"MFEM reads {delivered.GetNE()} hexahedra of order "
      f"{delivered.GetNodes().FESpace().GetOrder(0)} and the density in "
      f"{rho.FESpace().FEColl().Name()}")

# %% [markdown]
# The density is the model's own at each point of the reference body,
# so its integral over the reference mesh is the mass, whatever the
# mapping. The reference mesh is the file the build wrote, and the
# density's numbering is the same on both.

# %%
reference = mfem.Mesh(str(result.mesh_path), options["generate_edges"],
                      options["refine"], options["fix_orientation"])
density = mfem.GridFunction(reference, str(export.field_paths["rho"]))
form = mfem.LinearForm(density.FESpace())
form.AddDomainIntegrator(mfem.DomainLFIntegrator(
    mfem.GridFunctionCoefficient(density), 2, 8))
form.Assemble()
print(f"the mass of the density on the mesh is off PREM's by "
      f"{np.sum(form.GetDataArray()) / mass(model) - 1.0:.1e}")

# %% [markdown]
# The picture. On the left the delivered mesh cut by the plane through
# the poles, each cell in the colour of its mean density, the buffer
# pale, the interfaces drawn over it: the flattening is the mapping's,
# and the outer boundary is still a circle. On the right the size of
# the cells against radius: their radial size, which follows the rule
# and drops to each thin layer's thickness, and their width, which is
# set by `edge` and is proportional to radius.

# %%
mesh = read_mesh(export.mesh_path)
values = np.array(rho.GetDataArray()).reshape(delivered.GetNE(), -1).mean(axis=1)
in_body = np.isin(mesh.cell_attributes,
                  [e["attribute"] for e in card.layers if e["in_geometry"]])
colours = plt.get_cmap("YlOrBr")(values / values.max())
colours[~in_body] = plt.matplotlib.colors.to_rgba(BUFFER)
origin, plane = np.zeros(3), [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 6.4), layout="constrained",
                                  width_ratios=(1.25, 1.0))
mesh_section(left, plane_cut(mesh.points, mesh.cells, origin, plane),
             colours=colours, edgecolor=EDGES, linewidth=0.3)
for entry in card.interfaces:
    mesh_section(left, plane_cut(mesh.points, mesh.faces_with(entry["attribute"]),
                                 origin, plane, tensor=True), colours=INK,
                 linewidth=0.9)
left.set_axis_off()
left.set_title(f"PREM on a cubed sphere, {EDGE} cells an edge, flattened by "
               f"{FLATTENING:.0%}:\ndensity by cell, and the buffer outside")
fig.colorbar(plt.cm.ScalarMappable(
    norm=plt.Normalize(0.0, values.max() * model.scales.mass
                       / model.scales.length ** 3 / 1e3), cmap="YlOrBr"),
    ax=left, shrink=0.6, label="density, g/cm$^3$")

radii = spec.spheres * KM
right.stairs(np.diff(radii), radii, orientation="horizontal", baseline=None,
             color="#2a78d6", linewidth=1.5, label="radial size of a cell")
right.plot(0.5 * np.pi * radii / EDGE, radii, color="#eb6834",
           label=f"its width, (pi / 2) r / {EDGE}")
for entry in card.interfaces:
    right.axhline(entry["radius"] * KM, color=EDGES, linewidth=0.5)
right.set(xscale="log", xlabel="size, km", ylabel="radius, km",
          ylim=(0.0, b * KM), title="the sizes of the cells above the first sphere")
right.legend(frameon=False, loc="lower left")
fig.savefig(FIGURES / "meshing_08.png", dpi=120)
shutil.rmtree(work)
