# %% [markdown]
# # A fault in a mesh for MFEM
#
# The MFEM export of a mesh with a fault, read back by MFEM as a solver
# reads it. The fault's faces are boundary elements of the fault's own
# attribute, numbered after the interfaces', inside the mesh: each is a
# face of two elements, and MFEM keeps the order of its vertices, which
# says which of the two is on the hanging-wall side. MFEM's mesh keeps no
# curves; the manifest's rule reads them from the boundary elements. Here
# the thrust of example 03, under its hill, in the physical delivery
# (the nodes moved by the mapping), drawn in section from MFEM's own
# arrays: the elements by attribute, every boundary attribute where the
# section crosses it, and the elements on the fault's two sides.
#
# Needs: meshing, mfem, plot
# Network: none
# Run: poetry run python examples/meshing/06_a_fault_for_mfem.py

# %%
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mfem.ser as mfem
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton
from planetmodel.fault import Fault, GeographicFrame, Plane, Slip
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec,
                                UniformInterfaces, build_layered_mesh,
                                export_mfem_mesh, manifest)
from planetmodel.plotting import mesh_section, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
UNIT = GeographicFrame(1.0)
LAT, LON = 20.0, 30.0
LAYERS = {"core": "#c3c2b7", "mantle": "#e1e0d9", "lithosphere": "#f4f3f0"}
FAULT, INK, EDGES = "#2a78d6", "#52514e", "#a3a29b"
PLUS, MINUS = "#eda100", "#6250d6"

# %% [markdown]
# Example 03's ball, hill and thrust, built and exported.

# %%
def hill(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    c = (np.cos(theta) * np.cos(np.deg2rad(90.0 - LAT)) + np.sin(theta)
         * np.sin(np.deg2rad(90.0 - LAT)) * np.cos(phi - np.deg2rad(LON + 4.0)))
    return 0.06 * r ** 2 * np.exp(-(1.0 - c) / 0.006)


geometry = Geometry(Skeleton([0.0, 0.55, 0.75, 1.0]), layer_names=list(LAYERS),
                    interface_names=["cmb", "base", "surface"]).with_mapping(
    RadialStretch(CallableDisplacement(hill, name="hill"), rmax=1.0))
dip, width = 40.0, 0.24
plane = Plane.at_depth(UNIT, LAT, LON, 0.5 * width * np.sin(np.deg2rad(dip)),
                       strike=0.0, dip=dip, length=0.3, width=width,
                       slip=Slip.from_rake(90.0, 1.0))
thrust = Fault([plane])
work = Path(tempfile.mkdtemp(prefix="planetmodel_mfem_fault_"))
result = build_layered_mesh(
    MeshSpec(geometry, UniformInterfaces(0.1, 0.3, 0.3), order=2,
             faults=[EmbeddedFault(thrust, name="thrust",
                                   sizing=InterfaceSizing(0.03, 0.15, 0.1))]),
    work / "built")
export = export_mfem_mesh(result, work / "thrust", delivery="physical")
card = manifest.read(export.manifest_path)

# %% [markdown]
# The mesh as MFEM reads it, with the manifest's read options, and its
# arrays: the vertices (the corners, here physical), each element's and
# boundary element's vertices and attribute.

# %%
options = export.read_options
mesh = mfem.Mesh(str(export.mesh_path), options["generate_edges"], options["refine"],
                 options["fix_orientation"])
x = np.asarray(mesh.GetVertexArray())
cells = np.array([mesh.GetElementVertices(e) for e in range(mesh.GetNE())])
layer = np.array([mesh.GetAttribute(e) for e in range(mesh.GetNE())])
faces = np.array([mesh.GetBdrElementVertices(b) for b in range(mesh.GetNBE())])
attribute = np.array([mesh.GetBdrAttribute(b) for b in range(mesh.GetNBE())])
names = {e["attribute"]: e["name"] for e in card.interfaces + card.faults}
for a in np.unique(attribute):
    print(f"boundary attribute {a} ({names[a]}): {np.count_nonzero(attribute == a)} "
          "elements")

# %% [markdown]
# The two elements of each of the fault's boundary elements, from MFEM
# (`GetBdrElementFaceIndex`, `GetFaceElements`), and which is the plus
# one by the manifest's rule on the vertex order MFEM kept; located on
# the true fault, every plus element's centre lies on its hanging-wall
# side and every minus one's on its footwall side. Its curves, read from
# the boundary elements alone, are the free edge and the tip.

# %%
fault_rows = np.flatnonzero(attribute == card.fault_record("thrust")["attribute"])
pairs = np.array([mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(int(b)))
                  for b in fault_rows])
plus = manifest.plus_cell(x, faces[fault_rows], cells[pairs])
rows = np.arange(len(fault_rows))
side = np.zeros(len(cells), dtype=int)
side[pairs[rows, plus]], side[pairs[rows, 1 - plus]] = 1, -1
distance = thrust.locate(x[cells].mean(axis=1), strict=False).distance
print(f"{len(fault_rows)} fault faces: every plus element on the hanging-wall side "
      f"{bool(np.all(distance[side == 1] > 0))}, every minus element on the "
      f"footwall side {bool(np.all(distance[side == -1] < 0))}")
(curves,) = manifest.fault_curves([faces[fault_rows]],
                                  faces[attribute == card.surface_attribute])
print(f"its curves from the boundary elements: free edge {len(curves.free_edge)} "
      f"edges, tip {len(curves.tip)}, junction {len(curves.junction)}")

# %% [markdown]
# The section normal to the thrust's strike through its centroid, from
# MFEM's arrays: the elements by attribute, the fault's plus and minus
# elements over them, and each boundary attribute the section crosses.

# %%
axes = np.stack(np.moveaxis(UNIT.basis(plane.centroid), -1, 0)[[0, 2]])
fill = np.array(list(LAYERS.values()), dtype=object)[layer - 1]
fill[side == 1], fill[side == -1] = PLUS, MINUS
fig, ax = plt.subplots(figsize=(7.5, 7.6))
mesh_section(ax, plane_cut(x, cells, np.zeros(3), axes), colours=list(fill),
             edgecolor=EDGES, linewidth=0.5)
styles = {card.surface_attribute: dict(colors=INK, linewidth=1.2),
          card.interface_attribute("base"): dict(colors=INK, linewidth=1.2),
          card.fault_record("thrust")["attribute"]: dict(colors=FAULT, linewidth=3.0)}
for a, style in styles.items():
    mesh_section(ax, plane_cut(x, faces[attribute == a], np.zeros(3), axes), **style)
centre = plane.cartesian(0.0, 0.0) @ axes.T
ax.set(xlim=centre[0] + np.array([-0.16, 0.16]),
       ylim=centre[1] + np.array([-0.17, 0.11]), xticks=[], yticks=[])
ax.set_title("The exported mesh, as MFEM reads it")
fig.legend(handles=[Patch(facecolor=LAYERS[n], edgecolor=EDGES,
                          label=f"element attribute {card.layer_attribute(n)} ({n})")
                    for n in ("mantle", "lithosphere")]
           + [Patch(facecolor=PLUS, label="the thrust's plus elements"),
              Patch(facecolor=MINUS, label="the thrust's minus elements")]
           + [Line2D([], [], color=s["colors"], lw=s["linewidth"],
                     label=f"boundary attribute {a} ({names[a]})")
              for a, s in styles.items()],
           loc="lower center", ncol=2, frameon=False)
fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.2)
fig.savefig(FIGURES / "meshing_06.png", dpi=120)
shutil.rmtree(work)
