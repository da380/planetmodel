# %% [markdown]
# # A fault in a mesh for MFEM
#
# The MFEM export of a mesh with a fault, read back by MFEM as a solver
# reads it, and what a split-node code needs taken from it as a parallel
# code takes it: from the attributes and the manifest alone, with no
# geometry and no walk over neighbours, so that each rank can do it on
# the part of the mesh it holds. The fault's faces are boundary elements
# of the fault's own attribute, inside the mesh; the elements of its
# hanging wall that touch its slip, its collar, carry element attributes
# the manifest lists (`hanging_wall`); and its pinned edges, where its
# slip is zero, are found by counting how many of its faces use each
# edge. Here the thrust of example 03, under its hill, in the physical
# delivery (the nodes moved by the mapping), drawn in section from
# MFEM's own arrays: the elements by layer and by collar, and every
# boundary attribute where the section crosses it.
#
# Needs: meshing, mfem, plot
# Network: none
# Run: poetry run python examples/meshing/06_a_fault_for_mfem.py

# %%
import shutil
import tempfile
from collections import Counter
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
COLLAR = "#eb6834"

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
             optimise="Relocate3D",
             faults=[EmbeddedFault(thrust, name="thrust",
                                   sizing=InterfaceSizing.graded(0.03, 0.15))]),
    work / "built")
export = export_mfem_mesh(result, work / "thrust", delivery="physical")
card = manifest.read(export.manifest_path)

# %% [markdown]
# The mesh as MFEM reads it, with the manifest's read options, and its
# arrays: the vertices (the corners, here physical), each element's and
# boundary element's vertices and attribute. Each layer lists the element
# attributes its elements carry, its own and its collars'.

# %%
options = export.read_options
mesh = mfem.Mesh(str(export.mesh_path), options["generate_edges"], options["refine"],
                 options["fix_orientation"])
x = np.asarray(mesh.GetVertexArray())
cells = np.array([mesh.GetElementVertices(e) for e in range(mesh.GetNE())])
element = np.array([mesh.GetAttribute(e) for e in range(mesh.GetNE())])
faces = np.array([mesh.GetBdrElementVertices(b) for b in range(mesh.GetNBE())])
attribute = np.array([mesh.GetBdrAttribute(b) for b in range(mesh.GetNBE())])
names = {e["attribute"]: e["name"] for e in card.interfaces + card.faults}
for a in np.unique(attribute):
    print(f"boundary attribute {a} ({names[a]}): {np.count_nonzero(attribute == a)} "
          "elements")
for entry in card.layers:
    print(f"layer {entry['name']}: element attributes {entry['attributes']}")

# %% [markdown]
# The thrust's hanging wall is the elements whose attribute its record
# lists: an indicator each rank evaluates element by element. Its
# pinned edges come from counting, over its boundary elements, how many
# use each edge (`GetBdrElementEdges`; under MPI the counts are summed
# over the ranks sharing an edge): an edge used once is on its
# boundary, its free edge where the surface's boundary elements use it
# too and its tip, pinned, elsewhere. Its slip lives on the dofs of its
# boundary elements less those of its pinned edges, here in the mesh's
# own order-2 space. Then u = u_w + chi s jumps by the slip across the
# fault: element by element, chi s on the two sides of each interior
# face differs by exactly the slip at the face's dofs across the
# fault's faces, and not at all across any other.

# %%
record = card.fault_record("thrust")
chi = np.isin(element, record["hanging_wall"]).astype(float)
rows = np.flatnonzero(attribute == record["attribute"])


def edges_of(boundary: np.ndarray) -> Counter:
    """How many of the given boundary elements use each edge."""
    return Counter(e for b in boundary for e in mesh.GetBdrElementEdges(int(b))[0])


count = edges_of(rows)
surface = edges_of(np.flatnonzero(attribute == card.surface_attribute))
pinned_edges = [e for e, n in count.items() if n == 1 and e not in surface]
free_edges = [e for e, n in count.items() if n == 1 and e in surface]
fes = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(2, 3))
pinned = {d for e in pinned_edges for d in fes.GetEdgeDofs(e)}
carrying = sorted({d for b in rows for d in fes.GetBdrElementDofs(int(b))} - pinned)
slip = np.zeros(fes.GetNDofs())
slip[carrying] = np.random.default_rng(0).uniform(1.0, 2.0, len(carrying))
print(f"the thrust: {np.count_nonzero(chi)} elements in its hanging wall, of "
      f"attributes {record['hanging_wall']}; {len(pinned_edges)} pinned edges, "
      f"{len(free_edges)} on its free edge; {len(carrying)} slip dofs")
fault_faces = {mesh.GetBdrElementFaceIndex(int(b)): int(b) for b in rows}
across, elsewhere = 0.0, 0.0
for f in range(mesh.GetNumFaces()):
    e1, e2 = mesh.GetFaceElements(f)
    if e2 < 0:
        continue
    dofs = fes.GetFaceDofs(f)
    jump = (chi[e1] - chi[e2]) * slip[dofs]
    if f in fault_faces:
        side = manifest.fault_side(x, faces[fault_faces[f]], cells[e1])
        across = max(across, float(np.abs(jump - side * slip[dofs]).max()))
    else:
        elsewhere = max(elsewhere, float(np.abs(jump).max()))
print(f"chi s jumps by the slip across the thrust (largest error {across:g}) and "
      f"nowhere else (largest jump {elsewhere:g})")

# %% [markdown]
# A check by geometry, which the code does not need: located on the
# thrust, the centre of every element of its collar that lies over the
# fault is on its hanging-wall side.

# %%
found = thrust.locate(x[cells[chi > 0]].mean(axis=1), strict=False)
print(f"the collar's elements over the thrust on its hanging-wall side: "
      f"{bool(np.all(found.distance[found.inside] > 0))}")

# %% [markdown]
# The section normal to the thrust's strike through its centroid, from
# MFEM's arrays: the elements by layer, their layer read from their
# attribute through the manifest, the thrust's collar over them, and
# each boundary attribute the section crosses.

# %%
axes = np.stack(np.moveaxis(UNIT.basis(plane.centroid), -1, 0)[[0, 2]])
layer = np.array([card.layer_of(a)["name"] for a in element])
fill = np.array([LAYERS[n] for n in layer], dtype=object)
fill[chi > 0] = COLLAR
fig, ax = plt.subplots(figsize=(7.5, 7.6))
mesh_section(ax, plane_cut(x, cells, np.zeros(3), axes), colours=list(fill),
             edgecolor=EDGES, linewidth=0.5)
styles = {card.surface_attribute: dict(colors=INK, linewidth=1.2),
          card.interface_attribute("base"): dict(colors=INK, linewidth=1.2),
          record["attribute"]: dict(colors=FAULT, linewidth=3.0)}
for a, style in styles.items():
    mesh_section(ax, plane_cut(x, faces[attribute == a], np.zeros(3), axes), **style)
centre = plane.cartesian(0.0, 0.0) @ axes.T
ax.set(xlim=centre[0] + np.array([-0.16, 0.16]),
       ylim=centre[1] + np.array([-0.17, 0.11]), xticks=[], yticks=[])
ax.set_title("The exported mesh, as MFEM reads it")
fig.legend(handles=[Patch(facecolor=LAYERS[n], edgecolor=EDGES,
                          label=f"element attribute {card.layer_attribute(n)} ({n})")
                    for n in ("mantle", "lithosphere")]
           + [Patch(facecolor=COLLAR, edgecolor=EDGES,
                    label="element attribute "
                    f"{', '.join(map(str, record['hanging_wall']))} (the thrust's "
                    "collar)")]
           + [Line2D([], [], color=s["colors"], lw=s["linewidth"],
                     label=f"boundary attribute {a} ({names[a]})")
              for a, s in styles.items()],
           loc="lower center", ncol=2, frameon=False)
fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.2)
fig.savefig(FIGURES / "meshing_06.png", dpi=120)
shutil.rmtree(work)
