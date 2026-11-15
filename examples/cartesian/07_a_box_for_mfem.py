# %% [markdown]
# # A box with a fault, for MFEM
#
# The MFEM export of a box with a fault, read back by MFEM as a solver
# reads it, and what a split-node code needs taken from it as a
# parallel code takes it: from the attributes and the manifest alone,
# with no geometry and no walk over neighbours. The box's boundary is
# labelled: its free surface and the Moho are interfaces, its base and
# four sides are faces (`manifest.FACES`), each a boundary attribute of
# its own after the interfaces', the fault's faces one more after them;
# the elements of the fault's hanging wall that touch its slip, its
# collar, carry element attributes the manifest lists. The fault's
# curves come from counting how many of its boundary elements use each
# edge: once on its boundary, where the surface's boundary elements use
# it too its free edge, where a face's do its face edge, otherwise its
# tip, which alone is pinned. Here the thrust of example 03 under its
# range, in the physical delivery (the nodes moved by the mapping),
# drawn from MFEM's own arrays.
#
# Needs: meshing, mfem, plot
# Network: none
# Run: poetry run python examples/cartesian/07_a_box_for_mfem.py

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from planetmodel import Skeleton, cartesian
from planetmodel.fault import CartesianFrame, Fault, Plane, Slip
from planetmodel.mesh3d import (EmbeddedFault, MeshSpec, UniformInterfaces,
                                build_layered_mesh, export_mfem_mesh, manifest)
from planetmodel.plotting import BLOCK_VIEW, fault_curves, mesh_section, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
LOCAL = CartesianFrame()
LAYERS = {"crust": "#f4f3f0", "mantle": "#e1e0d9"}
FAULT, COLLAR, INK, EDGES = "#2a78d6", "#b3cdee", "#1a1a19", "#a3a29b"
FACES = {"surface": "#eda100", "base": "#52514e", "x_min": "#c3c2b7",
         "y_max": "#a3a29b"}


# %% [markdown]
# Example 03's box, range and thrust, built and exported physical.

# %%
def ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.08 * np.exp(-((x - 0.6) / 0.15) ** 2) * (1.0 + 0.25 * np.cos(np.pi * y))


box = cartesian.Geometry(Skeleton([0.0, 0.4, 1.0]),
                         cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                         layer_names=list(LAYERS), interface_names=["surface", "moho"])
geometry = box.stretched(cartesian.layer_linear(box.skeleton, [ridge, None, None]))
dip, width = 40.0, 0.3
plane = Plane.at_depth(LOCAL, 0.5 + 0.5 * width * np.cos(np.deg2rad(dip)), 0.0,
                       0.5 * width * np.sin(np.deg2rad(dip)), strike=0.0, dip=dip,
                       length=0.5, width=width, slip=Slip.from_rake(90.0, 1.0))
thrust = Fault([plane])
work = Path(tempfile.mkdtemp(prefix="planetmodel_box_mfem_"))
result = build_layered_mesh(
    MeshSpec(geometry, UniformInterfaces(0.1, 0.25, 0.3), order=2,
             faults=[EmbeddedFault(thrust, name="thrust", sizing=0.03)]),
    work / "built")
export = export_mfem_mesh(result, work / "thrust", delivery="physical")
card = manifest.read(export.manifest_path)

# %% [markdown]
# The mesh as MFEM reads it, with the manifest's read options, and its
# arrays: the vertices (the corners, physical), each element's and
# boundary element's vertices and attribute. The manifest names every
# boundary attribute: the interfaces, the box's faces with their
# planes, the fault.

# %%
options = export.read_options
mesh = mfem.Mesh(str(export.mesh_path), options["generate_edges"], options["refine"],
                 options["fix_orientation"])
x = np.asarray(mesh.GetVertexArray())
cells = np.array([mesh.GetElementVertices(e) for e in range(mesh.GetNE())])
element = np.array([mesh.GetAttribute(e) for e in range(mesh.GetNE())])
faces = np.array([mesh.GetBdrElementVertices(b) for b in range(mesh.GetNBE())])
attribute = np.array([mesh.GetBdrAttribute(b) for b in range(mesh.GetNBE())])
names = {e["attribute"]: e["name"] for e in card.interfaces + card.faces + card.faults}
for a in np.unique(attribute):
    print(f"boundary attribute {a} ({names[a]}): {np.count_nonzero(attribute == a)} "
          "elements")
for entry in card.layers:
    print(f"layer {entry['name']}: element attributes {entry['attributes']}")

# %% [markdown]
# The fault's hanging wall is the elements whose attribute its record
# lists; its two sides at each face are MFEM's face elements of the face
# the boundary element is, the plus one on the side its vertex order
# points to (`manifest.fault_side`). Its curves by counting edges over
# its boundary elements, the surface's and the box's faces'.

# %%
record = card.fault_record("thrust")
chi = np.isin(element, record["hanging_wall"])
rows = np.flatnonzero(attribute == record["attribute"])


def edges_of(boundary: np.ndarray) -> Counter:
    """How many of the given boundary elements use each edge."""
    return Counter(e for b in boundary for e in mesh.GetBdrElementEdges(int(b))[0])


count = edges_of(rows)
surface = edges_of(np.flatnonzero(attribute == card.surface_attribute))
sides = edges_of(np.flatnonzero(np.isin(attribute, card.face_attributes)))
once = [e for e, n in count.items() if n == 1]
free = [e for e in once if e in surface]
cut = [e for e in once if e in sides and e not in surface]
pinned = [e for e in once if e not in surface and e not in sides]
plus = []
for b in rows:
    e1, e2 = mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(int(b)))
    side = manifest.fault_side(x, faces[b], cells[e1])
    plus.append(e1 if side > 0 else e2)
print(f"the thrust: {rows.size} boundary elements, {np.count_nonzero(chi)} elements in "
      f"its collar ({record['hanging_wall']}); every plus element in it: "
      f"{bool(np.all(chi[plus]))}; {len(pinned)} pinned edges (its tip), {len(free)} "
      f"on its free edge, {len(cut)} on a face")


def vertex_edges(edge_ids: list[int]) -> np.ndarray:
    """MFEM's edges as pairs of vertex indices."""
    return np.array([mesh.GetEdgeVertices(e) for e in edge_ids],
                    dtype=np.int64).reshape(-1, 2)


# %% [markdown]
# The box's boundary from MFEM's arrays, from above the corner of least
# y and greatest x: its far faces (the base, x = 0, y = 1/2) by their
# attributes, the free surface's elements as a wireframe over the
# fault's boundary elements and its curves.

# %%
fig = plt.figure(figsize=(14, 6.0), layout="constrained")
grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
ax = fig.add_subplot(grid[0], projection="3d", computed_zorder=False)
for name in ("base", "x_min", "y_max"):
    tri = x[faces[attribute == card.face_attribute(name)]]
    ax.add_collection3d(Poly3DCollection(tri, facecolor=FACES[name],
                                         edgecolor="#fcfcfb", linewidth=0.3))
ax.add_collection3d(Poly3DCollection(x[faces[rows]], facecolor=FAULT,
                                     edgecolor="#fcfcfb", linewidth=0.3))
fault_curves(ax, x, {"tip": vertex_edges(pinned), "free_edge": vertex_edges(free)})
top = x[faces[attribute == card.surface_attribute]]
ax.add_collection3d(Poly3DCollection(top, facecolor="none",
                                     edgecolor=FACES["surface"], linewidth=0.4))
ax.set(xlim=(0, 1), ylim=(-0.5, 0.5), zlim=(-1.0, 0.1), xlabel="x", ylabel="y",
       zlabel="z")
ax.set_box_aspect((1.0, 1.0, 1.1), zoom=1.05)
ax.view_init(*BLOCK_VIEW)
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor("none")
ax.set_title("MFEM's boundary elements by attribute (unit lengths, z up)")

# %% [markdown]
# The section y = 0 from MFEM's arrays: the elements by layer, read from
# their attribute through the manifest, the thrust's collar over them,
# and each boundary attribute the section crosses.

# %%
sx = fig.add_subplot(grid[1])
origin, axes = np.zeros(3), np.array([[1.0, 0, 0], [0, 0, 1.0]])
fill = np.array([LAYERS[card.layer_of(a)["name"]] for a in element], dtype=object)
fill[chi] = COLLAR
mesh_section(sx, plane_cut(x, cells, origin, axes), colours=list(fill),
             edgecolor=EDGES, linewidth=0.5)
styles = {card.surface_attribute: dict(colors=FACES["surface"], linewidth=2.0),
          card.interface_attribute("moho"): dict(colors=INK, linewidth=1.2),
          record["attribute"]: dict(colors=FAULT, linewidth=3.0)}
for a, style in styles.items():
    mesh_section(sx, plane_cut(x, faces[attribute == a], origin, axes), **style)
sx.set(xlim=(0.3, 0.85), ylim=(-0.45, 0.12), xlabel="x (unit lengths)",
       ylabel="z, up (unit lengths)", title="The section y = 0, as MFEM reads it")
sx.legend(handles=[Patch(facecolor=LAYERS[n], edgecolor=EDGES,
                         label=f"element attribute {card.layer_attribute(n)} ({n})")
                   for n in LAYERS]
          + [Patch(facecolor=COLLAR, edgecolor=EDGES,
                   label=f"element attribute {record['hanging_wall'][0]} (the "
                   "thrust's collar)")]
          + [Line2D([], [], color=s["colors"], lw=s["linewidth"],
                    label=f"boundary attribute {a} ({names[a]})")
             for a, s in styles.items()]
          + [Patch(facecolor=FACES[n], label=f"boundary attribute "
                   f"{card.face_attribute(n)} ({n})") for n in FACES if n != "surface"],
          loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
fig.savefig(FIGURES / "cartesian_07.png", dpi=120, bbox_inches="tight")
print("figure written to", FIGURES / "cartesian_07.png")
shutil.rmtree(work)
