# %% [markdown]
# # A fault leaving the box through a side
#
# A box stands for an unbounded body, so a fault may leave it through
# its base or a side as well as through its free surface. Each edge of a
# fault where it ends is one of its curves, and the manifest records
# each kind as a curve group of its own: its buried tip, where its slip
# is pinned to zero; its free edge on the free surface, its trace, where
# its sides are free; and its face edge on the base or a side, an
# artificial boundary, where its sides are free too and nothing is
# pinned, the fault going on beyond the box. A consumer of a mesh that
# keeps no curves reads the same from the faces alone, given the box's
# faces (`manifest.fault_curves`); without them a face edge would be
# taken for a tip and pinned. Here a plane breaking the surface and
# leaving through the side x = 1: a block of the box with the fault's
# faces and its curves by kind, the fault square on with its outline,
# and the side seen square on with the face edge on it.
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/cartesian/05_a_fault_through_a_side.py

# %%
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from planetmodel import Skeleton, cartesian
from planetmodel.fault import CartesianFrame, Fault, Plane, Slip
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec,
                                UniformInterfaces, build_layered_mesh, manifest,
                                read_mesh)
from planetmodel.plotting import (FAULT_CURVE_STYLES, block_planes, clip_polygons,
                                  fault_curves, in_plane, mesh_block, mesh_section)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
LOCAL = CartesianFrame()
LAYERS = {"crust": "#f4f3f0", "mantle": "#e1e0d9"}
FAULT, COLLAR, INK, EDGES = "#2a78d6", "#b3cdee", "#1a1a19", "#a3a29b"
CURVES = ("tip", "free_edge", "face_edge")

# %% [markdown]
# A unit box of two layers, the Moho 0.4 deep. The plane strikes along
# +x and dips 50 degrees towards -y, its up-dip edge on the surface,
# 0.4 long, from x = 0.7 to 1.1: the box's side x = 1 cuts it, and the
# part beyond is trimmed.

# %%
geometry = cartesian.Geometry(Skeleton([0.0, 0.4, 1.0]),
                              cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                              layer_names=list(LAYERS),
                              interface_names=["surface", "moho"])
dip, width = 50.0, 0.25
plane = Plane.at_depth(LOCAL, 0.9, 0.0, 0.5 * width * np.sin(np.deg2rad(dip)),
                       strike=90.0, dip=dip, length=0.4, width=width,
                       slip=Slip.from_rake(0.0, 1.0))
fault = Fault([plane])

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_box_side_"))
result = build_layered_mesh(
    MeshSpec(geometry, UniformInterfaces(0.1, 0.25, 0.3), order=2,
             optimise="Relocate3D",
             faults=[EmbeddedFault(fault, name="fault",
                                   sizing=InterfaceSizing.graded(0.03, 0.15))]),
    work / "side")
mesh, card = read_mesh(result.msh_path), manifest.read(result.manifest_path)
record = card.fault_record("fault")

# %% [markdown]
# What the manifest records: the fault's boundary attribute, and one
# curve group of each kind it has, numbered 4k - 3 to 4k for the k-th
# fault (tip, free edge, junction, face edge), None where it has none;
# the box's faces after the interfaces, each with its plane.

# %%
print(f"fault {record['name']!r}: boundary attribute {record['attribute']}, tip "
      f"{record['tip']}, free edge {record['free_edge']}, junction "
      f"{record['junction']}, face edge {record['face_edge']}; collars "
      f"{record['hanging_wall']}")
for face in card.faces:
    print(f"  face {face['name']}: attribute {face['attribute']}, {face['axis']} = "
          f"{face['value']:g}")

# %% [markdown]
# The rule from the faces alone, as an MFEM code applies it: its pinned
# edges are its tip; its face edge, read only with the box's faces given,
# pins nothing, so the cells along it are in the fault's walls.

# %%
faces = mesh.faces_with(record["attribute"])[:, :3]
surface = mesh.faces_with(card.surface_attribute)[:, :3]
sides = mesh.boundary_of(card.face_attributes)
(found,) = manifest.fault_curves([faces], surface, faces=sides)
(blind,) = manifest.fault_curves([faces], surface)
for key in CURVES:
    same = sorted(map(tuple, getattr(found, key).tolist())) == sorted(
        map(tuple, np.sort(mesh.curves[record[key]][:, :2], axis=1).tolist()))
    print(f"  {key}: {len(getattr(found, key))} edges, as the MSH group holds: {same}")
print(f"pinned: {len(found.pinned)} edges, the tip's; without the box's faces "
      f"{len(blind.pinned)}, the face edge's {len(found.face_edge)} among them")

# %% [markdown]
# The block x from 0.55 to 1, y from -0.45 to 0.15, z from -0.4 to 0,
# seen from above its corner of least x and y: its walls x = 1, y = 0.15
# and z = -0.4 are the cells' sections there, by layer and collar, the
# fault's faces inside it and its curves by kind over them.

# %%
bounds = [(0.55, 1.0), (-0.45, 0.15), (-0.4, 0.0)]
fill = np.array(list(LAYERS.values()), dtype=object)[mesh.layers(card) - 1]
fill[np.isin(mesh.cell_attributes, record["hanging_wall"])] = COLLAR
interfaces = mesh.boundary_of([e["attribute"] for e in card.interfaces])
curves = {key: mesh.curves[record[key]] for key in CURVES}
fig = plt.figure(figsize=(14, 5.8), layout="constrained")
grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])
ax = fig.add_subplot(grid[0], projection="3d")
mesh_block(ax, mesh.points, mesh.cells, bounds, colours=list(fill), lines=interfaces,
           view=(28.0, -125.0), zoom=1.1, edgecolor=EDGES, linewidth=0.3)
tri, _ = clip_polygons(mesh.points[faces], *block_planes(bounds))
ax.add_collection3d(Poly3DCollection(tri, facecolor=FAULT, edgecolor="#fcfcfb",
                                     linewidth=0.3, alpha=0.9))
fault_curves(ax, mesh.points, curves)
ax.set(xlabel="x", ylabel="y", zlabel="z")
ax.set_title("A block of the box (unit lengths, z up)")

# %% [markdown]
# The fault square on, along strike (+x) to the right and up dip upward:
# its faces, and its outline, each part the kind of curve it is.

# %%
basis = plane.fault_basis(0.0, 0.0)
square = (plane.centroid, np.stack([basis[:, 0], basis[:, 1]]))
sq = fig.add_subplot(grid[1])
mesh_section(sq, in_plane(mesh.points, faces, *square), colours="#dce9f8",
             edgecolor=FAULT, linewidth=0.4)
fault_curves(sq, mesh.points, curves, plane=square)
sq.set(xlabel="along strike, +x (unit lengths)", ylabel="up dip (unit lengths)",
       title="The fault square on: its curves")

# %% [markdown]
# The side x = 1 seen square on, from outside: its boundary faces, each
# the colour of the cell behind it, and the fault's face edge on them.

# %%
side = mesh.faces_with(card.face_attribute("x_max"))[:, :3]
behind = mesh.face_cells(side)[:, 0]
outside = (np.array([1.0, 0.0, 0.0]), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
sd = fig.add_subplot(grid[2])
mesh_section(sd, in_plane(mesh.points, side, *outside), colours=list(fill[behind]),
             edgecolor=EDGES, linewidth=0.3)
fault_curves(sd, mesh.points, {"face_edge": curves["face_edge"]}, plane=outside)
sd.set(xlim=(-0.5, 0.5), ylim=(-1.0, 0.0), xlabel="y (unit lengths)",
       ylabel="z, up (unit lengths)", title="The side x = 1: the face edge")
fig.legend(handles=[Patch(facecolor=c, edgecolor=EDGES, label=n)
                    for n, c in LAYERS.items()]
           + [Patch(facecolor=COLLAR, edgecolor=EDGES, label="its collar"),
              Patch(facecolor=FAULT, label="the fault's faces")]
           + [Line2D([], [], color=INK, label=key.replace("_", " "),
                     **FAULT_CURVE_STYLES[key]) for key in CURVES],
           loc="outside lower center", ncol=7, frameon=False)
fig.savefig(FIGURES / "cartesian_05.png", dpi=120, bbox_inches="tight")
print("figure written to", FIGURES / "cartesian_05.png")
shutil.rmtree(work)
