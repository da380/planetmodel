# %% [markdown]
# # Several faults and their labels
#
# Three faults in one mesh: a thrust that breaks the surface, a splay
# rising from it into its hanging wall, and a buried fault some way off.
# Each is a sheet of boundary faces of its own attribute, and the
# manifest names its curves: the buried tip, where its sides are joined;
# the free edge, where it meets the surface; the junction, where it
# meets another fault. Here the faces are drawn coloured by fault inside
# a block of the mesh, its far walls the cells' sections and its top the
# outer boundary's elements as a wireframe; and on a section across the
# thrust, the cells on its two sides, the plus (hanging-wall) one of
# each face and the minus one, found by `manifest.plus_cell`.
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/meshing/05_several_faults.py

# %%
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from planetmodel import Geometry, Skeleton
from planetmodel.fault import Fault, GeographicFrame, Plane, Slip, StructuredPatch
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec,
                                UniformInterfaces, build_layered_mesh, manifest,
                                read_mesh)
from planetmodel.plotting import clip_polygons, mesh_section, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
UNIT = GeographicFrame(1.0)
LAYERS = {"core": "#c3c2b7", "mantle": "#e1e0d9", "lithosphere": "#f4f3f0"}
COLOURS = {"thrust": "#2a78d6", "splay": "#eb6834", "buried": "#1baf7a"}
PLUS, MINUS = "#eda100", "#6250d6"
INK, GUIDE, EDGES = "#1a1a19", "#52514e", "#a3a29b"
CURVES = {"tip": dict(linestyle="--", linewidth=1.5),
          "free_edge": dict(linestyle="-", linewidth=2.5),
          "junction": dict(linestyle=":", linewidth=3.0)}

# %% [markdown]
# The thrust strikes north and dips 45 degrees east from the surface to
# 0.18 down; the splay leaves it along a line 0.13 down and rises east,
# steepening from 40 to 65 degrees, to a buried tip; the buried fault,
# south of the thrust, 0.06 to 0.18 down, strikes 60 and dips 75
# degrees. The base of the lithosphere is 0.25 down, so the junction and
# every fault's tip stand a few elements clear of it.

# %%
geometry = Geometry(Skeleton([0.0, 0.55, 0.75, 1.0]), layer_names=list(LAYERS),
                    interface_names=["cmb", "base", "surface"])
dip, width = 45.0, 0.26
thrust = Plane.at_depth(UNIT, 20.0, 30.0, 0.5 * width * np.sin(np.deg2rad(dip)),
                        strike=0.0, dip=dip, length=0.36, width=width,
                        slip=Slip.from_rake(90.0, 1.0))
east, north, up = np.moveaxis(UNIT.basis(thrust.centroid), -1, 0)
u = np.linspace(-0.5, 0.5, 5)
rising = np.linspace(1.0, 0.0, 6)
angle = np.deg2rad(40.0 + 25.0 * rising)
turn = np.deg2rad(25.0)
foot = thrust.cartesian(-u * 0.2 / thrust.length, np.full(u.size, 0.2))
controls = foot[:, None] + 0.12 * (
    ((np.sin(angle) - np.sin(np.deg2rad(40.0))) / turn)[None, :, None] * east
    + ((np.cos(np.deg2rad(40.0)) - np.cos(angle)) / turn)[None, :, None] * up)
splay = StructuredPatch(UNIT, UNIT.coordinates(controls),
                        slip=Slip.from_rake(90.0, 0.5))
buried = Plane.at_depth(UNIT, 0.0, 33.0, 0.12, strike=60.0, dip=75.0, length=0.18,
                        width=0.12, slip=Slip.from_rake(0.0, 0.3))
faults = {"thrust": Fault([thrust]), "splay": Fault([splay]),
          "buried": Fault([buried])}

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_faults_"))
near = InterfaceSizing(0.035, 0.15, 0.1)
result = build_layered_mesh(
    MeshSpec(geometry, UniformInterfaces(0.12, 0.3, 0.3), order=2,
             faults=[EmbeddedFault(f, name=n, sizing=near) for n, f in faults.items()]),
    work / "faulted")
print(result.summary())
mesh, card = read_mesh(result.msh_path), manifest.read(result.manifest_path)
for record in card.faults:
    counts = ", ".join(f"{key} {len(mesh.curves[record[key]])}"
                       for key in CURVES if record[key] is not None)
    print(f"{record['name']}: attribute {record['attribute']}, "
          f"{len(mesh.faces_with(record['attribute']))} faces; edges of its "
          f"{counts}; meets {record['meets'] or 'nothing'}")

# %% [markdown]
# An MFEM mesh keeps no curve groups; `manifest.fault_curves` reads the
# same curves from the faces alone (with the outer boundary's faces),
# and finds the MSH file's groups edge for edge.

# %%
recovered = manifest.fault_curves(
    [mesh.faces_with(r["attribute"])[:, :3] for r in card.faults],
    mesh.faces_with(card.surface_attribute)[:, :3])


def pairs(edges: np.ndarray) -> set:
    """Edges as a set of corner pairs, the smaller first."""
    return {tuple(e) for e in np.sort(edges[:, :2], axis=1).tolist()}


same = all(pairs(getattr(found, key)) == (pairs(mesh.curves[record[key]])
                                          if record[key] is not None else set())
           for record, found in zip(card.faults, recovered) for key in CURVES)
print(f"the curves read from the faces alone are the MSH file's groups: {same}")

# %% [markdown]
# The block, in the thrust's frame at its centroid (x east, y north, z
# up): -0.25 < x < 0.3, -0.42 < y < 0.22, z > -0.25. Its western,
# northern and lower walls are the cells' sections by those planes,
# coloured by layer with the interfaces' curves on them, its top the
# outer boundary's elements, drawn as a wireframe over the faults' faces
# and curves.

# %%
frame = np.stack([east, north, up])
local = (mesh.points - thrust.cartesian(0.0, 0.0)) @ frame.T
WEST, EAST, SOUTH, NORTH, FLOOR = -0.25, 0.3, -0.42, 0.22, -0.25
BOX = ([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1]],
       [-WEST, EAST, -SOUTH, NORTH, -FLOOR])
interfaces = np.isin(mesh.face_attributes, [e["attribute"] for e in card.interfaces])
walls, colours, curves = [], [], []
for origin, axes in (([WEST, 0, 0], [[0, 1, 0], [0, 0, 1]]),
                     ([0, NORTH, 0], [[1, 0, 0], [0, 0, 1]]),
                     ([0, 0, FLOOR], [[1, 0, 0], [0, 1, 0]])):
    cut = plane_cut(local, mesh.cells[:, :4], origin, axes)
    parts, index = clip_polygons(cut.in_space(), *BOX)
    walls += parts
    colours += [list(LAYERS.values())[a - 1]
                for a in mesh.cell_attributes[cut.index[index]]]
    lines = plane_cut(local, mesh.faces[interfaces, :3], origin, axes).in_space()
    curves += clip_polygons(lines, *BOX)[0]
top, _ = clip_polygons(local[mesh.faces_with(card.surface_attribute)[:, :3]], *BOX)
wire = [np.vstack([p, p[:1]]) for p in top]

fig = plt.figure(figsize=(13, 6.4))
ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
ax.add_collection3d(Poly3DCollection(walls, facecolors=colours, edgecolors=EDGES,
                                     linewidths=0.3))
ax.add_collection3d(Line3DCollection(curves, color=GUIDE, linewidth=1.0))
for record in card.faults:
    tri = local[mesh.faces_with(record["attribute"])[:, :3]]
    ax.add_collection3d(Poly3DCollection(tri, facecolor=COLOURS[record["name"]],
                                         edgecolor="#fcfcfb", linewidth=0.25))
for key, style in CURVES.items():
    segments = [local[edge[:2]] for record in card.faults if record[key] is not None
                for edge in mesh.curves[record[key]]]
    ax.add_collection3d(Line3DCollection(segments, color=INK, **style))
ax.add_collection3d(Line3DCollection(wire, color=GUIDE, linewidth=0.4, alpha=0.5))
ax.set(xlim=(WEST, EAST), ylim=(SOUTH, NORTH), zlim=(FLOOR, 0.12))
ax.set_box_aspect((EAST - WEST, NORTH - SOUTH, 0.12 - FLOOR), zoom=1.15)
ax.view_init(elev=32, azim=-35)
ax.set_axis_off()
ax.set_title("Three faults in a block of the mesh")

# %% [markdown]
# The plus and minus cells of the thrust's faces: the two cells holding
# each face, the plus one on the side its vertex order points to. Each
# plus cell's centre, located on the thrust, lies on its hanging-wall
# side (a positive distance) and each minus cell's on its footwall side.

# %%
attribute = card.fault_record("thrust")["attribute"]
faces = mesh.faces_with(attribute)[:, :3]
pairs = mesh.face_cells(faces)
plus = manifest.plus_cell(mesh.points, faces, mesh.cells[pairs][:, :, :4])
side = np.zeros(len(mesh.cells), dtype=int)
side[pairs[np.arange(len(faces)), plus]] = 1
side[pairs[np.arange(len(faces)), 1 - plus]] = -1
centres = mesh.points[mesh.cells[:, :4]].mean(axis=1)
distance = faults["thrust"].locate(centres[side != 0], strict=False).distance
print(f"the thrust: {np.count_nonzero(side == 1)} plus cells, all on the hanging-"
      f"wall side: {bool(np.all(distance[side[side != 0] == 1] > 0))}; "
      f"{np.count_nonzero(side == -1)} minus cells, all on the footwall side: "
      f"{bool(np.all(distance[side[side != 0] == -1] < 0))}")

# %% [markdown]
# The section through the centre spanned by east and up at the thrust's
# centroid, across its strike: the cells by layer, the thrust's plus and
# minus cells over them, the interfaces, and the thrust and the splay
# where the section crosses them.

# %%
sx = fig.add_subplot(1, 2, 2)
axes = np.stack([east, up])
fill = np.array(list(LAYERS.values()), dtype=object)[mesh.cell_attributes - 1]
fill[side == 1], fill[side == -1] = PLUS, MINUS
mesh_section(sx, plane_cut(mesh.points, mesh.cells[:, :4], np.zeros(3), axes),
             colours=list(fill), edgecolor=EDGES, linewidth=0.5)
mesh_section(sx, plane_cut(mesh.points, mesh.faces[interfaces, :3], np.zeros(3),
                           axes), colours=GUIDE, linewidth=1.0)
for record in card.faults[:2]:
    mesh_section(sx, plane_cut(mesh.points, mesh.faces_with(record["attribute"])[:, :3],
                               np.zeros(3), axes),
                 colours=COLOURS[record["name"]], linewidth=2.5, capstyle="round")
centre = thrust.cartesian(0.0, 0.0) @ axes.T
sx.set(xlim=centre[0] + np.array([-0.17, 0.17]),
       ylim=centre[1] + np.array([-0.2, 0.1]), xticks=[], yticks=[])
sx.set_title("Across the thrust: its plus and minus cells")
x0, y0 = centre[0] - 0.155, centre[1] - 0.185
sx.plot([x0, x0 + 0.05], [y0, y0], color=INK, lw=2.0, solid_capstyle="butt")
sx.text(x0 + 0.025, y0 + 0.006, "0.05", ha="center", va="bottom")

fig.legend(handles=[Patch(color=c, label=n) for n, c in COLOURS.items()]
           + [Line2D([], [], color=INK, label=k.replace("_", " "), **s)
              for k, s in CURVES.items()]
           + [Patch(facecolor=PLUS, label="plus cell"),
              Patch(facecolor=MINUS, label="minus cell")],
           loc="lower center", ncol=8, frameon=False)
fig.subplots_adjust(left=0.0, right=0.98, top=0.93, bottom=0.1, wspace=0.05)
fig.savefig(FIGURES / "meshing_05.png", dpi=120)
shutil.rmtree(work)
