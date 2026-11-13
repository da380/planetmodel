# %% [markdown]
# # A curved megathrust
#
# A subduction interface as a structured patch swept from its trench
# (`StructuredPatch.swept`): the trace an arc of a small circle, convex
# away from its pole, the interface descending towards the pole at a dip
# that steepens with depth, from 15 degrees at the trench, through the
# base of the lithosphere, to its buried tip. Built into a layered ball,
# refined towards it, it is seen in a cutaway of the cap of the ball
# over it, half of the cap removed, and on a map of the outer boundary's
# elements, where its free edge is the trench.
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/meshing/04_a_curved_megathrust.py

# %%
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from planetmodel import Geometry, Skeleton
from planetmodel.fault import Fault, GeographicFrame, Slip, SmallCircle, StructuredPatch
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec,
                                UniformInterfaces, build_layered_mesh, manifest,
                                read_mesh)
from planetmodel.plotting import clip_polygons, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
UNIT = GeographicFrame(1.0)
LAYERS = {"core": "#c3c2b7", "mantle": "#e1e0d9", "lithosphere": "#f4f3f0"}
FAULT, INK, EDGES, TRUE = "#2a78d6", "#52514e", "#a3a29b", "#0b0b0b"

# %% [markdown]
# The trench: 60 degrees of the small circle 30 degrees from a pole at
# (15N, 40E), west of the pole, travelled northwards so that the dip side
# faces the pole. The dip is 15 + 250 z degrees at depth z (65 at the
# base of the lithosphere, 0.2 down), down to 0.28.

# %%
pole = (15.0, 40.0)
trace = SmallCircle(pole, 30.0, 240.0, 300.0)
slab = StructuredPatch.swept(UNIT, trace, dip=lambda z: 15.0 + 250.0 * z,
                             bottom=0.28, n_u=9, n_v=9,
                             slip=Slip.from_rake(90.0, 1.0))
geometry = Geometry(Skeleton([0.0, 0.55, 0.8, 1.0]), layer_names=list(LAYERS),
                    interface_names=["cmb", "base", "surface"])

# %% [markdown]
# The mesh, of order 2, refined to 0.03 on the interface. The trench is
# on the reference sphere, so the mesher carries the interface's upper
# edge across the outer boundary and trims it there: its free edge.

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_megathrust_"))
result = build_layered_mesh(
    MeshSpec(geometry, UniformInterfaces(0.12, 0.3, 0.3), order=2,
             faults=[EmbeddedFault(Fault([slab]), name="megathrust",
                                   sizing=InterfaceSizing(0.03, 0.12, 0.15))]),
    work / "megathrust")
print(result.summary())
mesh, card = read_mesh(result.msh_path), manifest.read(result.manifest_path)
record = card.fault_record("megathrust")
faces = mesh.faces_with(record["attribute"])[:, :3]
surface = mesh.faces_with(card.surface_attribute)[:, :3]
print(f"the megathrust: {len(faces)} faces in layers {record['layers']}; its "
      f"trench {len(mesh.curves[record['free_edge']])} edges, its tip "
      f"{len(mesh.curves[record['tip']])}")

# %% [markdown]
# The cutaway, a block of the body in a frame at the middle of the
# trench: x towards the pole, z up, y normal to the plane they span
# through the centre, which cuts the interface down its dip. The block
# is -0.2 < x < 0.45, 0 < y < 0.35 and z > 0.6 below the outer
# boundary: its elements on the outer boundary, the cells' sections by
# its front (y = 0) and eastern (x = 0.45) faces coloured by layer with
# the interfaces' curves on them, and in front of it the interface's
# faces on the removed side, y < 0, standing out of the front face, the
# trench along their top. The corners of the order-2 elements are
# drawn, their sides straight, each polygon clipped to the block.

# %%
middle, _ = trace.surface(UNIT, np.array([0.5]))
ez = middle[0]
p = UNIT.cartesian([pole[0], pole[1], 1.0])
ex = (p - (p @ ez) * ez) / np.linalg.norm(p - (p @ ez) * ez)
local = mesh.points @ np.stack([ex, np.cross(ez, ex), ez]).T
WEST, EAST, BACK, BOTTOM = -0.2, 0.45, 0.35, 0.6
BOX = ([[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]],
       [-WEST, EAST, BACK, 0.0, -BOTTOM])
interfaces = np.isin(mesh.face_attributes, [e["attribute"] for e in card.interfaces])


def face(origin: list, axes: list) -> tuple[list, list, list]:
    """A face of the block: the cells' sections by its plane with their
    layers' colours, and the interfaces' curves on it."""
    cut = plane_cut(local, mesh.cells[:, :4], origin, axes)
    polygons, index = clip_polygons(cut.in_space(), *BOX)
    colours = np.array(list(LAYERS.values()))[mesh.cell_attributes[cut.index] - 1]
    curves = plane_cut(local, mesh.faces[interfaces, :3], origin, axes)
    return polygons, list(colours[index]), clip_polygons(curves.in_space(), *BOX)[0]


front, front_colours, front_curves = face([0, 0, 0], [[1, 0, 0], [0, 0, 1]])
east, east_colours, east_curves = face([EAST, 0, 0], [[0, 1, 0], [0, 0, 1]])
top, _ = clip_polygons(local[surface], *BOX)
standing, _ = clip_polygons(local[faces], [0, 1, 0], 0.0)
dip_line = plane_cut(local, faces, [0, 0, 0], [[1, 0, 0], [0, 0, 1]]).in_space()
trench = local[mesh.curves[record["free_edge"]][:, :2]]

fig = plt.figure(figsize=(13, 6.2))
ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
ax.add_collection3d(Poly3DCollection(
    top + front + east,
    facecolors=[LAYERS["lithosphere"]] * len(top) + front_colours + east_colours,
    edgecolors=EDGES, linewidths=0.4))
ax.add_collection3d(Line3DCollection(front_curves + east_curves, color=INK,
                                     linewidth=1.0))
ax.add_collection3d(Line3DCollection(dip_line, color=FAULT, linewidth=2.5))
ax.add_collection3d(Poly3DCollection(standing, facecolors=FAULT,
                                     edgecolors="#fcfcfb", linewidths=0.4))
ax.add_collection3d(Line3DCollection(trench, color=FAULT, linewidth=2.5))
ax.set(xlim=(WEST, EAST), ylim=(-0.3, BACK), zlim=(BOTTOM, 1.0))
ax.set_box_aspect((EAST - WEST, BACK + 0.3, 1.0 - BOTTOM), zoom=1.2)
ax.view_init(elev=30, azim=-60)
ax.set_axis_off()
ax.set_title("A block over the megathrust, cut down its dip")

# %% [markdown]
# The map: the outer boundary's elements around the trench by latitude
# and longitude, the interface's free edge on them, and the trench as
# given; the buried tip, beneath, marks its extent down dip.

# %%
lat, lon = np.moveaxis(UNIT.coordinates(mesh.points)[:, :2], -1, 0)
mapped = np.stack([lon, lat], axis=-1)
near = np.all(((lon > 0.0) & (lon < 45.0) & (lat > -10.0) & (lat < 38.0))[surface],
              axis=1)
mp = fig.add_subplot(1, 2, 2)
mp.add_collection(PolyCollection(mapped[surface[near]], edgecolors=EDGES,
                                 facecolors=LAYERS["lithosphere"], linewidths=0.4))
for group, style in ((record["free_edge"], dict(lw=3.0)),
                     (record["tip"], dict(lw=1.5, ls=(0, (4, 2))))):
    segments = mapped[mesh.curves[group][:, :2]]
    mp.add_collection(PolyCollection(segments, closed=False, facecolors="none",
                                     edgecolors=FAULT, linewidths=style["lw"],
                                     linestyles=style.get("ls", "-")))
given = UNIT.coordinates(trace.surface(UNIT, np.linspace(0.0, 1.0, 200))[0])
mp.plot(given[:, 1], given[:, 0], color=TRUE, lw=1.2, ls=(0, (1, 2)))
mp.set(xlim=(2.0, 42.0), ylim=(-8.0, 36.0), xlabel="longitude (degrees)",
       ylabel="latitude (degrees)")
mp.set_aspect(1.0 / np.cos(np.deg2rad(14.0)))
mp.set_title("The outer boundary's elements: the trench on them")
fig.legend(handles=[Patch(facecolor=LAYERS[n], edgecolor=EDGES, label=n)
                    for n in ("mantle", "lithosphere")]
           + [Patch(facecolor=FAULT, label="megathrust"),
              Line2D([], [], color=FAULT, lw=3.0, label="its free edge"),
              Line2D([], [], color=FAULT, lw=1.5, ls=(0, (4, 2)), label="its tip"),
              Line2D([], [], color=TRUE, lw=1.2, ls=(0, (1, 2)),
                     label="the trench as given")],
           loc="lower center", ncol=7, frameon=False)
fig.subplots_adjust(left=0.0, right=0.98, top=0.93, bottom=0.16, wspace=0.05)
fig.savefig(FIGURES / "meshing_04.png", dpi=120)
shutil.rmtree(work)
