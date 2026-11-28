# %% [markdown]
# # Reaching the surface under topography: the pull-back
#
# A mesh is built on the reference ball and its nodes are moved by the
# geometry's mapping afterwards, so a fault given where it physically
# is must enter the mesh as its preimage under the mapping
# (`fault.pull_back`): built in that way, the moved faces lie on the
# true fault. Here a thrust breaks the surface under a smooth hill. The
# section normal to its strike is drawn three times: the reference mesh,
# with the preimage built in, curved, below the true plane; the mesh
# moved by the mapping, where the fault's faces lie on the true plane
# and meet the moved surface; and the negative control, the fault built
# in as given and the nodes moved after, which carries the fault off the
# true plane by the hill's lift. The true fault is drawn over each, with
# its plane continued up to the hill's surface.
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/meshing/03_the_pull_back.py

# %%
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton
from planetmodel.fault import Fault, GeographicFrame, Plane, Slip
from planetmodel.mesh3d import (EmbeddedFault, MeshSpec, UniformInterfaces,
                                build_layered_mesh, manifest, read_mesh)
from planetmodel.plotting import mesh_section, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
UNIT = GeographicFrame(1.0)
LAT, LON = 20.0, 30.0
LAYERS = {"core": "#c3c2b7", "mantle": "#e1e0d9", "lithosphere": "#f4f3f0"}
FAULT, INK, EDGES, TRUE = "#2a78d6", "#52514e", "#a3a29b", "#0b0b0b"

# %% [markdown]
# A unit ball of three layers under a hill 0.06 high at its crest, east
# of the fault's trace over its hanging wall, decaying as r squared with
# depth. The thrust strikes north and dips 40 degrees east, the middle
# of its up-dip edge at the reference surface, which the hill lifts
# above it.

# %%
def hill(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    c = (np.cos(theta) * np.cos(np.deg2rad(90.0 - LAT)) + np.sin(theta)
         * np.sin(np.deg2rad(90.0 - LAT)) * np.cos(phi - np.deg2rad(LON + 4.0)))
    return 0.06 * r ** 2 * np.exp(-(1.0 - c) / 0.006)


reference = Geometry(Skeleton([0.0, 0.55, 0.75, 1.0]), layer_names=list(LAYERS),
                     interface_names=["cmb", "base", "surface"])
geometry = reference.with_mapping(
    RadialStretch(CallableDisplacement(hill, name="hill"), rmax=1.0))
dip, width = 40.0, 0.24
plane = Plane.at_depth(UNIT, LAT, LON, 0.5 * width * np.sin(np.deg2rad(dip)),
                       strike=0.0, dip=dip, length=0.3, width=width,
                       slip=Slip.from_rake(90.0, 1.0))
fault = Fault([plane])

# %% [markdown]
# Two meshes of order 2, the fault's size 0.03 (graded to the background
# and relocated, `EmbeddedFault`'s rule): one built under the hill, the
# fault pulled back through it, and one built on the reference ball with
# the fault as given, its nodes moved by the same mapping afterwards (the
# pull-back off).

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_pullback_"))
size = 0.03
meshes = {}
for label, g in (("pulled back", geometry), ("built as given", reference)):
    result = build_layered_mesh(
        MeshSpec(g, UniformInterfaces(0.1, 0.3, 0.3), order=2,
                 faults=[EmbeddedFault(fault, name="thrust", sizing=size)]),
        work / label.replace(" ", "_"))
    meshes[label] = (read_mesh(result.mesh_path), manifest.read(result.manifest_path))
mapping = geometry.mapping
record = meshes["pulled back"][1].fault_record("thrust")
print(f"the preimage: {record['controls'][0]} control nodes, offset "
      f"{record['offset']:.2g}")

# %% [markdown]
# Every node of the fault's faces, mid-edge nodes among them, moved by
# the mapping and located on the true fault. Nodes located beyond its
# up-dip edge lie on the part carried across to the surface and are
# counted apart.

# %%
worst = {}
for label, (mesh, card) in meshes.items():
    nodes = np.unique(mesh.faces_with(card.fault_record("thrust")["attribute"]))
    found = fault.locate(mapping(mesh.points[nodes]), strict=False)
    carried = ~found.inside & found.on_v_low
    worst[label] = float(np.abs(found.distance[~carried]).max())
    print(f"{label}: {nodes.size} nodes, {int(np.count_nonzero(carried))} carried "
          f"across to the surface; the rest within {worst[label]:.2g} of the true "
          f"fault, {worst[label] / size:.2g} of an element")

# %% [markdown]
# The section through the centre spanned by east and up at the fault's
# centroid, zoomed on the fault. The true fault is the plane's line
# x(0, v), v in [-1/2, 1/2], straight in v, continued up dip to the
# hill's surface.

# %%
east, north, up = np.moveaxis(UNIT.basis(plane.centroid), -1, 0)
axes = np.stack([east, up])
v = np.linspace(-1.5, 0.5, 2001)
ends = plane.cartesian(0.0, np.array([-0.5, 0.5]))
line = ends[0] + (v + 0.5)[:, None] * (ends[1] - ends[0])
r = np.linalg.norm(line, axis=1)
lifted = 1.0 + hill(1.0, np.arccos(line[:, 2] / r), np.arctan2(line[:, 1], line[:, 0]))
top = np.interp(0.0, (r - lifted)[::-1], v[::-1])
on = (line[v >= -0.5]) @ axes.T
continued = line[(v >= top) & (v <= -0.5)] @ axes.T
centre = on.mean(axis=0) + [-0.01, 0.0]
box = np.array([[-0.17, 0.17]] * 2) + centre[:, None]


def draw(ax: plt.Axes, mesh: object, card: object, x: np.ndarray) -> None:
    """The section of a mesh at positions x, and the true fault over it."""
    colours = np.array(list(LAYERS.values()))[mesh.layers(card) - 1]
    mesh_section(ax, plane_cut(x, mesh.cells[:, :4], np.zeros(3), axes),
                 colours=colours, edgecolor=EDGES, linewidth=0.5)
    interfaces = np.isin(mesh.face_attributes,
                         [e["attribute"] for e in card.interfaces])
    mesh_section(ax, plane_cut(x, mesh.faces[interfaces, :3], np.zeros(3), axes),
                 colours=INK, linewidth=1.0)
    faces = mesh.faces_with(card.fault_record("thrust")["attribute"])[:, :3]
    mesh_section(ax, plane_cut(x, faces, np.zeros(3), axes), colours=FAULT,
                 linewidth=3.0, capstyle="round")
    ax.plot(*on.T, color=TRUE, lw=1.2, ls=(0, (4, 3)))
    ax.plot(*continued.T, color=TRUE, lw=1.2, ls=(0, (1, 2)))
    ax.set(xlim=box[0], ylim=box[1], xticks=[], yticks=[])


fig, panels = plt.subplots(1, 3, figsize=(12, 4.6))
cases = [("Reference mesh: the preimage built in", "pulled back", False),
         (f"Moved: within {worst['pulled back']:.1g} of the true fault",
          "pulled back", True),
         (f"Pull-back off: up to {worst['built as given']:.2g} from it",
          "built as given", True)]
for ax, (title, label, moved) in zip(panels, cases):
    mesh, card = meshes[label]
    draw(ax, mesh, card, mapping(mesh.points) if moved else mesh.points)
    ax.set_title(title)
x0, y0 = box[0, 0] + 0.015, box[1, 0] + 0.015
panels[0].plot([x0, x0 + 0.05], [y0, y0], color=TRUE, lw=2.0, solid_capstyle="butt")
panels[0].text(x0 + 0.025, y0 + 0.006, "0.05", ha="center", va="bottom")
fig.legend(handles=[Line2D([], [], color=INK, lw=1.0, label="interface"),
              Line2D([], [], color=FAULT, lw=3.0, label="fault in the mesh"),
              Line2D([], [], color=TRUE, lw=1.2, ls=(0, (4, 3)), label="true fault"),
              Line2D([], [], color=TRUE, lw=1.2, ls=(0, (1, 2)),
                     label="its plane, continued")],
           loc="lower center", ncol=4, frameon=False)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.1, wspace=0.06)
fig.savefig(FIGURES / "meshing_03.png", dpi=120)
shutil.rmtree(work)
