# %% [markdown]
# # Reaching the surface under topography: the pull-back, in a box
#
# The mesh of a flat geometry is built on the reference box and its
# nodes are moved by the geometry's mapping afterwards, here a vertical
# stretch raising a range on the free surface, so a fault given where
# it physically is enters the mesh as its preimage under the mapping
# (`fault.pull_back`): built in that way, the moved faces lie on the
# true fault. A thrust breaks the surface under the range. The section
# across its strike is drawn three times: the reference mesh, with the
# preimage built in, curved, below the true plane; the mesh moved by the
# mapping, where the fault's faces lie on the true plane and meet the
# moved surface; and the negative control, the fault built in as given
# and the nodes moved after, which carries it off the true plane by the
# range's lift. Beside them, every node of the fault's faces, moved, by
# its distance from the true fault.
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/cartesian/03_the_pull_back.py

# %%
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from planetmodel import Skeleton, cartesian
from planetmodel.fault import CartesianFrame, Fault, Plane, Slip
from planetmodel.mesh3d import (EmbeddedFault, MeshSpec, UniformInterfaces,
                                build_layered_mesh, manifest, read_mesh)
from planetmodel.plotting import mesh_section, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
LOCAL = CartesianFrame()
LAYERS = {"crust": "#f4f3f0", "mantle": "#e1e0d9"}
FAULT, INK, EDGES, TRUE = "#2a78d6", "#52514e", "#a3a29b", "#0b0b0b"
CASES = {"pulled back": FAULT, "built as given": "#eb6834"}


# %% [markdown]
# A unit box of two layers, its crust 0.4 deep, under a range along y
# over x = 0.6, 0.1 high at its crest (heights, positive up), the Moho
# and the base held: each layer is stretched linearly in depth between
# its top's relief and its bottom's. The thrust strikes along y and dips
# 40 degrees towards +x, its up-dip edge on the reference surface at
# x = 0.5, where the range stands 0.06 above it; it ends 0.19 deep,
# well inside the crust.

# %%
def ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.08 * np.exp(-((x - 0.6) / 0.15) ** 2) * (1.0 + 0.25 * np.cos(np.pi * y))


reference = cartesian.Geometry(Skeleton([0.0, 0.4, 1.0]),
                               cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                               layer_names=list(LAYERS),
                               interface_names=["surface", "moho"])
geometry = reference.stretched(cartesian.layer_linear(reference.skeleton,
                                                      [ridge, None, None]))
dip, width = 40.0, 0.3
plane = Plane.at_depth(LOCAL, 0.5 + 0.5 * width * np.cos(np.deg2rad(dip)), 0.0,
                       0.5 * width * np.sin(np.deg2rad(dip)), strike=0.0, dip=dip,
                       length=0.5, width=width, slip=Slip.from_rake(90.0, 1.0))
fault = Fault([plane])
print("the up-dip edge:", np.round(plane.cartesian(0.0, -0.5), 6), "the range above "
      f"it {ridge(0.5, 0.0):.3f}")

# %% [markdown]
# Two meshes of order 2, the fault's size 0.03 (graded to the
# background and the linear mesh relocated, `EmbeddedFault`'s rule): one
# under the range, the fault pulled back through its stretch, and one on
# the reference box with the fault as given, its nodes moved by the same
# mapping afterwards (the pull-back off).

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_box_pullback_"))
size = 0.03
meshes = {}
for label, g in (("pulled back", geometry), ("built as given", reference)):
    result = build_layered_mesh(
        MeshSpec(g, UniformInterfaces(0.1, 0.25, 0.3), order=2,
                 faults=[EmbeddedFault(fault, name="thrust", sizing=size)]),
        work / label.replace(" ", "_"))
    meshes[label] = (read_mesh(result.mesh_path), manifest.read(result.manifest_path))
mapping = geometry.mapping
record = meshes["pulled back"][1].fault_record("thrust")
print(f"the preimage: {record['controls'][0]} control nodes, offset "
      f"{record['offset']:.2g}; its free edge is curve group {record['free_edge']}")

# %% [markdown]
# Every node of the fault's faces, mid-edge nodes among them, moved by
# the mapping and located on the true fault. Nodes located beyond its
# up-dip edge lie on the part carried across to the surface, the plane
# continued, and are counted apart.

# %%
found = {}
for label, (mesh, card) in meshes.items():
    nodes = np.unique(mesh.faces_with(card.fault_record("thrust")["attribute"]))
    x = mapping(mesh.points[nodes])
    where = fault.locate(x, strict=False)
    carried = ~where.inside & where.on_v_low
    found[label] = (x[~carried], np.abs(where.distance[~carried]))
    print(f"{label}: {nodes.size} nodes, {int(np.count_nonzero(carried))} carried "
          f"across to the surface; the rest within {found[label][1].max():.2g} of the "
          f"true fault, {found[label][1].max() / size:.2g} of an element")

# %% [markdown]
# The section y = 0, zoomed on the fault, in its coordinates (x, z), z
# up. The true fault is the plane's line x(0, v), v in [-1/2, 1/2],
# continued up dip to the range's surface.

# %%
origin, axes = np.zeros(3), np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
v = np.linspace(-1.5, 0.5, 2001)
ends = plane.cartesian(0.0, np.array([-0.5, 0.5]))
line = ends[0] + (v + 0.5)[:, None] * (ends[1] - ends[0])
top = np.interp(0.0, (line[:, 2] - ridge(line[:, 0], line[:, 1]))[::-1], v[::-1])
on = line[v >= -0.5] @ axes.T
continued = line[(v >= top) & (v <= -0.5)] @ axes.T
box = np.array([[0.3, 0.75], [-0.3, 0.12]])


def draw(ax: plt.Axes, mesh: object, card: object, x: np.ndarray) -> None:
    """The section of a mesh at positions x, and the true fault over it."""
    colours = np.array(list(LAYERS.values()))[mesh.layers(card) - 1]
    mesh_section(ax, plane_cut(x, mesh.cells[:, :4], origin, axes), colours=colours,
                 edgecolor=EDGES, linewidth=0.5)
    interfaces = mesh.boundary_of([e["attribute"] for e in card.interfaces])
    mesh_section(ax, plane_cut(x, interfaces, origin, axes), colours=INK, linewidth=1.0)
    faces = mesh.faces_with(card.fault_record("thrust")["attribute"])[:, :3]
    mesh_section(ax, plane_cut(x, faces, origin, axes), colours=FAULT, linewidth=3.0,
                 capstyle="round")
    ax.plot(*on.T, color=TRUE, lw=1.2, ls=(0, (4, 3)))
    ax.plot(*continued.T, color=TRUE, lw=1.2, ls=(0, (1, 2)))
    ax.set(xlim=box[0], ylim=box[1])


fig = plt.figure(figsize=(14, 4.4), layout="constrained")
grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.9])
cases = [("Reference: the preimage built in", "pulled back", False),
         (f"Moved: within {found['pulled back'][1].max():.1g} of the true fault",
          "pulled back", True),
         (f"Pull-back off: up to {found['built as given'][1].max():.2g} from it",
          "built as given", True)]
for k, (title, label, moved) in enumerate(cases):
    ax = fig.add_subplot(grid[k])
    mesh, card = meshes[label]
    draw(ax, mesh, card, mapping(mesh.points) if moved else mesh.points)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (unit lengths)")
    ax.set_ylabel("z, up (unit lengths)" if k == 0 else "")
    if k:
        ax.tick_params(labelleft=False)
dist = fig.add_subplot(grid[3])
for label, colour in CASES.items():
    x, d = found[label]
    dist.scatter(np.maximum(d, 1e-12), x[:, 2], s=6, color=colour, label=label)
dist.set(xscale="log", xlim=(1e-12, 1.0), xlabel="distance from the true fault",
         ylabel="z, up (unit lengths)", title="Every node of the fault, moved")
dist.axvline(size, color=INK, lw=0.8, ls=":")
dist.text(0.7 * size, -0.2, "the fault's\nelement size", color=INK, fontsize=8,
          ha="right", va="bottom")
dist.legend(loc="upper center", frameon=False, fontsize=9)
fig.legend(handles=[Patch(facecolor=c, edgecolor=EDGES, label=n)
                    for n, c in LAYERS.items()]
           + [Line2D([], [], color=INK, lw=1.0, label="interface"),
              Line2D([], [], color=FAULT, lw=3.0, label="fault in the mesh"),
              Line2D([], [], color=TRUE, lw=1.2, ls=(0, (4, 3)), label="true fault"),
              Line2D([], [], color=TRUE, lw=1.2, ls=(0, (1, 2)),
                     label="its plane, continued")],
           loc="outside lower center", ncol=6, frameon=False)
fig.savefig(FIGURES / "cartesian_03.png", dpi=120, bbox_inches="tight")
print("figure written to", FIGURES / "cartesian_03.png")
shutil.rmtree(work)
