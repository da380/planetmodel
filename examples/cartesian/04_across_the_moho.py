# %% [markdown]
# # A fault across the Moho under a mountain root
#
# Topography as a stack of reliefs stretches each layer linearly in depth
# between its top's relief and its bottom's, so the mapping's slope
# jumps at every interface inside the body: a kink. A plane crossing the
# Moho under a range and its root has a preimage creased along the
# crossing, which no smooth patch follows closely; the mesher finds the
# curve along which the preimage crosses the Moho, cuts the patch there
# into two pieces, one in each layer, pulls each back through one
# control grid, and builds them in joined along that curve, which lies
# on the reference Moho. One fault in the mesh: one attribute, a collar
# in each layer, its tip the plane's edge alone, the join a curve inside
# it. Here the section across the plane, in the body's shape; the
# fault's faces seen square on in its plane, by piece, with the join and
# the true crossing of the Moho's relief; and every node of its faces,
# moved, by its distance from the plane, on either side of the join.
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/cartesian/04_across_the_moho.py

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
from planetmodel.plotting import in_plane, mesh_section, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
LOCAL = CartesianFrame()
LAYERS = {"crust": "#f4f3f0", "mantle": "#e1e0d9"}
PIECES = {"crust": "#86b6ef", "mantle": "#1c5cab"}
COLLARS = {"crust": "#dce9f8", "mantle": "#b3cdee"}
FAULT, JOIN, INK, EDGES, TRUE = "#2a78d6", "#eb6834", "#52514e", "#a3a29b", "#0b0b0b"


# %% [markdown]
# A unit box of two layers, the Moho 0.4 deep, under a range along y
# over x = 0.6 with its root on the Moho, 0.04 deeper beneath it, the
# base held. The plane strikes 20 degrees east of y and dips 40 degrees,
# its centroid at (0.6, 0.1) at the reference Moho's depth, 0.5 long
# and 0.4 wide: its upper half in the crust, its lower in the mantle,
# and its crossing of the Moho's relief a curve, the root deepening
# towards x = 0.6.

# %%
def ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.08 * np.exp(-((x - 0.6) / 0.15) ** 2) * (1.0 + 0.25 * np.cos(np.pi * y))


def root(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -0.04 * np.exp(-((x - 0.6) / 0.2) ** 2)


box = cartesian.Geometry(Skeleton([0.0, 0.4, 1.0]),
                         cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                         layer_names=list(LAYERS), interface_names=["surface", "moho"])
geometry = box.stretched(cartesian.layer_linear(box.skeleton, [ridge, root, None]))
plane = Plane.at_depth(LOCAL, 0.6, 0.1, 0.4, strike=20.0, dip=40.0, length=0.5,
                       width=0.4, slip=Slip.from_rake(90.0, 1.0))
fault = Fault([plane])
print("knots of the mapping (depths where its slope may jump):",
      geometry.mapping.knots)

# %% [markdown]
# The mesh of order 2, at the default tolerance of the pull-back (each
# piece within a millionth of the patch's size), the fault's size 0.03
# (graded to the background and the linear mesh relocated,
# `EmbeddedFault`'s rule). Built whole, the plane's preimage would need
# far more control nodes than the cap allows at this tolerance.

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_box_moho_"))
result = build_layered_mesh(
    MeshSpec(geometry, UniformInterfaces(0.1, 0.25, 0.3), order=2,
             faults=[EmbeddedFault(fault, name="thrust", sizing=0.03)]),
    work / "across")
pieces = result.preimages["thrust"]
mesh, card = read_mesh(result.msh_path), manifest.read(result.manifest_path)
record = card.fault_record("thrust")
print(f"{len(pieces)} pieces, through {record['controls'][0]} control nodes each "
      f"way, offset {record['offset']:.2g}; the fault in layers {record['layers']}, "
      f"its collars {record['hanging_wall']}, its tip curve group {record['tip']}, "
      f"no junction ({record['junction']})")

# %% [markdown]
# The faces of each piece are those whose cells lie in its layer; the
# join is the nodes both pieces' faces hold. Moved by the mapping, the
# join lies on the Moho's relief, and every node of either piece on the
# plane, to the pull-back's offset.

# %%
x = result.mapping(mesh.points)
faces = mesh.faces_with(record["attribute"])
layer = mesh.layers(card)[mesh.face_cells(faces[:, :3])[:, 0]]
sides = {name: faces[layer == k + 1] for k, name in enumerate(LAYERS)}
join = np.intersect1d(*(np.unique(f) for f in sides.values()))
moho = root(x[join, 0], x[join, 1]) - 0.4
print(f"the join: {join.size} nodes, within {np.abs(x[join, 2] - moho).max():.2g} of "
      "the Moho's relief")
distance = {}
for name, f in sides.items():
    nodes = np.setdiff1d(np.unique(f), join)
    distance[name] = np.abs(fault.locate(x[nodes], strict=False).distance)
    print(f"the {name} piece: {len(f)} faces, its nodes within "
          f"{distance[name].max():.2g} of the plane")
distance["join"] = np.abs(fault.locate(x[join], strict=False).distance)

# %% [markdown]
# The section through the centroid across the strike, as the body is
# (the nodes moved), zoomed on the plane, its coordinates the distance
# down the dip's horizontal direction from the centroid and z: the
# cells by layer,
# each layer's collar of the fault, the interfaces, the fault's faces by
# piece and the join where the section crosses it.

# %%
across = np.array([np.cos(np.deg2rad(20.0)), -np.sin(np.deg2rad(20.0)), 0.0])
origin = plane.centroid * [1.0, 1.0, 0.0]
axes = np.array([across, [0.0, 0.0, 1.0]])
fill = np.array([LAYERS[n] for n in np.array(list(LAYERS))[mesh.layers(card) - 1]],
                dtype=object)
for k, name in enumerate(LAYERS):
    fill[mesh.cell_attributes == record["hanging_wall"][k]] = COLLARS[name]
fig = plt.figure(figsize=(14, 4.9), layout="constrained")
grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.9])
ax = fig.add_subplot(grid[0])
mesh_section(ax, plane_cut(x, mesh.cells[:, :4], origin, axes), colours=list(fill),
             edgecolor=EDGES, linewidth=0.4)
interfaces = mesh.boundary_of([e["attribute"] for e in card.interfaces])
mesh_section(ax, plane_cut(x, interfaces, origin, axes), colours=INK, linewidth=1.0)
for name, f in sides.items():
    mesh_section(ax, plane_cut(x, f[:, :3], origin, axes), colours=PIECES[name],
                 linewidth=3.0, capstyle="round")
at = (x[join] - origin) @ axes.T
nearest = at[np.argmin(np.abs((x[join] - origin) @ np.cross(*axes)))]
ax.plot(*nearest, marker="o", ms=8, mfc=JOIN, mec=TRUE, zorder=5)
ax.set(xlim=(-0.22, 0.26), ylim=(-0.62, 0.12),
       xlabel="across strike, down dip (unit lengths)",
       ylabel="z, up (unit lengths)", title="Across the strike, as the body is")

# %% [markdown]
# Square on: the moved faces projected onto the plane, along strike to
# the right and up dip upward, coloured by piece, the join's edges
# heavy, and the true crossing of the plane with the Moho's relief.

# %%
basis = plane.fault_basis(0.0, 0.0)
square = np.stack([basis[:, 0], basis[:, 1]])
sq = fig.add_subplot(grid[1])
for name, f in sides.items():
    mesh_section(sq, in_plane(x, f[:, :3], plane.centroid, square, rtol=1e-5),
                 colours=PIECES[name], edgecolor="#fcfcfb", linewidth=0.3)
edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
both = edges[np.all(np.isin(edges, join), axis=1)]
segments = (x[both] - plane.centroid) @ square.T
for s in segments:
    sq.plot(*s.T, color=JOIN, lw=3.0, solid_capstyle="round")
u = np.linspace(-0.5, 0.5, 201)
v = np.linspace(-0.5, 0.5, 2001)
U, V = np.meshgrid(u, v, indexing="ij")
P = plane.cartesian(U, V)
gap = P[..., 2] - (root(P[..., 0], P[..., 1]) - 0.4)
vc = np.array([np.interp(0.0, -g, v) for g in gap])
true = (plane.cartesian(u, vc) - plane.centroid) @ square.T
sq.plot(*true.T, color=TRUE, lw=1.2, ls=(0, (4, 3)))
sq.set(xlabel="along strike (unit lengths)", ylabel="up dip (unit lengths)",
       title="The fault square on, moved: its two pieces and their join")

# %% [markdown]
# And every node of the fault's faces, moved, by its distance from the
# plane, on either side of the join and on it.

# %%
hist = fig.add_subplot(grid[2])
bins = np.logspace(-14, -5, 37)
for name, colour in (("crust", PIECES["crust"]), ("mantle", PIECES["mantle"]),
                     ("join", JOIN)):
    hist.hist(np.maximum(distance[name], 1e-14), bins=bins, histtype="step", lw=1.8,
              color=colour, label=f"{name}: up to {distance[name].max():.1g}")
hist.axvline(record["offset"], color=INK, lw=0.8, ls=":")
hist.text(0.8 * record["offset"], hist.get_ylim()[1] * 0.97, "the pull-back's\noffset",
          ha="right", va="top", fontsize=8, color=INK)
hist.set(xscale="log", xlabel="distance from the plane (unit lengths)",
         ylabel="nodes", title="Every node of the fault, moved")
hist.legend(loc="center left", frameon=False, fontsize=9)
fig.legend(handles=[Patch(facecolor=c, edgecolor=EDGES, label=n)
                    for n, c in LAYERS.items()]
           + [Patch(facecolor=COLLARS[n], edgecolor=EDGES, label=f"collar in the {n}")
              for n in LAYERS]
           + [Patch(facecolor=PIECES[n], label=f"the {n} piece") for n in LAYERS]
           + [Line2D([], [], color=JOIN, lw=3.0, marker="o", mfc=JOIN, mec=TRUE,
                     label="their join"),
              Line2D([], [], color=TRUE, lw=1.2, ls=(0, (4, 3)),
                     label="the plane meets the Moho's relief")],
           loc="outside lower center", ncol=9, frameon=False, fontsize=9)
fig.savefig(FIGURES / "cartesian_04.png", dpi=120, bbox_inches="tight")
print("figure written to", FIGURES / "cartesian_04.png")
shutil.rmtree(work)
