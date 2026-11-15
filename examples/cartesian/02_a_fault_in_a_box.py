# %% [markdown]
# # A fault in a box, in section
#
# A plane fault in the mantle of a box of two layers, built into the
# mesh: the mesh conforms to it, so a vertical section across its strike
# shows the fault as a line along which the element edges run. The
# fault is built three times: with no refinement towards it, its faces
# conforming at whatever size the interfaces' rule gives there; refined
# by a steep ramp of its own, a size on it growing to a far size over a
# fixed width (an `InterfaceSizing`, the size more than doubling from
# one element to the next), the linear mesh left as the mesher made it
# (`optimise=None`); and by the default, the size on it given alone
# (`EmbeddedFault`'s rule), the size growing by about 30 per cent an
# element until it meets the interfaces' sizing and the linear mesh
# relocated by gmsh's Relocate3D pass, which moves no node of a fault
# or an interface (the first mesh is relocated too). A cut shows the
# elements as the plane leaves them, thin shards beside polygons
# whatever their shape; a crinkle (`plane_cut(..., kind="crinkle")`)
# shows the faces of the cells the plane passes through whole, on a
# stepped surface an element or less in front of it. The shape is
# measured too (`mesh3d.cell_quality`, gmsh's minSICN: 1 for a regular
# tetrahedron, 0 for a flat one), and the fault's own faces are drawn
# square on, as they are.
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/cartesian/02_a_fault_in_a_box.py

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
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec,
                                UniformInterfaces, build_layered_mesh, cell_quality,
                                manifest, read_mesh)
from planetmodel.plotting import in_plane, mesh_section, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
LOCAL = CartesianFrame()
LAYERS = {"crust": "#f4f3f0", "mantle": "#e1e0d9"}
FAULT, INK, EDGES = "#2a78d6", "#52514e", "#a3a29b"
BUILDS = {"no sizing": "#a3a29b", "steep ramp": "#eb6834",
          "graded (the default)": "#2a78d6"}

# %% [markdown]
# A unit box, x from 0 to 1 and y from -1/2 to 1/2, its crust 0.4 deep
# over a mantle to 1, and a plane striking along y and dipping 50
# degrees towards +x, its centroid 0.62 deep: wholly in the mantle, its
# edges well clear of the Moho and the base.

# %%
geometry = cartesian.Geometry(Skeleton([0.0, 0.4, 1.0]),
                              cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                              layer_names=list(LAYERS),
                              interface_names=["surface", "moho"])
plane = Plane.at_depth(LOCAL, 0.5, 0.0, 0.62, strike=0.0, dip=50.0, length=0.4,
                       width=0.2, slip=Slip.from_rake(90.0, 1.0))
fault = Fault([plane])

# %% [markdown]
# The three meshes, of order 1; the interfaces' size is 0.12, growing to
# 0.3 away from them, and the fault's 0.025. The steep ramp grows to
# 0.12 over 0.1, a gradient of 0.95, and holds the whole mesh to 0.12;
# the default grows at 0.3 per unit distance until it meets the
# interfaces' sizing, reaching their largest size, 0.3, at most
# (0.3 - 0.025) / 0.3 = 0.92 from the fault.

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_box_section_"))
h = 0.025
specs = {"no sizing": ({}, {}),
         "steep ramp": (dict(sizing=InterfaceSizing(h, 0.12, 0.1)),
                        dict(optimise=None)),
         "graded (the default)": (dict(sizing=h), {})}
built = {}
for k, (label, (sizing, options)) in enumerate(specs.items()):
    result = build_layered_mesh(
        MeshSpec(geometry, UniformInterfaces(0.12, 0.3, 0.3), order=1, **options,
                 faults=[EmbeddedFault(fault, name="fault", **sizing)]),
        work / f"build_{k}")
    mesh, card = read_mesh(result.msh_path), manifest.read(result.manifest_path)
    quality = cell_quality(result.msh_path)
    centres = mesh.points[mesh.cells[:, :4]].mean(axis=1)
    near = np.abs(fault.locate(centres, strict=False).distance) <= 3 * h
    built[label] = (mesh, card, quality, near)
    q = quality[near]
    print(f"{label}: {len(mesh.cells)} cells, "
          f"{len(mesh.faces_with(card.fault_record('fault')['attribute']))} fault "
          f"faces; minSICN within {3 * h:g} of the fault: least {q.min():.3f}, 1% "
          f"{np.percentile(q, 1):.3f}, median {np.median(q):.3f}; elsewhere: least "
          f"{quality[~near].min():.3f}, median {np.median(quality[~near]):.3f}")

# %% [markdown]
# The section y = 0, across the fault's strike, as a cut (above) and as
# a crinkle (below): the cells by layer, and over them the interfaces'
# faces and the fault's cut by the same plane, the curves they leave in
# it, which the cut's edges run along and the crinkle's, an element in
# front, do not. In the plane's coordinates, (x, z), z up.

# %%
origin, axes = np.zeros(3), np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def draw(ax: plt.Axes, mesh: object, card: object, *, kind: str, lw: float) -> None:
    """The section of a mesh: cells by layer, interfaces, the fault."""
    colours = np.array(list(LAYERS.values()))[mesh.layers(card) - 1]
    mesh_section(ax, plane_cut(mesh.points, mesh.cells[:, :4], origin, axes, kind=kind),
                 colours=colours, edgecolor=EDGES, linewidth=lw)
    interfaces = mesh.boundary_of([e["attribute"] for e in card.interfaces])
    mesh_section(ax, plane_cut(mesh.points, interfaces, origin, axes), colours=INK,
                 linewidth=1.0)
    faces = mesh.faces_with(card.fault_record("fault")["attribute"])[:, :3]
    mesh_section(ax, plane_cut(mesh.points, faces, origin, axes), colours=FAULT,
                 linewidth=2.5, capstyle="round")


fig = plt.figure(figsize=(13, 7.4), layout="constrained")
grid = fig.add_gridspec(2, 4)
ends = np.array([plane.cartesian(0.0, v) for v in (-0.5, 0.5)]) @ axes.T
box = np.array([[-0.14, 0.14]] * 2) + ends.mean(axis=0)[:, None]
for row, kind in enumerate(("cut", "crinkle")):
    whole = fig.add_subplot(grid[row, 0])
    draw(whole, *built["graded (the default)"][:2], kind=kind, lw=0.2)
    whole.add_patch(plt.Rectangle(box[:, 0], *np.diff(box).ravel(), fill=False,
                                  edgecolor="#0b0b0b", linewidth=1.0))
    whole.set(xlabel="x (unit lengths)" if row else "", ylabel="z, up (unit lengths)",
              title=f"The section y = 0, a {kind}")
    for k, label in enumerate(specs):
        ax = fig.add_subplot(grid[row, k + 1])
        draw(ax, *built[label][:2], kind=kind, lw=0.6)
        ax.set(xlim=box[0], ylim=box[1], xticks=[], yticks=[], title=f"{label}, {kind}")
        x0, y0 = box[0, 0] + 0.015, box[1, 0] + 0.015
        ax.plot([x0, x0 + 0.05], [y0, y0], color="#0b0b0b", lw=2.0,
                solid_capstyle="butt")
        ax.text(x0 + 0.025, y0 + 0.006, "0.05", ha="center", va="bottom")
fig.legend(handles=[Patch(facecolor=c, edgecolor=EDGES, label=n)
                    for n, c in LAYERS.items()]
           + [Line2D([], [], color=INK, lw=1.0, label="interface"),
              Line2D([], [], color=FAULT, lw=2.5, label="fault")],
           loc="outside lower center", ncol=4, frameon=False)
fig.savefig(FIGURES / "cartesian_02_section.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# The fault's faces, its own triangles, seen square on in its plane
# (along strike to the right, down dip downward), for each mesh; and
# the shape of the cells within three fault sizes of it, the steep ramp
# against the default (the few cells there without a sizing are left
# out).

# %%
basis = plane.fault_basis(0.0, 0.0)
square = np.stack([basis[:, 0], basis[:, 1]])      # strike, up dip
fig = plt.figure(figsize=(12, 6.2), layout="constrained")
grid = fig.add_gridspec(2, 3, height_ratios=[0.75, 1.0])
for k, label in enumerate(specs):
    mesh, card, quality, near = built[label]
    faces = mesh.faces_with(card.fault_record("fault")["attribute"])[:, :3]
    ax = fig.add_subplot(grid[0, k])
    mesh_section(ax, in_plane(mesh.points, faces, plane.centroid, square),
                 colours="#dce9f8", edgecolor=FAULT, linewidth=0.5)
    ax.set(xlabel="along strike (unit lengths)",
           title=f"{label}: {len(faces)} faces")
    ax.set_ylabel("up dip (unit lengths)" if k == 0 else "")
hist = fig.add_subplot(grid[1, :])
bins = np.linspace(0.0, 1.0, 41)
for label, colour in list(BUILDS.items())[1:]:
    q = built[label][2][built[label][3]]
    hist.hist(q, bins=bins, density=True, histtype="step", color=colour, lw=1.8,
              label=f"{label}: least {q.min():.2f}, median {np.median(q):.2f}")
hist.set(xlabel="minSICN of the cells within three fault sizes of the fault "
                "(1: regular, 0: flat)", ylabel="density")
hist.legend(loc="upper left", frameon=False)
fig.savefig(FIGURES / "cartesian_02_quality.png", dpi=120, bbox_inches="tight")
print("figures written to", FIGURES)
shutil.rmtree(work)
