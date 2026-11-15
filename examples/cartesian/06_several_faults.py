# %% [markdown]
# # Several faults in a box
#
# Three faults in one box: a thrust that breaks the surface and leaves
# through the side y = 1/2, a back-thrust splaying from it into its
# hanging wall and ending on it, and a fault buried in the mantle. Each
# is a sheet of boundary faces of its own attribute, and the manifest
# names its curves by kind: its buried tip, its free edge on the
# surface, its junction with another fault (and the faults it ends on),
# and its face edge on the base or a side. The cells of each fault's
# hanging wall that touch its slip, its collar, carry element attributes
# of their own, which the manifest lists by layer and by fault. Here a
# block of the box with every fault's faces and curves, and a section
# across the thrust with the collars, read from the cells' attributes
# alone.
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/cartesian/06_several_faults.py

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
from planetmodel.mesh3d import (EmbeddedFault, MeshSpec, UniformInterfaces,
                                build_layered_mesh, manifest, read_mesh)
from planetmodel.plotting import (FAULT_CURVE_STYLES, block_planes, clip_polygons,
                                  fault_curves, mesh_block, mesh_section, plane_cut)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
LOCAL = CartesianFrame()
LAYERS = {"crust": "#f4f3f0", "mantle": "#e1e0d9"}
COLOURS = {"thrust": "#2a78d6", "splay": "#eb6834", "buried": "#eda100"}
COLLARS = {"thrust": "#b3cdee", "splay": "#f6c3ad", "buried": "#f6d98f"}
BOTH = "#9fdcc4"
INK, GUIDE, EDGES = "#1a1a19", "#52514e", "#a3a29b"

# %% [markdown]
# A unit box of two layers, the Moho 0.4 deep. The thrust strikes along
# y and dips 45 degrees towards +x, from the surface to 0.21 deep, and
# runs from y = 0.1 out through the side y = 1/2. The splay leaves it
# along the line 0.13 down, a tenth of its width below its centre, and
# rises steeply back towards +x, dipping 60 degrees towards -x, to a
# buried tip 0.04 below the surface. The buried fault, in the mantle,
# strikes 60 degrees and dips 75.

# %%
geometry = cartesian.Geometry(Skeleton([0.0, 0.4, 1.0]),
                              cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                              layer_names=list(LAYERS),
                              interface_names=["surface", "moho"])
dip, width = 45.0, 0.3
thrust = Plane.at_depth(LOCAL, 0.3 + 0.5 * width * np.cos(np.deg2rad(dip)), 0.35,
                        0.5 * width * np.sin(np.deg2rad(dip)), strike=0.0, dip=dip,
                        length=0.5, width=width, slip=Slip.from_rake(90.0, 1.0))
foot = thrust.cartesian(-0.1, 0.1)
splay = Plane(LOCAL, foot, strike=180.0, dip=60.0, length=0.25, width=0.1,
              slip=Slip.from_rake(90.0, 0.5))
splay = Plane(LOCAL, 2.0 * foot - splay.cartesian(0.0, 0.5), strike=180.0, dip=60.0,
              length=0.25, width=0.1, slip=Slip.from_rake(90.0, 0.5))
buried = Plane.at_depth(LOCAL, 0.7, -0.25, 0.6, strike=60.0, dip=75.0, length=0.25,
                        width=0.2, slip=Slip.from_rake(0.0, 0.3))
faults = {"thrust": Fault([thrust]), "splay": Fault([splay]),
          "buried": Fault([buried])}
print("the splay's down-dip edge on the thrust, its tip",
      f"{-splay.cartesian(0.0, -0.5)[2]:.3f} deep")

# %% [markdown]
# The mesh of order 2, every fault's size 0.03 (graded to the background
# and the linear mesh relocated, `EmbeddedFault`'s rule). Each fault's
# record: its attribute, its
# curve groups by kind (None where it has none), the faults it meets
# and ends on, and its collars.

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_box_faults_"))
result = build_layered_mesh(
    MeshSpec(geometry, UniformInterfaces(0.1, 0.25, 0.3), order=2,
             faults=[EmbeddedFault(f, name=n, sizing=0.03) for n, f in faults.items()]),
    work / "faulted")
print(result.summary())
mesh, card = read_mesh(result.msh_path), manifest.read(result.manifest_path)
for record in card.faults:
    kinds = ", ".join(f"{key} {record[key]}" for key in manifest.FLAT_FAULT_CURVES)
    print(f"{record['name']}: attribute {record['attribute']}; {kinds}; meets "
          f"{record['meets'] or 'nothing'}, ends on {record['ends_on'] or 'nothing'}; "
          f"collars {record['hanging_wall']}")
for entry in card.layers:
    print(f"layer {entry['name']}: element attributes {entry['attributes']}")

# %% [markdown]
# An MFEM mesh keeps no curve groups; `manifest.fault_curves` reads the
# same curves from the faces alone, given the surface's and the box's
# faces, and finds the MSH file's groups edge for edge.

# %%
recovered = manifest.fault_curves(
    [mesh.faces_with(r["attribute"])[:, :3] for r in card.faults],
    mesh.faces_with(card.surface_attribute)[:, :3],
    faces=mesh.boundary_of(card.face_attributes))


def pairs(edges: np.ndarray) -> set:
    """Edges as a set of corner pairs, the smaller first."""
    return {tuple(e) for e in np.sort(edges[:, :2], axis=1).tolist()}


same = all(pairs(getattr(found, key)) == (pairs(mesh.curves[record[key]])
                                          if record[key] is not None else set())
           for record, found in zip(card.faults, recovered)
           for key in manifest.FLAT_FAULT_CURVES)
print(f"the curves read from the faces alone are the MSH file's groups: {same}")

# %% [markdown]
# The collars, from the attributes alone: a cell is in a fault's hanging
# wall where its attribute is one that fault's record lists; a cell in
# the splay's collar above the thrust is in the thrust's too.

# %%
hanging = {r["name"]: np.isin(mesh.cell_attributes, r["hanging_wall"])
           for r in card.faults}
both = hanging["thrust"] & hanging["splay"]
fill = np.array(list(LAYERS.values()), dtype=object)[mesh.layers(card) - 1]
for name, cells in hanging.items():
    fill[cells] = COLLARS[name]
fill[both] = BOTH
print(", ".join(f"{n}: {np.count_nonzero(c)} cells" for n, c in hanging.items())
      + f"; {np.count_nonzero(both)} in the collars of both the thrust and the splay")

# %% [markdown]
# The block x from 0.15 to 1, the whole of y, z from -0.8 to 0, seen
# from above the thrust's hanging wall, from +x: its walls x = 0.15,
# y = 1/2 (the box's side, where the thrust leaves) and z = -0.8 are
# the cells' sections, by layer and collar; inside it every fault's
# faces and its curves by kind, the splay in front of the thrust.

# %%
bounds = [(0.15, 1.0), (-0.5, 0.5), (-0.8, 0.0)]
interfaces = mesh.boundary_of([e["attribute"] for e in card.interfaces])
fig = plt.figure(figsize=(14, 6.6), layout="constrained")
grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
ax = fig.add_subplot(grid[0], projection="3d")
mesh_block(ax, mesh.points, mesh.cells, bounds, colours=list(fill), lines=interfaces,
           view=(30.0, -20.0), zoom=1.1, edgecolor=EDGES, linewidth=0.25)
for record in card.faults:
    tri, _ = clip_polygons(mesh.points[mesh.faces_with(record["attribute"])[:, :3]],
                           *block_planes(bounds))
    ax.add_collection3d(Poly3DCollection(tri, facecolor=COLOURS[record["name"]],
                                         edgecolor="#fcfcfb", linewidth=0.25))
for record in card.faults:
    fault_curves(ax, mesh.points, {key: mesh.curves[record[key]]
                                   for key in manifest.FLAT_FAULT_CURVES
                                   if record[key] is not None}, color=INK)
ax.set(xlabel="x", ylabel="y", zlabel="z")
ax.set_title("Three faults in a block of the box (unit lengths, z up)")

# %% [markdown]
# The section y = 0.3 across the thrust and the splay, in (x, z), z up,
# as a crinkle (`plane_cut(..., kind="crinkle")`), the cells the plane
# passes through drawn whole, where a cut would shred them: the cells by
# layer, coloured over them by the collar their attribute names; and
# over them the interfaces and the faults where the section crosses
# them, which lie in the plane, an element or less behind the cells'
# faces drawn.

# %%
sx = fig.add_subplot(grid[1])
origin, axes = np.array([0.0, 0.3, 0.0]), np.array([[1.0, 0, 0], [0, 0, 1.0]])
mesh_section(sx, plane_cut(mesh.points, mesh.cells[:, :4], origin, axes,
                           kind="crinkle"),
             colours=list(fill), edgecolor=EDGES, linewidth=0.5)
mesh_section(sx, plane_cut(mesh.points, interfaces, origin, axes), colours=GUIDE,
             linewidth=1.0)
for record in card.faults[:2]:
    mesh_section(sx, plane_cut(mesh.points,
                               mesh.faces_with(record["attribute"])[:, :3], origin,
                               axes),
                 colours=COLOURS[record["name"]], linewidth=2.5, capstyle="round")
sx.set(xlim=(0.18, 0.72), ylim=(-0.32, 0.02), xlabel="x (unit lengths)",
       ylabel="z, up (unit lengths)",
       title="Across the thrust, y = 0.3: the collars, by element attribute")
sx.legend(handles=[Patch(facecolor=c, edgecolor=EDGES, label=n)
                    for n, c in LAYERS.items()]
           + [Patch(facecolor=COLOURS[n], label=n) for n in COLOURS]
           + [Patch(facecolor=COLLARS[n], edgecolor=EDGES, label=f"{n}'s collar")
              for n in COLLARS]
           + [Patch(facecolor=BOTH, edgecolor=EDGES, label="in both collars")]
           + [Line2D([], [], color=INK, label=k.replace("_", " "),
                     **FAULT_CURVE_STYLES[k]) for k in manifest.FLAT_FAULT_CURVES],
          loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
fig.savefig(FIGURES / "cartesian_06.png", dpi=120, bbox_inches="tight")
print("figure written to", FIGURES / "cartesian_06.png")
shutil.rmtree(work)
