# %% [markdown]
# # AdGIA's faulted box, in planetmodel's form
#
# The box of AdGIA's coseismic examples (`meshes/faulted_box.py` in
# AdGIA), 8 wide and 4 deep with a planar thrust 1.5 by 1.5 centred 1
# below the surface, dipping 30 degrees towards +x, here with every
# length divided by 8, flat and under AdGIA's range topography (its
# relief 2, the default in 3-D). AdGIA builds its box with the free
# surface as a spline and cuts the body in two by the fault's plane
# extended to the boundary, the footwall and the hanging wall two
# volumes; planetmodel builds the reference box, moves its nodes by the
# vertical stretch of the topography, and gives the fault's hanging wall
# where it touches the slip, its collar, an element attribute of its
# own, its edge a tip curve group, pinned. The sizes are AdGIA's, at
# unit scale: 0.01875 on the fault, 0.05 on the surface, 0.1 far away,
# growing at its gradation, 0.3 per unit distance, which is the gradient
# a fault sized by a number is graded at (`EmbeddedFault`'s rule).
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/cartesian/08_adgia_s_box.py

# %%
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
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
SCALE = 1.0 / 8.0
BODY, FAULT, COLLAR, INK, EDGES = "#e1e0d9", "#2a78d6", "#b3cdee", "#1a1a19", "#a3a29b"


# %% [markdown]
# AdGIA's range, in its own lengths, and at unit scale: a steep front
# winding along strike, a crest and a long back slope over the hanging
# wall, a foreland basin and rolling hills.

# %%
def range_front(x: np.ndarray, y: np.ndarray, relief: float) -> np.ndarray:
    front = -1.6 + 0.3 * np.sin(np.pi * y / 5.0)
    rise = 0.5 * (1.0 + np.tanh((x - front) / 0.35))
    back = np.exp(-(np.maximum(x + 0.7, 0.0) / 2.6) ** 2)
    along = 0.65 + 0.35 * np.cos(np.pi * y / 5.0)
    ridge = 0.32 * rise * back * along
    foreland = -0.06 * np.exp(-((x - front + 0.9) / 0.8) ** 2) * along
    hills = (0.045 * np.sin(2 * np.pi * x / 3.3 + 0.6)
             * np.cos(2 * np.pi * y / 4.1 + 1.1)
             + 0.03 * np.sin(2 * np.pi * (x + 0.6 * y) / 2.7 + 2.0)
             + 0.02 * np.cos(2 * np.pi * (0.4 * x - y) / 3.1 + 0.3))
    return relief * (ridge + foreland + hills)


def height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return SCALE * range_front(x / SCALE, y / SCALE, 2.0)


# %% [markdown]
# The box, 1 by 1 and 0.5 deep, one layer, and AdGIA's fault: centred at
# (0, 0) 0.125 deep, 0.1875 along dip and along strike. AdGIA's frame
# (down dip, strike, the normal into the hanging wall) is the plane's
# fault basis (-up dip, strike, normal).

# %%
flat = cartesian.Geometry(Skeleton([0.0, 4.0 * SCALE]),
                          cartesian.Box([(-0.5, 0.5), (-0.5, 0.5)]),
                          layer_names=["body"], interface_names=["surface"])
under = flat.stretched(cartesian.layer_linear(flat.skeleton, [height, None]),
                       name="range")
plane = Plane(LOCAL, [0.0, 0.0, -1.0 * SCALE], strike=0.0, dip=30.0,
              length=1.5 * SCALE, width=1.5 * SCALE, slip=Slip.from_rake(90.0, 1.0))
fault = Fault([plane])
strike, up_dip, normal = plane.fault_basis(0.0, 0.0).T
print("down dip", np.round(-up_dip, 4), "strike", np.round(strike, 4), "normal",
      np.round(normal, 4))

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_adgia_box_"))
built = {}
for label, g in (("flat", flat), ("under the range", under)):
    result = build_layered_mesh(
        MeshSpec(g, UniformInterfaces(0.05, 0.1, (0.1 - 0.05) / 0.3), order=2,
                 faults=[EmbeddedFault(fault, name="fault", sizing=0.15 * SCALE)]),
        work / label.replace(" ", "_"))
    mesh, card = read_mesh(result.msh_path), manifest.read(result.manifest_path)
    built[label] = (mesh, card, result.mapping(mesh.points))
    print(f"{label}: {result.counts['elements']} elements, minSICN "
          f"{result.validation.min_sicn:.2f}")

# %% [markdown]
# What a consumer of either reads in place of AdGIA's attributes (its
# domain: footwall 1, hanging wall 2; its boundary: the fault 1, its
# extension 2, the surface 3, the clamped footwall 4 and hanging wall
# 5): the body's element attribute and its collar's, the surface's, the
# base's and the sides' boundary attributes, the fault's, and its tip.

# %%
mesh, card, _ = built["flat"]
record = card.fault_record("fault")
print(f"element attributes: body {card.layer_attribute('body')}, the fault's collar "
      f"{record['hanging_wall']}")
print(f"boundary attributes: surface {card.surface_attribute}, "
      + ", ".join(f"{e['name']} {e['attribute']}" for e in card.faces)
      + f", fault {record['attribute']}; its tip, curve group {record['tip']}, "
      f"{len(mesh.curves[record['tip']])} edges, its whole edge")

# %% [markdown]
# The surface from above, AdGIA's map: the topography's height, the
# fault from above (dashed) and its up-dip edge (solid), the section
# line; and the section y = 0 over the fault, flat and under the range,
# as the body is: the body's elements, the collar over them, the fault
# and its tip where the section crosses it (the circles).

# %%
fig = plt.figure(figsize=(14, 4.6), layout="constrained")
grid = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.0, 1.0])
ax = fig.add_subplot(grid[0])
X, Y = np.meshgrid(np.linspace(-0.5, 0.5, 301), np.linspace(-0.5, 0.5, 301))
H = height(X, Y)
bound = float(np.abs(H).max())
filled = ax.contourf(X, Y, H, levels=21, cmap="RdBu_r",
                     norm=TwoSlopeNorm(0.0, -bound, bound))
fig.colorbar(filled, ax=ax, label="surface height (unit lengths)", shrink=0.9)
outline = plane.cartesian(np.array([-0.5, 0.5, 0.5, -0.5, -0.5]),
                          np.array([-0.5, -0.5, 0.5, 0.5, -0.5]))
ax.plot(outline[:, 0], outline[:, 1], "k--", lw=1.2, label="fault, from above")
edge = plane.cartesian(np.array([-0.5, 0.5]), np.array([-0.5, -0.5]))
ax.plot(edge[:, 0], edge[:, 1], "k-", lw=2.5, label="its up-dip edge")
ax.axhline(0.0, color=INK, lw=0.8, ls=":", label="the section y = 0")
ax.set(xlabel="x", ylabel="y", aspect="equal", title="The surface, under the range")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
origin, axes = np.zeros(3), np.array([[1.0, 0, 0], [0, 0, 1.0]])
for k, label in enumerate(built):
    mesh, card, x = built[label]
    record = card.fault_record("fault")
    fill = np.where(np.isin(mesh.cell_attributes, record["hanging_wall"]), COLLAR, BODY)
    sx = fig.add_subplot(grid[k + 1])
    mesh_section(sx, plane_cut(x, mesh.cells[:, :4], origin, axes), colours=list(fill),
                 edgecolor=EDGES, linewidth=0.4)
    mesh_section(sx, plane_cut(x, mesh.faces_with(card.surface_attribute)[:, :3],
                               origin, axes), colours=INK, linewidth=1.2)
    trace = plane_cut(x, mesh.faces_with(record["attribute"])[:, :3], origin, axes)
    mesh_section(sx, trace, colours=FAULT, linewidth=2.5)
    ends = trace.pieces.reshape(-1, 2)[np.argsort(trace.pieces.reshape(-1, 2)[:, 0])]
    sx.plot(*ends[[0, -1]].T, "o", ms=5, mfc="white", mec=FAULT)
    sx.set(xlim=(-0.3, 0.3), ylim=(-0.3, 0.12), xlabel="x", ylabel="z",
           title=f"The section y = 0, {label}")
fig.legend(handles=[Patch(facecolor=BODY, edgecolor=EDGES,
                          label=f"element attribute {card.layer_attribute('body')}"),
                    Patch(facecolor=COLLAR, edgecolor=EDGES,
                          label=f"element attribute {record['hanging_wall'][0]}, the "
                          "collar"),
                    Line2D([], [], color=INK, lw=1.2,
                           label=f"boundary attribute {card.surface_attribute}, the "
                           "surface"),
                    Line2D([], [], color=FAULT, lw=2.5,
                           label=f"boundary attribute {record['attribute']}, the "
                           "fault")],
           loc="outside lower right", ncol=4, frameon=False, fontsize=9)
fig.suptitle("AdGIA's faulted box at unit scale (lengths divided by 8; z up)",
             fontsize=11)
fig.savefig(FIGURES / "cartesian_08.png", dpi=120, bbox_inches="tight")
print("figure written to", FIGURES / "cartesian_08.png")
shutil.rmtree(work)
