# %% [markdown]
# # A cubed sphere
#
# A ball can be filled with hexahedra with no mesher. Put a cube at the
# centre, divide it into small cubes, and stack columns of cells on
# each of its six faces, out to the surface. The six stacks are the
# blocks, and the mesh is a cubed sphere. Every interface of the model
# is a sphere of the mesh, as it is in a mesh of tetrahedra, and the
# cells meet face to face throughout.
#
# `CubedSphereSpec` describes one by two things. `edge` is the number
# of cells along an edge of the central cube, and so across each block:
# a quarter of a great circle has `edge` cells, whatever the radius, so
# the cells are narrower the deeper they lie. The sizing rule is the
# one a mesh of tetrahedra takes, read here as radial sizes alone: each
# layer takes as many cells across it as the rule asks for, at least
# one. Here a three-layer unit ball, with cells 0.03 thick at the
# surface growing to 0.2 below, drawn three ways: its disc, which is
# the same construction in the plane, a central square and four
# blocks, described by a `SquaredCircleSpec` and built by
# `build_squared_circle`; the ball cut by a plane through its centre;
# and its surface.
#
# Needs: mfem, plot
# Network: none
# Run: poetry run python examples/meshing/07_a_cubed_sphere.py

# %%
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (CubedSphereSpec, InterfaceSizing, PerInterface,
                                SquaredCircleSpec, UniformInterfaces,
                                build_cubed_sphere, build_squared_circle, manifest,
                                read_mesh)
from planetmodel.plotting import in_plane, mesh_block, mesh_section, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
LAYERS = {"core": "#c3c2b7", "mantle": "#e1e0d9", "lithosphere": "#f4f3f0"}
BLOCKS = ["#c9dcf2", "#f6d3c3", "#d7e8cf"]
INK, EDGES = "#52514e", "#a3a29b"
EDGE = 8

geometry = Geometry(Skeleton([0.0, 0.5, 0.9, 1.0]), layer_names=list(LAYERS),
                    interface_names=["cmb", "base", "surface"])
sizing = PerInterface({"surface": InterfaceSizing(0.03, 0.2, 0.4)},
                      base=UniformInterfaces(0.2, 0.2, 0.2))

# %% [markdown]
# The spec says what the mesh will be before anything is built: the
# radii of its spheres, from the first one outward, and the number of
# its cells. The first sphere is the innermost interface, here the top
# of the core. The core itself is the cube and the blend, the cells
# between the cube's flat faces and that sphere.

# %%
spec = CubedSphereSpec(geometry, EDGE, sizing=sizing)
print(f"spheres at r = {np.round(spec.spheres, 3).tolist()}")
print(f"{spec.blend.size - 1} cells across the blend, {spec.levels} levels above "
      f"the cube, {spec.cells} cells: {EDGE ** 3} in the cube and "
      f"{6 * EDGE ** 2} in each level")

# %% [markdown]
# The build writes the reference mesh in MFEM's format, with its
# manifest, and reports the checks it made: the worst shape of a cell
# (1 for a cube), how far each layer's volume is from that of the shell
# it fills, and the largest aspect ratio. The mesh is of order 2, so
# its cells are curved, and its nodes on an interface lie on the sphere
# exactly.

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_cubed_"))
ball = build_cubed_sphere(spec, work / "ball")
disc = build_squared_circle(SquaredCircleSpec(geometry, EDGE, sizing=sizing),
                            work / "disc")
for name, result in (("ball", ball), ("disc", disc)):
    report = result.validation
    print(f"{name}: {result.counts['elements']} cells, {result.counts['nodes']} "
          f"nodes; worst shape {report.min_sicn:.2f}, layer volumes within "
          f"{report.max_layer_volume_error:.1e}, aspect ratio up to "
          f"{report.max_aspect_ratio:.1f}")

# %% [markdown]
# `read_mesh` reads either file as plain arrays, each cell by its
# corners. The disc lies in a plane and is drawn whole; the ball is cut
# by the plane through its centre normal to (1, 1, 1), which crosses
# all six blocks, and the faces of each interface are cut by
# the same plane, which leaves the circles. The cells are drawn through
# their corners, with straight sides.

# %%
shades = np.array(list(LAYERS.values()))
flat = read_mesh(disc.mesh_path)
flat_card = manifest.read(disc.manifest_path)
solid = read_mesh(ball.mesh_path)
card = manifest.read(ball.manifest_path)
origin = np.zeros(3)
plane = [[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), layout="constrained")
ax = axes[0]
mesh_section(ax, in_plane(flat.points, flat.cells, origin, np.eye(3)[:2]),
             colours=shades[flat.layers(flat_card) - 1], edgecolor=EDGES,
             linewidth=0.5)
for entry in flat_card.interfaces:
    mesh_section(ax, in_plane(flat.points, flat.faces_with(entry["attribute"]),
                              origin, np.eye(3)[:2]), colours=INK, linewidth=1.3)
ax.set_title(f"the disc: a square and four blocks, {disc.counts['elements']} cells")

ax = axes[1]
mesh_section(ax, plane_cut(solid.points, solid.cells, origin, plane),
             colours=shades[solid.layers(card) - 1], edgecolor=EDGES, linewidth=0.5)
for entry in card.interfaces:
    mesh_section(ax, plane_cut(solid.points, solid.faces_with(entry["attribute"]),
                               origin, plane, tensor=True), colours=INK,
                 linewidth=1.3)
ax.set_title("the ball, cut through its centre normal to (1, 1, 1)")

# %% [markdown]
# The surface is seen from far along (1, 1, 1), where three blocks meet
# at a corner of the cube. A face belongs to the block whose axis its
# centre lies nearest, and the faces turned towards the viewer are
# drawn, projected onto the plane across the line of sight.

# %%
ax = axes[2]
faces = solid.points[solid.faces_with(card.surface_attribute)]
centres = faces.mean(axis=1)
view = np.ones(3) / np.sqrt(3.0)
a = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
b = np.cross(view, a)
seen = centres @ view > 0.0
block = np.argmax(np.abs(centres), axis=1)
ax.add_collection(PolyCollection(
    np.stack([faces[seen] @ a, faces[seen] @ b], axis=-1),
    facecolors=np.array(BLOCKS)[block[seen]], edgecolors=INK, linewidths=0.5))
ax.set_aspect("equal")
ax.autoscale_view()
ax.set_title(f"the surface: {EDGE} cells across each block")
for ax in axes:
    ax.set_axis_off()
axes[0].legend(handles=[Patch(facecolor=c, edgecolor=EDGES, label=name)
                        for name, c in LAYERS.items()], loc="lower left",
               frameon=False)
fig.savefig(FIGURES / "meshing_07.png", dpi=120)

# %% [markdown]
# ## A cutaway
#
# `mesh_block` draws a block of the mesh in three dimensions: the box
# given by its bounds, with the three walls nearest the viewer taken
# away, so that the three walls behind show the cells they cut, each
# cut through its corners as in a section. Here the box is the octant
# of the ball whose far corner is the centre, so its three walls are
# three planes through the centre, and the faces of the interfaces
# leave their circles on them.

# %%
fig, ax = plt.subplots(figsize=(7.0, 7.0), subplot_kw={"projection": "3d"},
                       layout="constrained")
mesh_block(ax, solid.points, solid.cells, ((0.0, 1.0), (-1.0, 0.0), (0.0, 1.0)),
           colours=shades[solid.layers(card) - 1], lines=solid.faces,
           edgecolor=EDGES, linewidth=0.4)
ax.set_axis_off()
ax.set_title("an octant of the ball, seen from outside")
fig.savefig(FIGURES / "meshing_07_cutaway.png", dpi=120)

# %% [markdown]
# ## The centre
#
# One cell across the blend is a needle pointing at the centre: its
# width is set by `edge` and shrinks with radius, while its length is
# the whole distance from the cube to the first sphere. The finer the
# mesh, the thinner the needle, about half of `edge` times longer than
# it is wide. So a spec divides the blend until none of its cells is
# more than `elongation` times longer radially than it is wide, 4 by
# default; `elongation=None` leaves it as the sizing has it. The blend
# can also be divided outright: `knots` gives the radial levels of the
# mesh in place of a sizing rule, and those below the first sphere are
# levels of the blend, each by its radius along the ray through the
# centre of a face of the square. Here the same disc at 16 cells an
# edge three ways, its spheres given as `knots`, and only its core
# drawn: one cell across the blend, the blend under the cap, and the
# blend cut once, at radius 0.35.

# %%
SPHERES = [0.5, 0.7, 0.9, 1.0]
cases = (("elongation=None", {"knots": SPHERES, "elongation": None}),
         ("elongation=4", {"knots": SPHERES}),
         ("a knot at 0.35", {"knots": [0.35, *SPHERES]}))
fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), layout="constrained")
for k, (ax, (label, given)) in enumerate(zip(axes, cases)):
    fine = SquaredCircleSpec(geometry, 16, **given)
    result = build_squared_circle(fine, work / f"centre_{k}")
    mesh = read_mesh(result.mesh_path)
    layers = mesh.layers(manifest.read(result.manifest_path))
    core = layers == 1
    mesh_section(ax, in_plane(mesh.points, mesh.cells[core], origin, np.eye(3)[:2]),
                 colours=LAYERS["core"], edgecolor=INK, linewidth=0.5)
    cells = fine.blend.size - 1
    ax.set_title(f"{label}: {cells} cell{'s' if cells > 1 else ''} across the "
                 f"blend,\naspect ratio up to "
                 f"{result.validation.max_aspect_ratio:.1f}")
    ax.set_axis_off()
    print(f"{label}: {cells} across the blend, {result.counts['elements']} cells, "
          f"aspect ratio up to {result.validation.max_aspect_ratio:.1f}")
fig.savefig(FIGURES / "meshing_07_centre.png", dpi=120)
shutil.rmtree(work)
