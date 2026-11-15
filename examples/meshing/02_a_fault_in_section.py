# %% [markdown]
# # A fault in the mesh, seen in section
#
# A plane fault inside one layer of a layered ball, built into the mesh:
# the mesh conforms to it, so a plane cut through the volume mesh shows
# the fault as a line along which the element edges run. The cut is
# taken through the fault's centroid, normal to its strike; it passes
# through the centre, so the whole section is a disc. The same fault is
# built twice, with the refinement towards it that `EmbeddedFault`'s
# `sizing` asks for, the size on it given and the rest following by
# rule, and without, when its faces conform at whatever size the
# interfaces' rule gives there; and the section is drawn as a cut and as
# a crinkle, the cells it passes through whole.
#
# Needs: meshing, plot
# Network: none
# Run: poetry run python examples/meshing/02_a_fault_in_section.py

# %%
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from planetmodel import Geometry, Skeleton
from planetmodel.fault import Fault, GeographicFrame, Plane, Slip
from planetmodel.mesh3d import (EmbeddedFault, MeshSpec, UniformInterfaces,
                                build_layered_mesh, manifest, read_mesh)
from planetmodel.plotting import mesh_section, plane_cut

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
UNIT = GeographicFrame(1.0)
LAYERS = {"core": "#c3c2b7", "mantle": "#e1e0d9", "lithosphere": "#f4f3f0"}
FAULT, INK, EDGES = "#2a78d6", "#52514e", "#a3a29b"

# %% [markdown]
# A unit ball of three layers, and a plane striking north and dipping 50
# degrees east with its centroid 0.33 below the surface: wholly in the
# mantle, its top and bottom edges some way from the interfaces above
# and below it.

# %%
geometry = Geometry(Skeleton([0.0, 0.55, 0.8, 1.0]), layer_names=list(LAYERS),
                    interface_names=["cmb", "base", "surface"])
plane = Plane.at_depth(UNIT, 20.0, 30.0, 0.33, strike=0.0, dip=50.0, length=0.3,
                       width=0.16, slip=Slip.from_rake(90.0, 1.0))
fault = Fault([plane])

# %% [markdown]
# The mesh twice, of order 1, with and without the size 0.025 on the
# fault. By `EmbeddedFault`'s rule the size grows from it by about 30
# per cent an element until it meets the interfaces' sizing; either way
# the linear mesh is relocated by gmsh's Relocate3D pass, which moves no
# node of a fault or an interface.

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_section_"))
built = {}
for label, sizing in (("without", None), ("with", 0.025)):
    result = build_layered_mesh(
        MeshSpec(geometry, UniformInterfaces(0.12, 0.3, 0.3), order=1,
                 faults=[EmbeddedFault(fault, name="fault", sizing=sizing)]),
        work / label)
    mesh, card = read_mesh(result.msh_path), manifest.read(result.manifest_path)
    faces = mesh.faces_with(card.fault_record("fault")["attribute"])
    p = mesh.points[faces]
    edge = np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2).mean()
    print(f"{label} the fault's sizing: {result.counts['elements']} elements, "
          f"minSICN {result.validation.min_sicn:.3g}; the fault {len(faces)} "
          f"faces, their edges {edge:.3g} long on average")
    built[label] = (mesh, card, len(faces))

# %% [markdown]
# The section by the plane through the centre spanned by east and up at
# the fault's centroid, which holds the fault's dip direction, drawn
# twice. Above, the cut: the cells as the plane leaves them, coloured by
# layer, exact, the interfaces' faces and the fault's, cut by the same
# plane, the curves they leave in it, which the cut's edges run along;
# but a plane near a corner or an edge of a cell cuts a thin shard from
# it, whatever its shape. Below, the crinkle (`plane_cut(...,
# kind="crinkle")`): the faces of the cells the plane passes through,
# whole, on a stepped surface an element or less in front of the plane,
# with the same curves over them, which their edges do not run along.

# %%
east, north, up = np.moveaxis(UNIT.basis(plane.centroid), -1, 0)
axes = np.stack([east, up])


def draw(ax: plt.Axes, mesh: object, card: object, *, kind: str, lw: float) -> None:
    """The section of a mesh: cells by layer, interfaces, the fault."""
    colours = np.array(list(LAYERS.values()))[mesh.layers(card) - 1]
    mesh_section(ax, plane_cut(mesh.points, mesh.cells[:, :4], np.zeros(3), axes,
                               kind=kind),
                 colours=colours, edgecolor=EDGES, linewidth=lw)
    interfaces = np.isin(mesh.face_attributes,
                         [e["attribute"] for e in card.interfaces])
    mesh_section(ax, plane_cut(mesh.points, mesh.faces[interfaces, :3],
                               np.zeros(3), axes), colours=INK, linewidth=0.9)
    fault_faces = mesh.faces_with(card.fault_record("fault")["attribute"])[:, :3]
    mesh_section(ax, plane_cut(mesh.points, fault_faces, np.zeros(3), axes),
                 colours=FAULT, linewidth=2.5, capstyle="round")


fig, panels = plt.subplots(2, 3, figsize=(12, 9.0))
ends = np.array([plane.cartesian(0.0, v) for v in (-0.5, 0.5)]) @ axes.T
box = np.array([[-0.13, 0.13]] * 2) + ends.mean(axis=0)[:, None]
for (whole, *zooms), kind in zip(panels, ("cut", "crinkle")):
    draw(whole, *built["with"][:2], kind=kind, lw=0.25)
    whole.set_title(f"Normal to the fault's strike, a {kind}")
    whole.add_patch(plt.Rectangle(box[:, 0], *np.diff(box).ravel(), fill=False,
                                  edgecolor="#0b0b0b", linewidth=1.0))
    whole.set_axis_off()
    for ax, label in zip(zooms, ("without", "with")):
        mesh, card, count = built[label]
        draw(ax, mesh, card, kind=kind, lw=0.6)
        ax.set(xlim=box[0], ylim=box[1], xticks=[], yticks=[])
        ax.set_title(f"{label.capitalize()} its sizing: {count} faces, a {kind}")
        x0, y0 = box[0, 0] + 0.015, box[1, 0] + 0.015
        ax.plot([x0, x0 + 0.05], [y0, y0], color="#0b0b0b", lw=2.0,
                solid_capstyle="butt")
        ax.text(x0 + 0.025, y0 + 0.006, "0.05", ha="center", va="bottom")
fig.legend(handles=[Patch(facecolor=c, edgecolor=EDGES, label=n)
                    for n, c in LAYERS.items()]
           + [Line2D([], [], color=INK, lw=0.9, label="interface"),
              Line2D([], [], color=FAULT, lw=2.5, label="fault")],
           loc="lower center", ncol=5, frameon=False)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.06, wspace=0.06,
                    hspace=0.12)
fig.savefig(FIGURES / "meshing_02.png", dpi=120)
shutil.rmtree(work)
