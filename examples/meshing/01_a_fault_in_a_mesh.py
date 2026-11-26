# %% [markdown]
# # A fault in a mesh
#
# A fault is given where it physically is, and the mesh is built on the
# reference ball and moved by the geometry's mapping afterwards, so the
# mesher builds in the fault's preimage under the mapping: once the nodes
# are moved, the fault's faces lie on the fault. Here a small layered
# ball with a relief over the fault, a plane thrust that reaches the
# surface, and a curved back-thrust splaying off it into its hanging
# wall. The mesh conforms to both and keeps every node whole: each fault
# is a sheet of boundary elements of its own attribute, every face
# oriented from the footwall into the hanging wall, and the manifest
# names the curves where it ends (its buried tip, its free edge on the
# surface, the junction where the two meet, and which fault ends on
# which). The cells of each fault's hanging wall that touch its slip,
# its collar, carry element attributes of their own after the layers',
# which the manifest lists by layer and by fault. Splitting the nodes is
# the solver's: it takes each fault's hanging wall from those attributes,
# and the rules behind them (which of a face's two cells is the plus
# one, which edges are pinned, which cells are in which wall) are
# `planetmodel.mesh3d.manifest`'s (and the C++ library's). Examples 02 to
# 06 draw such meshes; 05 and 06 the collars.
#
# Needs: meshing
# Network: none
# Run: poetry run python examples/meshing/01_a_fault_in_a_mesh.py

# %%
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton
from planetmodel.fault import Fault, GeographicFrame, Plane, Slip, StructuredPatch
from planetmodel.mesh3d import (EmbeddedFault, MeshSpec, UniformInterfaces,
                                build_layered_mesh, manifest, read_mesh)

UNIT = GeographicFrame(1.0)
LAT, LON = 20.0, 30.0

# %% [markdown]
# A unit ball of three layers, its surface raised by a smooth bump over
# the fault (0.03 at its crest) and a gentle swell towards the poles,
# both decaying as r squared with depth.

# %%
def relief(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    c = (np.cos(theta) * np.cos(np.deg2rad(90.0 - LAT)) + np.sin(theta)
         * np.sin(np.deg2rad(90.0 - LAT)) * np.cos(phi - np.deg2rad(LON + 3.0)))
    return r ** 2 * (0.03 * np.exp(-(1.0 - c) / 0.01) + 0.01 * np.cos(theta) ** 2)


geometry = Geometry(Skeleton([0.0, 0.55, 0.8, 1.0]),
                    layer_names=["core", "mantle", "lithosphere"],
                    interface_names=["cmb", "base", "surface"]).with_mapping(
    RadialStretch(CallableDisplacement(relief, name="relief"), rmax=1.0))

# %% [markdown]
# The thrust is a plane striking north and dipping 50 degrees east, the
# middle of its up-dip edge at the reference surface; under the relief
# the surface stands above that edge, and the mesher carries the fault
# across it and trims it there. The splay is a structured patch through control
# positions: its down-dip edge a line on the thrust, 0.06 below its
# centre down dip, and from there it rises east into the thrust's
# hanging wall, steepening from 40 to 65 degrees, to a buried tip. Each
# carries a uniform reverse slip.

# %%
dip, width = 50.0, 0.3
thrust = Plane.at_depth(UNIT, LAT, LON, 0.5 * width * np.sin(np.deg2rad(dip)),
                        strike=0.0, dip=dip, length=0.5, width=width,
                        slip=Slip.from_rake(90.0, 1.0))
east, north, up = np.moveaxis(UNIT.basis(thrust.centroid), -1, 0)
u = np.linspace(-0.5, 0.5, 5)                 # along strike, to the south
rising = np.linspace(1.0, 0.0, 6)             # up from the bottom, as v runs down dip
angle = np.deg2rad(40.0 + 25.0 * rising)
turn = np.deg2rad(25.0)
foot = thrust.cartesian(-u * 0.3 / thrust.length, np.full(u.size, 0.2))
controls = foot[:, None] + 0.12 * (
    ((np.sin(angle) - np.sin(np.deg2rad(40.0))) / turn)[None, :, None] * east
    + ((np.cos(np.deg2rad(40.0)) - np.cos(angle)) / turn)[None, :, None] * up)
splay = StructuredPatch(UNIT, UNIT.coordinates(controls),
                        slip=Slip.from_rake(90.0, 0.5))
faults = {"thrust": Fault([thrust]), "splay": Fault([splay])}

# %% [markdown]
# The mesh, of order 2, refined towards both faults by the size on them,
# 0.05: by `EmbeddedFault`'s rule the size grows from it to the
# background's by about 30 per cent an element, and the linear mesh is
# relocated by gmsh's Relocate3D pass, which moves no node of a fault or
# an interface. The validation report holds the faults' own checks
# (every face a face of two cells, every face pointing into the hanging
# wall) and the warnings beside the usual ones; the manifest, of the
# schema of a mesh with faults, gains one record a fault, and each layer
# the list of element attributes its cells carry, its collars' after its
# own.

# %%
work = Path(tempfile.mkdtemp(prefix="planetmodel_faults_"))
size = 0.05
result = build_layered_mesh(
    MeshSpec(geometry, UniformInterfaces(0.12, 0.3, 0.3), order=2,
             faults=[EmbeddedFault(f, name=n, sizing=size) for n, f in faults.items()]),
    work / "faulted")
print(f"{result.counts['elements']} elements, {result.counts['nodes']} nodes")
print(result.validation)
card = manifest.read(result.manifest_path)
print(card.schema)
for entry in card.layers:
    print(f"layer {entry['name']}: element attributes {entry['attributes']}")
for record in card.faults:
    print(json.dumps(record))

# %% [markdown]
# The mesh as a solver reads it, as plain arrays (`read_mesh`; an MFEM
# code reads the same from its `mfem::Mesh`), its reference positions
# moved by the mapping. Every node of a fault's faces, the curved
# elements' mid-edge nodes among them, is located on the true fault; the
# thrust's nodes that `locate` finds beyond its up-dip edge lie on the
# part carried across the surface, and are left out. The slip is read
# at a few nodes.

# %%
mesh = read_mesh(result.mesh_path)
x = result.mapping(mesh.points)
for record in card.faults:
    fault = faults[record["name"]]
    nodes = np.unique(mesh.faces_with(record["attribute"]))
    found = fault.locate(x[nodes], strict=False)
    carried = ~found.inside & found.on_v_low & (record["free_edge"] is not None)
    worst = float(np.abs(found.distance[~carried]).max())
    print(f"{record['name']}: {nodes.size} nodes, {int(np.count_nonzero(carried))} "
          f"carried across the surface; the rest within {worst:.2g} of the fault "
          f"({worst / size:.2g} of an element; the pull-back's offset "
          f"{record['offset']:.2g})")
    slip = fault.slip_at(x[nodes[~carried][:3]], max_distance=size)
    print("  slip (east, north, up) at three of them:",
          np.array2string(slip, precision=3, suppress_small=True).replace("\n", ""))

# %% [markdown]
# The two sides of a face, by the vertex order the mesh keeps: of the
# two cells sharing the splay's face nearest its centre, the plus one is
# on the side (v1 - v0) x (v2 - v0) points to. Located on the splay,
# its centre lies on the hanging-wall side (a positive distance) and the
# other's on the footwall side.

# %%
splay_faces = mesh.faces_with(card.fault_record("splay")["attribute"])[:, :3]
middle = splay.cartesian(0.0, 0.0)
face = splay_faces[np.argmin(np.linalg.norm(x[splay_faces].mean(axis=1) - middle,
                                            axis=1))]
pair = mesh.face_cells(face[None])[0]
corners = mesh.cells[pair][:, :4]
plus = int(manifest.plus_cell(x, face[None], corners[None])[0])
sides = faults["splay"].locate(x[corners].mean(axis=1)).distance
print(f"face {face.tolist()}: plus cell {pair[plus]} at {sides[plus]:+.3g}, minus cell "
      f"{pair[1 - plus]} at {sides[1 - plus]:+.3g} from the splay")

# %%
shutil.rmtree(work)
