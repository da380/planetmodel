# planetmodel

Spherically layered planetary models: a skeleton of boundary radii, a
geometry that places it in the physical world through one continuous
mapping, fields on each layer, and the meshes that hand a model to a
solver.

The library is being rebuilt in stages, and the tree currently holds the
first: the skeleton, the geometry and its mappings, a radial
spectral-element mesh, and a 2D and 3D mesher with export to MFEM.
Fields, model classes, readers and files follow. The previous version is
kept under `archive/v0.5/` for reference and is not imported.

## The ideas in this stage

**A skeleton.** A strictly increasing list of boundary radii, possibly
starting above zero for a shell. It answers geometric questions
(intervals, where a radius lies, with the two sides of a boundary left to
the caller) and supports surgery: refine, truncate, hollow, extend,
coarsen.

**A geometry.** A skeleton, one mapping from the reference ball to the
physical body, and the names of layers and interfaces. The mapping must
be orientation-preserving, continuous, and kinked only on skeleton
boundaries; a geometry checks those invariants when it is built. The
shipped mapping is the radial stretch `m(X) = (r + h) e_r` driven by any
callable `h(r, theta, phi)`, with closed forms for its deformation
gradient, Jacobian, validity, inverse and linearisation; any object with
`__call__`, `deformation_gradient` and `jacobian` is a mapping.

**Numbers, not units.** Nothing inside names a unit. Radii are numbers,
every tolerance is relative, and the meshers hand the geometry's numbers
to gmsh unchanged. What the numbers mean is decided by whoever builds a
concrete model.

**Two meshers.** `planetmodel.mesh1d` lays Gauss-Lobatto-Legendre
elements along the radius with every skeleton boundary an element
boundary. `planetmodel.mesh3d` meshes a geometry, full or hollow, in 2D or
3D, with shells outside it, writes a JSON manifest saying what every
attribute means, and exports the mesh and the mapping's displacement to
MFEM.

**Executable contracts.** `planetmodel.testing` holds one `check_*`
function per protocol; the shipped implementations and yours are held to
the same call.

## Installing

```
pip install planetmodel                     # numpy and scipy only
pip install 'planetmodel[meshing]'          # 2D and 3D meshes via gmsh
pip install 'planetmodel[mfem]'             # export to MFEM (PyMFEM)
pip install 'planetmodel[plot]'             # matplotlib, for the figures
pip install 'planetmodel[notebook]'         # ipykernel, to run tutorials cell by cell
```

Python 3.12 or later. Nothing optional is imported by `import
planetmodel`.

## Twelve lines

```python
import numpy as np
from planetmodel import Geometry, Skeleton, RadialMesh, validity_lattice

sk = Skeleton([0.0, 0.19, 0.55, 0.99, 1.0])                # an Earth-like skeleton
flat = lambda r, t, p: -(r / 300.0) * 0.5 * (3 * np.cos(t) ** 2 - 1)
g = Geometry(sk, layer_names=["inner_core", "outer_core", "mantle", "crust"],
             interface_names=["icb", "cmb", "moho", "surface"]).stretched(flat)
print(g.interface("cmb"), g.validity())

X = np.array([[0.0, 0.0, 0.8]])
print(g.mapping(X), g.mapping.jacobian(X))                 # where the point goes

mesh = RadialMesh(g, ngll=5, drmax=0.05)                    # a radial mesh
print(mesh, np.bincount(mesh.layer))
```

## Where to go next

- `examples/tutorials/`: four walkthroughs, from a skeleton to a mesh
  read back by MFEM, each a `# %%` script that runs headless.
- `docs/formats/mesh_manifest.md`: the manifest beside every mesh, from
  the consumer's side.

## Tests

```
poetry run pytest -m "not gmsh and not slow" -q     # the fast suite
poetry run pytest -m "not slow" -q                  # with gmsh and MFEM
poetry run ruff check .
```

## Licence

BSD-3
