# planetmodel

Spherically layered planetary models: a skeleton of boundary radii, a
geometry that places it in the physical world through one continuous
mapping, fields on each layer, and the meshes that hand a model to a
solver.

The library is being rebuilt in stages, and the tree currently holds the
first two: the skeleton, the geometry and its mappings, a radial
spectral-element mesh, a 2D and 3D mesher with export to MFEM; and
fields on one interval, the model with its units, PREM from its
polynomials, gravity, sampling, and the export of a model's fields to
MFEM. Two sub-packages consume the radial mesh: `planetmodel.loading`
solves the loading and tidal problem and gives Love numbers, and
`planetmodel.randomfield` draws Matern random fields on balls, annuli
and layers. Readers, files and time-dependent rheology follow. The
previous version is kept under `archive/v0.5/` for reference and is not
imported.

## The ideas so far

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

**A field on one interval.** Data on one layer: an interval, a character
(the tensor rank and weight that say how it transforms), a name, and
`evaluate(r, theta, phi, *, frame)` giving components in the local
spherical frame or in Cartesian ones. A discontinuity is two layers
asked separately. Radial fields sit on layer functions whose algebra is
exact on polynomials, so PREM's moduli `rho v^2` are exact polynomials;
analytic formulas, pointwise compositions, and fields pushed forward
through a mapping are fields too.

**One model class.** A geometry with a bag of fields on every layer.
What a name means comes from the shipped vocabulary or the specs a model
is given; a named model such as `prem()` is an instance, and behaviour
shared by groups of models (fluidity, the elastic moduli, gravity) is a
free function of a layer or a model.

**Units in one place.** The model's `Scales` say what one stored unit
is in SI, and `converted` re-expresses the whole model by name, exactly
for polynomials; `G` is read in the model's units. Nothing else names a
unit: radii are numbers, every tolerance is relative, and the meshers
hand the geometry's numbers to gmsh unchanged.

**Two meshers.** `planetmodel.mesh1d` lays Gauss-Lobatto-Legendre
elements along the radius with every skeleton boundary an element
boundary, evaluates a model's fields and gravity on its nodes, and
samples a model on an angular grid. `planetmodel.mesh3d` meshes a
geometry, full or hollow, in 2D or 3D, with shells outside it, writes a
JSON manifest saying what every attribute means, and exports the mesh,
the mapping's displacement and the model's fields to MFEM.

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
from planetmodel import RadialMesh, elastic_moduli, flattening, gravity, is_fluid, prem

m = prem()                                                  # exact polynomials, SI
oc, mantle = m.layer("outer_core"), m.layer("lowermost_mantle")
cmb = m.geometry.interface("cmb").radius
print(oc["rho"](cmb), mantle["rho"](cmb), is_fluid(oc))     # both sides, fluidity
print(elastic_moduli(mantle)(cmb, 0.3, 0.0)[:3, :3] / 1e9)  # Voigt matrix, GPa
print(gravity(m, [cmb, 6371e3]))                            # from the fields, exact

nd = m.nondimensionalised().stretched(flattening(1 / 300, rmax=1.0))
mesh = RadialMesh(nd, ngll=5, drmax=0.05)
print(nd.G, mesh.nodal(nd, "rho").shape, nd.geometry.validity())
```

## Where to go next

- `examples/tutorials/`: eleven walkthroughs, from a skeleton to Love
  numbers and random fields, each a `# %%` script that runs headless.
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
