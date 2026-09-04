# planetmodel

Spherically layered planetary models: reference bodies and their
fields, the mapping that makes them aspherical, and the meshes that
solvers consume.

## Three ideas and two meshers

**A skeleton of layers carrying fields.** A model is a list of layers.
Each layer is an interval of radius, the fields it holds by name, and
whether it is solid, fluid or vacuum. A field belongs to one layer; the
body-wide field `body["rho"]` is a view assembled from the layers'
pieces, defined on the layers that hold one and refusing radii
elsewhere by name.

**Fields with character and dimensions, static or not.** Every field
carries a tensor character, which says how it transforms under a
mapping, and physical dimensions, which say what its numbers are. A
field may depend on frequency or on time as well as on position: a
viscoelastic modulus is a frequency-dependent field built by a law from
a layer's static fields, and the law survives as a record the file
formats copy.

**A mapping from the reference body to the physical one.** The
canonical state of a model is its fields on a spherically symmetric
reference body plus a mapping to the physical body. Fields cross by
the push-forward their character dictates, and a physical quantity is
pulled back into the reference state on construction. A body with no
mapping is spherical, so the one-dimensional model is the general case
with a trivial map.

Model classes say what a body guarantees: `ElasticModel`,
`ViscoelasticModel`, `ViscousModel`. Each checks its fields layer by
layer and survives every surgery.

Two meshers deliver a body to solvers. `planetmodel.mesh1d` builds
radial spectral-element meshes. `planetmodel.mesh3d` builds meshes of a
layered body with gmsh, either the physical body or the reference body
plus its mapping, writes a manifest saying what every attribute means,
and exports to MFEM. Spectral codes read a netCDF file laid out on
their own radial and angular nodes.

## Installing

```
pip install planetmodel                     # numpy and scipy only
pip install 'planetmodel[netcdf]'           # the netCDF model file
pip install 'planetmodel[meshing]'          # 3D meshes via gmsh
pip install 'planetmodel[mfem]'             # export to MFEM (PyMFEM)
pip install 'planetmodel[plot]'             # matplotlib
```

Python 3.12 or later. Nothing optional is imported by `import
planetmodel`; each extra is imported where it is used.

## Fifteen lines

```python
import numpy as np
from planetmodel import prem, AngularGrid, write_model, read_model

earth = prem()                                   # a ViscoelasticModel
core = earth.layer(1)                            # the outer core
print(core.state, core.field_names[:4])          # fluid ('rho', 'vpv', ...)

r = np.array([3.5e6, 5.0e6, 6.3e6])              # radii in metres
print(earth["rho"](r))                           # density, kg m^-3
print(earth.elastic_moduli(r)[0, 3, 3])          # L on the CMB side, Pa

moduli_100s = earth.moduli_at(2 * np.pi / 100.0)  # complex tensor at 100 s
print(moduli_100s(r)[0, 3, 3])

sample = earth.sample(AngularGrid.gauss_legendre(8))
write_model(earth, sample, "prem.nc")            # planetmodel.model/1
again, _ = read_model("prem.nc")
print(type(again).__name__, again.viscoelastic_moduli.domain)
```

## Where to go next

- `examples/tutorials/`: application-first scripts to read, from
  `01_prem.py` to a layered rheology and a mesh for MFEM.
- `examples/reference/`: one concept per script, with the checks that
  say what holds.
- `docs/overview.md`: the referential framework and the design.
- `docs/conventions.md`: coordinates, units, frames, the time
  convention, the field vocabulary.
- `docs/formats/`: the netCDF model file and the mesh manifest, written
  from the reader's side.
- `docs/extending.md`: how to add a field type, a law, a model class, a
  mapping, a reader or a topography.
- `docs/nondimensionalisation.md`: scales, and where numbers change.

The executable contracts in `planetmodel.testing` are the definition of
each protocol: run `check_field` on your field and you know whether the
library can use it.

## Licence

BSD-3

