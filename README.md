# planetmodel: A library for spherical and aspherical planetary models

[![CI](https://github.com/da380/planetmodel/actions/workflows/ci.yml/badge.svg)](https://github.com/da380/planetmodel/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/planetmodel.svg)](https://pypi.org/project/planetmodel/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/planetmodel/badge/?version=latest)](https://planetmodel.readthedocs.io/en/latest/?badge=latest)

Spherically layered planetary models — a planet described as concentric
layers with physical properties on each — and the meshes that hand such
a model to a numerical solver.

With this library you can build a model (PREM out of the box, simple
layered models, a published table of one, or a model type of your own), ask
it for its properties with the discontinuities treated honestly, deform
it with ellipticity or surface topography, and mesh it: a
spectral-element mesh along the radius, or a 2D or 3D finite-element
mesh for [MFEM](https://mfem.org), refined where you need it. A
sub-package, `planetmodel.randomfield`, draws Matérn random fields on
balls, annuli and layers.

## Design

**A skeleton.** The list of boundary radii, increasing, possibly
starting above zero for a hollow shell. It answers geometric questions
(which layer a radius lies in, the intervals between boundaries) and
supports surgery: refine, truncate, hollow, extend, coarsen.

**A geometry.** A skeleton placed in the physical world by one
continuous mapping from the reference (spherical) body to the physical
one, plus names for the layers and interfaces. Ellipticity or surface
topography is therefore a property of the whole model, stated once,
rather than something each field has to know about. The shipped mapping
is the radial stretch `m(X) = (r + h) e_r` driven by any callable
`h(r, theta, phi)`; any object with `__call__`,
`deformation_gradient` and `jacobian` serves as a mapping, and a
geometry checks the invariants (orientation preserved, continuous
across every boundary) when it is built. A mapping declares the radii
where its gradient jumps, its knots, which may be boundaries or lie
inside a layer.

**A field on one layer.** An interval, a name, a character (the tensor
rank and weight that say how the values transform), and
`evaluate(r, theta, phi, *, frame)` giving components in the local
spherical frame or in Cartesian ones. A discontinuity is simply two
layers: ask each side and get each side's answer. Radial fields sit on
an algebra that is exact on polynomials — PREM's moduli `rho v^2` are
exact polynomials, not fits — and fields may be complex-valued, which
is what a model frozen at a frequency holds.

**One model class.** A model is a geometry with named fields on every
layer; what a name like `"rho"` means comes from the shipped vocabulary
or from the specs the model is given. A model type (`PREM`,
`LayeredIsotropicElastic`, or your own) derives from `Model` alone —
there is no hierarchy of model types — and the mixins of
`planetmodel.behaviours` add the standard derivations: the Love
moduli beside the velocities, the elastic tensor and its Voigt average,
gravity, and the constant-Q and Maxwell rheologies frozen at a
frequency.

**One file for a model.** The model file (`planetmodel.modelfile`, the
`netcdf` extra) holds any model, radial or laterally varying, in one
netCDF file with what its numbers mean: the names and units of its
fields, the names of its layers and interfaces, its constants and its
behaviours, so `read_model` gives back a model that does what the one
written did. A text deck is an import and an export beside it: a
mineos or TauP table, or any table of named columns, is brought in as
an ordinary model (`read_deck`, `model_from_deck`), and a model is
written as one for a code that reads one (`sample_deck`, `write_deck`).

**Units in one place.** The model's `Scales` say what one stored unit
of length, mass and time is in SI, and `converted` re-expresses the
whole model, exactly for polynomials. Nothing else names a unit: radii
are numbers, tolerances are relative, and the meshers hand the
geometry's numbers to gmsh unchanged.

**Two meshers.** `planetmodel.mesh1d` lays Gauss–Lobatto–Legendre
elements along the radius with every boundary an element boundary,
and evaluates a model's fields and gravity on the nodes;
`planetmodel.sample` (in `planetmodel.sampling`) evaluates a model's
fields at those radii on an angular grid.
`planetmodel.mesh3d` meshes a geometry, full or hollow, in 2D or 3D,
with buffer shells outside it when a far-field condition needs room;
element sizes are controlled per interface, and refinements
(`NearPoints`, `near_curve`, or any size function of your own) make the
mesh finer towards a coastline, a source region or a station network
without coarsening anything elsewhere. A spherical geometry can also be
meshed with hexahedra and no mesher, as a cubed sphere
(`CubedSphereSpec`, `build_cubed_sphere`): a central cube and six blocks
above its faces that conform to every interface, sized by the number of
cells along an edge and the same sizing rules read radially, or by its
radial levels given outright; its disc is meshed the same way with
quadrilaterals (`SquaredCircleSpec`, `build_squared_circle`). Every mesh
comes with a JSON manifest saying what each attribute means, and exports
to MFEM together with the mapping's displacement and the model's fields.

**Executable contracts.** `planetmodel.testing` holds one `check_*`
function per protocol, and the shipped implementations and your own are
held to the same call.

## Installing

```
pip install planetmodel                     # numpy, scipy, ducc0 and finufft
pip install 'planetmodel[meshing]'          # 2D and 3D meshes via gmsh
pip install 'planetmodel[mfem]'             # export to MFEM, and cubed spheres (PyMFEM)
pip install 'planetmodel[netcdf]'           # the model, surface and fault files (netCDF4)
pip install 'planetmodel[plot]'             # matplotlib, for the figures
pip install 'planetmodel[maps]'             # matplotlib and cartopy, for the maps
pip install 'planetmodel[notebook]'         # ipykernel, to run tutorials cell by cell
```

Python 3.12 or later. ducc0 does the spherical-harmonic transforms and
finufft the non-uniform Fourier transforms of `planetmodel.fourier`;
each ships wheels for Linux on x86_64, macOS on Apple silicon and
Windows, and elsewhere pip builds it from source, which needs a C++17
compiler. `import planetmodel` loads numpy and scipy alone: ducc0,
finufft and every extra are imported where they are used.

## A first look

```python
from planetmodel import PREM, RadialMesh, flattening, gravity

m = PREM()                              # exact polynomials, SI units

# Fields live on layers, so a discontinuity gives two honest answers,
# one from each side.
r_cmb = m.geometry.interface("cmb").radius
print(m.layer("outer_core")["rho"](r_cmb))         # density, core side
print(m.layer("lowermost_mantle")["rho"](r_cmb))   # density, mantle side
print(gravity(m, [r_cmb, 6371e3]))                 # gravity at the CMB and surface

# Reshape the model: non-dimensionalise it, flatten it by 1/300, and
# lay a radial spectral-element mesh over it.
nd = m.nondimensionalised().stretched(flattening(1 / 300))
mesh = RadialMesh(nd, ngll=5, drmax=0.05)
print(mesh.nodal(nd, "rho").shape)      # density at every mesh node
```

## Where to go next

- `examples/tutorials/`: fifteen walkthroughs, from a skeleton to random
  fields, a model from a table, a laterally varying model in a model
  file, pictures, published models and sensitivity kernels, each a
  `# %%` script that runs headless and opens as a notebook, in Google
  Colab from the badges in `examples/README.md`.
- `examples/README.md`: the list of every tutorial and of the shorter
  examples, each a script on one idea (flat models, faults in a mesh,
  cubed spheres, conversions between velocity, temperature and
  viscosity), and the mini-app `examples/miniapps/viscosity3d/`, a
  longer program that builds a 3D viscosity model file.
- `cpp/README.md`: the C++ library, header-only, which reads and writes
  the model, surface and fault files, and reads the mesh manifest, for
  codes written in C++.
- `docs/source/formats/`: the model file, the surface file, the fault
  file and the manifest beside every mesh, described for the codes that
  read them (also on Read the Docs).
- `CONTRIBUTING.md`: the development setup, the hooks, the test
  selections and how a release is made.

## Tests

```
poetry run pytest                                   # the fast suite
poetry run pytest -m "not slow and not network"     # with gmsh and MFEM
poetry run ruff check .
```

## Licence

BSD-3
