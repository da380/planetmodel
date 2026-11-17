# planetmodel C++

The C++ side of [planetmodel](https://planetmodel.readthedocs.io): a
header-only C++23 library that reads what the Python library writes, and
writes files that the Python reads. It reads and writes:

- **model files**, a model in one netCDF-4 file (`modelfile.hpp`):
  - reading gives the numbers as stored, each variable on each layer as
    a field you can evaluate anywhere by the file's own interpolation
    rules, and the mapping;
  - writing builds a file from your own fields;
- **surface files**, fields on one spherical surface, static or in time
  (`surfacefile.hpp`): reading gives the numbers as stored, held to the
  structure, writing builds a file from your own fields
  (`SurfaceField`), whole or not at all, and `SurfaceHistory` serves a
  field's values at any date by the field's own time rule, with the
  Python's arithmetic;
- **fault files**, a fault's patches of surface and the slip on them,
  static or in time (`faultfile.hpp`): `ReadFaultFile` makes the Python
  reader's checks and gives a `Fault` (`fault.hpp`) that evaluates as
  the Python's does, to rounding: positions, normals, the fault basis
  and the area element at any (u, v), the slip at any (u, v) and date
  in fault, ambient or Cartesian components, a quadrature with the slip
  at a date (`Fault::Quadrature`) or of every slice, the geometry formed
  once (`Fault::SliceQuadrature`), and the location of physical points
  near it with the slip there (`Fault::Locate`, `Fault::SlipAt`);
  `WriteFaultFile` writes a fault, built in code or read, as the Python
  writes it, and `FaultSlip::FromRake` makes a patch's slip from a rake
  and a magnitude as the Python's `Slip.from_rake` does; the support and
  the time rule are the enumerations `TimeSupport` and `TimeRule`
  (`time.hpp`), and a date is read as `DateOptions` say: a date within
  `TIME_RTOL` (or the `time_rtol` given) times the scale of the time
  axis of a stored time is that time, the scale being the largest of the
  axis's span and the magnitudes of its first and last times, and a
  date outside the span is refused or, by `Extrapolation::Hold`, held
  at the nearer end (a `none` slip refusing any date not stored either
  way);
- **mesh manifests**, the JSON beside every mesh the Python's 3D mesher
  writes (`manifest.hpp`), on a strict JSON reader of its own
  (`json.hpp`), with the faults built into a mesh, the element
  attributes of their hanging walls, and the three rules of their faces
  (`PlusCell`, `FaultCurves`, `FaultWalls`);
- **decks**, mineos, Maxwell, prony and burgers tables of knots and
  TauP's `.nd` and `.tvel` tables (`deck.hpp`), read and written as the
  Python does, with the names their headers or labels give, a prony
  deck's number of branches (`PronyBranches`), and the Python's checks
  of a prony or burgers deck's values, each layer's columns joined into
  functions of the radius, through the values or through their
  logarithms (by default `MAXWELL_KINDS`, `PRONY_KINDS` and
  `BURGERS_KINDS`: a Maxwell deck's viscosity, a prony deck's relaxation
  times and a burgers deck's two viscosities by `log-linear`, the
  strengths and the Kelvin modulus fraction by `linear`); a burgers
  deck's Burgers body becomes its two Prony branches by `BurgersTerms`,
  and two branches the body by `BurgersParameters` (`prony.hpp`), as the
  Python's `conversion.prony` forms them;
- the units, scales and vocabulary of the Python (`units.hpp`,
  `vocabulary.hpp`);
- **flat models**, the files of `planetmodel.cartesian`, a body layered
  in depth over a box or a rectangle, in the namespace
  `planetmodel::cartesian` under `planetmodel/cartesian/` (the section
  on flat models below): the flat model file and the flat surface file,
  read, evaluated and written; the manifest of a flat mesh, with the
  faces of the box and a fault's face edge; and decks of depths.

The library knows no solver and no mesh library. Each code keeps its own
glue: opening the mesh, sampling fields onto its own discretisation.

## Getting it

The library is fetched from the Python repository at a release tag. Its
version is the Python package's (`planetmodel::Version()`):

```cmake
include(FetchContent)
FetchContent_Declare(planetmodel
  GIT_REPOSITORY https://github.com/da380/planetmodel.git
  GIT_TAG        v<version>
  SOURCE_SUBDIR  cpp)
FetchContent_MakeAvailable(planetmodel)
target_link_libraries(my_code PRIVATE planetmodel::planetmodel)
```

It also installs: `cmake --install` puts the headers and a CMake package
in the prefix (with Interpolation and NumericConcepts beside them), and a
code then finds the same target with `find_package(planetmodel)`. The
install rules are on as a subproject too (`PLANETMODEL_INSTALL`), so that
a code installing a library that links planetmodel can export it.

It needs a C++23 compiler (GCC 13 or Clang 18 and later) and CMake 3.24
or later. It fetches its one dependency,
[Interpolation](https://github.com/da380/Interpolation), at the revision
`PLANETMODEL_INTERPOLATION_TAG` names (`main` between releases; a release
commit pins it).
The model file needs netCDF-C, found through its CMake package or
pkg-config. The option `PLANETMODEL_WITH_NETCDF` is on by default, and
configuring then fails, saying so, if netCDF-C is not found. A code that
reads only manifests and decks can turn the option off.

## Reading a model file

```cpp
#include <planetmodel/modelfile.hpp>

const auto file = planetmodel::ReadModelFile("earth.nc");

// a scalar field on layer 3, its value and its derivatives in r, theta
// and phi at a point; theta is the colatitude, in radians
const auto vs = file.FieldOn(3, "vsv").AsScalar();
const auto value = vs.Value(r, theta, phi);
const auto [dr, dtheta, dphi] = vs.Gradient(r, theta, phi);

// a field of any rank, its components in the frame you name
const auto stress = file.FieldOn(3, "stress").Evaluate(r, theta, phi,
                                                       planetmodel::Frame::Cartesian);

// the mapping from the reference body to the physical one
const auto u = file.Mapping().Displacement({x, y, z});

// a field on an interface (a shape kernel, a boundary's shift), a
// function of the angles
const auto shift = file.SurfaceFieldOn(1, "K_d670").Value(theta, phi);
```

The numbers are in the file's own units. `file.GetScales()` gives the
size of one stored unit in SI, so `Factor(dimensions)` converts a value
to SI.

A field holds the interpolants it was built with and nothing else that
changes, so it is cheap to copy and safe to evaluate from many threads.
A typical code builds the fields it wants, samples them onto its own
mesh, and discards them.

## Reading and writing a fault file

```cpp
#include <planetmodel/faultfile.hpp>

namespace pm = planetmodel;
const auto file = pm::ReadFaultFile("rupture.nc");
const auto& fault = file.fault;

// a quadrature of 20 by 8 cells, two Gauss-Legendre points each way, the
// slip at a date: points, normals, weights (areas) and Cartesian slip
const auto q = fault.Quadrature(20, 8, 2, /*date=*/1.5);
for (auto i = std::size_t{0}; i < q.Size(); ++i) {
  // the moment is yours, with your own moduli at q.points[i]
}

// a date past the end of the time axis, held at its last slice
const auto held = fault.Quadrature(20, 8, 2, /*date=*/9.0,
                                   {.extrapolation = pm::Extrapolation::Hold});

// a mesh node near the fault: its patch, (u, v) and signed distance,
// and the slip there in (east, north, up)
const auto where = fault.Locate({x, y, z});
const auto slip = fault.SlipAt(nodes, /*max_distance=*/1e3,
                               {.date = 1.5, .basis = pm::SlipBasis::Ambient});

// the geometry once and the slip of every slice, combined at any date by
// the weights the date reads: the quadrature's slip at that date
const auto slices = fault.SliceQuadrature(20, 8, 2);
const auto at = slices.SlipOf(fault.Rows(1.5));
const auto late = slices.SlipOf(fault.Rows(9.0, {.extrapolation = pm::Extrapolation::Hold}));

// written back, or a fault built in code written: the Python's file
pm::WriteFaultFile("copy.nc", file);
```

A structured patch is read by the same tensor-product B-splines as the
Python's (scipy's knots, the coefficients solved here), so the two
languages agree to rounding wherever they are asked.  The Python's
constructors of large patches, `StructuredPatch.swept` (down a dip
profile from a `GreatCircle`, `SmallCircle` or `Polyline` trace) and
`StructuredPatch.from_depths` (from a grid of depths), are Python-only:
a code here builds a structured patch from its control positions
(`FaultPatch::Structured`), or reads one from a fault file the Python
wrote.

## Flat models

A flat model is layered in depth below one free surface, over a box, or
over a rectangle, the cross-section of a body invariant along y. Its
files are schemas of their own (`planetmodel.model/3`,
`planetmodel.surface/2`, the manifest's schema 7), read by the headers
under `planetmodel/cartesian/`, which also write the model file, the surface
file and the decks, and leave the spherical library as it is.
Each reader refuses the other's files by their schema and names the
reader that takes them.

Points are (x, y, z) with z up and the free surface z = 0 in the
reference body. The layered coordinate is the depth, positive down, and
the layers and interfaces are numbered from the surface down: interface
k is the top of layer k, and the base is a face of the box, not an
interface. Components are in the local frame (up, x, y), the same at
every point, or in the Cartesian one. A field of depth alone is the flat
radial field: in a name shared with the spherical side (`IsRadial`, the
attributes `radial_layers` and `radial_interpolation`), "radial" means of
the layered coordinate alone, depth here.

```cpp
#include <planetmodel/cartesian/modelfile.hpp>

namespace flat = planetmodel::cartesian;
const auto file = flat::ReadModelFile("box.nc");

// a scalar field on layer 1, at (depth, x, y) or at a point (x, y, z)
const auto vs = file.FieldOn(1, "vs").AsScalar();
const auto value = vs.Value(depth, x, y);
const auto same = vs.ValueAt({x, y, -depth});
const auto [ddepth, dx, dy] = vs.Gradient(depth, x, y);

// a field of any rank, its components in the frame you name
const auto stress = file.FieldOn(1, "stress").Evaluate(depth, x, y, flat::Frame::Cartesian);

// the mapping, a vertical stretch: the height a reference point rises by
const auto mapping = file.Mapping();
const auto height = mapping.Height(depth, x, y);
const auto physical = mapping({x, y, -depth});

// a field on an interface, a function of (x, y)
const auto shift = file.SurfaceFieldOn(1, "shift").Value(x, y);
```

- **The lateral rules** are doubly periodic over the domain's extent, its
  bounds widened by its padding: `linear` and `cubic`
  (`cartesian/rules.hpp`). A field stating `fourier`, a Fourier series of
  its grid, is the Python's alone: its numbers are read, and building it
  as a field is refused by name, as `harmonic` is on the sphere.
- **A rectangle** (`Domain::Rectangle`) has the one lateral axis x. Its
  fields are evaluated at (depth, x, y) all the same, y not being read,
  and a vector still has its three components.
- **Writing** is the spherical writer's with the flat types: a
  `flat::ModelFile` of a skeleton of depths and a `flat::Domain`, parts
  added from the surface down on `flat::Grid`s over the domain, fields
  from `flat::LayerSource`s (`Depth`, `Lateral`, or values at
  `PointsOf(layer)`), the mapping as the heights of surfaces
  (`SetStack`) or as its upward displacement (`SetDisplacement`), heights
  positive up, and `flat::WriteModelFile`.
- **Surface files** (`cartesian/surfacefile.hpp`): `ReadSurfaceFile`,
  `WriteSurfaceFile` and `SurfaceHistory`, which serves a field's values
  at a date and the field there as a function of (x, y).
- **Manifests** (`cartesian/manifest.hpp`): `cartesian::MeshManifest`
  reads schema 7, the layers and interfaces by depth, the faces of the
  box (`Faces()`, `FaceAttributes()`), and the faults with their
  `face_edge`, the curve where a fault leaves the box through its base
  or a side, its sides free there. `cartesian::FaultCurves` tells that
  edge from the tip by the faces of the box; the other rules of a
  fault's faces are the spherical ones.
- **Decks of depths** (`cartesian/deck.hpp`): `ELASTIC()`, `MAXWELL()`,
  `PRONY()` and `BURGERS()` with `depth` for `r`, read and written by
  `ReadDeck` and `WriteDeck` as the Python does, the layers and
  interfaces named from the surface down (`DepthNames`).
- **The fault file** serves as it is: its `cartesian` frame is the flat
  model's points.

A code that may be built against a release without the flat headers
tests for them with `__has_include(<planetmodel/cartesian/modelfile.hpp>)`.

## Writing a model file

A code that holds fields of its own builds a `ModelFile` and writes it:

1. Make the `ModelFile` from its skeleton (the boundary radii), its
   scales and its tolerance.
2. Cut its layers into parts. Each part is a run of layers laid out in
   elements of Gauss-Lobatto-Legendre nodes, on an angular grid of its
   own.
3. Add its fields, each layer's values from a `LayerSource`. A source is
   one of:
   - a function of the radius;
   - a function of the point, in a frame the code names;
   - the values at the points `PointsOf(layer)` lists.

The library evaluates the sources and stores the components in the
format's frames, with every attribute the format asks for. It checks the
file as the readers will before writing anything:

```cpp
namespace pm = planetmodel;
auto file = pm::ModelFile(pm::Skeleton({3480e3, 5701e3, 6371e3}));
file.AddPart(0, 1, pm::ElementEdges(file.GetSkeleton(), 0, 1, 200e3), 5,
             pm::GaussLegendre(16));
file.AddField("viscosity", {{0, pm::LayerSource::Radial(lower)},
                            {1, pm::LayerSource::Lateral(upper)}},
              {.lateral = pm::LateralRule::Cubic});
pm::WriteModelFile("viscosity.nc", file);
```

A field that lives on an interface rather than in a layer goes on a grid
set once for all of them, from a source per interface, evaluated at the
interface's radius:

```cpp
file.SetSurfaceGrid(pm::GaussLegendre(64));
file.AddSurfaceField("K_d670", spec, {{1, pm::LayerSource::Lateral(kernel)}});
```

A field the vocabulary does not name, a sensitivity kernel say, is given
with a `FieldSpec` of its own: its character, its dimensions and what it
means.

The file is written under a temporary name of its own beside the target
and renamed into place once it is complete. A failed write leaves no
file, and an existing file at the path stays as it was. The programs in
`examples/` write a model file, write the file a code computing
sensitivity kernels writes (`write_kernel.cpp`, which the Python's
tutorial 15 draws), and read any model file back; `write_flat_model.cpp`
and `read_flat_model.cpp` do the same for a box under topography.

## Versions and features

`planetmodel/version.hpp` gives the version as a string,
`planetmodel::Version()` and `PLANETMODEL_VERSION`, and as the integers
`PLANETMODEL_VERSION_MAJOR`, `PLANETMODEL_VERSION_MINOR` and
`PLANETMODEL_VERSION_PATCH`, which the CMake target defines for every
code it links, with `PLANETMODEL_VERSION_AT_LEAST(major, minor, patch)`
for the preprocessor:

```cpp
#include <planetmodel/version.hpp>
#if PLANETMODEL_VERSION_AT_LEAST(1, 6, 0)
// what 1.6.0 added
#endif
```

Between releases the version stays the last release's, so a feature is
tested by its own macro, which the header providing it defines:

| macro | header | what it provides |
| --- | --- | --- |
| `PLANETMODEL_HAS_FAULT_SLICE_QUADRATURE` | `fault.hpp` | `Fault::SliceQuadrature` and `FaultSliceQuadrature::SlipOf`; a fault's date read by `DateOptions` (`Fault::Rows`, `SlipVector`, `Quadrature`); `FaultSlip::FromRake` |
| `PLANETMODEL_HAS_FAULT_FILE_WRITER` | `faultfile.hpp` | `WriteFaultFile` beside `ReadFaultFile` |
| `PLANETMODEL_HAS_VISCOELASTIC_DECKS` | `deck.hpp` | `PRONY()` and `BURGERS()` beside `MAXWELL()`, their kinds `PRONY_KINDS` and `BURGERS_KINDS`, `PronyColumns`, `PronyBranches`, `PronyHeader`, `BurgersHeader`; `MAXWELL_KINDS` joining the viscosity by `log-linear` |
| `PLANETMODEL_HAS_LOG_KINDS` | `radial.hpp` | the kinds through the logarithms, `KnotKind::LogCubic`, `LogNatural` and `LogLinear` |
| `PLANETMODEL_HAS_LOG_RULES` | `rules.hpp` | the model file's radial rules in the logarithm, `RadialRule::LogPolynomial` and `LogLinear` |
| `PLANETMODEL_HAS_BURGERS_TERMS` | `prony.hpp` | `BurgersTerms` and `BurgersParameters` |

A header a code may lack altogether is tested first with
`__has_include`:

```cpp
#if __has_include(<planetmodel/fault.hpp>)
#include <planetmodel/fault.hpp>
#endif
#ifdef PLANETMODEL_HAS_FAULT_SLICE_QUADRATURE
// the slice quadrature
#endif
```

## Errors

Reading, checking and building throw `planetmodel::Error`, naming the
file and the place in it (a variable, a JSON path, a line). Evaluation
throws only for a point outside the field's layer by more than the
file's tolerance.

## Building it here

```sh
poetry run python cpp/tests/make_fixtures.py        # the fixtures, from the Python
cmake -S cpp -B build/cpp -DPLANETMODEL_WARNINGS_AS_ERRORS=ON
cmake --build build/cpp -j 8
PLANETMODEL_REQUIRE_FIXTURES=1 ctest --test-dir build/cpp -j 4
cmake -S cpp -B build/cpp -DPLANETMODEL_BUILD_DOCS=ON && cmake --build build/cpp -t planetmodel_docs
```

The tests check the C++ against fixtures that the Python writes. The
reference is built by Doxygen, and every warning fails the build.
