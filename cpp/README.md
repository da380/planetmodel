# planetmodel C++

The C++ side of [planetmodel](https://planetmodel.readthedocs.io): a
header-only C++23 library that reads what the Python library writes, and
writes model files that the Python reads. It reads and writes:

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
- **mesh manifests**, the JSON beside every mesh the Python's 3D mesher
  writes (`manifest.hpp`), on a strict JSON reader of its own
  (`json.hpp`);
- **decks**, mineos and GIA tables of knots (`deck.hpp`), each layer's
  columns joined into functions of the radius;
- the units, scales and vocabulary of the Python (`units.hpp`,
  `vocabulary.hpp`).

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

It needs a C++23 compiler (GCC 13 or Clang 18 and later) and CMake 3.24
or later. It fetches its one dependency,
[Interpolation](https://github.com/da380/Interpolation), at a pinned tag.
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
tutorial 15 draws), and read any model file back.

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
