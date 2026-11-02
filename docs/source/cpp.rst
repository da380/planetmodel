The C++ library
===============

The repository holds a C++ library beside the Python one, in ``cpp/``. It
reads what the Python writes and writes files that the Python reads, so
that a solver written in C++ can take a model from planetmodel and hand
its own fields back. It is header-only C++23, and its version is
the Python package's.

It reads and writes four of the Python's files:

* **Model files** (:doc:`formats/model_file`). Reading gives the numbers
  as stored, and each variable on each layer as a field that can be
  evaluated anywhere by the file's own interpolation rules, together with
  the mapping. Writing builds a file from a code's own fields: the code
  gives each layer's values as a function or as values at the points the
  library lists, and the library forms the file and checks it as the
  readers will.
* **Surface files** (:doc:`formats/surface_file`). Reading gives the
  fields on the surface, static or in time, each served at any date by
  its own time rule; writing builds a file from a code's own fields.
* **Mesh manifests** (:doc:`formats/mesh_manifest`). The library reads
  these on a strict JSON reader of its own.
* **Decks**, the mineos and Maxwell tables of knots
  (:doc:`formats/maxwell_deck`) and TauP's ``.nd`` and ``.tvel`` tables,
  with the names of the layers and interfaces their headers or labels
  give. Each layer's columns are joined into
  functions of the radius by the rules the Python uses, and a deck is
  written as the Python writes it, character for character.

The units, scales and vocabulary are the Python's, restated in C++. Tests
hold the two sides equal, against fixtures that the Python writes.

A C++ code fetches the library from this repository at a release tag,
with CMake's ``FetchContent``:

.. code-block:: cmake

   include(FetchContent)
   FetchContent_Declare(planetmodel
     GIT_REPOSITORY https://github.com/da380/planetmodel.git
     GIT_TAG        v<version>
     SOURCE_SUBDIR  cpp)
   FetchContent_MakeAvailable(planetmodel)
   target_link_libraries(my_code PRIVATE planetmodel::planetmodel)

The model file needs netCDF-C. A code that reads only manifests and
decks can build without it, by setting ``PLANETMODEL_WITH_NETCDF=OFF``.

Reading a field from a model file looks like this:

.. code-block:: cpp

   #include <planetmodel/modelfile.hpp>

   const auto file = planetmodel::ReadModelFile("earth.nc");
   const auto vs = file.FieldOn(3, "vsv").AsScalar();
   const auto value = vs.Value(r, theta, phi);   // theta the colatitude

The `reference <cpp-reference/index.html>`_ documents every name in the
library. Its main page says how to read and write a model file, and the
programs in ``cpp/examples`` do both.
