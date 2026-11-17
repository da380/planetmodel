The C++ library
===============

The repository holds a C++ library beside the Python one, in ``cpp/``. It
reads what the Python writes and writes files that the Python reads, so
that a solver written in C++ can take a model from planetmodel and hand
its own fields back. It is header-only C++23, and its version is
the Python package's, which a C++ preprocessor can test
(``PLANETMODEL_VERSION_AT_LEAST``); each feature added between releases
has a macro of its own, defined by the header providing it, which the
reference's main page lists.

It reads and writes the Python's files:

* **Model files** (:doc:`formats/model_file`). Reading gives the numbers
  as stored, and each variable on each layer as a field that can be
  evaluated anywhere by the file's own interpolation rules, together with
  the mapping. Writing builds a file from a code's own fields: the code
  gives each layer's values as a function or as values at the points the
  library lists, and the library forms the file and checks it as the
  readers will.  A flat model's file (``planetmodel.model/3``) is read
  and written by the flat side of the library (below); the spherical
  reader refuses it by its schema, naming that reader.
* **Surface files** (:doc:`formats/surface_file`). Reading gives the
  fields on the surface, static or in time, each served at any date by
  its own time rule; writing builds a file from a code's own fields.  A
  flat surface's file (``planetmodel.surface/2``) is read and written by
  the flat side of the library (below), and refused here by its schema.
* **Fault files** (:doc:`formats/fault_file`). Reading makes the Python
  reader's checks and gives the fault, which evaluates as the Python's
  does, to rounding: positions, normals, the fault basis and the area
  element anywhere on a patch, the slip at any point and date, a
  quadrature with the slip at a date or of every slice, and the location
  of a physical point near the fault (a mesh node on it) with the slip
  there. Writing makes the file the Python writes from the same fault,
  a fault built in code or read.
* **Mesh manifests** (:doc:`formats/mesh_manifest`). The library reads
  these on a strict JSON reader of its own, the faults built into a mesh
  and the element attributes of their hanging walls among them, and
  states the three rules of a fault's faces: which of a face's two cells
  is on the hanging wall, which edges are the fault's tip, free edge and
  junction and where it ends on another, and which cells are in its
  hanging wall and its footwall.  A flat mesh's manifest (schema 7) is
  read by the flat side of the library (below), and refused here by its
  schema.
* **Decks**, the mineos, Maxwell, prony and burgers tables of knots
  (:doc:`formats/maxwell_deck`, :doc:`formats/prony_deck`,
  :doc:`formats/burgers_deck`) and TauP's ``.nd`` and ``.tvel`` tables,
  with the names of the layers and interfaces their headers or labels
  give, a prony deck's number of branches, and the checks the Python
  makes of a prony or burgers deck's values, with its messages. Each
  layer's columns are joined into functions of the radius by the rules
  the Python uses, and a deck is written as the Python writes it,
  character for character. A burgers deck's Burgers body is turned into
  its two Prony branches, and two branches into the body, by the
  Python's arithmetic.

The units, scales and vocabulary are the Python's, restated in C++. Tests
hold the two sides equal, against fixtures that the Python writes.

**Flat models.**  The files of ``planetmodel.cartesian``, a body layered
in depth over a box or a rectangle, are read and written by the headers
under ``planetmodel/cartesian/``, in the namespace
``planetmodel::cartesian``, beside the spherical library, which they
leave as it is; each reader refuses the other's files by their schema,
naming the reader that takes them.  Points are ``(x, y, z)`` with ``z``
up, the layered coordinate is the depth, positive down, and components
are in the local frame (up, x, y) or the Cartesian one:

* the flat model file (``cartesian/modelfile.hpp``,
  :doc:`formats/model_file`, its flat models): ``ReadModelFile`` and a
  field evaluated at ``(depth, x, y)`` or at a point ``(x, y, z)`` by the
  doubly periodic ``linear`` and ``cubic`` rules (``fourier``, a Fourier
  series, is the Python's alone and refused by name, as ``harmonic`` is
  on the sphere), over a box and over a rectangle; the mapping, the
  vertical stretch, as the height a point rises by; the fields on
  interfaces; and ``WriteModelFile`` from a code's own fields, as on the
  sphere;
* the flat surface file (``cartesian/surfacefile.hpp``,
  :doc:`formats/surface_file`, its flat surfaces), read, written and
  served in time, a field at a date also as a function of ``(x, y)``;
* the flat mesh's manifest, schema 7 (``cartesian/manifest.hpp``,
  :doc:`formats/mesh_manifest`): the layers and interfaces by depth, the
  faces of the box, and the faults with their face edge, where a fault
  leaves the box through its base or a side, which
  ``cartesian::FaultCurves`` tells from its tip by the faces of the box;
* decks of depths (``cartesian/deck.hpp``): the elastic, Maxwell, prony
  and burgers formats with ``depth`` for ``r``, named from the surface
  down, read and written as the Python does, character for character.

The fault file needs no flat counterpart: its ``cartesian`` frame is the
flat model's points.

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

and from a flat one like this:

.. code-block:: cpp

   #include <planetmodel/cartesian/modelfile.hpp>

   namespace flat = planetmodel::cartesian;
   const auto box = flat::ReadModelFile("box.nc");
   const auto vs = box.FieldOn(1, "vs").AsScalar();
   const auto value = vs.Value(depth, x, y);           // depth positive down
   const auto lifted = box.Mapping()({x, y, -depth});  // the point, z up

The `reference <cpp-reference/index.html>`_ documents every name in the
library. Its main page says how to read and write a model file, and the
programs in ``cpp/examples`` do both.
