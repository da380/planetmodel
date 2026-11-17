The surface file
================

One netCDF-4 file holds named scalar fields on one spherical surface
and one angular grid, static or depending on time: an ice-thickness
history, a present-day topography, the surface kernels of a
measurement.  It is written by ``planetmodel.surfacefile.write_surface``
and read by ``read_surface_file`` (the numbers and metadata, nothing
interpolated) and by any netCDF reader: the file is CF-styled, so
xarray, Panoply and ncview open it as it stands.  The Python module is
the definition of the format, and ``validate_structure`` there checks a
file against it.  In C++, planetmodel's own library (:doc:`../cpp`)
reads a surface file whole, held to the structure, and writes one from
a code's own fields.  Fields on a horizontal plane of a flat body
(``planetmodel.cartesian``) are a file of their own schema, described
under `Flat surfaces`_ below.

**Schema string**: the global attribute ``schema``,
``planetmodel.surface/1``.  Readers check it; it is bumped only for an
incompatible change.  A flat surface's file is ``planetmodel.surface/2``,
which a reader of ``/1`` refuses by its schema; ``read_surface`` reads
both, ``read_surface_file`` the numbers of a spherical file and
``cartesian.surfacefile.read_surface_file`` those of a flat one, each
refusing the other kind by name.

**Units.**  As for the :doc:`model file <model_file>`: values are
stored in the writer's own units under ``scale_length``,
``scale_mass``, ``scale_time`` and ``scale_temperature``, a field's
``dimensions`` attribute holds the integer exponents of mass, length,
time and temperature where the writer declares them, and ``units`` is
written for CF tools and never read back.  Times are stored in the
file's time unit.

Global attributes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - attribute
     - meaning
   * - ``schema``
     - ``planetmodel.surface/1``
   * - ``Conventions``
     - ``CF-1.8``
   * - ``source``
     - the writing code and version; required
   * - ``created``
     - when the file was written, ISO 8601; required
   * - ``title``
     - a name for the file's contents, where one was given
   * - ``scale_length``, ``scale_mass``, ``scale_time``,
       ``scale_temperature``
     - what one stored unit is in SI
   * - ``radius``
     - the radius of the surface the fields live on, in the file's
       units; absent, the fields live on the outer surface of whatever
       model the reader pairs the file with
   * - ``grid_kind``, ``grid_lmax``
     - the angular grid's family and band, as a model-file part states
       them
   * - ``poles``
     - ``north south``, present where any field carries polar values
   * - ``citation``
     - the works the contents cite, one a line, where there are any
   * - ``meta_<key>``
     - the writer's free metadata, strings or numbers

The grid and the axes
---------------------

One grid per file and one surface per file.  The CF coordinate
variables ``latitude`` (degrees north, decreasing) and ``longitude``
(degrees east, in [0, 360)) hold the grid, with
``colatitude_weights(latitude)`` where the grid has weights — the same
representation a model-file part uses.

Time enters through up to two coordinates and one derived dimension:

* ``time(time)``: strictly increasing time nodes in the file's units.
* ``instant(instant)``: a second strictly increasing set, for fields
  tied to observation times rather than the nodes.
* the dimension ``interval``, present where a field needs it, of size
  one less than ``time``: interval ``k`` spans ``time[k]`` to
  ``time[k + 1]``.

How the stored times map onto a calendar (kiloyears before present,
say) is the writer's provenance — ``meta_<key>`` attributes — not the
format's.

Fields
------

One float64 variable per name, NaN marking the points a field does not
cover.  The ``time_support`` attribute says how a field sits in time,
and fixes its dimensions:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - ``time_support``
     - dimensions
     - holds
   * - ``static``
     - ``(latitude, longitude)``
     - one function of position
   * - ``node``
     - ``(time, latitude, longitude)``
     - a history at the time nodes
   * - ``interval``
     - ``(interval, latitude, longitude)``
     - one function per interval between adjacent nodes
   * - ``instant``
     - ``(instant, latitude, longitude)``
     - one function per observation time

Each field carries ``long_name``, ``units``, ``dimensions`` (where
declared), ``lateral_interpolation`` (as a model-file field states it),
and ``time_interpolation``: ``linear`` or ``previous`` for a ``node``
field a reader may sample between nodes, and ``none`` otherwise —
``none`` always, for the other supports, which must not be resampled in
time.  Where the grid lacks either pole, a field may carry
``name_poles``, north then south, on its leading time shape; a reader
without it falls back to its lateral rule.  Values are real; complex
fields are refused.

Reading and checking
--------------------

``read_surface_file`` validates the structure and gives the numbers and
metadata back as a ``SurfaceFile``; ``validate_structure`` checks a
file alone, naming the first thing wrong; and
``planetmodel.testing.check_surface_file`` holds a file to what it was
written from.

Reading as a history
--------------------

``read_surface`` gives a ``SurfaceHistory``, the file as a function of
time: ``values(name, date)`` serves a field's values on the grid at a
date by the field's own time rule — a ``node`` field interpolated
between its nodes (``linear``), read from the latest node not after
the date (``previous``), or served at its stored nodes alone
(``none``); an ``interval`` field from the interval holding the date,
the last closed at its end; an ``instant`` field at its stored
instants alone; a ``static`` field at any date.  Outside its span a
``linear`` or ``previous`` ``node`` field and an ``interval`` field
refuse the date, or hold their nearest value where the caller asks
(``extrapolation="hold"``); a ``none`` ``node`` field and an
``instant`` field refuse any date that is not one of their stored
times, held or not.

A date within a tolerance of a stored time is that time: the
tolerance is ``time_rtol`` times the scale of the axis, the largest of
its span (its last time less its first) and the magnitudes of its first
and last times, ``time_rtol`` being ``TIME_RTOL = 1e-12`` unless the
caller gives another, and zero matching exactly.  An axis of one time
has the magnitude of that time as its scale; an axis whose one time is
zero has a scale of zero, and a date matches it exactly.  So a date a
rounding error from a node reads that node alone under every rule
(``none`` and ``instant`` serve it, ``previous`` and ``interval`` read
from it, ``linear`` gives it weight one), and a date that close beyond
an end of the span is that end and is not refused; the nearest stored
time within the tolerance is taken, the earlier on a tie.  A reader
that converts its own clock to the file's units therefore meets the
stored times without matching them bit for bit: its rounding is
relative to the size of the times it forms, which the magnitudes in the
scale allow for however short the span (a time stored at -999 on an
axis of span 1, reached as -998.99999999999511, is read as -999).  ``function(name, date)`` is the same
field as a ``GriddedFunction`` by its stated lateral rule, stored
polar values carried along.  The C++ ``SurfaceHistory`` reads a file
by the same arithmetic (``TimeRowsAt``, its tolerance ``TIME_RTOL``),
so the two languages serve identical values.  A flat surface's file is
read and written in C++ by ``planetmodel::cartesian``'s
``ReadSurfaceFile`` and ``WriteSurfaceFile``
(``planetmodel/cartesian/surfacefile.hpp``), whose ``SurfaceHistory``
serves a field's values by the same arithmetic and the field at a date
as a function of ``(x, y)`` by the ``linear`` or the ``cubic`` rule;
each of the two readers refuses the other's files by their schema.

Flat surfaces
-------------

A flat surface file holds fields on one horizontal plane of a body of
``planetmodel.cartesian`` (a topography, an ice-thickness history, the
kernels of a measurement on the free surface), written by
``cartesian.surfacefile.write_surface`` and read by
``cartesian.surfacefile.read_surface_file`` and ``read_surface``.  Its
schema is ``planetmodel.surface/2``, and everything above holds for it
(the attributes ``source``, ``created``, ``title``, the scales,
``citation`` and ``meta_<key>``; the axes ``time``, ``instant`` and
``interval``; each field's supports, time rules, ``dimensions``,
``long_name`` and ``units``; the history's dates, tolerance and
extrapolation, by the same code) except where the surface differs:

- ``coordinate_system`` is ``flat``; ``depth``, where the writer states
  it, is the depth of the plane in the reference body in the file's
  units, and absent, the fields lie on the free surface;
- the lateral domain is stated as a flat model file states it
  (``domain``, ``domain_bounds``, ``domain_padding``;
  :doc:`model_file`, its flat models), and the grid is the CF
  coordinate variables ``x(x)`` and, over a box, ``y(y)``, its nodes
  over the domain's extent;
- a field is ``name(x, y)`` (``name(x)`` over a section) after its
  support's leading dimension, and states one of the flat lateral rules,
  ``linear``, ``cubic`` or ``fourier``;
- a flat grid has no poles: polar values, ``grid_kind``, ``grid_lmax``,
  colatitude weights, ``latitude``, ``longitude`` and ``radius`` are a
  spherical file's, refused on writing and on reading.

``read_surface`` gives a flat file's history, whose ``function(name,
date)`` is a ``cartesian.GriddedFunction`` on the file's grid by the
field's lateral rule.  An ice history over a box padded along x
(abbreviated from ``ncdump -h``):

.. code-block:: text

   netcdf flat_surface {
   dimensions: x = 10 ; y = 6 ; time = 3 ;
   variables:
       double x(x) ;   :units = "m" ; :axis = "X" ;
       double y(y) ;   :units = "m" ; :axis = "Y" ;
       double time(time) ;
       double topography(x, y) ;
           :units = "m" ; :time_support = "static" ;
           :time_interpolation = "none" ; :lateral_interpolation = "cubic" ;
       double ice_thickness(time, x, y) ;
           :units = "m" ; :time_support = "node" ;
           :time_interpolation = "linear" ; :lateral_interpolation = "linear" ;
   // global attributes:
       :schema = "planetmodel.surface/2" ; :Conventions = "CF-1.8" ;
       :source = "planetmodel 1.6.0" ; :created = "2026-11-15T20:09:54+00:00" ;
       :scale_length = 1. ; ... ;
       :coordinate_system = "flat" ; :domain = "box" ;
       :domain_bounds = 0., 400000., 0., 300000. ; :domain_padding = 50000., 0. ;
   }
