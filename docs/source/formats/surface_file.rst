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
file against it.

**Schema string**: the global attribute ``schema``,
``planetmodel.surface/1``.  Readers check it; it is bumped only for an
incompatible change.

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
