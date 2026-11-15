The fault file
==============

One netCDF-4 file holds one fault: a surface inside a body and the slip
on it, both evaluable anywhere, static or in time.  The surface is one
or more patches, each a map of a parameter square into space; the slip
is the displacement of the hanging wall relative to the footwall, a
vector field on each patch.  The file stores what evaluating the fault
anywhere takes and nothing more: a consumer samples it at points of its
own choosing (a quadrature of its own density, the nodes of its own
mesh), as it samples a model.  It is written by
``planetmodel.faultfile.write_fault`` and read by ``read_fault_file``
(the fault and the file's metadata) and ``read_fault`` (the fault),
which make every check below; ``planetmodel.testing.check_fault_file``
holds a file to the fault it was written from.  The file is CF-styled,
so ncdump, xarray and Panoply open it as it stands.  The Python module
is the definition of the format.  In C++, planetmodel's own library
(:doc:`../cpp`) reads a fault file with the same checks and evaluates
it as the Python does: positions, normals and slip at any point, a
quadrature, and the location of a point near the fault; and it writes
one (``WriteFaultFile``), the file the Python writes from the same
fault.  Either writer writes the file whole or not at all, with the
permissions a plain open for writing would give it.

**Schema string**: the global attribute ``schema``,
``planetmodel.fault/1``.  Readers check it; it is bumped for any change
to what a file may hold, since a reader refuses whatever the schema does
not name (Reading and checking, below).

**Units.**  As for the :doc:`model file <model_file>` and the
:doc:`surface file <surface_file>`: numbers are stored in the writer's
own units under ``scale_length``, ``scale_mass``, ``scale_time`` and
``scale_temperature``, what one stored unit is in SI.  Positions,
lengths, radii and slip are in the length unit, times in the time
unit; angles (latitude, longitude, strike, dip) are degrees.  A
variable's ``dimensions`` attribute holds the integer exponents of
mass, length, time and temperature, and ``units`` is written for CF
tools and never read back.

Conventions
-----------

**Frames.**  A fault's positions are in one frame, named by ``frame``.
A ``geographic`` frame places them on a body whose reference surface is
a sphere of radius ``reference_radius``: a position is (latitude,
longitude, radius), the Cartesian point r (cos lat cos lon, cos lat sin
lon, sin lat), z along the polar axis and x through longitude zero;
the ambient basis at a point is (east, north, up).  A ``cartesian``
frame is local: a position is (x, y, z), x east, y north and z up, the
ambient basis (x, y, z) everywhere, and the reference surface the plane
z = ``reference_surface``.  Its points are those of a flat-layered body
(``planetmodel.cartesian``), whose free surface is z = 0 and whose depth
is -z, so that a fault of a ``cartesian`` file is built into the mesh of
a box as it stands, as a ``geographic`` one is into a sphere's.  A depth
is measured below the reference surface, never below any topography on
it; the file stores radii (or z), not depths.

**The parameter square.**  Every patch maps the square of (u, v), each
in [-1/2, 1/2], into space: u runs along strike and v down dip.  With
x_u and x_v the tangents, the area element is the length of
x_u x x_v, and the unit normal is::

    n = (x_v x x_u) / |x_v x x_u|,

pointing from the footwall into the hanging wall.  The fault basis at
(u, v) is the strike s = x_u / norm(x_u), the up-dip direction a = n x s,
and the normal n: right-handed, with -a the direction of increasing v
within the tangent plane.

**A plane** is a flat rectangle in space, oriented at its centroid by
strike and dip as a finite-fault plane is (Aki & Richards 2002), not a
surface following the sphere: its points away from the centroid lie at
other radii.  With (east, north, up) the frame's basis at the centroid
c (a Cartesian point), the strike phi clockwise from north (in a
Cartesian frame, from +y towards +x) and the dip delta from the
horizontal, down to the right of the strike, in [0, 90]::

    s = sin phi east + cos phi north,   r = cos phi east - sin phi north,
    d = cos delta r - sin delta up,
    x(u, v) = c + u L s + v W d,

L the ``length`` along strike and W the ``width`` down dip, so that
(0, 0) is the centroid, the area element is L W everywhere and the
fault basis is (s, -d, n), n = sin delta r + cos delta up.  A dip of
zero is a horizontal plane (a sill), its strike orienting u.

**A structured patch** holds control positions at the nodes of a
uniform ``nu`` by ``nv`` grid over the square, corners included, node
(i, j) at u = -1/2 + i / (nu - 1), v = -1/2 + j / (nv - 1), in the
frame's coordinates.  They are turned into Cartesian points and
interpolated coordinate by coordinate by its ``interpolation``:
``linear``, the tensor-product B-spline of degree one (bilinear on each
cell), or ``cubic``, the tensor-product interpolating cubic B-spline
with not-a-knot ends (its knots the nodes, the second and the
second-to-last left out, each end repeated four times; at least four
nodes each way), scipy's ``make_interp_spline`` along u and then along
v.  Being Cartesian, the rule is indifferent to the date line and the
poles.  The area element must be positive at every control node.

**The slip** is held and interpolated in fault components: (strike-slip,
dip-slip, opening), dip-slip positive up dip (a reverse fault) and
opening along the normal, so that a rake lambda and a magnitude m are
(m cos lambda, m sin lambda, 0) and a dyke or a sill is a patch like
any other.  Each patch has its own slip grid of ``su`` by ``sv`` and a
``rule``: ``cell``, constant on each of su by sv equal cells of the
square, the values at the cells' centres (a point on a boundary between
cells reads the cell above it, the last cell closed at 1/2); or
``linear`` or ``cubic``, the values at the nodes of a uniform grid,
corners included, read between them by the rules above.  Each
component is read by the rule, and the vector at a point is rebuilt
from the fault basis there::

    slip(u, v) = c_strike s(u, v) + c_dip a(u, v) + c_open n(u, v).

So a slip without opening lies in the surface at every point of every
patch; a uniform (strike-slip, dip-slip) on a plane is one vector
everywhere; on a curved patch the rake is measured from the u-lines
point by point.  Slip need not match where patches meet, and nothing
tapers it at a patch's edge: both are the writer's business.

**Time.**  One time axis serves every patch.  A ``static`` fault has
one slip, read at any date.  A ``node`` fault has the coordinate
``time(time)``, strictly increasing, and a slip per node; between nodes
it is read by ``time_interpolation``, as a ``node`` field of the
surface file is: ``linear`` interpolates between the two nodes about
the date, ``previous`` reads the latest node not after it, and ``none``
serves the stored nodes alone.  Outside the span a reader of a
``linear`` or ``previous`` fault refuses the date, or holds the nearer
end where its caller asks; a ``none`` fault refuses any date that is not
one of its nodes, held or not.  A date within a tolerance of a node is
that node, and one that close beyond an end of the span is that end:
the tolerance is relative to the scale of the time axis, the largest of
its span and the magnitudes of its first and last times, ``1e-12`` of
it unless the caller gives another (zero is exact matching), by the
surface file's rule (:doc:`surface_file`, Reading as a history).  A ``static`` fault takes no date, and refuses one.  The
slip at a
node is always the cumulative slip, the total displacement of the
hanging wall since the fault's own start: afterslip is a series of
growing slices, and a rupture switched on is a static file placed on
the consumer's clock (or a zero slice and the rupture, read with
``previous``).  Supports on intervals or at instants, which the surface
file has, are not part of this version and are refused by name.

Global attributes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - attribute
     - meaning
   * - ``schema``
     - ``planetmodel.fault/1``
   * - ``Conventions``
     - ``CF-1.8``
   * - ``source``
     - the writing code and version; required
   * - ``created``
     - when the file was written, ISO 8601; required
   * - ``title``
     - a name for the file's contents, where one was given
   * - ``citation``
     - the works the contents cite, one a line, where there are any
   * - ``meta_<key>``
     - the writer's free metadata, strings or numbers (the meaning of
       the dates, say: the format has no calendar)
   * - ``scale_length``, ``scale_mass``, ``scale_time``,
       ``scale_temperature``
     - what one stored unit is in SI
   * - ``frame``
     - ``geographic`` or ``cartesian``
   * - ``reference_radius``
     - ``geographic`` only, and required there: the radius of the
       reference sphere, positive
   * - ``reference_surface``
     - ``cartesian`` only, and required there: the height z of the
       reference plane, finite (often zero)
   * - ``time_support``
     - ``static`` or ``node``
   * - ``time_interpolation``
     - ``none`` for ``static``; ``linear``, ``previous`` or ``none`` for
       ``node``
   * - ``patches``
     - the number of patch groups, an integer of at least one

A ``node`` file has, at the root, the dimension ``time`` and the
coordinate ``time(time)`` with ``long_name``, ``units`` and
``dimensions`` (0, 0, 1, 0); a ``static`` file has no dimension and no
variable at the root.

Patches
-------

Patch ``k`` is the group ``patch_<k>``, k = 0 to ``patches`` - 1, in
order; a file has exactly these groups and no others, and a patch
group has no groups of its own.  Its attributes: ``kind`` (``plane`` or
``structured``) and, optionally, ``name`` (a segment's label, free
text).

**A plane** states its parameters as attributes: ``centroid`` (three
numbers in the frame's coordinates: latitude in [-90, 90], longitude
and a positive radius, or x, y and z), ``strike`` and ``dip`` (degrees,
the dip in [0, 90]), ``length`` along strike and ``width`` down dip
(positive, in the length unit).  Its dimensions are the slip's alone.

**A structured patch** has the dimensions ``nu`` and ``nv`` (at least
two each way under ``linear``, four under ``cubic``), the attribute
``interpolation`` (``linear`` or ``cubic``), and its control positions
as three float64 variables over ``(nu, nv)``, CF auxiliary coordinates
with their own units:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - frame
     - variables
     - attributes
   * - ``geographic``
     - ``latitude``, ``longitude``, ``radius``
     - ``long_name``; ``units`` (``degrees_north``, ``degrees_east``,
       the length unit); ``standard_name`` for the angles,
       ``dimensions`` for the radius
   * - ``cartesian``
     - ``x``, ``y``, ``z``
     - ``long_name``, ``units``, ``dimensions``

Every position is finite, latitudes lie in [-90, 90] and radii are
positive.

**The slip**, in every patch: the dimensions ``su``, ``sv`` and
``component`` (always 3, opening written as zeros where there is
none), and the float64 variable ``slip`` over the support's leading
dimension (none for ``static``, ``time`` for ``node``) then
``(su, sv, component)``, with the attributes

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - attribute
     - meaning
   * - ``rule``
     - ``cell`` (su by sv cells, at least one each way), ``linear`` (su
       by sv nodes, at least two) or ``cubic`` (at least four)
   * - ``components``
     - ``strike_slip dip_slip opening``, the order of the last axis
   * - ``units``, ``dimensions``
     - the length unit; (0, 1, 0, 0)
   * - ``long_name``
     - ``displacement of the hanging wall relative to the footwall``

Its values are real and finite: slip is defined everywhere on its
patch.

Reading and checking
--------------------

A reader checks, in this order, and refuses the file naming the first
fault: the schema; that every global attribute is one the format names
(or ``meta_<key>``); ``source`` and ``created``; the four scales,
positive; the frame and its reference, the other frame's reference
refused; the time support (``interval`` and ``instant`` refused by
name, as not part of this version) and its rule; ``patches``; that the
root holds the ``time`` dimension and variable for ``node`` and nothing
for ``static``, the axis finite and strictly increasing; that the
groups are exactly ``patch_0`` to ``patch_<patches - 1>``; and in each
patch, its kind, that every attribute, dimension and variable is one
its kind has, a plane's parameters or a structured patch's rule, node
counts and finite positions, the slip's dimensions, ``components``,
``rule`` and node counts, finite values, and last a structured patch's
positive area element at every control node.  The variables' own
attributes are held to the lists above, ``_FillValue``, netCDF's own,
admitted on any variable.

Refusing whatever the schema does not name is deliberate: an extension
of the format (a triangulated patch, slip on intervals, non-uniform
grids) is a new schema, never an attribute a version-1 reader would
silently pass over.

The Python refusal is a ``ValueError`` reading ``malformed fault file
<path>: <where>: <what>``; the C++ one a ``planetmodel::Error`` reading
``<path>: <where>: <what>``, with the same ``<where>`` and ``<what>``.
``<where>`` names an attribute of the root by its name, a patch's by
``patch_<k>.<name>``, a variable by ``patch_<k>/<name>`` and its
attribute by ``patch_<k>/<name>.<attribute>``.

What a consumer does with it
----------------------------

**Sampling.**  At any (u, v) of any patch: the Cartesian point and the
point in the frame's coordinates, the tangents, the normal, the fault
basis and the area element; and at any date the slip, in fault
components, in the ambient basis at the point or in Cartesian
components.

**Quadrature.**  Each patch cut into n_u by n_v equal cells of the
square, with p Gauss-Legendre points each way in each cell (p = 1 is
the midpoint rule); a point's weight is the area it stands for, the
area element there times its weight in the square.  A plane's area
comes out exactly at any density, and for a smooth integrand on a
smooth patch the error falls as the cell size to the power 2p.  The
densities that suit a consumer's own discretisation are its business.
A fault's quadrature gives each point's patch beside it, and the slip
either at one date (``Fault.quadrature``, ``Fault::Quadrature``) or for
every slice at once, the geometry formed once
(``Fault.slice_quadrature``, ``Fault::SliceQuadrature``): a consumer
that superposes slices in time forms its sources a slice at a time and
combines them by the weights a date reads (``Fault.rows``,
``Fault::Rows``), which is the slip ``quadrature`` gives at that date,
number for number.

**The moment.**  The file holds no rigidity and no moment.  At each
quadrature point a consumer contracts its own elastic tensor at the
point with the slip and the normal and multiplies by the weight; for
an isotropic medium, with Lame parameters lambda and mu there::

    w (lambda (du . n) I + mu (du n + n du)),

so the same file on another model gives another moment.

**Locating points.**  A consumer's mesh nodes on a fault sit at
physical positions; ``locate`` gives each one's nearest point of the
fault (its patch and (u, v)), the distance signed positive on the
hanging-wall side, and whether it lies inside the square or is held on
an edge (and which), and ``slip_at`` reads the slip there, refusing a
point farther from the fault than a stated distance.

Deliberately absent
-------------------

No quadrature points or point cloud (a cloud is a quadrature of the
surface, a consumer's product); no cell counts for a consumer, no mesh
size; no rigidity, moment, magnitude or moment tensor; no rake or
other derived components; no Cartesian copies of the positions; no
depths (the radius is stored, the reference beside it); no tapering;
no calendar (the dates' meaning is the writer's ``meta_<key>``); no
rupture velocity or hypocentre (a kinematic rupture is slices in time,
or the consumer's business).

Examples
--------

A one-plane static rupture, in metres: 20 by 8 cells of slip at a rake
of 85 degrees.

.. code-block:: text
   :caption: rupture.nc

   netcdf rupture {

   // global attributes:
           :schema = "planetmodel.fault/1" ;
           :Conventions = "CF-1.8" ;
           :source = "planetmodel 1.6" ;
           :created = "2026-11-13T10:00:00+00:00" ;
           :scale_length = 1. ;
           :scale_mass = 1. ;
           :scale_time = 1. ;
           :scale_temperature = 1. ;
           :frame = "geographic" ;
           :reference_radius = 6371000. ;
           :time_support = "static" ;
           :time_interpolation = "none" ;
           :patches = 1 ;

   group: patch_0 {
     dimensions:
       su = 20 ;
       sv = 8 ;
       component = 3 ;
     variables:
       double slip(su, sv, component) ;
           slip:_FillValue = NaN ;
           slip:long_name = "displacement of the hanging wall relative to the footwall" ;
           slip:units = "m" ;
           slip:dimensions = 0, 1, 0, 0 ;
           slip:rule = "cell" ;
           slip:components = "strike_slip dip_slip opening" ;

     // group attributes:
           :kind = "plane" ;
           :centroid = 38.3, 142.4, 6351000. ;
           :strike = 195. ;
           :dip = 14. ;
           :length = 400000. ;
           :width = 160000. ;
     } // group patch_0
   }

Two patches with afterslip, in kilometres and years: a plane named
``main`` with its slip bilinear between 10 by 5 nodes, and a curved
splay of 9 by 6 control positions with a bicubic slip on 4 by 4
nodes, three slices linear in time.

.. code-block:: text
   :caption: afterslip.nc

   netcdf afterslip {
   dimensions:
       time = 3 ;
   variables:
       double time(time) ;
           time:_FillValue = NaN ;
           time:long_name = "time" ;
           time:units = "31557600 s" ;
           time:dimensions = 0, 0, 1, 0 ;

   // global attributes:
           :schema = "planetmodel.fault/1" ;
           :Conventions = "CF-1.8" ;
           :source = "planetmodel 1.6" ;
           :created = "2026-11-13T10:00:00+00:00" ;
           :scale_length = 1000. ;
           :scale_mass = 1. ;
           :scale_time = 31557600. ;
           :scale_temperature = 1. ;
           :frame = "geographic" ;
           :reference_radius = 6371. ;
           :time_support = "node" ;
           :time_interpolation = "linear" ;
           :patches = 2 ;

   group: patch_0 {
     dimensions:
       su = 10 ;
       sv = 5 ;
       component = 3 ;
     variables:
       double slip(time, su, sv, component) ;
           slip:_FillValue = NaN ;
           slip:long_name = "displacement of the hanging wall relative to the footwall" ;
           slip:units = "1000 m" ;
           slip:dimensions = 0, 1, 0, 0 ;
           slip:rule = "linear" ;
           slip:components = "strike_slip dip_slip opening" ;

     // group attributes:
           :kind = "plane" ;
           :name = "main" ;
           :centroid = 38.3, 142.4, 6351. ;
           :strike = 195. ;
           :dip = 14. ;
           :length = 400. ;
           :width = 160. ;
     } // group patch_0

   group: patch_1 {
     dimensions:
       nu = 9 ;
       nv = 6 ;
       su = 4 ;
       sv = 4 ;
       component = 3 ;
     variables:
       double latitude(nu, nv) ;
           latitude:_FillValue = NaN ;
           latitude:long_name = "latitude of the control positions" ;
           latitude:units = "degrees_north" ;
           latitude:standard_name = "latitude" ;
       double longitude(nu, nv) ;
           longitude:_FillValue = NaN ;
           longitude:long_name = "longitude of the control positions" ;
           longitude:units = "degrees_east" ;
           longitude:standard_name = "longitude" ;
       double radius(nu, nv) ;
           radius:_FillValue = NaN ;
           radius:long_name = "radius of the control positions" ;
           radius:units = "1000 m" ;
           radius:dimensions = 0, 1, 0, 0 ;
       double slip(time, su, sv, component) ;
           slip:_FillValue = NaN ;
           slip:long_name = "displacement of the hanging wall relative to the footwall" ;
           slip:units = "1000 m" ;
           slip:dimensions = 0, 1, 0, 0 ;
           slip:rule = "cubic" ;
           slip:components = "strike_slip dip_slip opening" ;

     // group attributes:
           :kind = "structured" ;
           :name = "splay" ;
           :interpolation = "cubic" ;
     } // group patch_1
   }
