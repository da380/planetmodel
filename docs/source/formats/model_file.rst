The model file
==============

One netCDF-4 file holds a planetary model: its skeleton of boundary
radii, its fields at the nodes of a radial layout times an angular
grid, and the mapping that places it in the physical world.  It is
written by ``planetmodel.modelfile.write_model`` and read by
``read_model`` (a ``Model``), by ``read_model_file`` (the numbers and
metadata, nothing interpolated) and by any netCDF reader: the file is
CF-styled, so xarray (``open_datatree`` for the whole file,
``open_dataset(path, group="part_0")`` for one part), Panoply and
ncview open it as it stands.  The Python module is the definition of
the format, and ``validate_structure`` there checks a file against it.
In C++, planetmodel's own library (:doc:`../cpp`) reads the file, its
fields evaluated by the rules below, and writes one from a code's own
fields, checked as the readers check it.  A flat model
(``planetmodel.cartesian``, a box or a rectangle layered in depth) is
written to a file of its own schema, described under `Flat models`_
below, the rest of this page holding for it as stated there.

**Schema string**: the global attribute ``schema``.  A file in which no
variable states a radial rule in the logarithm (``log-polynomial``,
``log-linear``, below) is ``planetmodel.model/1``; a file in which one
does is ``planetmodel.model/2``, so that a reader that knows ``/1``
alone refuses it by its schema before it reads a value by a rule it
does not know.  Readers check it, and a reader of both refuses a ``/1``
file stating a rule in the logarithm and a ``/2`` file stating none; a
number is added only for an incompatible change.  A flat model's file is
``planetmodel.model/3``, which a reader of the spherical schemas refuses
by its schema; ``read_model`` reads all three, giving a ``Model`` or a
``cartesian.Model`` by what the file states, ``read_model_file`` the
numbers of a spherical file and ``cartesian.modelfile.read_model_file``
those of a flat one, each refusing the other kind by name.

**Units.**  A file holds the model in the model's own units and says
what they are: one stored unit of length, mass, time and temperature is
``scale_length``, ``scale_mass``, ``scale_time`` and
``scale_temperature`` in SI (metres, kilograms, seconds, kelvin).
Writing never converts, and reading gives the model back in the file's
scales.  A field's ``dimensions`` attribute holds the integer exponents
of mass, length, time and temperature, in that order; its value in SI
is the stored number times the product of the scales to those powers.
The ``units`` attribute (``m s-1`` in SI, ``3865.3 m s-1`` under
geophysical scales, say) is written for CF tools and is never read
back.

Global attributes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - attribute
     - meaning
   * - ``schema``
     - ``planetmodel.model/1``, or ``planetmodel.model/2`` where a
       variable states a radial rule in the logarithm
   * - ``Conventions``
     - ``CF-1.8``
   * - ``source``
     - the writing code and version, stamped by the writers (absent in
       files written before it was)
   * - ``created``
     - when the file was written, ISO 8601, stamped by the writers
       (absent in older files)
   * - ``title``
     - ``planetmodel model`` and the model's name
   * - ``name``
     - the model's name, where it has one
   * - ``scale_length``, ``scale_mass``, ``scale_time``, ``scale_temperature``
     - what one stored unit is in SI
   * - ``rtol``
     - the geometry's tolerance, relative to the skeleton's span: radii
       closer than it are one radius, a point within it of a layer lies on
       the layer; ``1e-9`` where a file states none
   * - ``constant_<name>``
     - each constant of the model in its units (``constant_G``)
   * - ``constant_<name>_dimensions``
     - its dimensions, the integer exponents of mass, length, time and
       temperature, so that a reader can convert it
   * - ``constant_<name>_meaning``
     - what it is, where the model says
   * - ``citation``
     - the works the model cites, one a line, where it cites any: the
       papers of the models and data sets it was built from
   * - ``meta_<key>``
     - whatever the writer was given as ``meta``

The skeleton
------------

At the root, with dimensions ``layer``, ``interface`` and ``bound``
(two):

- ``layer_bounds(layer, bound)``: the inner and outer radius of each
  layer, from the centre outward;
- ``layer_name(layer)`` and ``interface_name(interface)``: strings,
  empty where the model gave no name;
- ``interface_radius(interface)``: the interfaces are the boundaries
  between layers and the outer boundary, plus the inner boundary of a
  hollow model;
- ``layer_part(layer)``: the part, ``part_<k>``, that holds the layer.

Parts
-----

The layers are cut into parts, contiguous runs of whole layers, each a
netCDF-4 group ``part_0``, ``part_1``, ... from the centre, with an
angular grid of its own; a model written on one grid is one part.  Its
attribute ``layers`` lists the layers it holds, and, where its grid
lacks a pole, the dimension ``pole`` (two) runs over the poles its
attribute ``poles`` names, ``north south``.  A part holds, for its
layers alone:

**The radial layout.**  Dimension ``node``: the Gauss-Lobatto-Legendre
nodes of every radial element, flattened element by element, so that
the boundary between two elements, and every interface, is two nodes at
one radius, each holding its own side's value; ``radius(node)``, an
auxiliary coordinate since it repeats; ``element_layer(element)``, the
layer of each element; ``element_bounds(element, bound)``; and the
attribute ``ngll``, the nodes per element.  Node ``j`` of element ``e``
is node ``e * ngll + j``, at the GLL point ``xi_j`` mapped onto the
element's bounds.

**The angular grid.**  ``latitude`` (degrees north, decreasing, since
colatitude increases) and ``longitude`` (degrees east, in [0, 360)) are
the CF coordinate variables; colatitude is ``90 - latitude`` in
degrees.  The attributes ``grid_kind`` (``gauss_legendre``,
``driscoll_healy``, ``equiangular`` or ``custom``) and ``grid_lmax``
(the band the grid resolves, where it has one) and the variable
``colatitude_weights(latitude)`` (quadrature weights against
``sin(theta) dtheta``, where it has them) say what the grid is.  Either
pole may be a node: a Driscoll-Healy grid has the north pole, an
equiangular grid either none or both.

**Fields.**  One variable per field:

- ``name(node)`` when the field depends on the radius alone on every
  layer of the part that holds it;
- ``name(node, latitude, longitude)`` otherwise;
- with trailing axes for components: ``component`` (3) for a vector,
  ``voigt`` (6) for a symmetric rank-2 tensor in Voigt form, and the
  pairs ``voigt_row``, ``voigt_column`` or ``component_row``,
  ``component_column`` for rank 4 and for a rank-2 tensor without
  Voigt form.  Components are in the local spherical frame ``(e_r,
  e_theta, e_phi)`` at the node, Voigt order ``(11, 22, 33, 23, 13,
  12)`` without engineering factors.

A field absent from a layer is NaN on that layer's nodes; a field
absent from every layer of a part is absent from its group.  The
attributes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - attribute
     - meaning
   * - ``long_name``
     - what the name means
   * - ``units``
     - the stored unit, for CF tools
   * - ``dimensions``
     - exponents of mass, length, time and temperature; absent where the
       model declares none
   * - ``rank``, ``weight``, ``voigt``
     - the field's character: its law under the mapping
   * - ``layers``
     - the layers holding the field
   * - ``radial_interpolation``
     - ``polynomial``, ``linear``, ``log-polynomial`` or ``log-linear``
       (below)
   * - ``lateral_interpolation``
     - ``linear``, ``cubic`` or ``harmonic`` (below), on a variable that
       varies with direction
   * - ``radial_layers``
     - on a variable that varies with direction, the layers on which the
       field does not: its values there repeat over the grid, and a
       reader may read one column of them as a field of the radius;
       absent where there are none
   * - ``coordinates``
     - ``radius``

**Perturbed fields.**  A field given as a radial reference with a
perturbation, on every layer of the part that holds it perturbed
(of one kind) or radial, is two variables: ``name(node)``, the reference, with the
attributes ``perturbation`` (the other variable's name) and
``perturbation_kind``; and ``name_perturbation``, NaN on layers where
the field is not perturbed; each states its own rules.  Otherwise the
field is one variable of its values.  The kinds:

- ``absolute``: ``reference + perturbation``;
- ``relative``: ``reference * (1 + perturbation)``, as a relative
  velocity perturbation is;
- ``log``: ``reference * exp(perturbation)``, the natural logarithm, as
  a viscosity's perturbation is.

A relative or logarithmic perturbation is a pure number.

**Polar values.**  Where a part's grid lacks either pole as a node,
every variable that varies with direction has ``name_poles(node, pole,
...)``, the field's value at the north and then the south pole of
every node, evaluated exactly by the writer, with components in the
Cartesian frame (the spherical frame has none at a pole).  A pole that
is a node of the grid is its own row, which holds one value: for a
scalar, the same number at every longitude; for a field with
components, the same Cartesian components, which the stored spherical
components at each longitude rotate into.

The mapping
-----------

A model's mapping, from the spherical reference body to the physical
one, is held in one of two forms, never both; an identity mapping is
neither.

- **Boundary displacements**, the group ``geometry``: dimension
  ``boundary``, ``boundary_radius(boundary)`` (the reference radii of
  the surfaces, which may lie inside the model's layers),
  ``boundary_displacement(boundary, latitude, longitude)`` (the radial
  displacement of each surface, zero where it stays spherical), on the
  group's own grid with ``boundary_displacement_poles(boundary, pole)``
  where that grid lacks a pole, and the attributes
  ``lateral_interpolation`` and ``between`` (``linear``: between two
  surfaces the displacement is the linear interpolation in radius of
  theirs, and the mapping moves each point along its radius).
- **A displacement field**, ``displacement(node, latitude, longitude,
  component)`` in every part: ``m(X) - X`` in the spherical frame at
  ``X``, with ``displacement_poles`` where the grid lacks a pole.  Its
  deformation gradient is the derivatives of its interpolation rules,
  which are continuous in angle and may jump in radius at element
  boundaries.  Every part is then on one grid, so that the parts'
  displacements agree where they meet.

Fields on interfaces
--------------------

A function of the angles on one of the skeleton's interfaces (a shape
kernel, the derivative of an objective along a shift of a boundary; or
the shift itself) is held in the optional group ``surfaces``, beside
the parts, and read by ``read_surfaces`` as one gridded function per
interface.  A file without the group holds no such fields.

- Dimension ``surface``, over the interfaces that hold any field, and
  ``surface_interface(surface)``, the index of each in the root's
  ``interface`` dimension, increasing.  Only the skeleton's interfaces
  hold fields, never a radius inside a layer.
- The group's own grid, as a part's: ``latitude``, ``longitude``,
  ``colatitude_weights`` where it has them, the attributes
  ``grid_kind`` and ``grid_lmax``, and dimension ``pole`` with the
  attribute ``poles = "north south"`` where the grid lacks a pole.
- One variable per field, ``name(surface, latitude, longitude)``, NaN on
  an interface that lacks it, with the attributes ``long_name``,
  ``units`` (for CF tools), ``dimensions`` (where declared),
  ``lateral_interpolation`` and ``interfaces`` (the interfaces holding
  it); and ``name_poles(surface, pole)``, its values at the north and
  south poles, where the grid lacks a pole.

A field on an interface is a number at each point: it has no
components and no law under the mapping, and is read at an interface's
reference radius.  A name is a field of the layers or a field on
interfaces, never both, so that its ``dimensions`` and ``long_name``
say one thing in the file.  The C++ writer adds one with
``ModelFile::AddSurfaceField`` and the reader gives it with
``ModelFile::SurfaceFieldOn``.

Interpolation between nodes
---------------------------

Every variable states how it is read between nodes, and a reader uses
the stated rules unless it is asked for others.  The writer states the
rule the fields hold themselves to, the same on every layer of the
variable (see below for which), and states ``harmonic`` only on a grid
of a shipped kind whose band is at least the field's.

**Radially**, within the element holding the radius (the element above,
at a shared boundary):

- ``polynomial``: the Lagrange polynomial through the element's GLL
  nodes;
- ``linear``: straight lines between consecutive nodes;
- ``log-polynomial`` and ``log-linear``, the rules in the logarithm: the
  natural logarithms of the values are read by ``polynomial`` or
  ``linear``, and the value is their exponential.  A reader evaluates it
  on element ``e`` as ``y_e exp(P(r))``, ``y_e`` the element's first
  value and ``P`` the rule's interpolant through ``log(y_k / y_e)`` at
  its nodes, which is the same function and gives an element of one
  value that value to the bit; its radial derivative is ``y_e exp(P(r))
  P'(r)``.  A variable stating one is of rank 0 and of the radius alone
  (``name(node)``), and every value it holds is positive and finite;
  NaN still marks a layer that lacks it.  A positive field whose
  logarithm is a piecewise polynomial, such as a deck's column joined
  in its logarithm, is held by it exactly.

A viscosity joined in its logarithm, through two layers of a part:

.. code-block:: text

   :schema = "planetmodel.model/2" ;
   ...
   double viscosity(node) ;
       :long_name = "dynamic viscosity" ; :units = "kg m-1 s-1" ;
       :dimensions = 1, -1, -1, 0 ; :rank = 0 ; :layers = 1, 2 ;
       :radial_interpolation = "log-polynomial" ; ...

**Laterally**, at every node of that element, on a *padded grid*: the
grid's longitudes gain a wrap column at ``phi_0 + 2 pi``, a copy of the
first, and its colatitudes gain a row at each pole the grid lacks,
holding the polar value across every column.  A point is read at its
colatitude in [0, pi] and its longitude reduced into ``[phi_0, phi_0 +
2 pi)``.

- ``linear``: the bilinear interpolant on the padded grid.
- ``cubic``: the tensor-product cubic spline on the padded grid,
  periodic in longitude, and in colatitude clamped at each pole to the
  slope ``a cos(phi) + b sin(phi)``, with ``(a, b)`` fitted by least
  squares, over the grid's longitudes, to the slopes at that pole of the
  not-a-knot cubic spline through each column's values.  The
  interpolant is then continuously differentiable across the seam and
  through the poles.
- ``harmonic``: analysis of the values on the grid into real
  orthonormal spherical harmonics to the grid's band, and synthesis at
  the point; exact for a field band-limited to the grid.

Components of rank 1 and above are interpolated as Cartesian components
and turned back into the spherical frame at the point.

**The layout is exact where it can be.**  The writer puts element edges
at every breakpoint of a field's piecewise-polynomial description, or
of its logarithm's under a rule in the logarithm, and at every knot of
the mapping, with ``ngll`` one more than the highest degree, so that a
tabulated model written and read back is the same functions:
polynomials, and the exponentials of polynomials of a column joined in
its logarithm.  A grid of a field's band holds a band-limited field
exactly.  A field with no such description (a function of other
fields, say) is sampled on elements of the spacing the grid's band
sets, with ``ngll`` at least 5, and holds to that sampling's accuracy.

**Which rule the writer states.**  A variable of the radius alone that
is, on some of its layers, a positive field whose logarithm is a
piecewise polynomial and which is no polynomial itself, and on the rest
a positive constant, a field with no description, or absent, states
``log-polynomial``; a constant comes back to the bit under it.  A
variable that would need a rule in the logarithm on one layer and a
rule in the values on another (a polynomial that varies, or a layer that
varies with direction) is refused by the writer, naming the layers:
in parts of their own each layer is held exactly, and by one rule
named for the variable (``interpolation=``) approximately where that
rule does not hold it.  Any variable may be
written by a rule named for it; a rule in the logarithm so named for a
reference with a perturbation is the reference's, the perturbation
stating the rule it names in the values.  A reference whose
perturbation varies with direction (a viscosity perturbed in its
logarithm about a radial profile) states a rule in the logarithm as any
radial variable does.

**For a consumer in C++**, ``planetmodel::ModelFile::FieldOn`` reads a
variable under any of the four radial rules, with its derivatives, and
``RadialInterpolant`` reads nodal values by them; ``WriteModelFile``
writes a variable added with ``Rules{.radial =
RadialRule::LogPolynomial}``, and the file's schema follows.  A flat
model's file is read and written by ``planetmodel::cartesian``'s
``ReadModelFile`` and ``WriteModelFile``
(``planetmodel/cartesian/modelfile.hpp``), a field evaluated at
``(depth, x, y)`` or at a point ``(x, y, z)`` in the local or the
Cartesian frame, the mapping the vertical stretch; each of the two
readers refuses the other's files by their schema, naming the reader
that takes them, and the flat one refuses a field stating ``fourier``
by name where the field is built, its numbers still read.

Flat models
-----------

A model of ``planetmodel.cartesian`` is a body layered in depth below
one free surface, over a lateral domain: a box, or a rectangle, the
cross-section of a body invariant along y.  Points are ``(x, y, z)``
with ``z`` up and the free surface ``z = 0`` in the reference body; the
layered coordinate is the depth, positive downward.  ``write_model``
writes it as it writes a spherical model, and ``read_model`` reads it
back as a ``cartesian.Model``.  Its file is of schema
``planetmodel.model/3``, whether or not a variable states a rule in the
logarithm, and holds everything this page describes (the units and
scales, the constants, citation and meta, parts of contiguous layers at
the GLL nodes of their elements, fields of every rank with their
characters and rules, references with perturbations, the radial rules,
those in the logarithm among them, the exact layout, the groups
``geometry`` and ``surfaces``) except where the geometry differs:

**The root** states the system and the domain beside the shared
attributes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - attribute
     - meaning
   * - ``coordinate_system``
     - ``flat``
   * - ``local_frame``
     - ``up x y``: the components of every field are in the local frame
       ``(e_up, e_x, e_y)``, the same at every point; the first is the
       upward one, and the frame is right-handed
   * - ``domain``
     - ``box``, or ``rectangle`` for the cross-section of a body invariant
       along y
   * - ``domain_bounds``
     - ``x0, x1, y0, y1`` of a box, ``x0, x1`` of a rectangle: the domain
       proper
   * - ``domain_padding``
     - ``px, py`` of a box, ``px`` of a rectangle: the periodic margin on
       each side of each axis.  The domain's extent, ``[x0 - px, x1 +
       px]`` (and so in y), is the period of every Fourier series on it

The skeleton is in depth from the free surface down:
``layer_bounds(layer, bound)`` holds each layer's top and bottom depths,
and ``interface_depth(interface)`` replaces ``interface_radius``.
Interface 0 is the free surface (or the top of a body whose skeleton
starts below it) and interface ``k`` the top of layer ``k``; the base of
the box, the skeleton's last depth, is a face and no interface.  Parts
run from the surface down, ``part_0`` holding the top layer.

**A part** holds ``depth(node)`` in place of ``radius(node)`` (CF
``standard_name = "depth"``, ``positive = "down"``) and the horizontal
grid as the CF coordinate variables ``x(x)`` and, over a box, ``y(y)``,
in the model's unit of length: the nodes ``x_i = x0 - px + i Lx / nx``,
``i = 0 .. nx - 1``, of the extent of length ``Lx``, the last one step
short of the far end, which the period closes on the first (and so in
y).  A field is ``name(node)`` of depth alone, or ``name(node, x, y)``
(``name(node, x)`` over a rectangle, the same at every y), with the
trailing axes of its components as above, in the local frame; its
``coordinates`` attribute is ``depth``.  No flat grid has poles, a kind,
a band or weights: ``latitude``, ``longitude``, ``colatitude_weights``,
``grid_kind``, ``grid_lmax``, ``pole``, ``poles``, ``radius``,
``interface_radius`` and every ``*_poles`` are a spherical file's, and
a reader of a flat file refuses one that has any of them.

**Lateral rules.**  A variable that varies laterally states one of the
flat rules, each periodic along every axis of the grid closed by its
period, a point read at its coordinates reduced into the period from
the first node:

- ``linear``: the bilinear interpolant through the nodes and the wrap
  node one period on (linear along a rectangle's one axis);
- ``cubic``: the tensor product of the periodic cubic splines through
  the nodes, continuous with two derivatives across the seam;
- ``fourier``: analysis of the values on the grid into its real
  orthonormal Fourier basis and synthesis at the point, exact for a
  field the grid's shape holds (wavenumbers ``|k_i| <= n_i / 2``, the
  Nyquist cosine of an even axis once).

A Fourier series (a ``cartesian.FourierField``, or a Fourier function on
an interface or as a relief) is held as a harmonic field is on the
sphere: its values at the grid's nodes, stating ``fourier`` where the
grid holds every one of its modes (it lies over the same domain and its
shape contains the series'), so that the analysis of the stored values
gives back the coefficients, and the default ``linear`` on a grid that
does not.  In depth the element edges are put at the breakpoints of its
coefficients' piecewise polynomial, so a series written and read back is
the same function, to the accuracy of the synthesis at points.  A field
with no description in depth is sampled on elements no wider than the
grid's finest node spacing.

**The mapping** is the vertical stretch, every point moved along the
vertical by its upward displacement ``w`` (a height, positive up), in one
of two forms, never both; the identity is neither.

- **Boundary reliefs**, the group ``geometry``: ``boundary_depth(boundary)``
  (the reference depths of the surfaces, the free surface first and the
  base last) and ``boundary_displacement(boundary, x, y)``, each
  surface's height above its reference place, with the attributes
  ``positive = "up"``, ``lateral_interpolation`` and ``between =
  "linear"`` (between two surfaces the height is the linear
  interpolation in depth of theirs), on the group's own grid over the
  domain.  A surface that keeps its place holds zeros.
- **A displacement field**, ``displacement(node, x, y)`` in every part:
  ``w`` at the node, ``positive = "up"``, read by its stated rules.  Every
  part is then on one grid.

A mapping of any other kind has no form in a flat file and is refused by
the writer.

**Fields on interfaces** are functions of ``(x, y)``, the group
``surfaces`` as above on a grid of its own over the domain, without
poles.

A two-layer model in a box padded along x, its shear speed perturbed in
the mantle by a Fourier series and its free surface raised by a relief
(abbreviated from ``ncdump -h``):

.. code-block:: text

   netcdf flat {
   dimensions: layer = 2 ; interface = 2 ; bound = 2 ;
   variables:
       double layer_bounds(layer, bound) ;
           :long_name = "depths of the layer's top and bottom below the free surface" ;
       string layer_name(layer) ;
       double interface_depth(interface) ;
       string interface_name(interface) ;
       int layer_part(layer) ;
   // global attributes:
       :schema = "planetmodel.model/3" ; :Conventions = "CF-1.8" ;
       :scale_length = 1. ; ... ; :constant_G = 6.6743e-11 ; ...
       :coordinate_system = "flat" ; :local_frame = "up x y" ;
       :domain = "box" ; :domain_bounds = 0., 400000., 0., 300000. ;
       :domain_padding = 50000., 0. ;

   group: part_0 {
     dimensions: node = 4 ; element = 2 ; bound = 2 ; x = 10 ; y = 6 ;
     variables:
       double depth(node) ;
           :units = "m" ; :standard_name = "depth" ; :positive = "down" ;
       int element_layer(element) ;
       double element_bounds(element, bound) ;
       double x(x) ;   :units = "m" ; :axis = "X" ;
       double y(y) ;   :units = "m" ; :axis = "Y" ;
       double vs(node) ;
           :rank = 0 ; :layers = 0, 1 ; :radial_interpolation = "polynomial" ;
           :perturbation = "vs_perturbation" ; :perturbation_kind = "relative" ;
           :coordinates = "depth" ; ...
       double vs_perturbation(node, x, y) ;
           :layers = 1 ; :radial_interpolation = "polynomial" ;
           :lateral_interpolation = "fourier" ; :coordinates = "depth" ; ...
     // group attributes:
       :ngll = 2 ; :layers = 0, 1 ;
     }

   group: geometry {
     dimensions: boundary = 3 ; x = 20 ; y = 12 ;
     variables:
       double boundary_depth(boundary) ;
       double x(x) ;
       double y(y) ;
       double boundary_displacement(boundary, x, y) ;
           :units = "m" ; :positive = "up" ;
           :lateral_interpolation = "cubic" ; :between = "linear" ;
     }
   }

The fault file needs no flat schema of its own: its ``cartesian`` frame
(:doc:`fault_file`) is the flat model's points, ``z`` up.

An example
----------

The structure of a two-layer model with one field varying laterally,
on one Gauss-Legendre grid of band 4, placed by boundary reliefs, with
a shape kernel on both its interfaces:

.. code-block:: text

   netcdf model {
   dimensions: layer = 2 ; interface = 2 ; bound = 2 ;
   variables:
       double layer_bounds(layer, bound) ;
       string layer_name(layer) ;
       double interface_radius(interface) ;
       string interface_name(interface) ;
       int layer_part(layer) ;
   // global attributes:
       :schema = "planetmodel.model/1" ; :Conventions = "CF-1.8" ;
       :scale_length = 1. ; :scale_mass = 1. ; :scale_time = 1. ;
       :scale_temperature = 1. ; :constant_G = 6.6743e-11 ;

   group: part_0 {
     dimensions: node = 8 ; element = 2 ; bound = 2 ; latitude = 5 ;
                 longitude = 9 ; pole = 2 ;
     variables:
       double radius(node) ;
       int element_layer(element) ;
       double element_bounds(element, bound) ;
       double latitude(latitude) ;     :units = "degrees_north" ;
       double longitude(longitude) ;   :units = "degrees_east" ;
       double colatitude_weights(latitude) ;
       double rho(node, latitude, longitude) ;
           :units = "kg m-3" ; :dimensions = 1, -3, 0, 0 ;
           :rank = 0 ; :weight = 1 ; :voigt = 1 ; :layers = 0, 1 ;
           :radial_interpolation = "polynomial" ;
           :lateral_interpolation = "cubic" ; :coordinates = "radius" ;
       double rho_poles(node, pole) ;
       double vs(node) ;
           :radial_interpolation = "polynomial" ; ...
     // group attributes:
       :ngll = 4 ; :grid_kind = "gauss_legendre" ; :grid_lmax = 4 ;
     }

   group: geometry {
     dimensions: boundary = 3 ; latitude = 17 ; longitude = 33 ; pole = 2 ;
     variables:
       double boundary_radius(boundary) ;
       double boundary_displacement(boundary, latitude, longitude) ;
           :lateral_interpolation = "cubic" ; :between = "linear" ;
       double boundary_displacement_poles(boundary, pole) ;
     }

   group: surfaces {
     dimensions: surface = 2 ; latitude = 9 ; longitude = 17 ; pole = 2 ;
     variables:
       int surface_interface(surface) ;
       double latitude(latitude) ;     :units = "degrees_north" ;
       double longitude(longitude) ;   :units = "degrees_east" ;
       double colatitude_weights(latitude) ;
       double K_shape(surface, latitude, longitude) ;
           :long_name = "a shape kernel" ; :units = "m-2" ;
           :dimensions = 0, -2, 0, 0 ; :lateral_interpolation = "linear" ;
           :interfaces = 0, 1 ;
       double K_shape_poles(surface, pole) ;
     // group attributes:
       :grid_kind = "gauss_legendre" ; :grid_lmax = 8 ; :poles = "north south" ;
     }
   }
