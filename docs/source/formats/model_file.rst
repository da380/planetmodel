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
fields, checked as the readers check it.

**Schema string**: the global attribute ``schema``,
``planetmodel.model/1``.  Readers check it; it is bumped only for an
incompatible change.

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
     - ``planetmodel.model/1``
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
     - ``polynomial`` or ``linear`` (below)
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
variable, and states ``harmonic`` only on a grid of a shipped kind
whose band is at least the field's.

**Radially**, within the element holding the radius (the element above,
at a shared boundary):

- ``polynomial``: the Lagrange polynomial through the element's GLL
  nodes;
- ``linear``: straight lines between consecutive nodes.

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
at every breakpoint of a field's piecewise-polynomial description and
at every knot of the mapping, with ``ngll`` one more than the highest
degree, so that a tabulated model written and read back is the same
polynomials; a grid of a field's band holds a band-limited field
exactly.

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
