# Changelog

The notes of the next release, drafted as the work lands; a release's
notes are published on its GitHub release, and the section here starts
again empty.

## Unreleased

### Added

- Cubed-sphere meshes: a spherical geometry meshed with hexahedra and no
  mesher. `mesh3d.CubedSphereSpec` describes one and
  `mesh3d.build_cubed_sphere` builds it: a central cube of `edge`^3 cells
  and six blocks of columns above its faces, conforming to every
  interface, for a ball, and the six blocks alone for a shell.
  `mesh3d.SquaredCircleSpec` and `mesh3d.build_squared_circle` are the
  same in two dimensions, the geometry's disc or annulus meshed with
  quadrilaterals; the dimension is the class's, and neither spec takes
  one. `edge`, the number of cells along an edge of the cube, is the
  lateral resolution. The radial knots follow the sizing rule a
  `MeshSpec` takes, read as radial sizes (each layer takes the cells the
  target size asks for, at least one, so a layer of any thinness is
  kept), or are given outright as `knots`. The innermost layer of a ball
  is the cube and the blend out to the first sphere. The blend is
  divided until none of its cells is more than `elongation` times longer
  radially than it is wide (4 by default), or outright by knots below
  the first sphere, each the radius of a level along the ray through the
  centre of a face, which the cap then leaves alone. The geometry is of any order from 1; the nodes on an
  interface lie on its sphere exactly. The mesh is written in MFEM's
  format through PyMFEM (the `mfem` extra) and needs no gmsh; its
  manifest is that of any spherical mesh, schema 5 unchanged, and
  `export_mfem_mesh` and `export_mfem` take it as they take a mesh of
  tetrahedra. Before it is written a cubed sphere is held to MFEM's
  orientation of its elements, the sign of its Jacobian, the facing and
  the radius of every interface, and each layer's volume against that of
  the shell it fills. `ValidationReport` gains `max_layer_volume_error`
  and `max_aspect_ratio`, which a builder that does not measure them
  leaves as None, and `measure`, the name of the shape measure it holds:
  gmsh's minSICN, or "shape" for a cubed sphere's cells through their
  corners, which its summary prints under that name.
- `HarmonicFunction.scaled(k)`: a harmonic relief on a scaled body
  stays a `HarmonicFunction`, where `LayerLinear.scaled` used to wrap
  it. `testing.check_displacement` holds a scaled angular gradient that
  vanishes to the tolerance of the values.
- `faultfile.validate_structure`, `faultfile.schema_of` and
  `faultfile.malformed`, as `modelfile` and `surfacefile` have them.
- `SphericalGRF.on_grid(vector, grid)` and `SphericalGRF.covariance(a,
  b)`, the covariance between points (r, theta, phi) summed from the
  basis, as `BoxGRF` has them; `scale`, `coefficient_std` and `size` on
  `RadialGRF` and `SphericalGRF`. `size=` on `SphericalGRF.sample`, on
  `sample_grid` of both `SphericalGRF` and `BoxGRF`, and on `sample` of
  both `LayeredGRF`.
- C++: the spherical `SurfaceHistory` has `GetGrid()` and
  `Function(name, date)`, a field at a date as a function of (theta,
  phi) by its lateral rule with its stored polar values carried through
  time; a field stating `harmonic`, or without stored polar values on a
  grid lacking a pole, is refused as a function and still served by
  `Values`.
- C++: `ReadMeshManifest(path)` reads a manifest, beside `ReadModelFile`,
  `ReadSurfaceFile` and `ReadFaultFile`; the same in
  `planetmodel::cartesian`.
- `modelfile.write_model` needs no `grid` for a radial model: where
  every field written is of the radius alone (of depth alone in a flat
  model) and the mapping is the identity, it writes the file the grid of
  one node writes, `gauss_legendre(0)` or `cartesian.Grid(domain, 1)`;
  any other model given no grid is refused, naming the field that varies
  laterally or the mapping. `modelfile.radial_grid` is that grid, or the
  refusal, and `export_mfem(model_file=True)` uses it.
- `model_from_deck(deck, *, behaviours=..., kind=..., kinds=...,
  layer_names=..., interface_names=..., constants=..., scales=...,
  name=..., coordinate=..., domain=...)`: a table of knots as an
  ordinary model, its layers from the repeated knots and its columns
  joined by the kinds asked for, with the behaviours, names, constants
  and scales the caller gives; a table of depths is a flat model with
  `coordinate=cartesian.DEPTH` and the `domain` it is built over.
  Nothing is read from a header.
- `deck.mineos_header(deck)` reads a mineos header's values and
  `deck.mineos_constants(deck)` its reference period as the constant
  `omega_ref`, for a caller to pass on, beside `mineos_names` and
  `nd_names`.
- `write_deck(path, deck, format, *, header=...)`: `header` is True for
  the format's own header, generated from the table and the model
  sampled (the default), lines written as they are, or False for none;
  `format` may be the column names of a plain table.
- An `.nd` table of a model is written with TauP's labels `mantle`,
  `outer-core` and `inner-core` at its interfaces `moho`, `cmb` and
  `icb`; a `.tvel` table under two title lines and without labels.
- `scripts/migrate_deck.py` writes a Maxwell, prony or burgers deck of
  version 2, of radii or of depths, or a bare table given its columns,
  as a model file with its names and behaviours.
- `InterfaceSizing.at`: the size a sizing gives at a distance from its
  interface.
- Sections of hexahedra and quadrilaterals. `plotting.plane_cut` cuts
  hexahedra (rows of eight corners) into polygons of up to six sides,
  and with `tensor=True` reads rows of four as quadrilaterals and cuts
  them into segments; `plotting.in_plane` takes quadrilaterals whole;
  `mesh_section` draws either. `plotting.mesh_block` draws a cutaway of
  hexahedra, rows of eight corners, with quadrilaterals for the curves
  on its walls.
- `mesh3d.read_mesh` reads a file in MFEM's format (`.mesh`) through
  PyMFEM as it reads an MSH file through gmsh, each element by its
  corners, and `mesh3d.cell_quality` gives the shape of the cells of
  such a file of hexahedra or quadrilaterals. `MeshArrays.tensor` says
  whether the cells are hexahedra or quadrilaterals; the rules stated
  for simplices (`layers_by_position`, `walls`, `face_cells`) refuse
  them by name.
- Examples `meshing/07_a_cubed_sphere.py` and
  `meshing/08_prem_on_a_cubed_sphere.py`, and the Earth-scale script
  `scripts/cubed_sphere_prem.py`.

### Changed

- `MeshResult.counts` counts as the MFEM export does, whichever builder
  made the mesh: `elements` are the cells alone and `boundary_elements`
  the faces that carry attributes, so a mesh's build and its export
  report the same two numbers. A mesh built through gmsh used to count
  both under `elements`, and its `dim0` to `dim3` keys are gone.
  `mesh3d.export.mesh_counts` is the count of a mesh MFEM holds.
- `MeshResult.counts` has the `order` of the mesh from every builder.
- `rheology.is_linear_rheology_name` says which names a linear rheology,
  and so a frozen model, is read from, and `mesh1d.gravity.radii_within`
  holds radii to a model's range; both were private names shared between
  modules.
- `conversion.prony.with_burgers` holds a relaxation time that varies
  laterally as a log perturbation of the same branch's time over the
  layer's unperturbed viscosity and shear modulus, as `with_branches`
  does; its values are unchanged.
- `RegionalGrid.extent` gives a band round the circle a whole turn in
  longitude from its first.
- The `step` of `conversion.lithosphere`'s `isotherm`,
  `viscosity_threshold` and `maxwell_threshold` is documented as metres,
  whatever the model's scales.
- `slices.section` at an `azimuth` from a pole counts it from the
  meridian of the longitude given with `start`; the refusal that could
  never fire is gone, and the arc's points within `mapping.SIN_FLOOR` of
  the polar axis lie on it. `slices.runs` takes `lateral=`, the lines'
  own lateral coordinates, and `slices.evaluate_layers` takes `near=`.
- `surfacefile.HistoryBase.variable` gives a field's stored values and
  attributes; it was a private name the flat history shared.
  `modelfile.attribute_text` and `modelfile.attribute_number` read a
  netCDF attribute as text or as one finite number, None otherwise;
  `modelfile.Layout` gains `interfaces` and `surfacefile.SurfaceLayout`
  gains `check_rule`.
- **The inner boundary of a hollow body points out of the body, towards
  the centre, in every mesh file.** MFEM turns the elements on a mesh's
  true boundary that way as it reads a file, so an MSH file from
  `build_layered_mesh` or `build_offset_mesh`, which wrote that boundary
  away from the centre like an interface, disagreed there with its own
  MFEM delivery and with a cubed sphere. The MSH file of a hollow
  geometry is written as MFEM holds it, and in two dimensions the lines
  on every circle are oriented by the same rule, where they were left as
  gmsh made them. The vertex order of those boundary elements changes
  in the MSH file; a delivery read through MFEM is as it was. The format
  page states the rule, `ValidationReport.inward_faces` counts against
  it for every builder, and the checks of MFEM's verdict on boundary
  elements, which could not fail, are gone.
- `HeldField`, `fields.held` and `RadialField.held` take the interval
  first, as every constructor of a layer function or a field does:
  `HeldField(source, at, interval)` is now `HeldField(interval, source,
  at)`, `held(field, at, interval)` is `held(interval, field, at)`, and
  `field.held(at, interval)` is `field.held(interval, at)`.
- `refined(radii, names=...)` is now `refined(radii,
  interface_names=...)` and `extended(radii, names=...)` is
  `extended(radii, layer_names=...)`, on a geometry and on a model, on
  the sphere and in a box; `truncated` and `hollowed` keep `name=`.
- `Geometry.coarsened` takes what `Model.coarsened` takes: `keep` and
  `drop` are interfaces, by index or name, where they were indices of
  the interior boundaries. On a full ball an integer selects the
  boundary it did; on a hollow ball and in a box, where interface k is
  boundary k, `drop=[k]` was the boundary that is now `drop=[k + 1]`,
  and a negative index counts interfaces. `Skeleton.coarsened` still
  counts interior boundaries, and a `CoarseningMap` reports them under
  that name: `kept_interfaces` and `dropped_interfaces` are now
  `kept_interior` and `dropped_interior`. The refusal of a hollow body's
  inner boundary speaks of a hollow geometry.
- `Geometry.knots` is a property, as `knots` is on every mapping and
  displacement: `geometry.knots()` is now `geometry.knots`.
- `RadialMesh.element_at` is gone: `Mesh1D.element_of`, which it called,
  is the one name.
- A mapping's `is_valid(...)` is now `validity(...)`, the name of the
  free function `mapping.validity` and of `Geometry.validity`; it
  returns a `ValidityReport` as before. A mapping written elsewhere with
  an `is_valid` of its own is no longer asked and is judged by J > 0.
  `testing.check_mapping` holds a mapping's `validity` to its report.
- The sizing rules name their sizes as `InterfaceSizing` does:
  `UniformInterfaces(h_min, h_max, decay_width)` is now `(size,
  far_size, decay_width)`, `CappedInterfaces(h, h_far, decay_width,
  ...)` is `(size, far_size, decay_width, ...)`, and
  `AngularResolution(h_ref, h_far, ...)` is `(size, far_size, ...)`, its
  `size` the size at `r_ref`. The order is unchanged, so positional
  calls stand.
- `plotting.plane_cut` and `plotting.in_plane` call their second
  argument `elements`, not `simplices`, and the refusal of a crinkle of
  anything but tetrahedra names what it was given.
- `export_mfem_mesh(result, path_base, ...)` and `export_mfem(result,
  path_base, ...)` are now `(result, path, ...)`, as every builder names
  its basename.
- `conversion.prony.burgers_series(*, strength, viscosity_ratio)` is now
  `burgers_series(*, kelvin_modulus_fraction, viscosity_ratio)`, the
  Kelvin element given as `burgers_terms`, `burgers_parameters` and
  `with_burgers` give it: the fraction is 1 / the old strength
  (`strength=5.0` is `kelvin_modulus_fraction=0.2`), and the series'
  name states it.
- `randomfield.SpectralBasis.degree` is now `SpectralBasis.label`, the
  operator family's word for what may be a degree, an order or a lateral
  eigenvalue.
- `modelfile.read_file(path, layouts)` is now `modelfile.read_with(path,
  layouts)`, as `surfacefile` names its reader by layouts.
- **The spherical random-field samplers are coefficient-first, as the
  flat ones are.** `randomfield.RadialGRF`, `SphericalGRF` and
  `LayeredGRF`: `sample` returns the coefficient vector in the order of
  the sampler's basis (`size` entries, independent, of standard
  deviations `coefficient_std`), and `synthesise`, `to_layer`,
  `to_field` and `coefficient_functions` take that vector.
  `RadialGRF.sample` used to return the values at the nodes and
  `SphericalGRF.sample` the coefficient functions, shape (2, lmax + 1,
  lmax + 1, r.size); each is now `synthesise` of a sample. A seeded
  sample is the same field as before, to rounding.
- `randomfield.LayeredGRF.sample` returns one vector, the layers' end to
  end (`block`, `size`), and no longer takes `character`;
  `to_fields(vector, character=...)` gives the fields it used to return.
- `RadialGRF.factor` is a method, as `DepthGRF.factor` is.
- `cartesian.DepthGRF.synthesise` takes one coefficient vector or one
  batch of them, shape (k, nmodes), where it took any leading axes.
- `randomfield.fields.nodal_ppoly` and `coefficient_vectors` are what
  the spherical and flat samplers share; the first was private to
  `cartesian.randomfield`. `marginal_scale` returns the scale read-only,
  as it does sigma.
- C++: `SurfaceFieldOptions::support`, `time_interpolation` and
  `lateral_interpolation` are the enumerations `TimeSupport`, `TimeRule`
  and `LateralRule`, as the fault file's and the model file's are;
  `SurfaceField` writes them as the names a file states.
- C++: `SurfaceHistory::Values` (and the flat `Function`) take a date's
  extrapolation and tolerance as a `DateOptions`, as `Fault::Rows` does,
  and `SurfaceHistory::Support` returns a `TimeSupport`; the same in
  `planetmodel::cartesian`.
- `write_surface` and `write_fault` take `source` as `write_model` does,
  this library and its version unless given
  (`modelfile.stamped_source`); the C++ `WriteSurfaceFile` and
  `WriteFaultFile` stamp `planetmodel C++ <version>` where `source` is
  empty.
- The C++ library fetches Interpolation at a release tag by default
  (`PLANETMODEL_INTERPOLATION_TAG`, `v1.1.2`), changed when planetmodel
  needs a newer one; the variable builds against another revision.
- **`cartesian.Rectangle` answers per axis, as `cartesian.Box` does.**
  Its `bounds` are `((x0, x1),)`, its `padding` `(p,)`, its `extent`
  `((lo, hi),)` and its periods `lengths`, `(L,)`, where they were a
  pair, a number, a pair and `length`; `Rectangle.length` is gone. Its
  constructor takes the pair and the one padding as before, and each as
  the rectangle holds it. What is built over a domain runs over its
  `axes`, and a flat model file and a flat surface file state a domain
  exactly as they did.
- `cartesian.domains.DomainBase` holds what `Box` and `Rectangle` share:
  `bounds`, `padding`, `extent`, `lengths`, `contains`, `scaled` and
  equality, each per axis of the leaf's `axes`.
- `fourier.PaddedDomain.of` reads a domain through `bounds` and
  `padding` held per axis alone; an object whose `bounds` is one bare
  pair is refused.
- `PREM` declares its reference frequency, the constant `omega_ref` of
  2 pi rad/s (a period of 1 s): it is in PREM's model file and its
  mineos export says `tref 1`; `rheology.omega_ref_constant` makes the
  constant for any model, and no numerical result changes.
- **A model file says what kind of model it holds.** The root attribute
  `behaviours` names the behaviours of the model written
  (`Model.behaviours`, the mixins of its class in the order of its
  bases), and `read_model` gives back a model that has those of
  `behaviours.BEHAVIOURS`, so that `moduli`, `gravity`, `frozen` and the
  rest are called on it as on the original. A name the reader lacks is
  kept, written again unchanged, and warned of once; a file without the
  attribute reads as a plain `Model`, as before. `read_model` of a file
  written from here on no longer returns a plain `Model` where the model
  written had behaviours; `read_model(path, model_type=())` does.
- `read_model(path, model_type=...)` says what to give back instead: a
  model type of the caller's own, or the behaviours of one. A model is
  read without any constructor being called (`model.restored`) and holds
  the fields the file holds.
- `behaviours.model_type(base, behaviours)`: the model type with the
  behaviours given over `Model` or `cartesian.Model`, one class for each
  combination.
- `testing.check_model` holds a model's `behaviours`, and
  `check_model_file` holds the file's attribute and the behaviours of
  the model read back.
- C++: `ModelFile::Behaviours()` and `SetBehaviours` in `planetmodel` and
  `planetmodel::cartesian`; a file read and written back names the same
  behaviours.
- **A text deck is an import and an export, and the model file the form
  of every model.** The model types of the decks are removed (see
  Removed); a table is read with `read_deck`, made a model with
  `model_from_deck` and saved with `modelfile.write_model`, and
  `read_model` gives it back with its behaviours.
- `read_deck` keeps a file's header lines as they stand under
  `deck.header["lines"]` and reads nothing from them: a deck read no
  longer holds `name`, `tref`, `nic`, `noc`, `layers` or `interfaces` in
  its header.
- The mineos header `write_deck` generates takes `nknot`, `nic` and
  `noc` from the table, the outer core its innermost layer whose S
  velocity is zero throughout (`0 0` for a body with no fluid layer),
  and `tref` from the model's declared `omega_ref`, 0 where it declares
  none; it used to name the first two layers the core and default `tref`
  to 1.
- `write_deck` writes a deck in a format without a header of its own
  under no header; the lines a deck was read with are written by passing
  `header=deck.header["lines"]`.
- `DeckFormat` has `columns`, `name`, `header_lines`,
  `write_header(deck, names)`, `fill`, `read_table` and `write_table`;
  `DeckCoordinate` has `word`, `base` and `geometry`.
- `datasets.DeckReader(format, *, behaviours=..., names=...,
  constants=..., kind=...)`: an entry states its model's behaviours, how
  its layers are named (a rule, or the names) and the rule that reads
  its constants; `model=` is gone, and STW105 is a model of
  `behaviours.model_type` with the four behaviours.
- No column is joined in its logarithm by default:
  `kinds={"viscosity": "log-linear"}` asks.
- Tutorial 11 brings a mineos deck in, saves it as a model file, reads
  it back and writes it as a deck again; it needs the `netcdf` extra.
- **C++: decks are removed.** `deck.hpp`, `cartesian/deck.hpp` and
  `radial.hpp` are gone, and with them `Deck`, `DeckFormat`, `ReadDeck`,
  `WriteDeck`, `DeckLayers`, the formats `MINEOS()`, `MAXWELL()`,
  `PRONY()`, `BURGERS()`, `ND()`, `TVEL()` and the flat `ELASTIC()`,
  their naming and header functions, `RadialFunction`, `KnotKind` and
  `ThroughKnots`, and the macros `PLANETMODEL_HAS_VISCOELASTIC_DECKS`
  and `PLANETMODEL_HAS_LOG_KINDS`. A C++ code reads the model file,
  which holds every model, a radial one among them; the C++ library
  reads every format a solver reads at run time, the model file, the
  surface file, the fault file and the mesh manifest. `BurgersTerms`
  and `BurgersParameters` (`prony.hpp`) stay.
- `MeshResult.msh_path` is now `MeshResult.mesh_path`: the file is MSH
  where gmsh built the mesh and MFEM's format for a cubed sphere. Code
  that reads `result.msh_path`, or builds a `MeshResult` by that
  keyword, takes the new name.
- `export_mfem_mesh` and `export_mfem` refuse a delivery whose `.mesh`
  file would be the reference mesh it is made from, which a cubed
  sphere's reference mesh, itself a `.mesh` file, makes possible: such a
  delivery takes another basename.
- `require_mapping_on_shells` and `require_spherical_outer_boundary`
  moved from `mesh3d.layered` to `mesh3d.spec`, and `check_sizing_scale`
  from `mesh3d._sizing` to `mesh3d.spec`, where both kinds of spec share
  them with `shell_radii`, `shell_names`, `check_shells` and
  `computational_domain`. In `mesh3d.export`, `load_mesh`, `mesh_order`,
  `node_array` and `write_vector` are public names (they had a leading
  underscore), and `PRECISION` is the number of digits a mesh and a
  field are written with.
- The mesh manifest's format page says what a mesh's elements may be and
  that the manifest does not record it, and notes MFEM's default
  quadrature on a geometry of high order.

### Removed

- `MineosModel`, `MaxwellDeckModel`, `PronyDeckModel`, `BurgersDeckModel`
  and `cartesian`'s `ElasticDeckModel`, `MaxwellDeckModel`,
  `PronyDeckModel`, `BurgersDeckModel`; `Tabulated` with `keep_deck`,
  `to_deck` and `restated`.
- The Maxwell, prony and burgers deck formats and the elastic deck of
  depths: `MAXWELL`, `PRONY`, `BURGERS`, `cartesian.ELASTIC`, their
  `*_format`, `*_names`, `*_COLUMNS` and `*_KINDS`, `prony_columns`,
  `prony_branches`, `prony_layers`, `deck_from`, `cartesian.depth_names`,
  `STRENGTH_ROUNDING`, and their three format pages.
- `conversion.prony.prony_deck`, `burgers_deck` and `burgers_layers`;
  `with_branches`, `with_burgers` and the series stay.

### Fixed

- `gravity`, `mass`, `pressure`, `layer_masses`, `with_gravity` and
  `RadialMesh.nodal_gravity` raised `AttributeError` on a density that
  is radial without being a `RadialField` (a perturbed, composed or held
  one); such a density is integrated by quadrature.
- `units.parse_unit` read "deg K", "degrees K" and "degrees Kelvin" as
  pi/180 of a kelvin; a degree followed by K or kelvin is the kelvin
  ("degK", "degree_K" and "°K" among its spellings), and "degrees" alone
  is still the angle.
- `RadialMesh(edges=...)` kept an edge given within `rtol` of a skeleton
  boundary beside it, so an element reached into the next layer and
  `nodal` refused its radii; such an edge is the boundary.
- A layer appended to an `Elastic` model by `extended` with its fields
  given held no Love moduli; it is completed as a layer given on
  construction is, on a model that is `SelfGravitating` as well in
  either order of the bases.
- A frozen model refused `stress_exponent` and `transition_stress` as
  fields its moduli were derived from, which freezing does not read;
  they are attached as any other field.
- A `ComposedField` of rank 0 over radial sources of higher rank
  reported `is_radial` and then refused to be called with the radius
  alone.
- `RadialMesh.nodal(..., nu=1)` raised `AttributeError` on a field that
  cannot state its derivative (the velocities of a layer described by
  its moduli); it is refused by name.
- A number divided by a `PolynomialLayer` raised `TypeError`; it is a
  `NumericLayer`, as it is for the other layer functions.
- A `CubedSphereSpec` or `SquaredCircleSpec` took knots within the
  geometry's tolerance of one another or of the cube's surface and built
  slivers or failed on a folded cell; they are refused by name.
- `MeshArrays.boundary_of` cut the quadrilateral faces of a mesh of
  hexahedra down to three corners; it returns them whole.
- `MeshSpec` took an `order` that is not a whole number.
- The mesh manifest's format page showed the displacement of a
  referential delivery with `"ordering": "byNODES"`; it is written
  `byVDIM`, the ordering of the mesh's own nodal space, and has two
  components on a two-dimensional mesh.
- `datasets.transplanted` and `rereferenced` read a source in other
  scales than the model as though it shared them, so values kept under
  `mode="absolute"` came out wrong and a model in another length unit
  was not reached; the source and its reference are converted to the
  model's scales.
- `datasets.merged` blended a regional grid's SI radii and values into a
  model in other scales as they stood; they are put in the model's
  units.
- `RegionalGrid.edge_distance` measured to the whole great circle
  through a side in longitude, too short in a box wider than 180
  degrees; it is the angle to the meridian arc. A grid whose longitudes
  go round the circle short of the poles kept a seam with two sides; it
  is a band, read across the join.
- `datasets.read_grid` did not find the missing values of a packed
  variable, comparing the scaled data with the stored `_FillValue`; the
  fill is found among the stored values and the scaling applied after.
- `Deck == Deck` and `hash(deck)` raised; a deck is compared and hashed
  by identity.
- `PolarStereographic.inverse` documented longitudes in (-180, 180];
  they are in [-180, 180).
- A model read from a file and written again restated a variable read
  by the harmonic rule as `linear`, and so a field on an interface held
  as a `GriddedFunction` by that rule; a field read by the harmonic rule
  on a grid of a shipped kind is of that grid's band, and states the
  rule on a grid of at least it.
- The flat writers wrote a layer field, a field on an interface or a
  surface field named as a spherical file's variable is (`radius`,
  `latitude`, `longitude`, `colatitude_weights`, `interface_radius`,
  `boundary_radius`), which their own readers refuse; the writers refuse
  it by name.
- A surface file holding a `node`, `interval` or `instant` field without
  its coordinate variable `time` or `instant` passed validation and
  failed with `TypeError` on use, and polar values over the wrong
  leading dimension passed too; both are refused as malformed.
- `SurfaceHistory.function` read a field stating `harmonic` by the
  bicubic rule; it reads it by the harmonic rule.
- `write_surface` and `cartesian.surfacefile.write_surface` wrote a
  lateral rule the grid cannot carry (`harmonic` on a grid without a
  band, `cubic` on a grid too small for it); they refuse it as
  `write_model` does, and a reader refuses a model or surface file
  stating one, where it used to fail when the field was built.
- `write_surface` and `write_fault` took a citation given as one string,
  read back as its characters, and a work of several lines, read back as
  several works; both are refused, as a model's citation is.
- The model, surface and fault readers raised errors that were not their
  malformed-file refusal on a scale that is infinite, text or several
  numbers, on `ngll` that is not a whole number, on an array-valued
  `schema`, `time_support`, `time_interpolation` or
  `lateral_interpolation`, and on a grid whose coordinates make none;
  each is refused naming the file and what is wrong.
- A model file whose `interface_radius` (`interface_depth` in a flat
  file) is not the interfaces of its `layer_bounds` was read without
  complaint; it is refused.
- `testing.check_model_file` did not hold a spherical file's name,
  constants, the polar values of a perturbation, of the geometry group
  or of a displacement, the geometry group's radii, or the perturbation,
  displacement and reliefs of the model read back; it does, and holds a
  flat file's name and constants too.
- Without netCDF4, `write_surface` and `write_fault` said that the model
  file needs it; the message names netCDF files and the `netcdf` extra.
- A slice departing from another model (`relative_to=` a model) and a
  `correlation` of two models compared raw numbers at the same numeric
  position whatever the two models' `Scales`; the other model is
  converted to the first's (`slices.in_scales_of`), and one that cannot
  be converted is refused naming both scales.
- `slices.profile` at a pole, and at the centre at physical points, read
  a component in the frame of longitude zero; it is read at the angles
  given. A physical `slices.sphere` reads a polar node of its grid at
  the node's own longitude, and `slices.section` a point on the polar
  axis at the longitude of the arc beside it.
- The spherical slices returned numbers for a flat model; they refuse
  one, naming both coordinate systems.
- `slices.normalised` left the `bands` of a flat mean profile undivided.
- A radius or depth within the geometry's tolerance of an interface,
  accepted as on it, was passed unsnapped to the field, which refused
  it; it is read on the boundary.
- `slices.integrated` under-integrated a relative or logarithmic
  `PerturbedField` and ignored `n` for it; the quadrature is exact for a
  relative perturbation's degree (`PerturbedField.quadrature_degree`)
  and takes `n` points on each piece of a logarithmic one.
- A `GriddedDisplacement` read by `radial="linear"`, on the sphere or in
  a box, declared only its element boundaries as knots though its
  derivative jumps at every node; every node is a knot, so
  `Geometry.knots` and a `RadialMesh` put an edge there.
- `GriddedFunction.scaled` dropped the `rtol` the function was built
  with, so a relief accepted at construction could be refused when its
  geometry was rescaled.
- `Packing.component_of` refused a cosine named by the conjugate of its
  pair's representative, with a message about sines; it returns the
  representative's cosine, and refuses a sine so named with a message
  that names the representative.
- `expansions.ppoly_of` failed on a B-spline of degree 0.
- The C++ `netcdf::ReadDoubles` read a variable's fill as an ordinary
  number, so a model, surface or fault file marking absent values by a
  `_FillValue` other than NaN was misread (a field held on layers that
  lack it); a value equal to the fill, the variable's own or its type's
  default, is NaN, as the Python reads it.
- The C++ `GllNodes` computed an element's last node from its first, so
  `ModelFile::AddPart` with edges such as 0, 0.3, 0.9, 1.5 wrote a file
  whose fields the C++ then refused to evaluate; the first and last node
  are the element's ends.
- The C++ model-file readers accepted an `element_layer` that disagreed
  with `element_bounds` and then failed or gave other numbers on
  evaluation; each element's bounds must lie in the layer it is given.
- The C++ `ReadSurfaceFile` did not hold a field to its dimensions, so a
  field stored as (longitude, latitude), or over another dimension of
  its support's length, was read as if it were in order; a field runs
  over its support's dimension and then latitude and longitude, and its
  polar values over that dimension and the poles.
- The C++ `FaultWalls` read past the end of its vertices for a cell
  indexing none after a corner that carries slip; every corner is
  checked and the cell refused.
- The C++ readers kept the NUL bytes of a text attribute, so an
  attribute written as the empty string counted as present (a fault
  file's `source`); they are dropped, as netCDF4-python drops them.
- The flat C++ `ModelFile::SetStack` and `ValidateModelFile` took a
  stack whose first surface lies above the free surface, and wrote a
  file the Python cannot read as a model; a negative first depth is
  refused.
- The C++ `WriteSurfaceFile`, spherical and flat, failed on a file it
  had read, copying `_FillValue` after the data; netCDF's own attributes
  are left to the writer.
- `deck.nd_names` gave a label at the innermost boundary to the
  outermost layer, and on a solid body put its interface's name over
  `surface`; such a label is refused by name.
- The Python's model-file validator (`validate_structure`,
  `read_model_file`, spherical and flat) took an `element_layer` giving
  an element to a layer its bounds lie outside; it is refused at
  `part_<k>/element_layer`, to 1e-12 of the part's span, as the C++
  refuses it.
- A flat model file whose `geometry/boundary_depth` starts above the
  free surface passed `validate_structure` and
  `cartesian.modelfile.read_model_file` and was refused only by
  `read_model`, with a skeleton's message; it is refused by name as a
  malformed file.
- The C++ readers of model, surface and fault files, spherical and flat,
  refused a scale that is not finite through `Scales`, naming neither
  the file nor the attribute; it is refused at `scale_<base>` as missing
  or not positive and finite.
- The C++ `ReadModelFile` and `cartesian::ReadModelFile` took an
  `interface_radius` (`interface_depth`) that is not the interfaces of
  `layer_bounds`; they are held to them, to 1e-12 of the skeleton's
  span, as the Python holds them.
- The C++ `ValidateModelFile`, and so the readers and writers of model
  files, spherical and flat, took a variable stating a lateral rule the
  grid of its group cannot carry (`harmonic` on a `custom` grid or one
  stating no `grid_lmax`, `cubic` on a grid too small for it); it is
  refused at the variable.
- The C++ `ValidateSurfaceFile`, and so `ReadSurfaceFile` and
  `WriteSurfaceFile`, spherical and flat, took a field stating a lateral
  rule the grid cannot carry; it is refused as the Python refuses it.
- The C++ `WriteSurfaceFile` (spherical and flat) and `WriteFaultFile`
  wrote a citation as given; each work is trimmed, an empty one or a
  repeat dropped, and a work of more than one line refused, as
  `ModelFile::SetCitation` and the Python's writers do.
- The flat C++ writers took a layer field, a field on an interface or a
  surface field named as a spherical file's variable is (`radius`,
  `latitude`, `longitude`, `colatitude_weights`, `interface_radius`,
  `boundary_radius`) and wrote a file the readers refuse; the name is
  refused, and `cartesian::RESERVED_NAMES` holds the six.
- The C++ readers of model files truncated an `ngll` that is not a whole
  number; it is refused at `part_<k>.ngll`.
- The model file's format page did not say that each element lies within
  the layer `element_layer` gives it, nor that a flat stack's first
  depth is not negative.
- `cartesian.BoxGRF.sample(size=k)` returned a batch that its own
  `synthesise`, `on_grid`, `coefficient_functions` and `to_field`
  refused, as `DepthGRF.to_layer` and `to_field` and
  `LayeredGRF.to_fields` refused theirs; every method of a sampler that
  takes a coefficient vector takes a batch, shape (k, size), an array
  coming back with the batch axis leading and a field as a tuple over
  the batch.
- `read_model_file`, `read_model` and `validate_structure` took whether
  a model file needs polar values from the presence of a `pole`
  dimension, and read a file without them by computing them; the grid's
  nodes decide it, as in C++, and a part, a displacement, the `geometry`
  group or the `surfaces` group without them, or with them of the wrong
  size or over other dimensions, is refused as malformed.
- A model file whose `geometry/boundary_radius` (a flat file's
  `boundary_depth`) does not increase strictly, or holds one surface,
  was read by `read_model_file` and refused by `read_model` with a
  skeleton's message; it is refused as malformed, by name.
- `read_surface_file` read polar values on a grid holding both poles,
  which `write_surface` refuses to write; it refuses them.
- C++: `ReadSurfaceFile` read a file whose `pole` dimension is not of
  length two where no field carries polar values.
- C++: `ReadModelFile` refused negative or non-finite `layer_bounds` as
  `Skeleton: ...`, naming neither the file nor the variable, and read
  past the end of a file with no layers; both are refused at
  `layer_bounds`. The Python raised `IndexError` on a file with no
  layers.
- `write_model` wrote a `source` that is not text; it is refused.
- Both readers refuse the same files: the C++ a grid with a latitude
  outside [-90, 90] (it clamped it), the Python a surface file whose
  `source` or `created` is not text or whose field's `dimensions` are
  not four whole numbers; the C++ also refuses a `dimensions` that is
  text or fractional.
- `testing.check_slice` reads a point on the polar axis or at the centre
  at the angles its slice did (`slices.read_at`, and the new `lateral`
  of a sphere, section and profile slice), so a component in the
  spherical frame there is no longer wrongly failed.
- `MeshResult.counts["nodes"]` of a mesh with faults built through gmsh
  counted nodes of gmsh's model that no cell uses, more than the file
  holds; it counts the nodes of the cells, which are those written and
  those MFEM reads.
