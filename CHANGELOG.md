# Changelog

The notes of the next release, drafted as the work lands; a release's
notes are published on its GitHub release, and the section here starts
again empty.

## Unreleased

### Added

- The stratification of a spherically symmetric model
  (`planetmodel.mesh1d.stratification`): `bullen_parameter`, its excess
  over one `bullen_excess`, and the square of the buoyancy frequency
  `brunt_vaisala_squared`, at any radii, from the density, the bulk
  modulus and gravity; also methods of `SelfGravitating`. The vocabulary
  gains the name `bullen_excess`, in the Python and in the C++.
- `stratified_density`: the density of a layer from its Bullen excess and
  its P velocity or bulk modulus, fixed by its density at the bottom, its
  density at the top or its mass, as a piecewise polynomial that holds
  the solution to a stated relative tolerance, with a breakpoint wherever
  the excess or the velocity has one.
- A fluid layer may be described by its stratification in place of its
  density. `SelfGravitating.with_bullen_excess` attaches a Bullen excess
  to a fluid layer, nowhere negative and zero where the layer is
  neutrally stratified, and derives the layer's density from it; the
  layer keeps its mass unless a mass or a density at one end is given.
  On a model that is also `Elastic`, `with_field` keeps such a density in
  step: a new velocity or excess on the layer derives it again, a new
  density beneath it derives it again under the changed gravity, and a
  density given to the layer directly becomes its description, the
  excess dropped. `is_neutral` says whether the excess a layer holds
  vanishes; `layer_masses` gives the mass of every layer.
  `with_stratification` reads the excess where a layer holds one, and
  takes no derivative there. `coarsened` refuses a merge that includes
  such a layer; `hollowed`, and `extended` by continuation, give the
  description up.
- `SelfGravitating.stabilised` makes the unstably stratified fluid layers
  of a model stable. Each is replaced by the stable layer nearest to it
  of the same P velocity, radii, mass and moment of inertia
  (`stable_layer`): every unstable interval becomes neutral together
  with as much of the fluid beside it as that takes, and the layer is
  then described by its Bullen excess. It returns the model and a
  `Stabilisation` saying, for each layer changed, where it is neutral
  and how far its density moved. A layer that cannot keep its moment of
  inertia, as a uniform one cannot, is refused unless the mass alone is
  asked to be held; a stable model is returned as it is. On PREM's
  outer core the density changes by at most 2.9e-4.
- A model file carries a layer's `bullen_excess` like any other radial
  field, its breakpoints element edges, so the stretches of a layer
  that are neutral are read back as exactly that; the schema is
  unchanged and a reader that does not know the name loses nothing. The
  format page says what the variable means, and the C++ reader is held
  to a fixture that carries it.
- Cubed-sphere meshes: a spherical geometry meshed with hexahedra and no
  mesher. `mesh3d.CubedSphereSpec` describes one and
  `mesh3d.build_cubed_sphere` builds it: a central cube of `edge`^3 cells
  and six blocks of columns above its faces, conforming to every
  interface, for a ball, and the six blocks alone for a shell; with
  `dimension=2`, the disc or annulus of quadrilaterals. `edge`, the
  number of cells along an edge of the cube, is the lateral resolution.
  The radial knots follow the sizing rule a `MeshSpec` takes, read as
  radial sizes (each layer takes the cells the target size asks for, at
  least one, so a layer of any thinness is kept), or are given outright
  as `knots`. The innermost layer of a ball is the cube and the blend
  out to the first sphere, and the blend is divided until none of its
  cells is more than `aspect` times longer radially than it is wide (4
  by default). The geometry is of any order from 1; the nodes on an
  interface lie on its sphere exactly. The mesh is written in MFEM's
  format through PyMFEM (the `mfem` extra) and needs no gmsh; its
  manifest is that of any spherical mesh, schema 5 unchanged, and
  `export_mfem_mesh` and `export_mfem` take it as they take a mesh of
  tetrahedra. Before it is written a cubed sphere is held to MFEM's
  orientation of its elements, the sign of its Jacobian, the facing and
  the radius of every interface, and each layer's volume against that of
  the shell it fills. `ValidationReport` gains `max_layer_volume_error`
  and `max_aspect_ratio`, which a builder that does not measure them
  leaves as None.
- `InterfaceSizing.at`: the size a sizing gives at a distance from its
  interface.
- Sections of hexahedra and quadrilaterals. `plotting.plane_cut` cuts
  hexahedra (rows of eight corners) into polygons of up to six sides,
  and with `tensor=True` reads rows of four as quadrilaterals and cuts
  them into segments; `plotting.in_plane` takes quadrilaterals whole;
  `mesh_section` draws either.
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

### Fixed

- A radial field that jumps at a breakpoint inside its layer is sampled
  from inside each element. The model file holds the boundary between
  two elements as two nodes, each with its own side's value; the writer
  and `RadialMesh.nodal` took both from the side above, so the element
  below was read back as a polynomial through one wrong value. An
  element's last node now takes the limit from below where the two
  sides differ (`fields.from_inside`, `PolynomialLayer.below`,
  `RadialField.below`). A field whose pieces meet, which is every field
  the library made before the Bullen excess, is sampled as it was, and
  its files are written bit for bit as before.
