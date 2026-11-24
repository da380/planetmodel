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

### Fixed

- The C++ header `planetmodel/fault.hpp` compiles under Clang 18 with
  OpenMP enabled: `ParameterRule`'s lambda takes the quadrature's points
  and weights by init-capture, where it captured a structured binding,
  which that compiler refuses in a translation unit built with
  `-fopenmp`. Nothing changes in what the function returns.
