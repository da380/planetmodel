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

### Fixed

- The C++ header `planetmodel/fault.hpp` compiles under Clang 18 with
  OpenMP enabled: `ParameterRule`'s lambda takes the quadrature's points
  and weights by init-capture, where it captured a structured binding,
  which that compiler refuses in a translation unit built with
  `-fopenmp`. Nothing changes in what the function returns.
