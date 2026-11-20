# Changelog

The notes of the next release, drafted as the work lands; a release's
notes are published on its GitHub release, and the section here starts
again empty.

## Unreleased

### Fixed

- The C++ header `planetmodel/fault.hpp` compiles under Clang 18 with
  OpenMP enabled: `ParameterRule`'s lambda takes the quadrature's points
  and weights by init-capture, where it captured a structured binding,
  which that compiler refuses in a translation unit built with
  `-fopenmp`. Nothing changes in what the function returns.
