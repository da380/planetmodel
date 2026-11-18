# Changelog

The notes of the next release, drafted as the work lands; a release's
notes are published on its GitHub release, and the section here starts
again empty.

## Unreleased

### Added

- Transforms at prepared points (`planetmodel.fourier`):
  `Transform.at(points, *, route=, epsilon=, nthreads=)` prepares a set
  of points once to be applied many times, a `fourier.PointTransform`
  whose `synthesise`, `synthesise_gradient` and `accumulate` take no
  points and give the results of the transform's methods of those names
  (to the bit, finufft on one thread). The points' angles in the box are
  taken once, and on finufft's route the first application of each type
  of transform to each number of fields at once makes finufft's plan, the
  points sorted, and keeps it; on the direct route the angles alone are
  kept. `Transform.synthesise`, `synthesise_gradient` and `accumulate`
  are each one application of such a preparation, their results and
  refusals as before. A preparation is copied and pickled without its
  plans, made again on first use, and applications from several threads
  take turns on them. Measured on one thread at 20 000 points on a
  64 x 64 grid: an application 2.4 to 2.8 ms against 3.4 to 4.3 ms with
  the points given (about a third less, the points' angles 0.5 ms and
  finufft's planning and sorting the rest); five fields at once 13.5 to
  14.6 ms against 14.8 to 15.8; 25 points on 16 x 16, 0.03 ms against
  0.3.

### Changed

- `fourier.Transform.from_rfft` and `to_rfft` take one array or vector
  without batch axes by their own steps, the batched ones' arithmetic on
  the same entries, so the values are as before to the bit: on a
  16 x 16 grid 3.0 microseconds against 6.5 and 5.5 against 11.7, about
  half on every grid measured (one to three axes, to 64 x 64).
  `Packing.size` is held rather than recomputed.
