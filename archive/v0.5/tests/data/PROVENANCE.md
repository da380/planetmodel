# Reference data

## `prem.nocrust`

An isotropic mineos-style deck, copied from `mfemElasticity/data/` (David
Al-Attar's C++ finite-element library, the consumer this meshing work
serves). Used to exercise `read_isotropic_deck` and, from M4, as the model
behind the acceptance benchmark.

- 3 header lines: a title; `ifanis tref ifdeck`; `nknot nic noc`.
- Then knots, one per line: `r rho vp vs qkappa qmu`.
- SI units, radius in **metres**. A repeated radius marks a discontinuity.
- The outermost radius is 6346600 m -- PREM's Moho. The crust is absent by
  construction, which is why the acceptance benchmark supplies crustal
  structure from CRUST-1.0 instead.
- `vs == 0` holds on exactly one layer, the fluid outer core.

## CRUST-1.0

`crust-1.0/crsthk.xyz` and `crust-1.0/depthtomoho.xyz`, copied at M4 from
`mfemElasticity/data/crust-1.0/`. Both are `lon lat value` on a 1-degree
grid with values in **kilometres** and latitude **descending** through the
file; readers must sort rather than assume order.

- `depthtomoho.xyz` is the depth to the Moho about sea level, so negative
  everywhere: -74.81 km at its deepest, under the Tibetan plateau.
- `crsthk.xyz` is the crustal thickness, positive everywhere.
- Their **sum is the surface elevation**, which is how the acceptance
  geometry gets both boundaries from one pair of files.
- Area-weighted means: Moho -21.42 km, surface -2.38 km, so a mean crust
  19.04 km thick. The unweighted mean of the Moho grid is -22.90 km --
  1.48 km deeper, because a lon-lat grid packs cells towards the poles
  and the poles carry thick crust. That difference is the radius the
  Moho would be placed at, which is why planetmodel weights by area.
- sha256:
  `18729df6312b6df303dea4445a6757d7c7e69c5ce086cba5937fa67108c45ace`
  (crsthk) and
  `167a9cbd698ab709dbab9ec65e9701454533d264e9a446e424f4d4f02ceeb98c`
  (depthtomoho).

CRUST-1.0's mean Moho, 6349.58 km, sits **above** `prem.nocrust`'s outer
radius of 6346.6 km: a deck with no crust has to grow to meet it rather
than be cut down to it.

Tests marked `data` are the ones that need these files.

Nothing here is imported from mfemElasticity at runtime: it is a
read-only reference checkout, and these are copies.
