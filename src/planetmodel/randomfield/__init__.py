"""Gaussian random fields of Matern type on balls, annuli and layers.

The core numerics of the SPDE construction on a radial spectral-element
mesh, in three tiers.  `RadialOperatorFamily` discretises the
degree-indexed radial operators A_l of A = 1 - div(kappa grad) with
their spectra, powers, inverses and white noise, under the r^2 dr
measure of a ball or annulus or the plain measure of an interval.
`padded_mesh` and `restriction` build the mesh such a family is put on,
longer than the physical interval so that its boundary conditions act
at a distance, and `SpectralBasis` is the truncated eigenbasis of one
degree on it: coefficients to nodal values and back, evaluation at any
radius, integration over the physical interval, and `theta`, of which
every Sobolev metric and Matern covariance is a power; `SphericalBasis`
holds the bases of degrees 0 ... lmax and fixes the order of their
coefficients as one vector.  That is what a space, an operator and a
Gaussian measure are made of.  `RadialGRF`, `SphericalGRF` and
`LayeredGRF` are the first consumers: they sample fields of radius, of
a shell and of the layers of a skeleton by Karhunen-Loeve expansion in
those bases, with an exact marginal standard deviation.  The shell
fields are expanded in the real orthonormal harmonics of
`planetmodel.harmonics`, whose grid synthesis needs the `harmonics`
extra (pyshtools); everything else here is numpy and scipy.  `fields`
states the construction of the samplers, `basis` that of the bases.

    grf = RadialGRF(r1, r2, nu, lam, sigma=sigma)
    grf.to_field(grf.sample(rng=rng))              # a RadialField
    shell = SphericalGRF(r1, r2, nu, lam, lmax=24)
    shell.to_field(shell.sample(rng=rng))          # an AnalyticField
    shell.sample_grid(gauss_legendre(24), rng=rng)   # nodes x grid, pyshtools
    LayeredGRF(model, nu, lam, layers=mantle).sample(rng=rng)

    mesh = padded_mesh(a, b, pad=2 * matern_reach(nu, lam), weight="one")
    family = RadialOperatorFamily(mesh, kappa=lam ** 2, weight="one")
    basis = SpectralBasis(family, 0, restrict=restriction(mesh, a, b), nmodes=64)
    u = basis.synthesise(basis.theta ** -beta * basis.white_noise(rng=rng))
"""
from .basis import SpectralBasis, SphericalBasis
from .fields import LayeredGRF, RadialGRF, SphericalGRF
from .mesh import Restriction, matern_reach, padded_mesh, restriction
from .operator import RadialOperatorFamily

__all__ = [
    "RadialOperatorFamily",
    "matern_reach", "padded_mesh", "Restriction", "restriction",
    "SpectralBasis", "SphericalBasis",
    "RadialGRF", "SphericalGRF", "LayeredGRF",
]
