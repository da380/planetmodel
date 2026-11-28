"""Gaussian random fields of Matern type on balls, annuli and layers.

The core numerics of the SPDE construction on a radial spectral-element
mesh, in three tiers.  `SeparableOperatorFamily` discretises the
operators A = 1 - div(kappa grad) separates into, one per lateral
eigenvalue, with their spectra, powers, inverses and white noise: the
degree-indexed radial operators A_l under the r^2 dr measure of a ball
or annulus by default, those of a disk, of a flat layer or of an
interval of the line by its `weight`, `lateral` and `index`.
`padded_mesh` and `restriction` build the mesh such a family is put on,
longer than the physical interval so that its boundary conditions act
at a distance, and `SpectralBasis` is the truncated eigenbasis of one
degree on it: coefficients to nodal values and back, evaluation at any
radius, integration over the physical interval, and `theta`, of which
every Sobolev metric and Matern covariance is a power; `SphericalBasis`
holds the bases of degrees 0 ... lmax and fixes the order of their
coefficients as one vector.  That is what a space, an operator and a
Gaussian measure are made of.  `RadialGRF`, `SphericalGRF` and
`LayeredGRF` are the samplers built on them: fields of radius, of
a shell and of the layers of a skeleton, by Karhunen-Loeve expansion in
those bases, with an exact marginal standard deviation.  A sample is
the vector of the field's coefficients in its basis, and the values at
the nodes, the values on a grid and the field a model holds are views
of a vector, or of a batch of them.  The shell fields are expanded in
the real orthonormal harmonics of `planetmodel.harmonics`, synthesised
through ducc0; everything else here is numpy and scipy.  `fields`
states the construction of the samplers, `basis` that of the bases.
The samplers of flat layers, Fourier series over a box with the family
in depth, are in `planetmodel.cartesian.randomfield`, on the same
family, bases and helpers and with the same methods.

::

    grf = RadialGRF(r1, r2, nu, lam, sigma=sigma)
    c = grf.sample(rng=rng)                        # the coefficient vector
    grf.synthesise(c)                              # values at the nodes grf.r
    grf.to_field(c)                                # a RadialField
    shell = SphericalGRF(r1, r2, nu, lam, lmax=24)
    c = shell.sample(rng=rng)
    shell.to_field(c)                              # a HarmonicField
    shell.on_grid(c, gauss_legendre(24))           # nodes x grid, ducc0
    layered = LayeredGRF(model, nu, lam, layers=mantle)
    layered.to_fields(layered.sample(rng=rng))     # a RadialField per layer

    mesh = padded_mesh(a, b, pad=2 * matern_reach(nu, lam), weight="one")
    family = SeparableOperatorFamily(mesh, kappa=lam ** 2, weight="one")
    basis = SpectralBasis(family, 0, restrict=restriction(mesh, a, b), nmodes=64)
    u = basis.synthesise(basis.theta ** -beta * basis.white_noise(rng=rng))
"""
from .basis import SpectralBasis, SphericalBasis
from .fields import LayeredGRF, RadialGRF, SphericalGRF
from .mesh import Restriction, matern_reach, padded_mesh, restriction
from .operator import SeparableOperatorFamily

__all__ = [
    "SeparableOperatorFamily",
    "matern_reach", "padded_mesh", "Restriction", "restriction",
    "SpectralBasis", "SphericalBasis",
    "RadialGRF", "SphericalGRF", "LayeredGRF",
]
