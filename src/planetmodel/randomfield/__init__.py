"""Gaussian random fields of Matern type on balls, annuli and layers.

The core numerics of the SPDE construction on a radial spectral-element
mesh: `RadialOperatorFamily` discretises the degree-indexed radial
operators A_l of A = 1 - div(kappa grad) with their spectra, powers,
inverses and white noise, which is what a space, an operator and a
Gaussian measure are made of; `RadialGRF`, `SphericalGRF` and
`LayeredGRF` sample fields of radius, of a shell and of the layers of a
skeleton by Karhunen-Loeve expansion with an exact marginal standard
deviation; `harmonics` holds the real orthonormal spherical harmonics
the shell fields are expanded in.  `fields` states the construction.

    grf = RadialGRF(r1, r2, nu, lam, sigma=sigma)
    grf.to_field(grf.sample(rng=rng))              # a RadialField
    shell = SphericalGRF(r1, r2, nu, lam, lmax=24)
    shell.to_field(shell.sample(rng=rng))          # an AnalyticField
    LayeredGRF(model, nu, lam, layers=mantle).sample(rng=rng)
"""
from .fields import LayeredGRF, RadialGRF, SphericalGRF
from .harmonics import real_harmonics, synthesise
from .operator import RadialOperatorFamily

__all__ = [
    "RadialOperatorFamily",
    "RadialGRF", "SphericalGRF", "LayeredGRF",
    "real_harmonics", "synthesise",
]
