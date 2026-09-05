"""The quasi-static loading and tidal problem of a spherically layered body.

A spherically symmetric, self-gravitating, hydrostatically pre-stressed
body responds to a surface load or an external tidal potential by a
displacement and a perturbation of its gravitational potential.  This
sub-package solves that problem degree by degree on a `RadialMesh` by
the reduced weak form of Al-Attar & Tromp (2014), Appendix D, with
transversely isotropic elasticity, fluid regions characterised by the
potential alone, and the exterior closed by Dirichlet-to-Neumann terms;
`assembly` states the form.  The results are the generalised Love
numbers of Al-Attar et al. (2024), the conventional load and tidal
numbers derived from them, the radial solutions themselves, and the
file pyslfp reads; `love` states the conventions.

The material enters through `Material`, which reads a model on a mesh
once: density, its gradient, gravity, fluidity and the five moduli at
the nodes.  The model is anything holding density and what
`planetmodel.moduli` reads the five moduli from, real or complex: a
viscoelastic body is a model frozen at a frequency by
`planetmodel.frozen`, and its Love numbers are complex.  Everything is
in the model's units with the model's G, and `LoveNumbers.in_si()` or
`write` converts at the end.

    material = Material(mesh, model)
    love = love_numbers(material, 64)
    love.conventional()["h"]                  # h' by degree
    love.write("love.dat")                    # for pyslfp
    solve_degree(material, 2, forcing="tide").evaluate(radii)
    love_numbers(Material(mesh, frozen(model, omega)), 2).tidal()["k"]
"""
from .assembly import DegreeSystem
from .love import (FORCINGS, DegreeSolution, LoveNumbers, love_numbers,
                   read_love_numbers, solve_degree)
from .material import Material, NodalModuli, nodal_moduli

__all__ = [
    "Material", "NodalModuli", "nodal_moduli",
    "DegreeSystem",
    "FORCINGS", "DegreeSolution", "solve_degree",
    "LoveNumbers", "love_numbers", "read_love_numbers",
]
