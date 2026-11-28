"""Named flat model types: a layered isotropic model and a crust over a
mantle.

Each is a class derived from `cartesian.Model` alone, with the
behaviours of `planetmodel.behaviours` that are pointwise mixed in:
`Elastic` attaches the five Love moduli beside the velocities a layer
is given (or the other way round), `ConstantQ` disperses them where a
layer holds quality factors, and `Viscoelastic` freezes the model at a
frequency where a layer holds a viscosity or Prony branches.  None is
`SelfGravitating`: gravity from the mass inside a radius is the
sphere's, and a flat model's gravity is prescribed
(`Model.with_uniform_gravity`).  Every model is in SI, converted to
other scales by `converted`, and built over a lateral `Box` or
`Rectangle` given to its constructor: the box is as large as the
problem it serves needs, which no default could know.  Below its base a
model is padded by `extended(depths, fields="hold")`.

`LayeredIsotropicElastic` is the flat counterpart of the spherical one:
`rho`, `vp` and `vs` per layer, a constant or the coefficients of a
polynomial in depth / scale (`catalogue.isotropic_layers`), a fluid
layer one with vs = 0; `homogeneous` is a single layer from the surface
down.

`CrustOverMantle` is the reference flat model of the tests and
examples: round values typical of a continental crust over the
uppermost mantle, in SI::

    layer                 depth (km)   rho (kg/m^3)   vp (m/s)   vs (m/s)
    crust                 0 - 30       2800           6400       3700
    lithospheric_mantle   30 - 100     3300           8000       4500
    asthenosphere         100 - base   3300           8000       4500

with the interfaces `surface`, `moho` and `lab` (the base of the
lithosphere) and the base at 200 km unless `base` says otherwise; with
`viscosity` (Pa s) the asthenosphere is a Maxwell body, its elastic
values unrelaxed.  It holds no quality factors; one put on a layer
(`with_field`) disperses its moduli by `moduli_at`.
"""
from __future__ import annotations

from collections.abc import Sequence

from ..behaviours import ConstantQ, Elastic, Viscoelastic
from ..catalogue import LayerValue, isotropic_layers
from ..layerfunction import constant_layer
from ..skeleton import Skeleton
from ..units import Scales
from .domains import Domain
from .geometry import Geometry
from .model import Model

__all__ = ["LayeredIsotropicElastic", "CrustOverMantle", "CRUST_OVER_MANTLE"]

#: `CrustOverMantle`'s layers from the surface down: name, top depth (m),
#: and rho (kg/m^3), vp and vs (m/s).
CRUST_OVER_MANTLE = (("crust", 0.0, 2800.0, 6400.0, 3700.0),
                     ("lithospheric_mantle", 30e3, 3300.0, 8000.0, 4500.0),
                     ("asthenosphere", 100e3, 3300.0, 8000.0, 4500.0))

#: `CrustOverMantle`'s interfaces from the surface down.
_CRUST_OVER_MANTLE_INTERFACES = ("surface", "moho", "lab")


class LayeredIsotropicElastic(Elastic, Model):
    """An isotropic flat model of exact polynomial layers between `depths`
    over the lateral `domain`.

    `depths` are the skeleton's, from the surface down (a first depth
    below zero is refused, one above it gives a body whose top lies
    below the free surface); `rho`, `vp` and `vs` give one entry per
    layer, a number for a constant or a sequence of ascending polynomial
    coefficients in depth / scale, and a layer with vs = 0 throughout is
    fluid.  `name` is the model's.
    """

    def __init__(self, depths: Sequence[float], domain: Domain, *,
                 rho: Sequence[LayerValue], vp: Sequence[LayerValue],
                 vs: Sequence[LayerValue], scale: float = 1.0,
                 layer_names: Sequence[str | None] | None = None,
                 interface_names: Sequence[str | None] | None = None,
                 scales: Scales = Scales.SI, name: str | None = None) -> None:
        geometry = Geometry(Skeleton(depths), domain, layer_names=layer_names,
                            interface_names=interface_names)
        layers = isotropic_layers(geometry, rho=rho, vp=vp, vs=vs, scale=scale)
        super().__init__(geometry, layers, scales=scales, name=name)

    @classmethod
    def homogeneous(cls, depth: float, domain: Domain, *, rho: float, vp: float,
                    vs: float, name: str | None = None,
                    scales: Scales = Scales.SI) -> "LayeredIsotropicElastic":
        """A uniform isotropic slab from the surface to `depth` over
        `domain`: constant rho, vp and vs; `name` names the model and its
        one layer."""
        return cls([0.0, depth], domain, rho=[rho], vp=[vp], vs=[vs],
                   layer_names=None if name is None else [name], scales=scales,
                   name=name)


class CrustOverMantle(Elastic, ConstantQ, Viscoelastic, Model):
    """A crust over the uppermost mantle, over the lateral `domain`, in SI;
    see the module docstring for its values.

    `base` is the depth of the base in metres, below the base of the
    lithosphere at 100 km; `viscosity`, in Pa s, makes the asthenosphere
    a Maxwell body of that constant viscosity, None leaving the model
    elastic.
    """

    def __init__(self, domain: Domain, *, base: float = 200e3,
                 viscosity: float | None = None) -> None:
        tops = [top for _, top, *_ in CRUST_OVER_MANTLE]
        if not float(base) > tops[-1]:
            raise ValueError(f"the base lies below the base of the lithosphere at "
                             f"{tops[-1]:g} m; got {base:g}")
        if viscosity is not None and not float(viscosity) > 0.0:
            raise ValueError(f"the viscosity is positive, got {viscosity!r}")
        geometry = Geometry(Skeleton(tops + [float(base)]), domain,
                            layer_names=[row[0] for row in CRUST_OVER_MANTLE],
                            interface_names=_CRUST_OVER_MANTLE_INTERFACES)
        layers = isotropic_layers(geometry,
                                  rho=[row[2] for row in CRUST_OVER_MANTLE],
                                  vp=[row[3] for row in CRUST_OVER_MANTLE],
                                  vs=[row[4] for row in CRUST_OVER_MANTLE])
        name = "crust over mantle"
        if viscosity is not None:
            iv = geometry.skeleton.interval(-1)
            layers[-1]["viscosity"] = Geometry.radial_field_type(
                iv, constant_layer(iv, float(viscosity)), name="viscosity")
            name += " (Maxwell asthenosphere)"
        super().__init__(geometry, layers, scales=Scales.SI, name=name)
