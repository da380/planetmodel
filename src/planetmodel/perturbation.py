"""A radial reference with a perturbation: a kind of field, not two names.

A laterally varying model is often stated as a one-dimensional
reference and a perturbation of it, a tomographic model as relative
perturbations of a reference's velocities, a viscosity as the logarithm
of its departure from a radial profile.  `PerturbedField(reference,
perturbation, *, kind)` is such a field, and the kind is how the two
combine::

    absolute   reference + perturbation
    relative   reference (1 + perturbation)
    log        reference exp(perturbation), the natural logarithm

The reference is a radial field of rank 0; the perturbation is a field
of rank 0 on the same interval, of the reference's character when it is
absolute and dimensionless (`SCALAR`) otherwise.  The field has the
reference's character and is radial exactly when the perturbation is.
`reference`, `perturbation` and `kind` are attributes, so that a writer
of files and a spherical mean ask the field for them rather than a
central module inspecting types.  Under a change of units the reference
rescales by its dimensions and the perturbation by them only when it is
absolute: a relative or logarithmic perturbation is a pure number.

`perturbed(model, name, perturbation, *, kind)` wraps the field of that
name on every layer the perturbation's interval covers, restating the
perturbation on each by `on_interval`, so that one perturbation over the
whole mantle perturbs each of its layers; a layer covered only in part
is refused by name.  It goes through `with_field(replace=True)`, so a
model type's own rule runs: on an `Elastic` model a perturbed velocity
re-derives the Love moduli from it.  `unperturbed(model)` puts every
perturbed field's reference back the same way.  A fully 3D field is
simply a field that is not perturbed; the reference and the spherical
mean of a perturbed field differ in general and are named apart.  A
flat perturbed field (a `DepthField` reference, any flat perturbation)
has a horizontal mean, by the same rules (`horizontal_mean`), and no
spherical one.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from .character import SCALAR
from .coordinates import SPHERICAL, common_coordinates, require_coordinates
from .fields import ComposedField, Field, FieldBase, RadialField
from .layerfunction import same_interval
from .model import Layer, Model

if TYPE_CHECKING:
    from .cartesian.grid import Grid
    from .sampling import AngularGrid

__all__ = ["KINDS", "PerturbedField", "perturbed", "unperturbed"]

#: How a reference and its perturbation combine.
KINDS = ("absolute", "relative", "log")


class PerturbedField(FieldBase):
    """A radial reference of rank 0 combined with a perturbation by `kind`.

    `kind` is one of `KINDS`; `name` is the reference's unless given.
    Both fields must lie on the same interval and in one coordinate
    system, which is the field's, and the perturbation must have the
    reference's character when absolute and be `SCALAR` otherwise.
    """

    def __init__(self, reference: Field, perturbation: Field, *, kind: str,
                 name: str | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if (reference.character.rank or not getattr(reference, "is_radial", False)
                or getattr(reference, "perturbation", None) is not None):
            raise ValueError(
                f"the reference of a perturbed field is a radial field of rank "
                f"0; {reference!r} is not (a perturbed field is perturbed once: "
                "combine the perturbations, or perturb its reference)")
        if perturbation.character.rank:
            raise ValueError(f"a perturbation is of rank 0, not "
                             f"{perturbation.character}")
        want = reference.character if kind == "absolute" else SCALAR
        if perturbation.character != want:
            raise ValueError(
                f"an {kind} perturbation has the character {want}, got "
                f"{perturbation.character}" if kind == "absolute" else
                f"a {kind} perturbation is a pure number, of character {want}; "
                f"got {perturbation.character}")
        rtol = float(getattr(reference, "rtol", 1e-9))
        if not same_interval(reference.interval, perturbation.interval, rtol=rtol):
            raise ValueError(
                f"the perturbation lies on {perturbation.interval}, the reference "
                f"on {reference.interval}; restate it with on_interval")
        self._coordinates = common_coordinates((reference, perturbation))
        self._reference, self._perturbation, self._kind = reference, perturbation, kind
        self._interval = tuple(float(x) for x in reference.interval)
        self._character = reference.character
        self._name = reference.name if name is None else name
        self._rtol = rtol

    @property
    def reference(self) -> Field:
        """The radial reference."""
        return self._reference

    @property
    def perturbation(self) -> Field:
        """The perturbation of the reference."""
        return self._perturbation

    @property
    def kind(self) -> str:
        """How the two combine, one of `KINDS`."""
        return self._kind

    @property
    def is_radial(self) -> bool:
        """Whether the field depends on the radius alone: when the
        perturbation does."""
        return bool(getattr(self._perturbation, "is_radial", False))

    def _values(self, r: np.ndarray, theta: np.ndarray | None,
                phi: np.ndarray | None) -> np.ndarray:
        ref = self._reference.evaluate(r, theta, phi)
        dp = self._perturbation.evaluate(r, theta, phi)
        if self._kind == "absolute":
            return ref + dp
        if self._kind == "relative":
            return ref * (1.0 + dp)
        return ref * np.exp(dp)

    def radial_pieces(self) -> tuple[np.ndarray, int] | None:
        """The breakpoints and degree that hold both parts exactly, each
        being held apart: the union of their breakpoints and the higher
        of their degrees, or None when either part has none."""
        parts = [getattr(f, "radial_pieces", lambda: None)()
                 for f in (self._reference, self._perturbation)]
        if any(p is None for p in parts):
            return None
        return (np.unique(np.concatenate([p[0] for p in parts])),
                max(p[1] for p in parts))

    def spherical_mean(self, *, grid: AngularGrid | None = None) -> RadialField:
        """The mean from the parts: reference + mean(perturbation) when
        absolute, reference (1 + mean(perturbation)) when relative, and
        reference mean(exp(perturbation)) when logarithmic, the last by
        quadrature on `grid`, or on the perturbation's own grid where it
        has weights; a radial perturbation's mean is itself.  A field of
        another system than the sphere's is refused."""
        from .means import quadrature_mean, spherical_mean
        require_coordinates(self, SPHERICAL, "a spherical mean")
        return self._mean(spherical_mean, quadrature_mean, grid,
                          lambda g: getattr(g, "weights", None) is not None,
                          "a grid with weights")

    def horizontal_mean(self, *, grid: Grid | None = None) -> RadialField:
        """The mean over a box's period at every depth from the parts, by
        the rules of `spherical_mean` with the horizontal mean
        (`cartesian.horizontal_mean`) in place of the spherical one, the
        quadrature of a log perturbation on `grid`, a `cartesian.Grid`, or
        on the perturbation's own grid.  A field of another system than
        the flat one is refused."""
        from .cartesian.coordinates import FLAT
        from .cartesian.grid import Grid
        from .cartesian.means import horizontal_mean, quadrature_mean
        require_coordinates(self, FLAT, "a horizontal mean")
        return self._mean(horizontal_mean, quadrature_mean, grid,
                          lambda g: isinstance(g, Grid), "a cartesian.Grid")

    def _mean(self, mean: Callable[..., RadialField],
              quadrature: Callable[[Field, Any], RadialField], grid: Any,
              usable: Callable[[Any], bool], wanted: str) -> RadialField:
        """The mean from the parts by `mean`, the system's mean of a field,
        and `quadrature`, its mean on a grid that is `usable`."""
        ref, d = self._reference, self._perturbation
        if self._kind == "absolute":
            out = ref + mean(d, grid=grid)
        elif self._kind == "relative":
            out = ref * (1.0 + mean(d, grid=grid))
        else:
            ex = ComposedField(np.exp, [d], character=SCALAR)
            if getattr(d, "is_radial", False):
                ex_mean = mean(ex)
            else:
                g = grid if grid is not None else getattr(d, "grid", None)
                if g is None or not usable(g):
                    raise ValueError(
                        "the mean of a log perturbation is the mean of its "
                        "exponential, taken by quadrature: pass grid=, "
                        f"{wanted}")
                ex_mean = quadrature(ex, g)
            out = ref * ex_mean
        return out.renamed(self._name)

    def on_interval(self, lo: float, hi: float) -> "PerturbedField":
        """Both parts on [lo, hi]."""
        return PerturbedField(self._reference.on_interval(lo, hi),
                              self._perturbation.on_interval(lo, hi),
                              kind=self._kind, name=self._name)

    def rescaled(self, *, k: float, v: float) -> "PerturbedField":
        """v f(r / k, theta, phi): the reference by (k, v), the perturbation
        by (k, v) when absolute and by (k, 1) otherwise."""
        dv = v if self._kind == "absolute" else 1.0
        return PerturbedField(self._reference.rescaled(k=k, v=v),
                              self._perturbation.rescaled(k=k, v=dv),
                              kind=self._kind, name=self._name)

    def renamed(self, name: str | None) -> "PerturbedField":
        """The same field under `name`."""
        return PerturbedField(self._reference, self._perturbation, kind=self._kind,
                              name=name)

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return (f"PerturbedField({nm}{self._kind}, {self._character} on "
                f"[{lo:g}, {hi:g}])")


def perturbed(model: Model, name: str, perturbation: Field, *, kind: str,
              layers: Iterable[int | str] | None = None) -> Model:
    """`model` with the field `name` perturbed on the layers the
    perturbation's interval covers, or on `layers` when given.

    Each layer's field becomes `PerturbedField(field, perturbation
    restated on the layer, kind=kind)`, attached through
    `with_field(replace=True)`.  A layer the interval covers only in
    part, a chosen layer it does not cover, a layer lacking `name`, and
    an interval covering no layer are refused by name.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    lo, hi = (float(x) for x in perturbation.interval)
    tol = model.geometry.rtol * model.skeleton.span

    def covered(layer: Layer) -> bool | None:
        a, b = layer.interval
        if a >= lo - tol and b <= hi + tol:
            return True
        if b <= lo + tol or a >= hi - tol:
            return False
        return None                                 # in part

    if layers is None:
        chosen = []
        for layer in model.layers:
            c = covered(layer)
            if c is None:
                raise ValueError(
                    f"the perturbation on [{lo:g}, {hi:g}] covers layer "
                    f"{layer.index} ({layer.name}) on [{layer.interval[0]:g}, "
                    f"{layer.interval[1]:g}] only in part; a perturbation "
                    "perturbs whole layers")
            if c:
                chosen.append(layer)
        if not chosen:
            raise ValueError(f"the perturbation on [{lo:g}, {hi:g}] covers no "
                             "layer of the model")
    else:
        chosen = [model.layer(w) for w in layers]
        for layer in chosen:
            if not covered(layer):
                raise ValueError(
                    f"layer {layer.index} ({layer.name}) on [{layer.interval[0]:g},"
                    f" {layer.interval[1]:g}] is not covered by the perturbation "
                    f"on [{lo:g}, {hi:g}]")
    out = model
    for layer in chosen:
        if name not in layer:
            raise KeyError(f"layer {layer.index} ({layer.name}) holds no field "
                           f"{name!r} to perturb")
        a, b = layer.interval
        field = PerturbedField(layer[name], perturbation.on_interval(a, b),
                               kind=kind)
        out = out.with_field(layer.index, name, field, replace=True)
    return out


def unperturbed(model: Model) -> Model:
    """`model` with every perturbed field replaced by its reference, through
    `with_field(replace=True)`."""
    out = model
    for layer in model.layers:
        for name in layer.names:
            reference = getattr(layer[name], "reference", None)
            if reference is not None:
                out = out.with_field(layer.index, name, reference, replace=True)
    return out
