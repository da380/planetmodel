"""base.py -- what it means to be a field on the reference body.

A Field is values with a tensor character, evaluable anywhere on the
reference body.  The contract is deliberately structural: an object with
`skeleton`, `character`, `name` and `evaluate` *is* a Field, whether or
not it inherits from anything here, so a random-field sample, a wrapper
round an external model, or a two-line closure all qualify without
planetmodel knowing about them.

`evaluate` returns an array of shape

    broadcast(r, theta, phi).shape + character.component_shape

with ranks 2 and 4 Voigt-reduced.  At a discontinuity both one-sided
values are meaningful, so `layer=` names a side explicitly and `side=`
breaks the tie when it does not.

Components are in the frame the coordinates imply.  `evaluate` speaks
(r, theta, phi), so it returns components in the local
(e_r, e_theta, e_phi) frame; `frame="cartesian"` asks for the other, by
conjugation with the frame matrix R = [e_r, e_theta, e_phi] and, for a
Voigt matrix, by its Bond transformation (materials.bond_matrix).  An
implementation may support only its natural frame and raise ValueError
on the other, but it may not quietly ignore the argument and return the
wrong components under the right name.

`FieldBase` (composite.py) is the optional [extra] tier: it supplies the
algebra -- +, -, unary -, scalar * and /, closed over the composites --
and `is_radial`, `domain`, `restricted`, `on_interval` and `rescaled`,
to implementations that want them for free.  Every shipped field,
composites included, inherits it.

`Assemblable` is the second, smaller contract, between a field type and
the body that stores it.  A body keeps single-layer pieces and hands
back body-wide views; surgery re-states pieces on new intervals; a
change of scales converts them.  A field type that says what it is
built from (`operands`) and how to build itself again (`rebuilt_from`,
`assembled`) gets all of that generically; `RadialField`,
`AnalyticField` and `ElasticField` implement the three directly.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Field", "Assemblable"]


@runtime_checkable
class Field(Protocol):
    """Values of a given Character on a Skeleton, evaluable at points."""

    skeleton: object          # Skeleton; typed loosely to avoid a cycle
    character: object         # Character; every shipped field sets one
    name: str | None

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical"):
        """Values at broadcast (r, theta, phi), in the requested frame.

        `r`, `theta` and `phi` broadcast together and the result has the
        broadcast shape followed by the component shape, whether or not
        the field depends on the angles: a radial field given angles
        returns values of the full broadcast shape.  Angles may be
        omitted only by fields that do not depend on them.  `layer`
        selects one side at a discontinuity; with layer=None the radius
        is resolved against the skeleton and `side` ("upper" or
        "lower") breaks ties on the skeleton's boundaries only, not on
        boundaries a merged layer once had.  `frame` is
        "spherical" (the default, the frame the coordinates imply) or
        "cartesian"; a field that cannot supply one raises ValueError
        naming the frame.
        """
        ...


@runtime_checkable
class Assemblable(Protocol):
    """A field that can be taken apart and put together again.

    `operands()` are the fields this one is built from, in a fixed
    order; `rebuilt_from(operands)` is the same construction on other
    operands, which is how restriction to a layer, re-statement on an
    interval and a change of scales are done generically: apply the
    operation to each operand and rebuild.  `assembled(skeleton,
    pieces)` is the classmethod a body calls to turn single-layer
    pieces of this type into one field on its skeleton, returning
    NotImplemented when the pieces do not fit together, in which case
    the body falls back to a generic layerwise view.

    `matches(other)`, optional, says whether two pieces are the same
    construction and so may be assembled operand by operand; the
    default compares the type and the number of operands.
    """

    def operands(self) -> tuple:
        ...

    def rebuilt_from(self, operands, *, name: str | None = None):
        ...

    @classmethod
    def assembled(cls, skeleton, pieces, *, name: str | None = None):
        ...
