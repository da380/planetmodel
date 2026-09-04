"""time.py -- fields that depend on time: `evaluate(..., t=)`.

A time-dependent field is a Field with one more keyword, `t`, real, in
the body's own units (seconds while the scales are SI).  Values are
real.  The relaxation and creep functions of a time-domain code are
the intended occupants; none ships yet, and the kind exists so that
they have a home with the same lifts, algebra, composition and
restriction as the frequency kind, and so that passing between the two
is an explicit transform rather than a convention.

    lifted_to_time(field)         a static field that ignores t
    ComposedTimeField(fn, ...)    fn(t, *values)
    at_time(field, t)             the field at one t, as a static Field
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .dependent import ComposedDependentField, FrozenField, LiftedField

__all__ = ["TimeDependentField", "ComposedTimeField", "lifted_to_time",
           "at_time"]


@runtime_checkable
class TimeDependentField(Protocol):
    """Values of a Character on a Skeleton, at points and a time."""

    skeleton: object
    character: object
    name: str | None
    kind: str                 # "time"

    def evaluate(self, r, theta=None, phi=None, *, t, layer=None,
                 side: str = "upper", frame: str = "spherical"):
        """Values at broadcast (r, theta, phi) and one time t."""
        ...


class ComposedTimeField(ComposedDependentField):
    """fn(t, *values): a relaxation or creep law applied to its parameters."""

    def __init__(self, fn, sources, *, character, kind: str = "time",
                 dimensions=None, name: str | None = None,
                 omega_domain: str = "complex", law=None) -> None:
        if kind != "time":
            raise ValueError(f"a ComposedTimeField is of kind 'time', not "
                             f"{kind!r}")
        super().__init__(fn, sources, character=character, kind=kind,
                         dimensions=dimensions, name=name, law=law)


def lifted_to_time(field, *, name: str | None = None):
    """A static field as a time-dependent one that ignores t."""
    if getattr(field, "kind", "static") == "time":
        return field
    return LiftedField(field, kind="time", name=name)


def at_time(field, t) -> FrozenField:
    """A time-dependent field at one t, as a static Field."""
    return FrozenField(field, t)
