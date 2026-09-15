"""Radial profiles of spherically symmetric models, drawn one way.

A profile of a radial field against radius is drawn with the radius on
the vertical axis increasing upward, the value along the horizontal,
one segment per layer, and at every interior boundary a horizontal
line joining the two one-sided values, so that a discontinuity reads
as a step and the profile as one unbroken curve; a faint horizontal
guide at every layer boundary sits beneath the curves.  `profile` draws
a sequence of radial fields, one per layer in order; `radial_profile`
draws a named field of a model, skipping the layers that lack it.  Both
return the matplotlib lines they drew and leave the labelling to the
caller.  matplotlib is imported here and nowhere else in the library:
the `plot` extra.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .fields import Field

if TYPE_CHECKING:
    from .model import Model

__all__ = ["profile", "radial_profile", "BOUNDARY_STYLE"]


#: The style of the guide line drawn at a layer boundary, beneath the curves.
BOUNDARY_STYLE = {"color": "0.85", "lw": 0.8, "zorder": 0.5}


def profile(ax: Any, fields: Sequence[Field | None], *, scale: float = 1.0,
            value_scale: float = 1.0, n: int = 200, label: str | None = None,
            joiners: bool = True, boundaries: bool = True, **style: Any) -> list[Any]:
    """Draw radial fields against radius on `ax`, radius upward.

    `fields` are radial fields of rank 0, one per layer in order from the
    centre, or None for a layer to leave blank.  Radii are multiplied by
    `scale` (1e-3 for kilometres from metres) and values by
    `value_scale` before drawing; `n` points per layer; `label` goes on
    the first segment; every segment takes the same colour, from
    `style` or the axes' cycle.  Where two consecutive layers meet, a
    horizontal line joining their one-sided values is drawn in the same
    style, unless `joiners` is False; with `boundaries`, a faint
    horizontal guide (`BOUNDARY_STYLE`) is drawn at every end of every
    field's interval, beneath the curves.  Returns the profile lines
    drawn; a joiner carries the label "_joiner", which keeps it out of
    legends, and the guides are not returned.
    """
    if boundaries:
        for b in sorted({end for f in fields if f is not None for end in f.interval}):
            ax.axhline(b * scale, **BOUNDARY_STYLE)
    lines = []
    color = style.pop("color", style.pop("c", None))
    previous: tuple[float, float] | None = None
    for field in fields:
        if field is None:
            previous = None
            continue
        if not getattr(field, "is_radial", False) or field.character.rank != 0:
            raise ValueError(f"{field!r} is not a radial field of rank 0")
        lo, hi = field.interval
        r = np.linspace(lo, hi, n)
        v = field(r)
        if np.iscomplexobj(v):
            raise TypeError(f"{field!r} is complex; draw its real or imaginary "
                            "part as a field of its own")
        line, = ax.plot(v * value_scale, r * scale, color=color,
                        label=label if not lines else None, **style)
        color = line.get_color()
        meets = previous is not None and abs(lo - previous[0]) <= 1e-9 * (hi - lo)
        if joiners and meets:
            jump = ax.plot([previous[1] * value_scale, v[0] * value_scale],
                           [lo * scale, lo * scale], color=color, label="_joiner",
                           **style)
            lines.extend(jump)
        lines.append(line)
        previous = (hi, float(v[-1]))
    return lines


def radial_profile(ax: Any, model: Model, name: str, *, scale: float = 1.0,
                   value_scale: float = 1.0, n: int = 200, label: str | None = None,
                   joiners: bool = True, boundaries: bool = True,
                   **style: Any) -> list[Any]:
    """Draw the field `name` of a model against radius on `ax`, radius
    upward, one segment per layer holding it, a line across every
    discontinuity and a guide at every boundary of the skeleton; see
    `profile` for the arguments.  A layer that lacks the name is left
    blank."""
    if boundaries:
        for b in model.skeleton.boundaries:
            ax.axhline(float(b) * scale, **BOUNDARY_STYLE)
    fields = [layer[name] if name in layer else None for layer in model.layers]
    return profile(ax, fields, scale=scale, value_scale=value_scale, n=n,
                   label=name if label is None else label, joiners=joiners,
                   boundaries=False, **style)
