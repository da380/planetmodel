"""Profiles: a field against its layered coordinate, drawn one way.

A profile is drawn with the layered coordinate on the vertical axis,
the radius increasing upward and a depth downward (the axis inverted,
for the fields of a flat model), so that up on the page is out of the
body either way; the value runs along the horizontal, one segment per
layer, and at every interior boundary a horizontal line joins the two
one-sided values, so that a discontinuity reads as a step and the
profile as one unbroken curve; a faint horizontal guide at every layer
boundary sits beneath the curves.  Which way the coordinate runs is
asked of the fields' coordinate system (`coordinates.increases_outward`).
`profile` draws either a sequence of radial fields, one per layer in
order, or a `slices.ProfileSlice` (a profile at a point, or the
spherical mean), its layers as they were sliced; `radial_profile` draws
a named field of a model, spherical or flat, skipping the layers that
lack it.  Each returns the matplotlib lines it drew and leaves the
labelling to the caller.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..coordinates import coordinates_of, increases_outward
from ..fields import Field
from ..slices import ProfileSlice
from .style import BOUNDARY_STYLE

if TYPE_CHECKING:
    from ..model import ModelBase

__all__ = ["profile", "radial_profile"]

#: One layer's run of a profile: its radii and values, or None to leave blank.
type Run = tuple[np.ndarray, np.ndarray] | None


def _runs_of(fields: Sequence[Field | None] | ProfileSlice, n: int) -> list[Run]:
    """The runs of radial fields evaluated at `n` points per layer, or of a
    profile slice's layers."""
    if isinstance(fields, ProfileSlice):
        return [(lay.radius, lay.values) for lay in fields.layers]
    runs: list[Run] = []
    for field in fields:
        if field is None:
            runs.append(None)
            continue
        if not getattr(field, "is_radial", False) or field.character.rank != 0:
            raise ValueError(f"{field!r} is not a radial field of rank 0")
        lo, hi = field.interval
        r = np.linspace(lo, hi, n)
        v = field(r)
        if np.iscomplexobj(v):
            raise TypeError(f"{field!r} is complex; draw its real or imaginary "
                            "part as a field of its own")
        runs.append((r, v))
    return runs


def _downward(fields: Sequence[Field | None] | ProfileSlice) -> bool:
    """Whether the fields' layered coordinate increases into the body (a
    depth), read from the first field's coordinates; a profile slice is
    spherical."""
    if isinstance(fields, ProfileSlice):
        return False
    first = next((f for f in fields if f is not None), None)
    return first is not None and not increases_outward(coordinates_of(first))


def _point_down(ax: Any) -> None:
    """The vertical axis inverted, once: a depth increases downward."""
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def profile(ax: Any, fields: Sequence[Field | None] | ProfileSlice, *,
            scale: float = 1.0, value_scale: float = 1.0, n: int = 200,
            label: str | None = None, joiners: bool = True, boundaries: bool = True,
            **style: Any) -> list[Any]:
    """Draw a profile against the layered coordinate on `ax`: radius
    upward, depth downward.

    `fields` are radial fields of rank 0 (of the layered coordinate
    alone), one per layer in the skeleton's order, or None for a layer to
    leave blank, each drawn at `n` points; or a `ProfileSlice`, drawn at
    its points.  Where the fields' coordinate is a depth the vertical
    axis is inverted, unless it already is.  Coordinates are multiplied by
    `scale` (1e-3 for kilometres from metres) and values by `value_scale`
    before drawing; `label` goes on the first segment; every segment
    takes the same colour, from `style` or the axes' cycle.  Where two
    consecutive layers meet, a horizontal line joining their one-sided
    values is drawn in the same style, unless `joiners` is False; with
    `boundaries`, a faint horizontal guide (`BOUNDARY_STYLE`) is drawn at
    every end of every layer, beneath the curves.  Returns the profile
    lines drawn; a joiner carries the label "_joiner", which keeps it out
    of legends, and the guides are not returned.
    """
    runs = _runs_of(fields, n)
    if _downward(fields):
        _point_down(ax)
    if boundaries:
        ends = {float(e) for run in runs if run is not None
                for e in (run[0][0], run[0][-1])}
        for b in sorted(ends):
            ax.axhline(b * scale, **BOUNDARY_STYLE)
    lines = []
    color = style.pop("color", style.pop("c", None))
    previous: tuple[float, float] | None = None
    for run in runs:
        if run is None:
            previous = None
            continue
        r, v = np.asarray(run[0], dtype=float), np.asarray(run[1], dtype=float)
        lo, hi = float(r[0]), float(r[-1])
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


def radial_profile(ax: Any, model: ModelBase, name: str, *, scale: float = 1.0,
                   value_scale: float = 1.0, n: int = 200, label: str | None = None,
                   joiners: bool = True, boundaries: bool = True,
                   **style: Any) -> list[Any]:
    """Draw the field `name` of a model against its layered coordinate on
    `ax`, radius upward on the sphere and depth downward in a box, one
    segment per layer holding it, a line across every discontinuity and
    a guide at every boundary of the skeleton; see `profile` for the
    arguments.  A layer that lacks the name is left blank."""
    if not increases_outward(model.geometry.coordinates):
        _point_down(ax)
    if boundaries:
        for b in model.skeleton.boundaries:
            ax.axhline(float(b) * scale, **BOUNDARY_STYLE)
    fields = [layer[name] if name in layer else None for layer in model.layers]
    return profile(ax, fields, scale=scale, value_scale=value_scale, n=n,
                   label=name if label is None else label, joiners=joiners,
                   boundaries=False, **style)
