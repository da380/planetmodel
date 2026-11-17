"""Sections: a great-circle section drawn as a wedge of the annulus.

`section` draws a `slices.SectionSlice` on matplotlib's polar axes: the
angle is the distance along the arc, the wedge centred at the top with
the start on the left, and the radius runs from the centre at true
scale, so that a section of a shell is a wedge of an annulus and a whole
circle is a disc.  Each layer is drawn on its own, so that a jump at an
interface is a sharp edge, and each interface is a faint arc
(`style.BOUNDARY_STYLE`).  A slice taken at physical points is a wedge
as it stands, its interfaces curved where the mapping moved them; a
slice taken at reference points is drawn at its mapped positions with
`physical=True`, on plain axes of equal aspect in the plane of the
section.  The colour scale is shared by the layers (`style.colour_scale`),
chosen with `diverging=` and `clip=` as a map's is, and `normalised=True`
draws the slice divided by its largest absolute value.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..coordinates import increases_outward
from ..slices import SectionSlice, normalised as _normalised
from .style import BOUNDARY_STYLE, colour_extend, colour_scale

__all__ = ["section"]


def _in_plane(slice_: SectionSlice, points: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """The distance along the arc and the distance from the centre of
    points, in the plane of the section."""
    a, t = slice_.plane
    x, y = points @ a, points @ t
    return np.unwrap(np.arctan2(y, x), axis=-1), np.hypot(x, y)


def section(ax: Any, slice_: SectionSlice, *, scale: float = 1.0,
            physical: bool = False, cmap: str | None = None,
            vmin: float | None = None, vmax: float | None = None,
            diverging: bool = False, clip: float | None = None,
            normalised: bool = False, interfaces: bool = True,
            colorbar: bool = False, **style: Any) -> list[Any]:
    """Draw a section on `ax` by the rules of the module docstring.

    `ax` is polar axes (`subplot_kw={"projection": "polar"}`), or plain
    ones for `physical=True`; radii are multiplied by `scale`; `cmap`,
    `vmin` and `vmax` override the colour scale, `diverging` and `clip`
    choose it, `normalised` divides by the largest absolute value;
    `interfaces` draws the faint arcs; `colorbar` adds a colour bar
    labelled with what the slice shows, with arrows at the ends the
    values run beyond; `style` goes to `pcolormesh`.  Returns the meshes
    drawn, one per layer.
    """
    if not increases_outward(slice_.coordinates):
        raise ValueError("a section of a flat model is drawn by vertical_section")
    polar = getattr(ax, "name", None) == "polar"
    if physical and slice_.at == "physical":
        raise ValueError("this slice is at physical points already: draw it on polar "
                         "axes without physical=True")
    if physical == polar:
        raise ValueError("a section is drawn on polar axes (subplot_kw={'projection': "
                         "'polar'}), or with physical=True on plain axes")
    if normalised:
        slice_ = _normalised(slice_)
    length = slice_.length
    cmap, lo, hi = colour_scale([lay.values for lay in slice_.layers], slice_.shown,
                                cmap=cmap, vmin=vmin, vmax=vmax, diverging=diverging,
                                clip=clip)
    meshes = []
    for lay in slice_.layers:
        if physical:
            angle, radius = _in_plane(slice_, lay.physical)
        else:
            angle = np.broadcast_to(slice_.distance, lay.radius.shape)
            radius = lay.radius
        # the wedge centred at the top, the start on the left
        turn = angle - 0.5 * length
        if polar:
            mesh = ax.pcolormesh(turn, radius * scale, lay.values, shading="gouraud",
                                 cmap=cmap, vmin=lo, vmax=hi, **style)
        else:
            x, y = radius * scale * np.sin(turn), radius * scale * np.cos(turn)
            mesh = ax.pcolormesh(x, y, lay.values, shading="gouraud", cmap=cmap,
                                 vmin=lo, vmax=hi, **style)
        meshes.append(mesh)
        if interfaces:
            for row in (0, -1):
                if polar:
                    ax.plot(turn[row], radius[row] * scale, **BOUNDARY_STYLE)
                else:
                    ax.plot(x[row], y[row], **BOUNDARY_STYLE)
    radii = np.concatenate([lay.radius.ravel() for lay in slice_.layers])
    if polar:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        if length < 2.0 * np.pi - 1e-12:
            ax.set_thetamin(-0.5 * np.degrees(length))
            ax.set_thetamax(0.5 * np.degrees(length))
        ax.set_rorigin(0.0)
        ax.set_rlim(float(radii.min()) * scale, float(radii.max()) * scale)
    else:
        ax.set_aspect("equal")
    if colorbar:
        ax.figure.colorbar(meshes[0], ax=ax, label=slice_.label,
                           extend=colour_extend([lay.values for lay in slice_.layers],
                                                lo, hi))
    return meshes
