"""Pictures of flat models: a plane as a map, a vertical section, a block.

The flat counterparts of `sphere_map`, `section` and `cutaway`, drawing
the slices of `planetmodel.cartesian.slices`.  `plane_map` draws a
`PlaneSlice` as a map in (x, y), x to the right and y up at equal
aspect, each value filling the cell around its node (`grid_cell_edges`:
midway between neighbouring nodes, the grid's period tiled exactly), a
value NaN (a physical plane above the surface) left blank.
`vertical_section` draws a `SectionSlice`, the distance along the
section to the right and the depth downward (the vertical axis inverted,
as a flat profile is drawn), layer by layer, so that a jump at an
interface is a sharp edge, and each layer's top and bottom a faint line
over the colours (`style.BOUNDARY_STYLE`).  A slice taken at reference
points is drawn flat, or with `physical=True` at its points' physical
positions, the body as the mapping shapes it (under topography, its top
the physical surface); a slice taken at physical points is drawn as it
stands.  Either way a point X is drawn at (X - path[0]) @ plane.T, the
section's own coordinates (distance, depth), which are those
`plane_cut(points, cells, slice_.path[0], slice_.plane)` cuts a mesh in:
the cells, interfaces and faults of a box's mesh, its reference points
or its physical ones, drawn by `mesh_section` on the same axes, lie on
the section of the model.

`block` draws a `BlockSlice` on matplotlib's three-dimensional axes: the
box's top and its two sides toward the viewer, each face a surface whose
patches take the mean colour of their corners, z up, the box at its true
proportions, seen from its corner, and the interfaces drawn on the
sides, over them (the axes' `computed_zorder` turned off: the faces
drawn are the ones seen, so nothing drawn hides another); the depth is
labelled downward on the vertical axis (its ticks read -z).  With
`physical=True` the faces are drawn at their physical positions, the
top as the topography.

The colour scale is `style.colour_scale`'s, shared by every part of a
picture: a departure, and a signed field with `diverging=True`, drawn
symmetric about zero, the limits put at a percentile with `clip=`, and
`normalised=True` draws the slice divided by its largest absolute value
(`slices.normalised`); `colorbar=True` labels a colour bar with what the
slice shows.  Each function returns what it drew and leaves the figure,
and the axes' labels, to the caller.  matplotlib is not imported here
until a picture needs a part of it beyond the axes it is given.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..slices import Shown, normalised as _normalised
from .meshes import BLOCK_VIEW
from .style import BOUNDARY_STYLE, colour_extend, colour_scale

if TYPE_CHECKING:
    from ..cartesian.grid import Grid
    from ..cartesian.slices import BlockSlice, PlaneSlice, SectionSlice

__all__ = ["plane_map", "vertical_section", "block", "grid_cell_edges",
           "BLOCK_VIEWS"]

#: The azimuth, in degrees, of matplotlib's three-dimensional view toward
#: each corner of a block, by compass (x east, y north): the view from
#: the southeast is matplotlib's own default.
BLOCK_VIEWS = {"southeast": BLOCK_VIEW[1], "southwest": -120.0, "northeast": 30.0,
               "northwest": 150.0}

#: The elevation of a block's view, in degrees.
_ELEVATION = BLOCK_VIEW[0]


def grid_cell_edges(grid: Grid) -> tuple[np.ndarray, ...]:
    """The edges of the cells around a grid's nodes along each axis: midway
    between neighbouring nodes, n + 1 of them over n nodes, spanning one
    period from half a step before the first node."""
    return tuple(axis[0] + (np.arange(axis.size + 1) - 0.5) * step
                 for axis, step in zip(grid.axes, grid.spacing))


def plane_map(ax: Any, slice_: PlaneSlice, *, scale: float = 1.0,
              cmap: str | None = None, vmin: float | None = None,
              vmax: float | None = None, diverging: bool = False,
              clip: float | None = None, normalised: bool = False,
              colorbar: bool = False, **style: Any) -> Any:
    """Draw a plane slice on `ax` as a map, by the rules of the module
    docstring.

    x and y are multiplied by `scale` (1e-3 for kilometres from metres);
    `cmap`, `vmin` and `vmax` override the colour scale, `diverging` and
    `clip` choose it, `normalised` divides by the largest absolute value;
    `colorbar` adds a colour bar labelled with what the slice shows, with
    arrows at the ends the values run beyond; `style` goes to
    `pcolormesh`.  A slice over a section's grid, which has one axis, is
    refused.  Returns the mesh drawn.
    """
    grid = slice_.grid
    if grid.dimension != 2:
        raise ValueError("a map is of a plane over a box, whose grid has two axes; a "
                         "section's plane is a line: draw a vertical section instead")
    if normalised:
        slice_ = _normalised(slice_)
    v = np.asarray(slice_.values, dtype=float)
    xe, ye = grid_cell_edges(grid)
    cmap, lo, hi = colour_scale([v], slice_.shown, cmap=cmap, vmin=vmin, vmax=vmax,
                                diverging=diverging, clip=clip)
    mesh = ax.pcolormesh(xe * scale, ye * scale, np.ma.masked_invalid(v.T),
                         shading="flat", cmap=cmap, vmin=lo, vmax=hi, **style)
    ax.set_aspect("equal")
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=slice_.label,
                           extend=colour_extend([v], lo, hi))
    return mesh


def _in_plane(slice_: SectionSlice, points: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """The distance along the section and the depth of points, in the
    section's plane: (X - path[0]) @ plane.T."""
    st = (np.asarray(points, dtype=float) - slice_.path[0]) @ slice_.plane.T
    return st[..., 0], st[..., 1]


def _layered(ax: Any, runs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
             shown: Shown, *, scale: float, cmap: str | None, vmin: float | None,
             vmax: float | None, diverging: bool, clip: float | None,
             interfaces: bool, colorbar: bool, **style: Any) -> list[Any]:
    """Draw (distance, depth, values) runs, one per layer, each (n, m), depth
    downward, on one colour scale."""
    cmap, lo, hi = colour_scale([v for _, _, v in runs], shown, cmap=cmap,
                                vmin=vmin, vmax=vmax, diverging=diverging, clip=clip)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    meshes = []
    for s, d, v in runs:
        meshes.append(ax.pcolormesh(s * scale, d * scale, v, shading="gouraud",
                                    cmap=cmap, vmin=lo, vmax=hi, **style))
        if interfaces:
            for row in (0, -1):
                ax.plot(s[row] * scale, d[row] * scale, **BOUNDARY_STYLE)
    if colorbar and meshes:
        ax.figure.colorbar(meshes[0], ax=ax, label=shown.label,
                           extend=colour_extend([v for _, _, v in runs], lo, hi))
    return meshes


def vertical_section(ax: Any, slice_: SectionSlice, *, scale: float = 1.0,
                     physical: bool = False, cmap: str | None = None,
                     vmin: float | None = None, vmax: float | None = None,
                     diverging: bool = False, clip: float | None = None,
                     normalised: bool = False, interfaces: bool = True,
                     colorbar: bool = False, **style: Any) -> list[Any]:
    """Draw a vertical section on plain axes, depth downward, by the rules
    of the module docstring.

    `physical` draws a slice taken at reference points at its points'
    physical positions; distances and depths are multiplied by `scale`;
    `cmap`, `vmin` and `vmax` override the colour scale, shared by the
    layers, `diverging` and `clip` choose it, `normalised` divides by the
    largest absolute value; `interfaces` draws each layer's top and
    bottom as a faint line; `colorbar` adds a colour bar labelled with
    what the slice shows; `style` goes to `pcolormesh`.  The vertical
    axis is inverted unless it already is.  Returns the meshes drawn, one
    per layer.
    """
    if getattr(ax, "name", "") != "rectilinear":
        raise ValueError("a vertical section is drawn on plain axes")
    if physical and slice_.at == "physical":
        raise ValueError("this slice is at physical points already: draw it without "
                         "physical=True")
    if normalised:
        slice_ = _normalised(slice_)
    runs = []
    for lay in slice_.layers:
        if physical:
            s, d = _in_plane(slice_, lay.physical)
        else:
            s = np.broadcast_to(slice_.distance, lay.radius.shape)
            d = lay.radius
        runs.append((np.asarray(s), np.asarray(d), np.asarray(lay.values)))
    return _layered(ax, runs, slice_.shown, scale=scale, cmap=cmap, vmin=vmin,
                    vmax=vmax, diverging=diverging, clip=clip, interfaces=interfaces,
                    colorbar=colorbar, **style)


def _patches(values: np.ndarray) -> np.ndarray:
    """The mean of each patch's four corners, (n - 1, m - 1) of (n, m)."""
    v = np.asarray(values, dtype=float)
    return 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])


def block(ax: Any, slice_: BlockSlice, *, scale: float = 1.0, physical: bool = False,
          cmap: str | None = None, vmin: float | None = None,
          vmax: float | None = None, diverging: bool = False,
          clip: float | None = None, normalised: bool = False,
          interfaces: bool = True, colorbar: bool = False,
          view: tuple[float, float] | None = None, **style: Any) -> list[Any]:
    """Draw a block on three-dimensional axes, by the rules of the module
    docstring.

    `ax` is made with `subplot_kw={"projection": "3d"}`; positions are
    multiplied by `scale`; `physical` draws the faces at their physical
    positions; `cmap`, `vmin` and `vmax` override the colour scale,
    shared by the faces, `diverging` and `clip` choose it, `normalised`
    divides by the largest absolute value; `interfaces` draws the
    interfaces and the edges of the faces as faint lines; `colorbar` adds
    a colour bar labelled with what the slice shows; `view` is the
    (elevation, azimuth) of the view in degrees, by default 25 degrees
    up and toward the block's corner (`BLOCK_VIEWS`); `style` goes to
    `plot_surface`.  Returns the surfaces drawn, one per piece of a face.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter
    import matplotlib.pyplot as plt
    if getattr(ax, "name", "") != "3d":
        raise ValueError("a block is drawn on three-dimensional axes, made with "
                         "subplot_kw={'projection': '3d'}")
    if normalised:
        slice_ = _normalised(slice_)
    # the faces drawn are the ones toward the viewer, so nothing drawn hides
    # them and the lines on them are drawn over them, in the order drawn
    ax.computed_zorder = False
    parts = [lay.values for lay in slice_.layers]
    cmap, lo, hi = colour_scale(parts, slice_.shown, cmap=cmap, vmin=vmin, vmax=vmax,
                                diverging=diverging, clip=clip)
    norm, colours = Normalize(lo, hi), plt.get_cmap(cmap)
    surfaces = []
    for face, lay in zip(slice_.faces, slice_.layers):
        X = (lay.physical if physical else lay.reference) * scale
        rgba = colours(norm(np.ma.masked_invalid(_patches(lay.values))))
        surfaces.append(ax.plot_surface(X[..., 0], X[..., 1], X[..., 2],
                                        facecolors=rgba, shade=False, linewidth=0,
                                        antialiased=False, rstride=1, cstride=1,
                                        **style))
        if interfaces:
            rows = (0, -1) if face != "top" else ()
            for row in rows:
                ax.plot(X[row, :, 0], X[row, :, 1], X[row, :, 2], **BOUNDARY_STYLE)
            if face == "top":
                edges = (X[0], X[-1], X[:, 0], X[:, -1])
                for e in edges:
                    ax.plot(e[:, 0], e[:, 1], e[:, 2], **BOUNDARY_STYLE)
    every = np.concatenate([((lay.physical if physical else lay.reference)
                             * scale).reshape(-1, 3) for lay in slice_.layers])
    span = np.nanmax(every, axis=0) - np.nanmin(every, axis=0)
    ax.set_box_aspect(tuple(float(s) for s in np.where(span > 0, span, 1.0)))
    elevation, azimuth = ((_ELEVATION, BLOCK_VIEWS[slice_.corner]) if view is None
                          else view)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{-z + 0.0:g}"))
    if colorbar:
        ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                           label=slice_.label, shrink=0.6, pad=0.12,
                           extend=colour_extend(parts, lo, hi))
    return surfaces
