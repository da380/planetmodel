"""Cutaways: the body with a wedge cut from it, or discs through it, the
field on the faces.

`cutaway` draws a `slices.CutawaySlice`, of `slices.cutaway` or of
`slices.discs`, on plain matplotlib axes as one image, the pixels of the
slice's orthographic view coloured by what each looks at: the faces of
the cut, or the discs, always show the field; the sphere left standing
in a cut shows the field or a plain colour (`inner=`);
and the surface shows the field (`surface="field"`), a plain colour, the
land and the sea in two colours from Natural Earth's land polygons
(`"land"`), Natural Earth's shaded relief, the raster cartopy ships
(`"image"`), or any image of the whole surface given as an array (rows
from the north pole to the south, columns from 180 degrees west
eastward, RGB or RGBA).  Plain colours and images are shaded by the
light of `LIGHT`, so that the sphere reads as one; the field's colours
are never shaded.  `coastlines=True` draws Natural Earth's coastlines
where they are seen, through `CutawaySlice.visible`, and
`interfaces=True` the model's interfaces as faint arcs on the faces
(`style.BOUNDARY_STYLE`); a slice showing a sphere within the body
wears its dressing and coastlines on that sphere, the faces running
past it.  The edge of the body is antialiased by the
slice's coverage, a pixel of the edge whose centre misses the body
taking the colour of its nearest neighbour that sees it.  The colour
scale is `style.colour_scale`'s over every pixel that shows the field,
chosen with `diverging=` and `clip=` as a map's is; `normalised=True`
draws the slice divided by its largest absolute value.  cartopy (the
`maps` extra) is imported where land, relief or coastlines are drawn,
and only then.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..frames import cartesian_points, from_geographic, geographic
from ..slices import CutawaySlice, normalised as _normalised
from .style import BOUNDARY_STYLE, colour_extend, colour_scale

__all__ = ["cutaway", "LIGHT", "LAND", "SEA", "COASTLINE_STYLE"]

#: The direction the light comes from, in the picture's frame (right, up,
#: toward the viewer), for shading plain colours and images.
LIGHT = (-0.3, 0.5, 1.0)

#: The depth of the shading: a plain colour is multiplied by 1 - SHADE at
#: a grazing surface and by 1 where the light falls square on it.
_SHADE = 0.45

#: The colours of the land and the sea of `surface="land"`.
LAND = "#d8ceb1"
SEA = "#c9dcea"

#: The style of a coastline.
COASTLINE_STYLE = {"color": "0.15", "lw": 0.5}

#: The points along each interface arc drawn on a face.
_ARC_POINTS = 181


def _rgb(colour: Any) -> np.ndarray:
    """A matplotlib colour as RGBA floats."""
    from matplotlib.colors import to_rgba
    return np.asarray(to_rgba(colour), dtype=float)


def _shading(slice_: CutawaySlice, points: np.ndarray, normals: np.ndarray
             ) -> np.ndarray:
    """The factor (...,) a plain colour at `points` is multiplied by for
    the light of `LIGHT` on a surface of outward `normals`."""
    light = np.asarray(LIGHT, dtype=float) @ slice_.view
    light /= np.linalg.norm(light)
    cos = np.clip(normals @ light, 0.0, 1.0)
    return 1.0 - _SHADE * (1.0 - cos)


def _sample_image(image: np.ndarray, theta: np.ndarray, phi: np.ndarray
                  ) -> np.ndarray:
    """The pixels of an image of the whole surface, rows from the north
    pole, columns from 180 degrees west, nearest to (theta, phi)."""
    lat, lon = geographic(theta, phi)
    rows, cols = image.shape[:2]
    i = np.clip(((90.0 - lat) / 180.0 * rows).astype(int), 0, rows - 1)
    j = np.clip(((lon + 180.0) % 360.0 / 360.0 * cols).astype(int), 0, cols - 1)
    out = image[i, j]
    if out.shape[-1] == 3:
        out = np.concatenate([out, np.ones(out.shape[:-1] + (1,))], axis=-1)
    return out


def _natural_earth_image() -> np.ndarray:
    """Natural Earth's shaded relief, the raster cartopy ships, as floats."""
    import os
    import cartopy
    from matplotlib.image import imread
    path = os.path.join(cartopy.config["repo_data_dir"], "raster", "natural_earth",
                        "50-natural-earth-1-downsampled.png")
    return np.asarray(imread(path), dtype=float)


def _on_land(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Whether points lie on Natural Earth's land, at 110 m."""
    import shapely
    from cartopy.feature import NaturalEarthFeature
    lat, lon = geographic(theta, phi)
    lon = (lon + 180.0) % 360.0 - 180.0
    out = np.zeros(lat.shape, dtype=bool)
    for geometry in NaturalEarthFeature("physical", "land", "110m").geometries():
        shapely.prepare(geometry)
        out |= shapely.contains_xy(geometry, lon, lat)
    return out


def _coastlines(slice_: CutawaySlice, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Natural Earth's coastlines at 110 m in the picture, the runs that
    are seen separated by NaN."""
    from cartopy.feature import NaturalEarthFeature
    xs, ys = [], []
    for geometry in NaturalEarthFeature("physical", "coastline",
                                        "110m").geometries():
        lines = getattr(geometry, "geoms", [geometry])
        for line in lines:
            lon, lat = np.asarray(line.coords, dtype=float).T
            theta, phi = from_geographic(lat, lon)
            X = cartesian_points(slice_.outer, theta, phi)
            seen = slice_.visible(X)
            if not np.any(seen):
                continue
            x, y = slice_.project(X)
            xs.append(np.where(seen, x * scale, np.nan))
            ys.append(np.where(seen, y * scale, np.nan))
            xs.append([np.nan])
            ys.append([np.nan])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def _interface_arcs(slice_: CutawaySlice, scale: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The model's interfaces between the cut's radii as arcs on the two
    faces, in the picture, the runs that are seen separated by NaN."""
    xs, ys = [], []
    for radius in slice_.boundaries:
        if not slice_.inner < radius < slice_.top:
            continue
        for centre, a, b, start, end in slice_.arcs:
            s = np.linspace(start, end, _ARC_POINTS)[:, None]
            X = radius * (centre + a * np.cos(s) + b * np.sin(s))
            seen = slice_.visible(X)
            if not np.any(seen):
                continue
            x, y = slice_.project(X)
            xs += [np.where(seen, x * scale, np.nan), [np.nan]]
            ys += [np.where(seen, y * scale, np.nan), [np.nan]]
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def _surface_colours(slice_: CutawaySlice, surface: Any, hit: np.ndarray
                     ) -> np.ndarray | None:
    """The RGBA of the surface pixels `hit`, or None for the field."""
    if isinstance(surface, str) and surface == "field":
        return None
    X = slice_.reference[hit]
    r = np.linalg.norm(X, axis=-1)
    theta = np.arccos(np.clip(X[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(X[:, 1], X[:, 0])
    if isinstance(surface, str) and surface == "image":
        rgba = _sample_image(_natural_earth_image(), theta, phi)
    elif isinstance(surface, str) and surface == "land":
        rgba = np.where(_on_land(theta, phi)[:, None], _rgb(LAND), _rgb(SEA))
    elif isinstance(surface, np.ndarray) and surface.ndim == 3:
        image = surface.astype(float)
        if surface.dtype.kind in "ui":
            image = image / 255.0
        if image.shape[2] not in (3, 4):
            raise ValueError("an image of the surface has RGB or RGBA pixels, "
                             f"got shape {surface.shape}")
        rgba = _sample_image(image, theta, phi)
    else:
        try:
            rgba = np.broadcast_to(_rgb(surface), (X.shape[0], 4)).copy()
        except ValueError:
            raise ValueError("surface is 'field', 'image', 'land', an RGB(A) "
                             f"image of the surface or a colour, got {surface!r}"
                             ) from None
    rgba = np.array(rgba, dtype=float)
    rgba[:, :3] *= _shading(slice_, X, X / r[:, None])[:, None]
    return rgba


def cutaway(ax: Any, slice_: CutawaySlice, *, surface: Any = "field",
            inner: Any = "field", coastlines: bool = False,
            interfaces: bool = True, scale: float = 1.0, cmap: str | None = None,
            vmin: float | None = None, vmax: float | None = None,
            diverging: bool = False, clip: float | None = None,
            normalised: bool = False, colorbar: bool = False, **style: Any) -> Any:
    """Draw a cutaway on plain axes by the rules of the module docstring.

    `surface` is `"field"`, `"image"`, `"land"`, an RGB(A) image of the
    whole surface, or a colour; `inner` `"field"` or a colour for the
    sphere left standing in the cut; `coastlines` draws the coastlines;
    `interfaces` the interfaces as faint arcs on the faces; positions
    are multiplied by `scale`; `cmap`, `vmin` and `vmax` override the
    colour scale, `diverging` and `clip` choose it, `normalised` divides
    by the largest absolute value; `colorbar` adds a colour bar labelled
    with what the slice shows, with arrows at the ends the values run
    beyond; `style` goes to `imshow`.  Returns the image drawn.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    if (getattr(ax, "projection", None) is not None
            or getattr(ax, "name", "") == "polar"):
        raise ValueError("a cutaway is drawn on plain axes")
    if normalised:
        slice_ = _normalised(slice_)
    field = slice_.face > 0
    if not (isinstance(surface, str) and surface == "field"):
        field &= ~slice_.mask("surface")
    if not (isinstance(inner, str) and inner == "field") and "inner" in slice_.faces:
        field &= ~slice_.mask("inner")
    values = slice_.values[field]
    cmap, lo, hi = colour_scale([values], slice_.shown, cmap=cmap, vmin=vmin,
                                vmax=vmax, diverging=diverging, clip=clip)
    norm = Normalize(lo, hi)
    colours = plt.get_cmap(cmap)
    n = slice_.n
    rgba = np.zeros((n, n, 4))
    rgba[field] = colours(norm(values))
    top = _surface_colours(slice_, surface, slice_.mask("surface"))
    if top is not None:
        rgba[slice_.mask("surface")] = top
    if not (isinstance(inner, str) and inner == "field") and "inner" in slice_.faces:
        core = slice_.mask("inner")
        X = slice_.reference[core]
        r = np.linalg.norm(X, axis=-1)
        plain = np.broadcast_to(_rgb(inner), (X.shape[0], 4)).copy()
        plain[:, :3] *= _shading(slice_, X, X / r[:, None])[:, None]
        rgba[core] = plain
    # a pixel of the edge whose centre misses the body takes the colour of its
    # nearest neighbour that sees it, at its coverage
    seen = slice_.face > 0
    partial = ~seen & (slice_.coverage > 0.0)
    if np.any(partial):
        from scipy.ndimage import distance_transform_edt
        _, (rows, cols) = distance_transform_edt(~seen, return_indices=True)
        rgba[partial] = rgba[rows[partial], cols[partial]]
    rgba[..., 3] *= slice_.coverage
    e = slice_.extent * scale
    image = ax.imshow(rgba, extent=(-e, e, -e, e), origin="upper",
                      interpolation="antialiased", **style)
    if interfaces:
        x, y = _interface_arcs(slice_, scale)
        if x.size:
            ax.plot(x, y, **BOUNDARY_STYLE)
    if coastlines:
        x, y = _coastlines(slice_, scale)
        if x.size:
            ax.plot(x, y, **COASTLINE_STYLE)
    ax.set_aspect("equal")
    ax.set_xlim(-e, e)
    ax.set_ylim(-e, e)
    ax.set_axis_off()
    if colorbar:
        ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                           label=slice_.label, extend=colour_extend([values], lo, hi))
    return image
