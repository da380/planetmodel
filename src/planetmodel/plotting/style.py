"""The look every picture shares: boundary guides and colour scales.

A boundary of the skeleton is a faint guide (`BOUNDARY_STYLE`), drawn
over a picture's colours and beneath its curves: above an image (zorder
0), a colour mesh (1) and the axes' grid (1.5, matplotlib's default for
`axes.axisbelow`), so that an interface shows on a section, and below
lines (2), so that a profile runs over its guides.  A departure, and any
field asked for with `diverging=True` (a kernel, any signed field), is
drawn on a diverging colour map (`DIVERGING`, negative red and positive
blue) with limits symmetric about zero, the largest absolute value;
anything else on a sequential one (`SEQUENTIAL`), the limits the finite
range of what is drawn.  `clip=p` puts the limits at the p-th percentile
of the absolute values instead (and of the values, from the
(100 - p)-th, on a sequential scale), so that a spike beside a source
does not wash out the rest; a percentile of zero, where most of a field
is, falls back to the largest value.  The caller's limits win over both.
`colour_scale` decides them once for a picture of several parts, so that
every layer of a section shares one scale, and `colour_extend` says
which ends of a colour bar the values run beyond, to be drawn as arrows.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..slices import Shown

__all__ = ["BOUNDARY_STYLE", "DIVERGING", "SEQUENTIAL", "colour_scale",
           "colour_extend"]

#: The style of the guide drawn at a layer boundary: over colours, beneath
#: curves (the module docstring).
BOUNDARY_STYLE = {"color": "0.85", "lw": 0.8, "zorder": 1.75}

#: The colour map of a departure: negative red, positive blue.
DIVERGING = "RdBu"

#: The colour map of anything else.
SEQUENTIAL = "viridis"


def _finite(values: Iterable[ArrayLike]) -> np.ndarray:
    """The finite numbers of a picture's parts, flattened."""
    parts = [np.asarray(v, dtype=float) for v in values]
    return np.concatenate([v[np.isfinite(v)].ravel() for v in parts])


def colour_scale(values: Iterable[ArrayLike], shown: Shown, *, cmap: str | None = None,
                 vmin: float | None = None, vmax: float | None = None,
                 diverging: bool = False, clip: float | None = None
                 ) -> tuple[str, float, float]:
    """The colour map and limits of a picture of `values`, the arrays of
    its parts: the caller's where given, else by the rule of the module
    docstring for what `shown` says the values are, `diverging` and
    `clip`, a percentile in (50, 100]."""
    if clip is not None and not 50.0 < float(clip) <= 100.0:
        raise ValueError(f"clip is a percentile in (50, 100], got {clip!r}")
    finite = _finite(values)
    if finite.size == 0:
        raise ValueError(f"{shown.label} has no finite value to draw")
    if shown.is_departure or diverging:
        size = np.abs(finite)
        bound = float(size.max() if clip is None else np.percentile(size, clip))
        bound = bound if bound > 0.0 else float(size.max())
        bound = bound if bound > 0.0 else 1.0
        lo, hi = -bound, bound
        default = DIVERGING
    else:
        lo, hi = ((float(finite.min()), float(finite.max())) if clip is None else
                  (float(np.percentile(finite, 100.0 - float(clip))),
                   float(np.percentile(finite, float(clip)))))
        if lo == hi:
            lo, hi = lo - 0.5 * max(abs(lo), 1.0), hi + 0.5 * max(abs(hi), 1.0)
        default = SEQUENTIAL
    return (default if cmap is None else cmap, lo if vmin is None else float(vmin),
            hi if vmax is None else float(vmax))


def colour_extend(values: Iterable[ArrayLike], lo: float, hi: float) -> str:
    """Which ends of a colour bar from `lo` to `hi` the finite values run
    beyond: matplotlib's `extend`, "neither", "min", "max" or "both"."""
    finite = _finite(values)
    below = bool(finite.size) and float(finite.min()) < lo
    above = bool(finite.size) and float(finite.max()) > hi
    return {(False, False): "neither", (True, False): "min",
            (False, True): "max", (True, True): "both"}[(below, above)]
