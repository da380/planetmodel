"""Spectra and correlations drawn.

`spectrum` draws a `slices.Spectrum`'s power by degree (or its
amplitude, the square root) as a line on a logarithmic scale, from a
first degree, 0 by default and 1 to leave the mean out.  `correlation`
draws a `slices.Correlation` as an image, degree along and radius up,
each value filling its cell, on the diverging colour map from -1 to 1,
a degree without power left blank; `correlation_total` draws the
correlation over all degrees against radius, radius upward as in a
profile.

The spectra and correlations of a flat model
(`cartesian.slices.Spectrum`, `Correlation`) are drawn by the same
functions, which ask the object: one that has `wavenumbers` is drawn
against its bins' horizontal wavenumber `|k|` (lengths multiplied by
`scale`, so `|k|` divided by it), a spectrum on logarithmic axes, the mean
(k = 0) left out, with the wavelength `2 pi / |k|` on a second axis along
the top; and one whose layered coordinate is a depth (its
`coordinates`) is drawn depth downward.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..coordinates import increases_outward
from .style import DIVERGING

if TYPE_CHECKING:
    from ..cartesian import slices as flat
    from ..slices import Correlation, Spectrum

__all__ = ["spectrum", "correlation", "correlation_total"]


#: The id of the axis of wavelengths along the top of a flat spectrum's axes.
_WAVELENGTHS = "planetmodel wavelengths"


def _wavelength_axis(ax: Any, label: str) -> None:
    """The wavelength `2 pi / |k|` along the top of axes of `|k|`, once, under
    `label`."""
    held = [child for child in ax.child_axes if child.get_gid() == _WAVELENGTHS]
    if held:
        held[0].set_xlabel(label)
        return

    def inverse(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0.0, 2.0 * np.pi / np.where(v > 0.0, v, 1.0), np.inf)

    from matplotlib.ticker import NullFormatter
    top = ax.secondary_xaxis("top", functions=(inverse, inverse))
    top.xaxis.set_minor_formatter(NullFormatter())
    top.set_gid(_WAVELENGTHS)
    top.set_xlabel(label)


def spectrum(ax: Any, spectrum_: Spectrum | flat.Spectrum, *, amplitude: bool = False,
             first: int = 0, scale: float = 1.0, label: str | None = None,
             wavelengths: str | None = "wavelength", **style: Any) -> Any:
    """Draw a spectrum on `ax`: the power by degree, or the amplitude with
    `amplitude=True`, from degree `first`, the vertical scale logarithmic;
    a flat spectrum against `|k|` by the module docstring, from bin `first`,
    lengths multiplied by `scale`, the wavelength along the top under the
    label `wavelengths` (None for no such axis); `label` for a legend;
    `style` to `plot`.  Returns the line drawn."""
    if not 0 <= int(first) <= int(spectrum_.degrees[-1]):
        raise ValueError(f"first must be a degree in 0..{int(spectrum_.degrees[-1])}, "
                         f"got {first}")
    values = spectrum_.amplitude if amplitude else spectrum_.power
    keep = spectrum_.degrees >= int(first)
    k = getattr(spectrum_, "wavenumbers", None)
    if k is None:
        line, = ax.plot(spectrum_.degrees[keep], values[keep], label=label, **style)
        ax.set_yscale("log")
        return line
    from matplotlib.ticker import NullFormatter
    keep &= k > 0.0
    line, = ax.plot(k[keep] / scale, values[keep], label=label, **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # a span of a decade or two labels the minor ticks too, over one another
    ax.xaxis.set_minor_formatter(NullFormatter())
    if wavelengths is not None:
        _wavelength_axis(ax, wavelengths)
    return line


def _edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges midway between centres, the end cells as wide as their
    neighbours'."""
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def _point_down(ax: Any, correlation_: Correlation | flat.Correlation) -> None:
    """The vertical axis inverted, once, where the layered coordinate is a
    depth."""
    if not increases_outward(correlation_.coordinates) and not ax.yaxis_inverted():
        ax.invert_yaxis()


def correlation(ax: Any, correlation_: Correlation | flat.Correlation, *,
                scale: float = 1.0, first: int = 1, cmap: str | None = None,
                colorbar: bool = False, **style: Any) -> Any:
    """Draw a correlation by degree and radius on `ax` as an image, from
    degree `first` (1 by default, the mean left out); radii multiplied by
    `scale`; a flat correlation by `|k|` and depth, depth downward, `|k|`
    divided by `scale`; `cmap` in place of the diverging one; `colorbar`
    a colour bar; `style` to `pcolormesh`.  Returns the mesh drawn."""
    degrees = correlation_.degrees
    if not 0 <= int(first) <= int(degrees[-1]):
        raise ValueError(f"first must be a degree in 0..{int(degrees[-1])}, got "
                         f"{first}")
    keep = degrees >= int(first)
    k = getattr(correlation_, "wavenumbers", None)
    along = degrees[keep] if k is None else k[keep] / scale
    order = np.argsort(correlation_.radii)
    radii = correlation_.radii[order] * scale
    values = np.ma.masked_invalid(correlation_.by_degree[order][:, keep])
    mesh = ax.pcolormesh(_edges(along), _edges(radii), values, shading="flat",
                         cmap=DIVERGING if cmap is None else cmap, vmin=-1.0, vmax=1.0,
                         **style)
    _point_down(ax, correlation_)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=correlation_.label)
    return mesh


def correlation_total(ax: Any, correlation_: Correlation | flat.Correlation, *,
                      scale: float = 1.0, label: str | None = None,
                      **style: Any) -> Any:
    """Draw the correlation over all degrees against radius on `ax`, radius
    upward (a flat correlation's depth downward), radii multiplied by
    `scale`; `label` for a legend; `style` to `plot`.  Returns the line
    drawn."""
    order = np.argsort(correlation_.radii)
    line, = ax.plot(correlation_.total[order], correlation_.radii[order] * scale,
                    label=label, **style)
    ax.set_xlim(-1.0, 1.0)
    _point_down(ax, correlation_)
    return line
