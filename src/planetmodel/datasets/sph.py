"""The Woodhouse-Ritsema coefficient files: spherical harmonics on radial splines.

S20RTS, S40RTS, S12RTS and SP12RTS are distributed as `.sph` files.  The
first line is the header, `lmax degree-mask nslot slot-mask`, the masks
one digit (1 present, 0 absent) per degree 0 ... lmax and per slot; then,
for each present slot in order and each present degree l = 0 ... lmax,
the 2l + 1 coefficients a_l0, a_l1, b_l1, ..., a_ll, b_ll, each degree
starting a line.  A first line holding lmax alone means every degree and
slot is present, and a file of numbers alone (SGLOBE-rani's copies) is
read with `lmax=` given, its blocks taken as the mantle's splines.  The
24 slots are three of the crust, absent from these files, and the 21
radial splines of the mantle, from the Moho down to the CMB.

The model is ::

    f(r, theta, phi) = sum_k S_k(x) sum_lm X_lm(theta)
                                      (a_klm cos(m phi) + b_klm sin(m phi)),

with X_lm the complex orthonormal Legendre functions carrying the
Condon-Shortley phase and no sqrt 2 (`"woodhouse"` with `csphase=-1` in
`planetmodel.harmonics`), x = -1 + 2 (r - r_cmb) / (r_moho - r_cmb), and
S_k the cardinal cubic spline of knot k of `SPLINE_KNOTS`: the cubic
spline through one at that knot and zero at the others, clamped at each
end to the slope of the quadratic through the three end knots, C2
between, as Woodhouse's `rspln` builds it.  Being piecewise cubic in r
it is held exactly, and `sph_field` is a `HarmonicField` on [r_cmb,
r_moho], the perturbation zero beyond.  The values are fractions, dlnVs
(or dlnVp for SP12RTS's P part), degree 0 kept as the file holds it.
`SPH_RADII` are the CMB and Moho of these models in metres, 3480 and
6346.6 km (6346.619 km in Ritsema's own code, 19 m higher).

`SphReader(parameters)` reads a data set of such files, one name per
file (`vs`, `vp`), into the model `models.over_reference` makes over the
entry's reference.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline, PPoly

from ..expansions import HarmonicField
from ..harmonics import from_convention
from ..model import Model
from ..units import Scales
from .models import over_reference, reference_of

if TYPE_CHECKING:
    from .registry import Dataset

__all__ = ["SPLINE_KNOTS", "SPH_RADII", "SLOTS", "MANTLE_SLOTS", "SphFile",
           "read_sph", "sph_splines", "sph_field", "SphReader"]

#: The 21 knots of the radial splines in x in [-1, 1], from the CMB up.
SPLINE_KNOTS = np.array([
    -1.00000, -0.78631, -0.59207, -0.41550, -0.25499, -0.10909, 0.02353,
    0.14409, 0.25367, 0.35329, 0.44384, 0.52615, 0.60097, 0.66899, 0.73081,
    0.78701, 0.83810, 0.88454, 0.92675, 0.96512, 1.00000])
SPLINE_KNOTS.setflags(write=False)

#: The radii of the CMB and the Moho the splines span, in metres.
SPH_RADII = (3480.0e3, 6346.6e3)

#: The slots of a file: three of the crust and the 21 mantle splines.
SLOTS = 24

#: The slots (from one) of the mantle splines, from the Moho down.
MANTLE_SLOTS = tuple(range(4, SLOTS + 1))


@dataclass(frozen=True, eq=False)
class SphFile:
    """A `.sph` file's numbers as it holds them: the degree `lmax`, which
    degrees are present, which of `nslots` slots each block is (from one,
    in file order), and `blocks` of shape (blocks, (lmax + 1)^2) laid out
    l = 0 ... lmax, a_l0, a_l1, b_l1, ..., zeros for absent degrees."""

    lmax: int
    degrees: tuple[bool, ...]
    slots: tuple[int, ...]
    blocks: np.ndarray
    _: KW_ONLY
    nslots: int = SLOTS
    header: str = ""
    source: str | None = None


def _mask(digits: str, n: int, what: str, origin: str) -> tuple[bool, ...]:
    """A string of n 0/1 digits as flags, or the error."""
    if len(digits) != n or set(digits) - {"0", "1"}:
        raise ValueError(f"{origin}: the {what} mask {digits!r} is not {n} digits of "
                         "0 and 1")
    return tuple(c == "1" for c in digits)


def read_sph(source: str | os.PathLike[str] | Iterable[str], *,
             lmax: int | None = None) -> SphFile:
    """A `.sph` file, from a path or from lines: its header read, its
    coefficients counted against it, absent degrees as zeros.  `lmax`
    reads a file of numbers alone, every degree and the 21 mantle splines
    present, and must agree with a header where there is one."""
    if isinstance(source, (str, os.PathLike)):
        origin = os.fspath(source)
        with open(source) as fh:
            lines = fh.read().splitlines()
    else:
        origin, lines = "the .sph lines", list(source)
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        raise ValueError(f"{origin}: no lines")
    head = lines[0].split()
    header, body = "", lines
    nslots = SLOTS
    if len(head) == 4 and head[0].isdigit() and head[2].isdigit():
        header, body = lines[0], lines[1:]
        file_lmax, nslots = int(head[0]), int(head[2])
        degrees = _mask(head[1], file_lmax + 1, "degree", origin)
        present = _mask(head[3], nslots, "slot", origin)
        slots = tuple(i + 1 for i, p in enumerate(present) if p)
    elif len(head) == 1 and head[0].isdigit():
        header, body = lines[0], lines[1:]
        file_lmax, degrees, slots = int(head[0]), None, None
    else:
        if lmax is None:
            raise ValueError(f"{origin}: no header; give lmax= to read a file of "
                             "coefficients alone")
        file_lmax, degrees, slots = int(lmax), None, None
    if lmax is not None and int(lmax) != file_lmax:
        raise ValueError(f"{origin}: the header says lmax {file_lmax}, not {lmax}")
    if degrees is None:
        degrees = (True,) * (file_lmax + 1)
    values = np.array(" ".join(body).split(), dtype=float)
    per = sum(2 * l + 1 for l in range(file_lmax + 1) if degrees[l])
    if slots is None:
        if per == 0 or values.size % per:
            raise ValueError(f"{origin}: {values.size} coefficients are not whole "
                             f"blocks of {per}")
        count = values.size // per
        slots = (MANTLE_SLOTS if count == len(MANTLE_SLOTS)
                 else tuple(range(1, count + 1)))
    if values.size != per * len(slots):
        raise ValueError(f"{origin}: {values.size} coefficients, but the header asks "
                         f"{len(slots)} blocks of {per}")
    blocks = np.zeros((len(slots), (file_lmax + 1) ** 2))
    starts = np.cumsum([0] + [2 * l + 1 for l in range(file_lmax + 1)])
    taken = values.reshape(len(slots), per)
    at = 0
    for l in range(file_lmax + 1):
        if degrees[l]:
            blocks[:, starts[l]:starts[l + 1]] = taken[:, at:at + 2 * l + 1]
            at += 2 * l + 1
    blocks.setflags(write=False)
    return SphFile(file_lmax, degrees, tuple(slots), blocks, nslots=nslots,
                   header=header, source=origin if origin != "the .sph lines" else None)


def sph_splines(*, cmb: float = SPH_RADII[0], moho: float = SPH_RADII[1]) -> PPoly:
    """The 21 radial splines as one piecewise polynomial of radius between
    `cmb` and `moho`, its values of shape (21,), the k-th the cardinal
    spline of the k-th knot from the CMB."""
    r = cmb + 0.5 * (SPLINE_KNOTS + 1.0) * (moho - cmb)
    n = r.size

    def end_slope(x: np.ndarray, y: np.ndarray) -> float:
        # the slope at x[0] of the quadratic through the three points
        h1, h2 = x[1] - x[0], x[2] - x[0]
        return float((y[1] - y[0]) * h2 / (h1 * (h2 - h1))
                     - (y[2] - y[0]) * h1 / (h2 * (h2 - h1)))

    c = np.zeros((4, n - 1, n))
    for k in range(n):
        y = np.zeros(n)
        y[k] = 1.0
        bc = ((1, end_slope(r[:3], y[:3])), (1, end_slope(r[::-1][:3], y[::-1][:3])))
        c[..., k] = CubicSpline(r, y, bc_type=bc).c
    return PPoly(c, r)


def sph_field(data: SphFile, *, cmb: float = SPH_RADII[0],
              moho: float = SPH_RADII[1], name: str | None = None) -> HarmonicField:
    """The file's field on [cmb, moho], in the library's harmonics, exact:
    the mantle splines' coefficients converted from the files' convention
    and carried by `sph_splines`.  A file of other slots is refused."""
    if data.nslots != SLOTS or tuple(data.slots) != MANTLE_SLOTS:
        raise ValueError(
            f"the radial splines of the .sph format are the 21 mantle slots "
            f"{MANTLE_SLOTS[0]}...{MANTLE_SLOTS[-1]} of {SLOTS}; this file holds "
            f"slots {data.slots} of {data.nslots}")
    L = data.lmax
    a = np.zeros((len(MANTLE_SLOTS), 2, L + 1, L + 1))
    i = 0
    for l in range(L + 1):
        a[:, 0, l, 0] = data.blocks[:, i]
        a[:, 0, l, 1:l + 1] = data.blocks[:, i + 1:i + 2 * l + 1:2]
        a[:, 1, l, 1:l + 1] = data.blocks[:, i + 2:i + 2 * l + 2:2]
        i += 2 * l + 1
    c = from_convention(np.moveaxis(a, 0, -1), normalization="woodhouse", csphase=-1)
    c = c[..., ::-1]                                # blocks run from the Moho down
    splines = sph_splines(cmb=cmb, moho=moho)
    coefficients = np.einsum("pik,slmk->pislm", splines.c, c)
    return HarmonicField((cmb, moho), PPoly(coefficients, splines.x), name=name)


@dataclass(frozen=True)
class SphReader:
    """The reader of a data set of `.sph` files, one perturbed name per
    file in its sources' order (`vs`, or `vp` for a P model), relative to
    the entry's reference."""

    _: KW_ONLY
    parameters: tuple[str, ...] = ("vs",)

    def read(self, paths: tuple[Path, ...], dataset: Dataset, *,
             reference: Model | None = None, scales: Scales | None = None) -> Model:
        """The data set as a model over its reference (`reference` when
        given), in SI or `scales`."""
        if len(paths) != len(self.parameters):
            raise ValueError(f"{dataset.name}: {len(paths)} files for the parameters "
                             f"{self.parameters}")
        fields = {name: sph_field(read_sph(p), name=name)
                  for name, p in zip(self.parameters, paths)}
        model = over_reference(reference_of(dataset, reference=reference), fields,
                               kind="relative", name=dataset.name,
                               citation=dataset.citation)
        return model if scales is None else model.converted(scales)
