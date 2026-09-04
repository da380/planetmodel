"""topography.py -- shapes on the sphere.

A Topography is a function of direction alone: given a colatitude and
longitude it returns a height.  It says nothing about where it sits;
placing it at a radius is the Surface's job (surface.py).  Keeping the
two apart is what stops a crustal-thickness grid being silently treated
as a radius.

The protocol is one method.  A gradient, a mean and known bounds are
optional enrichments, discovered by attribute and supplied numerically
when absent, so a bare `lambda theta, phi: ...` is a perfectly good
topography and a spherical-harmonic expansion is a better one.  The
rule for using one is *adapt, then call*: `as_topography(x)` returns an
object that has every method, exact where `x` supplied it and numerical
otherwise, and nothing else in the library probes for attributes.

Provenance.  A shape built from files says so: `provenance()` reports
the files, the factor they were scaled by and the interpolation used,
walking through sums, scalings and centring, so that a mesh manifest can
record how a relief was made without knowing how the classes nest.

Coordinates.  Everything here speaks the library's convention --
colatitude theta in [0, pi] and longitude phi in (-pi, pi], both in
radians.  Gridded data arrives in degrees of latitude and longitude, and
exactly one private helper converts, so the boundary between the two
conventions is a single function rather than a habit.
"""
from __future__ import annotations

import numpy as np
from typing import Protocol, runtime_checkable

from ..registry import register

__all__ = [
    "Topography", "as_topography", "GriddedTopography",
    "AnalyticTopography", "HarmonicTopography", "SumTopography",
    "ScaledTopography", "CentredTopography", "ZeroTopography",
]


def _no_provenance() -> dict:
    """What a shape built from no file reports."""
    return {"files": [], "exaggeration": 1.0, "interpolation": None}


@runtime_checkable
class Topography(Protocol):
    """A height as a function of direction.

    Required: __call__(theta, phi) -> array, broadcasting over its
    arguments.  Optional and discovered when present:

        gradient(theta, phi) -> (d/dtheta, d/dphi)
        mean() -> float                 area-weighted
        bounds() -> (min, max)
    """

    def __call__(self, theta, phi): ...


def _angles(theta, phi):
    """Broadcast (theta, phi) to a common shape as float arrays."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return np.broadcast_arrays(theta, phi)


def _to_lonlat_degrees(theta, phi):
    """(colatitude, longitude) in radians -> (lon, lat) in degrees.

    The single crossing point between the library's convention and the
    degrees-and-latitude convention of gridded data files.
    """
    lat = 90.0 - np.degrees(theta)
    lon = np.degrees(phi)
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, lat


class _TopographyAlgebra:
    """Arithmetic shared by the concrete topographies."""

    def provenance(self) -> dict:
        """The files this shape was built from, and how.

        A dict with `files` (a list of `{"file", "scale_to_m"}` entries),
        `exaggeration` (the product of the scalings applied outside any
        sum) and `interpolation` (the gridded interpolation in use, or
        None).  A shape built from no file reports an empty list.
        """
        return _no_provenance()

    def __add__(self, other):
        if not callable(other):
            return NotImplemented
        return SumTopography((self, other))

    def __sub__(self, other):
        if not callable(other):
            return NotImplemented
        return SumTopography((self, ScaledTopography(other, -1.0)))

    def __neg__(self):
        return ScaledTopography(self, -1.0)

    def __mul__(self, k):
        """Scaling: this is what a topography exaggeration factor is."""
        if callable(k):
            raise TypeError("topography * topography is not defined")
        return ScaledTopography(self, k)

    __rmul__ = __mul__

    def __truediv__(self, k):
        return ScaledTopography(self, 1.0 / float(k))


@register("topography", "zero")
class ZeroTopography(_TopographyAlgebra):
    """The flat shape; the default wherever no topography is attached."""

    def __call__(self, theta, phi):
        theta, _ = _angles(theta, phi)
        return np.zeros(theta.shape)

    def gradient(self, theta, phi):
        theta, _ = _angles(theta, phi)
        z = np.zeros(theta.shape)
        return z, z.copy()

    def mean(self) -> float:
        return 0.0

    def bounds(self) -> tuple[float, float]:
        return 0.0, 0.0

    def __repr__(self) -> str:
        return "ZeroTopography()"


@register("topography", "analytic")
class AnalyticTopography(_TopographyAlgebra):
    """Any callable of (theta, phi), with numerical enrichments.

    The landing point for a random-field sample, a spherical harmonic
    evaluated in closed form, or a test shape.  Supply `gradient` if you
    have it; otherwise it is taken by central differences, with the pole
    singularity of the phi component handled by the caller (Mapping
    clips sin(theta), see mapping.py).
    """

    def __init__(self, fn, *, gradient=None, name: str | None = None,
                 dstep: float = 1e-6) -> None:
        """Bind a callable; a gradient or mean it carries itself is kept."""
        if not callable(fn):
            raise TypeError(f"expected a callable, got {type(fn).__name__}")
        self._fn = fn
        self._grad = gradient if gradient is not None else getattr(
            fn, "gradient", None)
        self._mean = getattr(fn, "mean", None)
        self._h = float(dstep)
        self.name = name

    def __call__(self, theta, phi):
        theta, phi = _angles(theta, phi)
        return np.asarray(self._fn(theta, phi), dtype=float)

    def gradient(self, theta, phi):
        """(d/dtheta, d/dphi), analytic if given, else central differences."""
        if self._grad is not None:
            gt, gp = self._grad(*_angles(theta, phi))
            return np.asarray(gt, dtype=float), np.asarray(gp, dtype=float)
        theta, phi = _angles(theta, phi)
        h = self._h
        dt = (self(theta + h, phi) - self(theta - h, phi)) / (2.0 * h)
        dp = (self(theta, phi + h) - self(theta, phi - h)) / (2.0 * h)
        return dt, dp

    def mean(self, *, n_theta: int = 64, n_phi: int = 128) -> float:
        """The area-weighted mean, by Gauss-Legendre in cos(theta).

        The quadrature the sphere asks for rather than a fine grid: see
        `_quadrature`.  The defaults integrate any shape band-limited at
        degree 20 exactly, which is what makes the zero-mean contract of
        `ReferenceBody.with_surface` a statement about the shape and not
        about the grid it was sampled on.  A callable carrying its own
        exact `mean` is asked instead.
        """
        if self._mean is not None:
            return float(self._mean())
        return _quadrature(self, n_theta, n_phi)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"AnalyticTopography({self._fn!r}{nm})"


@register("topography", "harmonic")
class HarmonicTopography(_TopographyAlgebra):
    """A shape given by real spherical-harmonic coefficients.

    The band-limited topography: exact values, exact gradients, an exact
    mean, and the form a spectral consumer already thinks in.  It is
    what `AnalyticTopography.mean`'s Gauss-Legendre rule was built to
    integrate exactly.

    **Layout.**  `coeffs[0, l, m]` multiplies the cosine harmonic and
    `coeffs[1, l, m]` the sine one, for 0 <= m <= l <= lmax -- the real
    layout of pyshtools, so an `SHCoeffs.coeffs` array drops straight
    in.  Entries with m > l name no harmonic and `coeffs[1, l, 0]`
    multiplies sin(0 phi) = 0; both must be zero rather than ignored,
    since a caller who filled them meant something by them.

    **Convention.**  Orthonormal, with the Condon-Shortley phase
    *included* -- the convention GSHTrans uses.  Explicitly: the complex
    harmonics are

        Y_lm = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!) P_l^m(cos theta)
               exp(i m phi),

    with `P_l^m` carrying the phase (-1)^m, and the real basis this
    class expands in is

        Y_l0,       Y_lm^c = sqrt(2) Re Y_lm,
                    Y_lm^s = sqrt(2) Im Y_lm      (m > 0),

    orthonormal on the sphere under the *unweighted* integral, so that
    the integral of a product of two of them is delta.  In particular
    Y_00 = 1 / sqrt(4 pi), which is why `mean()` is the degree-zero
    coefficient divided by sqrt(4 pi) exactly, and why every other
    harmonic integrates to zero -- the statement the zero-mean contract
    of `ReferenceBody.with_surface` needs.

    (pyshtools names the same convention `normalization="ortho"` with
    its `csphase` flag set to include the phase; its flag value for
    "included" is -1, since 1 means "do not apply".  The formulas
    above, not a flag value, are what this class implements.)

    **Evaluation.**  A three-term recurrence in l for the normalised
    associated Legendre functions, seeded along the sectoral diagonal,
    with the theta derivative obtained by differentiating that same
    recurrence -- so the gradient is exact, not a difference, and costs
    one more array per step.  d/dphi is the analytic m (cos, sin) swap.
    Nothing is materialised over (l, m): the sum is accumulated as the
    recurrence walks, so the working set is a few arrays the size of the
    evaluation points.
    """

    def __init__(self, coeffs, *, lmax: int | None = None,
                 name: str | None = None) -> None:
        """Bind real coefficients in the pyshtools layout, optionally cut."""
        c = np.asarray(coeffs, dtype=float)
        if c.ndim != 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
            raise ValueError(
                "coeffs must have the real layout (2, lmax + 1, lmax + 1), "
                f"cosine and sine, got shape {c.shape}")
        L = c.shape[1] - 1
        if lmax is not None:
            L = int(lmax)
            if L < 0:
                raise ValueError(f"lmax must be non-negative, got {lmax}")
            keep = min(L, c.shape[1] - 1) + 1
            padded = np.zeros((2, L + 1, L + 1))
            padded[:, :keep, :keep] = c[:, :keep, :keep]
            c = padded
        else:
            c = c.copy()

        upper = np.triu(np.ones((L + 1, L + 1), dtype=bool), 1)
        if np.any(c[:, upper]):
            raise ValueError(
                "coeffs has non-zero entries with m > l, which name no "
                "harmonic: the layout is coeffs[kind, l, m] with m <= l")
        if np.any(c[1, :, 0]):
            raise ValueError(
                "coeffs[1, :, 0] is non-zero, but it multiplies sin(0 phi): "
                "the m = 0 harmonic has a cosine part only")

        self.coeffs = c
        self.lmax = L
        self.name = name

    # -- the expansion ------------------------------------------------------

    def _expand(self, theta, phi):
        """(f, df/dtheta, df/dphi) at the given directions.

        One pass over the (l, m) triangle.  For each m the sectoral
        function seeds the ladder in l,

            Ptil_0^0 = 1/sqrt(4 pi)
            Ptil_m^m = -sqrt((2m+1)/(2m)) sin(theta) Ptil_{m-1}^{m-1}
            Ptil_{m+1}^m = sqrt(2m+3) cos(theta) Ptil_m^m
            Ptil_l^m = a cos(theta) Ptil_{l-1}^m - b Ptil_{l-2}^m,

            a = sqrt((2l-1)(2l+1) / ((l-m)(l+m)))
            b = sqrt((2l+1)(l+m-1)(l-m-1) / ((2l-3)(l-m)(l+m))),

        where Ptil carries the full 4-pi orthonormal factor and the
        Condon-Shortley phase (the minus sign on the sectoral step is
        that phase).  Differentiating each line with respect to theta,
        using d(sin)/dtheta = cos and d(cos)/dtheta = -sin, gives the
        matching recurrence for dPtil/dtheta -- exact, and stable in the
        same way, because it is the same recurrence.
        """
        theta, phi = _angles(theta, phi)
        x, s = np.cos(theta), np.sin(theta)
        c = self.coeffs
        root2 = np.sqrt(2.0)

        f = np.zeros(theta.shape)
        ft = np.zeros(theta.shape)
        fp = np.zeros(theta.shape)

        # The sectoral function and its theta derivative, walked in m.
        pmm = np.full(theta.shape, 1.0 / np.sqrt(4.0 * np.pi))
        dmm = np.zeros(theta.shape)

        for m in range(self.lmax + 1):
            if m:
                k = np.sqrt((2.0 * m + 1.0) / (2.0 * m))
                pmm, dmm = -k * s * pmm, -k * (x * pmm + s * dmm)
            if not np.any(c[:, m:, m]):
                continue
            cm, sm = np.cos(m * phi), np.sin(m * phi)

            p, d = pmm, dmm
            pp = dp = None                    # the l - 2 terms
            for l in range(m, self.lmax + 1):
                if l == m + 1:
                    a = np.sqrt(2.0 * m + 3.0)
                    p, d, pp, dp = a * x * p, a * (x * d - s * p), p, d
                elif l > m:
                    a = np.sqrt((2 * l - 1) * (2 * l + 1)
                                / ((l - m) * (l + m)))
                    b = np.sqrt((2 * l + 1) * (l + m - 1) * (l - m - 1)
                                / ((2 * l - 3) * (l - m) * (l + m)))
                    p, d, pp, dp = (a * x * p - b * pp,
                                    a * (x * d - s * p) - b * dp, p, d)
                if m == 0:
                    w = c[0, l, 0]
                    f += w * p; ft += w * d
                else:
                    cc, ss = c[0, l, m], c[1, l, m]
                    w = root2 * (cc * cm + ss * sm)
                    f += w * p; ft += w * d
                    fp += p * (root2 * m * (ss * cm - cc * sm))
        return f, ft, fp

    def __call__(self, theta, phi):
        """The height in the given directions."""
        return self._expand(theta, phi)[0]

    def gradient(self, theta, phi):
        """(d/dtheta, d/dphi), both exact.

        The phi derivative is m times the cosine-sine swap, so it
        vanishes at the poles for every m -- as it must, since the
        harmonics with m > 0 vanish there and the limit of a continuous
        shape cannot depend on the meridian of approach.
        """
        _, gt, gp = self._expand(theta, phi)
        return gt, gp

    def mean(self) -> float:
        """The area-weighted mean: the degree-zero coefficient, exactly.

        Y_00 = 1/sqrt(4 pi) is constant and every other harmonic
        integrates to zero over the sphere, so no quadrature is
        involved and the zero-mean contract is exact for a band-limited
        shape rather than exact to a rule.
        """
        return float(self.coeffs[0, 0, 0] / np.sqrt(4.0 * np.pi))

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"HarmonicTopography(lmax={self.lmax}{nm})"


class SumTopography(_TopographyAlgebra):
    """The sum of topographies, evaluated pointwise.

    Each term is adapted on construction, so the sum's gradient and
    mean are exact wherever the terms' are.
    """

    def __init__(self, terms, *, name: str | None = None) -> None:
        terms = tuple(as_topography(t) for t in terms)
        if not terms:
            raise ValueError("SumTopography needs at least one term")
        self._terms = terms
        self.name = name

    @property
    def terms(self):
        return self._terms

    def __call__(self, theta, phi):
        out = None
        for t in self._terms:
            v = np.asarray(t(theta, phi), dtype=float)
            out = v if out is None else out + v
        return out

    def gradient(self, theta, phi):
        gt = gp = None
        for t in self._terms:
            a, b = t.gradient(theta, phi)
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            gt = a if gt is None else gt + a
            gp = b if gp is None else gp + b
        return gt, gp

    def mean(self) -> float:
        return float(sum(t.mean() for t in self._terms))

    def provenance(self) -> dict:
        """The terms' files pooled; a sum is not an exaggeration of anything."""
        files: list = []
        interpolation = None
        for t in self._terms:
            p = t.provenance()
            files.extend(p["files"])
            interpolation = interpolation or p["interpolation"]
        return {"files": files, "exaggeration": 1.0,
                "interpolation": interpolation}

    def __repr__(self) -> str:
        return f"SumTopography({len(self._terms)} terms)"


class ScaledTopography(_TopographyAlgebra):
    """A topography times a constant -- an exaggeration factor, usually."""

    def __init__(self, topo, factor: float, *, name: str | None = None) -> None:
        self._topo = as_topography(topo)
        self.factor = float(factor)
        self.name = name

    @property
    def shape(self):
        """The topography being scaled."""
        return self._topo

    def __call__(self, theta, phi):
        return self.factor * np.asarray(self._topo(theta, phi), dtype=float)

    def gradient(self, theta, phi):
        gt, gp = self._topo.gradient(theta, phi)
        return (self.factor * np.asarray(gt, dtype=float),
                self.factor * np.asarray(gp, dtype=float))

    def mean(self) -> float:
        return self.factor * self._topo.mean()

    def bounds(self):
        b = getattr(self._topo, "bounds", None)
        if b is None:
            return None
        lo, hi = b()
        return (self.factor * lo, self.factor * hi) if self.factor >= 0 else (
            self.factor * hi, self.factor * lo)

    def provenance(self) -> dict:
        """The scaled shape's provenance, with this factor multiplied in."""
        p = dict(self._topo.provenance())
        p["exaggeration"] = self.factor * p["exaggeration"]
        return p

    def __repr__(self) -> str:
        return f"ScaledTopography({self.factor:g} x {self._topo!r})"


class CentredTopography(_TopographyAlgebra):
    """A shape with a constant removed: `shape(theta, phi) - shift`.

    What `Surface.centred` and `ReferenceBody.with_surface` produce, so
    that the relief carried by an interface has zero mean and the mean
    lives in the interface radius.  It keeps `shape` and `shift` as
    values rather than folding the constant into a sum, so that a
    provenance walk sees the relief it was built from -- its files, its
    exaggeration -- and never meets the constant, which is a mean radius
    in metres and not an exaggeration of anything.
    """

    def __init__(self, shape, shift: float, *, name: str | None = None) -> None:
        self.shape = as_topography(shape)
        self.shift = float(shift)
        self.name = name

    def __call__(self, theta, phi):
        return np.asarray(self.shape(theta, phi), dtype=float) - self.shift

    def gradient(self, theta, phi):
        """The shape's gradient: a constant has none."""
        return self.shape.gradient(theta, phi)

    def mean(self) -> float:
        return float(self.shape.mean()) - self.shift

    def bounds(self):
        b = getattr(self.shape, "bounds", None)
        if b is None:
            return None
        lo, hi = b()
        return lo - self.shift, hi - self.shift

    def provenance(self) -> dict:
        return self.shape.provenance()

    def __repr__(self) -> str:
        return f"CentredTopography({self.shape!r} - {self.shift:g})"








def _quadrature(fn, n_theta: int, n_phi: int) -> float:
    """(1/4pi) times the integral of fn over the sphere, by quadrature.

    Gauss-Legendre in x = cos(theta), equispaced in phi.  The two
    directions are not alike and the rule reflects it.  In x the measure
    is flat -- the area element sin(theta) dtheta dphi is dx dphi -- so
    an n-point Gauss rule integrates a polynomial of degree 2n - 1 in x
    exactly, and a shape band-limited at degree L is exactly that, of
    degree L.  In phi the integrand is periodic, so the rectangle rule
    on n_phi equispaced points is exact for every azimuthal order below
    n_phi and, in particular, annihilates every m != 0 term exactly.
    Together they make the mean of a band-limited shape exact rather
    than merely converged, and the nodes avoid both poles, where a
    topography's phi gradient is singular.
    """
    n_theta, n_phi = int(n_theta), int(n_phi)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x)[:, None]
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)[None, :]
    # Broadcast rather than trust: a callable that ignores its arguments
    # and returns one number is a legitimate topography, and weighting a
    # scalar as though it were the grid would scale it by 1 / n_phi.
    v = np.broadcast_to(np.asarray(fn(theta, phi), dtype=float),
                        (n_theta, n_phi))
    return float(np.sum(w[:, None] * v) / (2.0 * n_phi))


def as_topography(fn) -> Topography:
    """Adapt any callable of (theta, phi) to the Topography protocol.

    An object that already provides a gradient, a mean and a provenance
    is returned unchanged, so wrapping a gridded or harmonic topography
    never costs it its exact enrichments; anything else is wrapped in
    an `AnalyticTopography`, which keeps whichever of those the callable
    carries and supplies the rest numerically.
    """
    if all(hasattr(fn, a) for a in ("gradient", "mean", "provenance")):
        return fn
    return AnalyticTopography(fn)


@register("topography", "gridded")
class GriddedTopography(_TopographyAlgebra):
    """A topography sampled on a longitude-latitude tensor grid.

    Built for CRUST-1.0-style data: `lon lat value` on a regular grid,
    values in kilometres, latitude running downwards through the file.
    Nothing is assumed about ordering -- both coordinates are sorted on
    read.

    Three details decide whether this behaves at the awkward places:

    * **The seam.**  Longitude wraps, so a query between the last and
      first columns interpolates across the join rather than clamping.
    * **The poles.**  A lat-lon grid has no sample at a pole, and any
      value there must be independent of longitude or the field is
      discontinuous.  The pole value is therefore the mean of the
      outermost ring, and latitude interpolates linearly between the
      ring and the pole.
    * **The mean.**  Cells shrink towards the poles, so an unweighted
      average over-weights them.  mean() weights by cell area,
      proportional to the difference of the sines of the cell's latitude
      edges, which is why mean() of a Y_1^0-shaped field vanishes.
    """

    def __init__(self, lons, lats, values, *, interpolation: str = "bilinear",
                 name: str | None = None) -> None:
        """Bind an ascending lon-lat grid and its values, shape (nlat, nlon)."""
        self.lons = np.asarray(lons, dtype=float)
        self.lats = np.asarray(lats, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if self.lons.ndim != 1 or self.lats.ndim != 1:
            raise ValueError("lons and lats must be 1-d")
        if self.values.shape != (self.lats.size, self.lons.size):
            raise ValueError(
                f"values must have shape (nlat, nlon) = "
                f"({self.lats.size}, {self.lons.size}), got {self.values.shape}")
        # Longitudes are held in [-180, 180): a 0..360 grid is the same
        # data, and evaluation wraps queries into that range, so a grid
        # left on its own convention would extrapolate across half the
        # planet from a single padded column.
        wrapped = (self.lons + 180.0) % 360.0 - 180.0
        if not np.array_equal(wrapped, self.lons):
            order = np.argsort(wrapped)
            self.lons = wrapped[order]
            self.values = self.values[:, order]
        if not (np.all(np.diff(self.lons) > 0) and np.all(np.diff(self.lats) > 0)):
            raise ValueError("lons and lats must be strictly increasing")
        if interpolation not in ("bilinear", "bicubic"):
            raise ValueError("interpolation must be 'bilinear' or 'bicubic'")
        self.interpolation = interpolation
        self.name = name
        self._spline = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_xyz(cls, path, *, scale: float = 1.0, **kw) -> "GriddedTopography":
        """Read a `lon lat value` text file onto its tensor grid.

        `scale` multiplies the values, which is how kilometres become
        metres: CRUST-1.0 ships km and the model layer works in SI.
        Input that is not a complete tensor grid is rejected rather than
        silently gap-filled.
        """
        raw = np.loadtxt(path, dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 3:
            raise ValueError(f"{path}: expected columns lon lat value")
        lon, lat, val = raw[:, 0], raw[:, 1], raw[:, 2]
        lons, ix = np.unique(lon, return_inverse=True)
        lats, iy = np.unique(lat, return_inverse=True)
        if lons.size * lats.size != raw.shape[0]:
            raise ValueError(
                f"{path}: {raw.shape[0]} rows do not fill a "
                f"{lats.size} x {lons.size} tensor grid")
        values = np.full((lats.size, lons.size), np.nan)
        values[iy, ix] = val * float(scale)
        if np.isnan(values).any():
            raise ValueError(f"{path}: grid has holes after regridding")
        topo = cls(lons, lats, values, name=str(path), **kw)
        # Kept so a manifest can say how the file was read, and a consumer
        # can read it the same way.
        topo.scale_to_m = float(scale)
        return topo

    # -- the awkward places -------------------------------------------------

    def _pole_values(self) -> tuple[float, float]:
        """(south, north) pole values: the means of the outermost rings.

        Longitude-independent by construction, which is the only choice
        that leaves the field continuous at the pole.
        """
        return float(self.values[0].mean()), float(self.values[-1].mean())

    def _augmented(self):
        """The grid extended by a wrapped longitude column at each end and
        a pole row at each end -- the form on which interpolation is
        just a lookup, with no special cases.
        """
        lons = np.concatenate(([self.lons[-1] - 360.0], self.lons,
                               [self.lons[0] + 360.0]))
        vals = np.column_stack([self.values[:, -1], self.values,
                                self.values[:, 0]])
        south, north = self._pole_values()
        lats = np.concatenate(([-90.0], self.lats, [90.0]))
        vals = np.vstack([np.full(lons.size, south), vals,
                          np.full(lons.size, north)])
        return lons, lats, vals

    def __call__(self, theta, phi):
        """The height in the given directions."""
        theta, phi = _angles(theta, phi)
        lon, lat = _to_lonlat_degrees(theta, phi)
        if self.interpolation == "bicubic":
            return self._bicubic(lon, lat)
        return self._bilinear(lon, lat)

    def _bilinear(self, lon, lat):
        """Bilinear interpolation on the augmented grid."""
        lons, lats, vals = self._augmented()
        lat = np.clip(lat, -90.0, 90.0)

        i = np.clip(np.searchsorted(lons, lon, side="right") - 1,
                    0, lons.size - 2)
        j = np.clip(np.searchsorted(lats, lat, side="right") - 1,
                    0, lats.size - 2)
        x0, x1 = lons[i], lons[i + 1]
        y0, y1 = lats[j], lats[j + 1]
        a = np.where(x1 > x0, (lon - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
        b = np.where(y1 > y0, (lat - y0) / np.where(y1 > y0, y1 - y0, 1.0), 0.0)
        return ((1 - a) * (1 - b) * vals[j, i]
                + a * (1 - b) * vals[j, i + 1]
                + (1 - a) * b * vals[j + 1, i]
                + a * b * vals[j + 1, i + 1])

    def _bicubic(self, lon, lat):
        """Bicubic interpolation, on a seam-padded copy of the grid."""
        if self._spline is None:
            from scipy.interpolate import RectBivariateSpline
            pad = 2
            lons = np.concatenate((self.lons[-pad:] - 360.0, self.lons,
                                   self.lons[:pad] + 360.0))
            vals = np.column_stack([self.values[:, -pad:], self.values,
                                    self.values[:, :pad]])
            south, north = self._pole_values()
            lats = np.concatenate(([-90.0], self.lats, [90.0]))
            vals = np.vstack([np.full(lons.size, south), vals,
                              np.full(lons.size, north)])
            self._spline = RectBivariateSpline(lats, lons, vals)
        lat = np.clip(lat, -90.0, 90.0)
        return self._spline.ev(lat, lon)

    def gradient(self, theta, phi, *, h: float = 1e-6):
        """(d/dtheta, d/dphi) by central differences.

        Bilinear interpolation has kinks at cell edges, so this is
        piecewise constant and discontinuous there.  That is acceptable
        under the smoothness doctrine -- coefficients need only be
        piecewise continuous -- but a mapping wanting smooth angular
        derivatives should use interpolation="bicubic".
        """
        theta, phi = _angles(theta, phi)
        dt = (self(theta + h, phi) - self(theta - h, phi)) / (2.0 * h)
        dp = (self(theta, phi + h) - self(theta, phi - h)) / (2.0 * h)
        return dt, dp

    # -- reductions ---------------------------------------------------------

    def _cell_weights(self) -> np.ndarray:
        """Cell areas: dlon times the difference of the sines of the edges."""
        edges = np.empty(self.lats.size + 1)
        edges[1:-1] = 0.5 * (self.lats[:-1] + self.lats[1:])
        edges[0] = max(-90.0, self.lats[0] - 0.5 * (self.lats[1] - self.lats[0]))
        edges[-1] = min(90.0, self.lats[-1] + 0.5 * (self.lats[-1] - self.lats[-2]))
        dsin = np.sin(np.radians(edges[1:])) - np.sin(np.radians(edges[:-1]))

        dlon = np.empty(self.lons.size)
        if self.lons.size > 1:
            mid = 0.5 * (self.lons[:-1] + self.lons[1:])
            dlon[1:-1] = mid[1:] - mid[:-1]
            dlon[0] = mid[0] - (self.lons[0] - 0.5 * (self.lons[1] - self.lons[0]))
            dlon[-1] = ((self.lons[-1] + 0.5 * (self.lons[-1] - self.lons[-2]))
                        - mid[-1])
        else:
            dlon[:] = 360.0
        return dsin[:, None] * dlon[None, :]

    def mean(self) -> float:
        """The area-weighted mean of the field over the sphere."""
        w = self._cell_weights()
        return float(np.sum(w * self.values) / np.sum(w))

    def bounds(self) -> tuple[float, float]:
        return float(self.values.min()), float(self.values.max())

    def provenance(self) -> dict:
        """The file this grid was read from, where `from_xyz` built it."""
        files = []
        if self.name:
            files.append({"file": str(self.name),
                          "scale_to_m": float(getattr(self, "scale_to_m", 1.0))})
        return {"files": files, "exaggeration": 1.0,
                "interpolation": self.interpolation}

    # -- combination --------------------------------------------------------

    def same_grid_as(self, other) -> bool:
        """Whether another gridded topography shares this grid exactly."""
        return (isinstance(other, GriddedTopography)
                and np.array_equal(self.lons, other.lons)
                and np.array_equal(self.lats, other.lats))

    def regridded_to(self, other) -> "GriddedTopography":
        """This field resampled onto another's grid, explicitly."""
        lon, lat = np.meshgrid(other.lons, other.lats)
        theta = np.radians(90.0 - lat)
        phi = np.radians(lon)
        return GriddedTopography(other.lons, other.lats, self(theta, phi),
                                 interpolation=self.interpolation,
                                 name=self.name)

    def __add__(self, other):
        """Sum on a shared grid; mismatched grids must be resampled first.

        Silently adopting one operand's coordinates -- the obvious
        shortcut -- turns a units or resolution mistake into a plausible
        looking field, so it is refused and names the fix.
        """
        if isinstance(other, GriddedTopography):
            if not self.same_grid_as(other):
                raise ValueError(
                    "cannot add topographies on different grids: "
                    f"{self.lats.size}x{self.lons.size} and "
                    f"{other.lats.size}x{other.lons.size}. Resample first, "
                    "e.g. a + b.regridded_to(a)")
            return GriddedTopography(self.lons, self.lats,
                                     self.values + other.values,
                                     interpolation=self.interpolation)
        return super().__add__(other)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return (f"GriddedTopography({self.lats.size}x{self.lons.size}, "
                f"{self.interpolation}{nm})")
