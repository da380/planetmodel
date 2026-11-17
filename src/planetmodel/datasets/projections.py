"""Map projections of gridded sources: the polar stereographic.

A regional grid of the polar regions is often published on a projected
plane, x and y in metres, rather than in latitude and longitude:
Antarctic models on EPSG:3031, Arctic ones on EPSG:3413 and their kin.
`PolarStereographic` is that projection of an ellipsoid, in EPSG's two
variants (Guidance Note 7-2): B, true to scale on a standard parallel
(EPSG:3031: 71 S, WGS 84), and A, a scale factor at the pole.  For the
south pole the easting x runs along the meridian 90 degrees east of the
central one and the northing y along the central meridian; for the north
pole y runs along the meridian opposite the central one.  With the
ellipsoid's eccentricity e and, for a geodetic latitude phi measured
toward the projection's pole::

    t   = tan(pi/4 - phi/2) / ((1 - e sin phi) / (1 + e sin phi))^(e/2),
    rho = a m_c t / t_c                              (variant B),
    rho = 2 a k_0 t / sqrt((1 + e)^(1 + e) (1 - e)^(1 - e))   (variant A),

m_c = cos phi_c / sqrt(1 - e^2 sin^2 phi_c) and t_c the same t at the
standard parallel phi_c.  The inverse is closed: the conformal latitude
chi = pi/2 - 2 atan t, then phi from chi by EPSG's series in e^2, good to
well below a millimetre.

The library's colatitude is geocentric, the projection's latitude
geodetic: `from_angles` and `to_angles` convert between them on the
projection's ellipsoid (tan(geodetic) = tan(geocentric) / (1 - e^2)),
so that a point of a model built in geodetic coordinates, as a code on
an ellipsoidal Earth writes it, lands where its makers put it.
`PolarStereographic.from_proj4` reads a projection from a file's
`proj4text`, the attribute the Earth Model Collaboration's projected
files carry, refusing anything but a polar stereographic one.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["ELLIPSOIDS", "PolarStereographic"]

#: Ellipsoids by the names proj gives them: (semi-major axis in metres,
#: inverse flattening).
ELLIPSOIDS = {"WGS84": (6378137.0, 298.257223563), "GRS80": (6378137.0,
                                                             298.257222101)}


@dataclass(frozen=True)
class PolarStereographic:
    """The polar stereographic projection of the module docstring: `pole`
    -1 for the south and +1 for the north; `standard_parallel` (degrees,
    variant B) or `scale` (the scale factor at the pole, variant A), one
    of them; `central_meridian` (degrees); the ellipsoid's
    `semi_major_axis` (metres) and `inverse_flattening`; `false_easting`
    and `false_northing` (metres)."""

    pole: int
    _: KW_ONLY
    standard_parallel: float | None = None
    scale: float | None = None
    central_meridian: float = 0.0
    semi_major_axis: float = ELLIPSOIDS["WGS84"][0]
    inverse_flattening: float = ELLIPSOIDS["WGS84"][1]
    false_easting: float = 0.0
    false_northing: float = 0.0

    def __post_init__(self) -> None:
        if self.pole not in (-1, 1):
            raise ValueError(f"pole is -1 (south) or 1 (north), got {self.pole!r}")
        if (self.standard_parallel is None) == (self.scale is None):
            raise ValueError("give one of standard_parallel (variant B) and scale "
                             "(variant A)")
        if self.standard_parallel is not None:
            sp = float(self.standard_parallel)
            if not 0.0 < self.pole * sp <= 90.0:
                raise ValueError(f"the standard parallel {sp} is not in the "
                                 f"{'south' if self.pole < 0 else 'north'}ern "
                                 "hemisphere")
        if self.scale is not None and not self.scale > 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if not (self.semi_major_axis > 0.0 and self.inverse_flattening > 1.0):
            raise ValueError("the ellipsoid's semi-major axis and inverse flattening "
                             "must be positive, the latter above one")

    @property
    def eccentricity(self) -> float:
        """The ellipsoid's first eccentricity."""
        f = 1.0 / self.inverse_flattening
        return float(np.sqrt(f * (2.0 - f)))

    def _t(self, phi: np.ndarray) -> np.ndarray:
        """t of the module docstring at geodetic latitudes `phi` (radians)
        measured toward the projection's pole."""
        e = self.eccentricity
        s = np.sin(phi)
        return (np.tan(0.25 * np.pi - 0.5 * phi)
                / ((1.0 - e * s) / (1.0 + e * s)) ** (0.5 * e))

    def _factor(self) -> float:
        """rho / t: a m_c / t_c, or 2 a k_0 / sqrt(...)."""
        a, e = self.semi_major_axis, self.eccentricity
        if self.standard_parallel is not None:
            pc = np.radians(self.pole * float(self.standard_parallel))
            if np.isclose(pc, 0.5 * np.pi, rtol=0.0, atol=1e-15):
                k0 = 1.0
            else:
                mc = np.cos(pc) / np.sqrt(1.0 - (e * np.sin(pc)) ** 2)
                return float(a * mc / self._t(np.array(pc)))
        else:
            k0 = float(self.scale)
        return float(2.0 * a * k0 / np.sqrt((1.0 + e) ** (1.0 + e)
                                            * (1.0 - e) ** (1.0 - e)))

    def forward(self, latitude: ArrayLike, longitude: ArrayLike
                ) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) in metres of geodetic latitudes and longitudes (degrees),
        broadcast together."""
        lat, lon = np.broadcast_arrays(np.asarray(latitude, dtype=float),
                                       np.asarray(longitude, dtype=float))
        if np.any(np.abs(lat) > 90.0):
            raise ValueError("latitudes must lie in [-90, 90]")
        phi = np.radians(self.pole * lat)                # toward the pole
        rho = self._factor() * self._t(phi)
        dl = np.radians(lon - self.central_meridian)
        x = self.false_easting + rho * np.sin(dl)
        y = self.false_northing - self.pole * rho * np.cos(dl)
        return x, y

    def inverse(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Geodetic latitudes and longitudes (degrees, the longitudes in
        (-180, 180] about none in particular) of points (x, y) in metres,
        broadcast together, by the closed form of the module docstring."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        dx, dy = x - self.false_easting, y - self.false_northing
        rho = np.hypot(dx, dy)
        t = rho / self._factor()
        chi = 0.5 * np.pi - 2.0 * np.arctan(t)
        e2 = self.eccentricity ** 2
        phi = (chi + (e2 / 2 + 5 * e2 ** 2 / 24 + e2 ** 3 / 12 + 13 * e2 ** 4 / 360)
               * np.sin(2 * chi)
               + (7 * e2 ** 2 / 48 + 29 * e2 ** 3 / 240 + 811 * e2 ** 4 / 11520)
               * np.sin(4 * chi)
               + (7 * e2 ** 3 / 120 + 81 * e2 ** 4 / 1120) * np.sin(6 * chi)
               + (4279 * e2 ** 4 / 161280) * np.sin(8 * chi))
        lon = self.central_meridian + np.degrees(np.arctan2(dx, -self.pole * dy))
        lon = np.where(rho > 0.0, lon, self.central_meridian)
        return self.pole * np.degrees(phi), (lon + 180.0) % 360.0 - 180.0

    def geodetic_latitude(self, theta: ArrayLike) -> np.ndarray:
        """Geodetic latitudes (degrees) on the projection's ellipsoid of the
        library's geocentric colatitudes (radians)."""
        psi = 0.5 * np.pi - np.asarray(theta, dtype=float)
        e2 = self.eccentricity ** 2
        return np.degrees(np.arctan2(np.sin(psi), (1.0 - e2) * np.cos(psi)))

    def from_angles(self, theta: ArrayLike, phi: ArrayLike
                    ) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) in metres of the library's geocentric colatitudes and
        longitudes (radians), broadcast together."""
        theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                         np.asarray(phi, dtype=float))
        return self.forward(self.geodetic_latitude(theta), np.degrees(phi))

    def to_angles(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """The library's geocentric colatitudes and longitudes (radians) of
        points (x, y) in metres, broadcast together, the longitudes in
        [0, 2 pi)."""
        lat, lon = self.inverse(x, y)
        e2 = self.eccentricity ** 2
        g = np.radians(lat)
        psi = np.arctan2((1.0 - e2) * np.sin(g), np.cos(g))
        return 0.5 * np.pi - psi, np.mod(np.radians(lon), 2.0 * np.pi)

    def scale_factor(self, latitude: ArrayLike) -> np.ndarray:
        """The projection's scale factor at geodetic latitudes (degrees):
        lengths on the plane are this many times those on the ellipsoid."""
        lat = np.asarray(latitude, dtype=float)
        phi = np.radians(np.clip(self.pole * lat, -89.999999, 89.999999))
        e = self.eccentricity
        m = np.cos(phi) / np.sqrt(1.0 - (e * np.sin(phi)) ** 2)
        return self._factor() * self._t(phi) / (self.semi_major_axis * m)

    @classmethod
    def from_proj4(cls, text: str) -> PolarStereographic:
        """The projection a proj4 string states: `+proj=stere` with `+lat_0`
        at a pole, `+lat_ts` (variant B) or `+k`/`+k_0` (variant A),
        `+lon_0`, `+x_0`, `+y_0`, and the ellipsoid by `+datum` or
        `+ellps` (WGS84, GRS80) or `+a` with `+rf` or `+b`, in metres
        (`+units=m`, the default).  Anything else is refused by name."""
        words = {}
        for w in text.split():
            if w.startswith("+"):
                k, _, v = w[1:].partition("=")
                words[k] = v
        if words.get("proj") != "stere":
            raise ValueError(f"{text!r} is not a stereographic projection "
                             "(+proj=stere)")
        lat0 = float(words.get("lat_0", "nan"))
        if abs(abs(lat0) - 90.0) > 1e-9:
            raise ValueError(f"{text!r} is not polar: +lat_0 is {lat0}, not +-90")
        if words.get("units", "m") != "m":
            raise ValueError(f"{text!r} is in {words['units']}, not metres")
        a, rf = ELLIPSOIDS["WGS84"]
        name = words.get("datum") or words.get("ellps")
        if name is not None:
            if name not in ELLIPSOIDS:
                raise ValueError(f"{text!r}: the ellipsoid {name!r} is not one of "
                                 f"{sorted(ELLIPSOIDS)}")
            a, rf = ELLIPSOIDS[name]
        if "a" in words:
            a = float(words["a"])
            if "rf" in words:
                rf = float(words["rf"])
            elif "b" in words:
                rf = a / (a - float(words["b"]))
        pole = -1 if lat0 < 0 else 1
        k = words.get("k", words.get("k_0"))
        lat_ts = words.get("lat_ts")
        if lat_ts is not None:
            return cls(pole, standard_parallel=float(lat_ts),
                       central_meridian=float(words.get("lon_0", 0.0)),
                       semi_major_axis=a, inverse_flattening=rf,
                       false_easting=float(words.get("x_0", 0.0)),
                       false_northing=float(words.get("y_0", 0.0)))
        return cls(pole, scale=float(k if k is not None else 1.0),
                   central_meridian=float(words.get("lon_0", 0.0)),
                   semi_major_axis=a, inverse_flattening=rf,
                   false_easting=float(words.get("x_0", 0.0)),
                   false_northing=float(words.get("y_0", 0.0)))
