"""Degree solutions, Love numbers, and the file pyslfp reads.

A degree is solved for one forcing: a unit surface density through its
traction-like piece, its attraction piece, or both, or a unit external
potential psi = (r/a)^l.  The surface values of U, V and phi per unit
forcing are the generalised Love numbers of Al-Attar et al. (2024),
eq. (62): with the load split into a traction-like term and a
potential-source term,

    u_lm   = h_l sigma_lm + h^u_l zeta^u_lm + h^phi_l zeta^phi_lm,
    phi_lm = k_l sigma_lm + k^u_l zeta^u_lm + k^phi_l zeta^phi_lm,

a true surface load acting through both channels, so h = h^u + h^phi
and k = k^u + k^phi; the tangential numbers l^u, l^phi follow V in the
same way, and h^t, l^t, k^t are the response to the unit external
potential; the load sums are the properties `h`, `l_load` and `k`.  By
the symmetry of the bilinear form g h^phi_l = k^u_l, a check on every
calculation.  All numbers are in the material's units
with phi the physical potential perturbation, negative near added mass;
the conventional dimensionless load numbers are

    h' = h g (2l + 1) / (4 pi G a),   l' = l g (2l + 1) / (4 pi G a),
    k' = -k (2l + 1) / (4 pi G a) - 1,

and the geodetic tidal numbers k^T = k^t, h^T = -g h^t, l^T = -g l^t.
Degree 1 is in the centre-of-mass frame, where the potential channel
vanishes and k' = -1; at degree 0 the potential-only fluid reduction is
invalid, so fluid regions are mu = 0 elastic media and the total k_0
satisfies k^u + k^phi = -4 pi G a.

The pyslfp file is plain text, one row per degree from 0 with columns
l, h_u, k_u, h_phi, k_phi, h_t, k_t in SI: h per unit surface density
in m^3 kg^-1, k likewise in m^4 kg^-1 s^-2, h_t in s^2 m^-1 and k_t
dimensionless.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field, replace

import numpy as np

from ..mesh1d.gll import lagrange_basis
from ..units import FREQUENCY, Dimensions, Scales
from .assembly import DegreeSystem
from .material import Material

__all__ = ["FORCINGS", "DegreeSolution", "solve_degree", "LoveNumbers",
           "love_numbers", "read_love_numbers"]

FORCINGS = ("load", "load_force", "load_potential", "tide")

#: Dimensions of each Love number: displacement or potential per unit
#: surface density, or per unit potential.
_PER_LOAD_LENGTH = Dimensions(mass=-1, length=3)
_PER_LOAD_POTENTIAL = Dimensions(mass=-1, length=4, time=-2)
_PER_POTENTIAL_LENGTH = Dimensions(length=-1, time=2)
_DIMENSIONS = {
    "h_u": _PER_LOAD_LENGTH, "l_u": _PER_LOAD_LENGTH,
    "k_u": _PER_LOAD_POTENTIAL,
    "h_phi": _PER_LOAD_LENGTH, "l_phi": _PER_LOAD_LENGTH,
    "k_phi": _PER_LOAD_POTENTIAL,
    "h_t": _PER_POTENTIAL_LENGTH, "l_t": _PER_POTENTIAL_LENGTH,
    "k_t": Dimensions(),
}
_COLUMNS = ("h_u", "k_u", "h_phi", "k_phi", "h_t", "k_t")


@dataclass(frozen=True)
class DegreeSolution:
    """The radial solution of one degree for one forcing.

    `U`, `V` and `phi` are nodal arrays of shape (nspec, ngll) over the
    whole mesh, zero below the solved sub-mesh and where a component has
    no dof; `evaluate(radii)` interpolates them within their elements.
    `surface` is (U, V, phi) at the outer boundary.
    """

    l: int
    forcing: str
    mesh: object = field(repr=False)
    U: np.ndarray = field(repr=False)
    V: np.ndarray = field(repr=False)
    phi: np.ndarray = field(repr=False)

    @property
    def surface(self) -> tuple[complex, complex, complex]:
        return (self.U[-1, -1].item(), self.V[-1, -1].item(),
                self.phi[-1, -1].item())

    def evaluate(self, radii) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(U, V, phi) at `radii`, each element's polynomial evaluated on
        its own interval; a radius on an element boundary takes the
        element above it."""
        r = np.asarray(radii, dtype=float)
        flat = r.reshape(-1)
        mesh = self.mesh
        if flat.size and (flat.min() < mesh.left[0] or flat.max() > mesh.right[-1]):
            raise ValueError("radii must lie within the mesh")
        out = [np.zeros(flat.shape, dtype=a.dtype) for a in (self.U, self.V, self.phi)]
        es = np.array([mesh.element_of(x) for x in flat], dtype=int)
        for e in np.unique(es):
            m = es == e
            basis = lagrange_basis(mesh.r[e], flat[m])
            for o, a in zip(out, (self.U, self.V, self.phi)):
                o[m] = basis @ a[e]
        return tuple(o.reshape(r.shape) for o in out)


def _material(model_or_material, *, mesh, ngll, lmax) -> Material:
    if isinstance(model_or_material, Material):
        if mesh is not None:
            raise ValueError("a Material already fixes the mesh")
        return model_or_material
    if mesh is None:
        from ..mesh1d import RadialMesh
        mesh = RadialMesh(model_or_material, ngll=ngll, lmax=max(lmax, 1))
    return Material(mesh, model_or_material)


def solve_degree(model_or_material, l: int, *, forcing: str = "load",
                 mesh=None, ngll: int = 5, eps: float = 1e-8) -> DegreeSolution:
    """The degree-l solution of a model, or of a ready `Material`, for
    one forcing of `FORCINGS`: a unit surface density through both
    channels, its traction-like or attraction piece alone, or a unit
    external potential.  Given a model, the mesh is built with the
    `lmax=l` rule unless supplied."""
    if forcing not in FORCINGS:
        raise ValueError(f"forcing must be one of {FORCINGS}, got {forcing!r}")
    material = _material(model_or_material, mesh=mesh, ngll=ngll, lmax=l)
    system = DegreeSystem(material, l, eps=eps)
    if forcing == "tide":
        b = system.tide()
    else:
        part = {"load": "both", "load_force": "force",
                "load_potential": "potential"}[forcing]
        b = system.load(part=part)
    U, V, phi = system.expand(system.solve(b))
    return DegreeSolution(int(l), forcing, material.mesh, U, V, phi)


@dataclass(frozen=True)
class LoveNumbers:
    """Love numbers by degree, in the units of `scales`; see the module
    docstring for what each is.  `radius`, `surface_gravity` and `G` are
    the body's, in the same units; `omega` is the frequency of a frozen
    viscoelastic model, whose numbers are complex, and None for an
    elastic one."""

    degree: np.ndarray
    h_u: np.ndarray
    l_u: np.ndarray
    k_u: np.ndarray
    h_phi: np.ndarray
    l_phi: np.ndarray
    k_phi: np.ndarray
    h_t: np.ndarray
    l_t: np.ndarray
    k_t: np.ndarray
    radius: float
    surface_gravity: float
    G: float
    _: KW_ONLY
    scales: Scales = Scales.SI
    omega: float | None = None

    def __post_init__(self) -> None:
        n = len(self.degree)
        for name in ("degree",) + tuple(_DIMENSIONS):
            a = np.array(getattr(self, name))
            if a.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    @property
    def lmax(self) -> int:
        return int(self.degree[-1])

    @property
    def is_complex(self) -> bool:
        return self.h_u.dtype.kind == "c"

    # -- the load sums and the conventional forms -----------------------------

    @property
    def h(self) -> np.ndarray:
        """h = h^u + h^phi, the radial response to a unit surface density."""
        return self.h_u + self.h_phi

    @property
    def l_load(self) -> np.ndarray:
        """l = l^u + l^phi, the tangential response to a unit surface density."""
        return self.l_u + self.l_phi

    @property
    def k(self) -> np.ndarray:
        """k = k^u + k^phi, the potential response to a unit surface density."""
        return self.k_u + self.k_phi

    def conventional(self) -> dict:
        """The dimensionless load Love numbers h', l', k' by degree."""
        fac = (2.0 * self.degree + 1.0) / (4.0 * np.pi * self.G * self.radius)
        g = self.surface_gravity
        return {"h": self.h * g * fac, "l": self.l_load * g * fac,
                "k": -self.k * fac - 1.0}

    def tidal(self) -> dict:
        """The geodetic tidal Love numbers k^T, h^T, l^T by degree."""
        g = self.surface_gravity
        return {"k": self.k_t.copy(), "h": -g * self.h_t, "l": -g * self.l_t}

    def reciprocity_residual(self) -> np.ndarray:
        """|k^u - g h^phi| per degree, relative to the largest of |k^u|,
        |g h^phi| and |k^phi|; zero where all vanish (degree 1)."""
        ku, hphi = self.k_u, self.surface_gravity * self.h_phi
        scale = np.maximum(np.maximum(np.abs(ku), np.abs(hphi)), np.abs(self.k_phi))
        safe = np.where(scale > 0.0, scale, 1.0)
        return np.where(scale > 0.0, np.abs(ku - hphi) / safe, 0.0)

    # -- units and files ----------------------------------------------------

    def converted(self, scales: Scales) -> LoveNumbers:
        """The same numbers under other scales."""
        if scales == self.scales:
            return self
        changes = {}
        for name, dims in _DIMENSIONS.items():
            ratio = self.scales.factor(dims) / scales.factor(dims)
            changes[name] = getattr(self, name) * ratio
        one = Dimensions(length=1)
        length = self.scales.factor(one) / scales.factor(one)
        accel = Dimensions(length=1, time=-2)
        grav = Dimensions(mass=-1, length=3, time=-2)
        changes["radius"] = self.radius * length
        changes["surface_gravity"] = (
            self.surface_gravity * self.scales.factor(accel) / scales.factor(accel))
        changes["G"] = self.G * self.scales.factor(grav) / scales.factor(grav)
        if self.omega is not None:
            changes["omega"] = (self.omega * self.scales.factor(FREQUENCY)
                                / scales.factor(FREQUENCY))
        return replace(self, scales=scales, **changes)

    def in_si(self) -> LoveNumbers:
        return self.converted(Scales.SI)

    def write(self, path) -> None:
        """Write the pyslfp file: rows from degree 0 to lmax, columns
        l, h_u, k_u, h_phi, k_phi, h_t, k_t in SI.  Refused for complex
        numbers and for a table not starting at degree 0."""
        if self.is_complex:
            raise ValueError("the pyslfp file holds real Love numbers")
        if not np.array_equal(self.degree, np.arange(self.lmax + 1)):
            raise ValueError("the pyslfp file needs every degree from 0 to lmax")
        si = self.in_si()
        cols = np.column_stack([si.degree] + [getattr(si, n) for n in _COLUMNS])
        header = ("elastic Love numbers, SI, physical-potential sign convention\n"
                  "l  h_u  k_u  h_phi  k_phi  h_t  k_t\n"
                  "load columns per unit surface density, tidal columns per unit "
                  "external potential (r/a)^l, degree 1 in the centre-of-mass frame")
        np.savetxt(path, cols, fmt=["%6d"] + ["%+.15e"] * 6, header=header)

    def __repr__(self) -> str:
        kind = "complex " if self.is_complex else ""
        at = "" if self.omega is None else f", omega={self.omega:g}"
        return (f"LoveNumbers({kind}degrees {int(self.degree[0])}..{self.lmax}, "
                f"{self.scales!r}{at})")


def read_love_numbers(path) -> LoveNumbers:
    """A `LoveNumbers` in SI from a pyslfp file; the tangential numbers,
    which the file does not hold, are NaN, and the radius, gravity and
    G are those of the header when the file was written here, else
    NaN."""
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 7:
        raise ValueError(f"expected 7 columns, got {data.shape[1]}")
    n = data.shape[0]
    nan = np.full(n, np.nan)
    cols = {name: data[:, j + 1] for j, name in enumerate(_COLUMNS)}
    return LoveNumbers(data[:, 0].astype(int), l_u=nan, l_phi=nan, l_t=nan,
                       radius=np.nan, surface_gravity=np.nan, G=np.nan,
                       scales=Scales.SI, **cols)


def love_numbers(model_or_material, lmax: int, *, mesh=None, ngll: int = 5,
                 eps: float = 1e-8) -> LoveNumbers:
    """The Love numbers of a model, or of a ready `Material`, for every
    degree from 0 to `lmax`.

    Each degree is assembled once and solved for the three forcings.
    Given a model, the mesh is built with the `lmax` rule
    (`RadialMesh(model, ngll=ngll, lmax=lmax)`) unless supplied.  A
    model frozen at a frequency gives complex numbers.
    """
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    material = _material(model_or_material, mesh=mesh, ngll=ngll, lmax=lmax)
    dtype = complex if material.is_complex else float
    out = {name: np.zeros(lmax + 1, dtype=dtype) for name in _DIMENSIONS}
    for l in range(lmax + 1):
        system = DegreeSystem(material, l, eps=eps)
        B = np.column_stack([system.load(part="force"),
                             system.load(part="potential"), system.tide()])
        X = system.solve(B)
        for j, channel in enumerate(("u", "phi", "t")):
            for letter, component in (("h", "U"), ("l", "V"), ("k", "phi")):
                i = system.surface_dof(component)
                if i >= 0:
                    out[f"{letter}_{channel}"][l] = X[i, j]
    return LoveNumbers(np.arange(lmax + 1), radius=material.radius,
                       surface_gravity=material.surface_gravity, G=material.G,
                       scales=material.scales, omega=material.omega, **out)
