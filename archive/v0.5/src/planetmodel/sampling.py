"""sampling.py -- a body evaluated on a consumer's nodes.

The one abstraction the export layer adds.  A consumer's
unknowns live on a product of a radial node set and an angular node set,
and planetmodel evaluates there, once, in a vectorised call per layer; the
result is a `Sample`, which the netCDF writer and the MFEM exporter both
consume and which is what a spectral code reads back.

**Layout.**  Every sampled array is `(node, colatitude, longitude,
components...)`, node outermost and longitude fastest, which is
GSHTrans's `LayeredSpinField` storage and gplspec's
`[element][node][spatial]`.  The radial nodes are the per-element GLL
nodes of a `RadialMesh`, flattened element by element, so the flat
`radius(node)` array repeats every element boundary: an interface is
two nodes at one radius, one on each side, and each carries its own
layer's one-sided value.  That is how a discontinuous field survives
sampling without a second array saying where the jumps are.  A field
that does not depend on direction -- one declaring `is_radial` -- is
sampled on `(node,)` alone and the shape is the mark; a consumer
broadcasts.  A frequency-dependent field sampled at a chosen `omega`
carries one more trailing axis of length 2, the real part and
the imaginary part: a sample is float64 throughout, and a complex field
is two real numbers per component rather than a second dtype.

**Components** are in the spherical frame (e_r, e_theta, e_phi) at the
sample point, with ranks 2 and 4 Voigt-reduced as everywhere in planetmodel,
and the frame is recorded per field.  The canonical (-, 0, +)
components the spectral codes use are three lines on their side and are
not written, so that no file carries a complex convention.  The
displacement of the mapping, `m(X) - X`, is sampled the same way: a
vector field on the reference body, in the spherical frame, which for a
`RadialStretch` is `(h, 0, 0)` -- a fact the tests check rather than a
fact the code assumes.

**Angular grids** are node arrays first.  `kind` and `lmax` are hints
for a consumer that wants to know what band a grid resolves; the arrays
are the truth, and a consumer with its own grid hands its nodes over as
a `custom` grid.  The Gauss-Legendre constructor follows GSHTrans: the
longitude quadrature is the trapezoid rule on `nphi` equally spaced
points, exact for `exp(i (m - m') phi)` only while `|m - m'| < nphi`, so
resolving orders `|m| <= lmax` needs `nphi >= 2 lmax + 1`, and the
smallest fast FFT length at or above that bound is used.  The weight
convention shared by every kind is

    integral over S^2 of f dOmega  ~=  sum_i w_i sum_j (2 pi / nphi) f_ij,

so `weights` are colatitude weights against sin(theta) dtheta -- the
Legendre weights on x = cos(theta) for a Gauss grid -- and the
longitude rule is the consumer's own trapezoid.
"""
from __future__ import annotations

from collections.abc import Mapping as _MappingABC
from dataclasses import KW_ONLY, dataclass, field as _dc_field

import numpy as np
from scipy.fft import next_fast_len

from .mesh1d.mesh import RadialMesh
from .model.character import Character
from .model.fields.base import Field
from .model.frames import spherical_frame
from .model.mapping import Mapping
from .model.skeleton import Skeleton
from .model.units import Dimensions, Scales

__all__ = ["AngularGrid", "Sample", "SampleMetadata", "sample_body"]

_KINDS = ("gauss_legendre", "equiangular", "custom")


def _readonly(a, *, name: str) -> np.ndarray:
    """A read-only float64 1-d copy of a node array, or a ValueError."""
    a = np.array(a, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-d array, got shape "
                         f"{a.shape}")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class AngularGrid:
    """A product angular node set: colatitudes times longitudes.

    `colatitudes` are strictly increasing inside the open interval
    (0, pi) and `longitudes` strictly increasing inside [0, 2 pi), both
    in radians.  `kind` names the family the nodes came from and `lmax`
    the band the grid resolves, when it is that kind of grid; `weights`,
    when present, are colatitude weights in the convention stated in
    the module docstring.  The constructors `gauss_legendre` and
    `equiangular` build the two shipped families; anything else is a
    `custom` grid, and nothing downstream treats one differently.
    """

    colatitudes: np.ndarray
    longitudes: np.ndarray
    _: KW_ONLY
    kind: str = "custom"
    lmax: int | None = None
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Coerce to read-only float64 and validate the ranges."""
        theta = _readonly(self.colatitudes, name="colatitudes")
        phi = _readonly(self.longitudes, name="longitudes")
        if not np.all(np.diff(theta) > 0.0):
            raise ValueError("colatitudes must be strictly increasing")
        if theta[0] <= 0.0 or theta[-1] >= np.pi:
            raise ValueError("colatitudes must lie strictly inside (0, pi): "
                             "the poles are not sample points")
        if not np.all(np.diff(phi) > 0.0):
            raise ValueError("longitudes must be strictly increasing")
        if phi[0] < 0.0 or phi[-1] >= 2.0 * np.pi:
            raise ValueError("longitudes must lie in [0, 2 pi)")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        lmax = self.lmax
        if lmax is not None:
            lmax = int(lmax)
            if lmax < 0:
                raise ValueError(f"lmax must be non-negative, got {lmax}")
        w = self.weights
        if w is not None:
            w = _readonly(w, name="weights")
            if w.shape != theta.shape:
                raise ValueError(f"weights must have one entry per colatitude, "
                                 f"{theta.size}, got {w.size}")
        object.__setattr__(self, "colatitudes", theta)
        object.__setattr__(self, "longitudes", phi)
        object.__setattr__(self, "lmax", lmax)
        object.__setattr__(self, "weights", w)

    @property
    def ntheta(self) -> int:
        """Number of colatitudes."""
        return self.colatitudes.size

    @property
    def nphi(self) -> int:
        """Number of longitudes."""
        return self.longitudes.size

    @classmethod
    def gauss_legendre(cls, lmax: int, *, nphi: int | None = None
                       ) -> "AngularGrid":
        """The grid GSHTrans's `GaussLegendreGrid(lmax)` produces.

        lmax + 1 Gauss-Legendre colatitudes in x = cos(theta), sorted so
        that theta increases, with the Legendre weights alongside; and
        `nphi` equally spaced longitudes from zero, defaulting to the
        smallest fast FFT length at or above 2 lmax + 1.  An explicit
        `nphi` below that bound is refused, because at 2 lmax the orders
        +lmax and -lmax are one discrete mode and cannot be told apart.
        """
        lmax = int(lmax)
        if lmax < 0:
            raise ValueError(f"lmax must be non-negative, got {lmax}")
        x, w = np.polynomial.legendre.leggauss(lmax + 1)
        theta = np.arccos(x)
        order = np.argsort(theta)
        need = 2 * lmax + 1
        if nphi is None:
            nphi = int(next_fast_len(need))
        elif int(nphi) < need:
            raise ValueError(
                f"nphi={nphi} cannot resolve orders |m| <= {lmax}: the "
                f"trapezoid rule separates them only for nphi >= 2 lmax + 1 "
                f"= {need}")
        nphi = int(nphi)
        phi = 2.0 * np.pi * np.arange(nphi) / nphi
        return cls(theta[order], phi, kind="gauss_legendre", lmax=lmax,
                   weights=w[order])

    @classmethod
    def equiangular(cls, ntheta: int, nphi: int) -> "AngularGrid":
        """Midpoint colatitudes and equally spaced longitudes.

        theta_i = pi (i + 1/2) / ntheta keeps both poles out, as the
        open interval demands; the weights (pi / ntheta) sin(theta_i)
        make the midpoint rule in theta consistent with the module's
        weight convention.  No band is claimed.
        """
        ntheta, nphi = int(ntheta), int(nphi)
        if ntheta < 1 or nphi < 1:
            raise ValueError("ntheta and nphi must be positive")
        theta = np.pi * (np.arange(ntheta) + 0.5) / ntheta
        phi = 2.0 * np.pi * np.arange(nphi) / nphi
        return cls(theta, phi, kind="equiangular",
                   weights=(np.pi / ntheta) * np.sin(theta))

    def __repr__(self) -> str:
        band = "" if self.lmax is None else f", lmax={self.lmax}"
        return f"AngularGrid({self.kind}, {self.ntheta} x {self.nphi}{band})"


@dataclass(frozen=True)
class SampleMetadata:
    """What each sampled array is: per-field character, dimensions,
    frame and domain, and the scales and skeleton it was taken on.

    `domains[name]` is the tuple of skeleton layer indices the field is
    defined on (a field belongs to one layer, and a
    body-wide view has a domain).  Nodes of elements in any other layer
    hold NaN, which is what the netCDF writer turns into `_FillValue`:
    a hole in a field is carried, never filled.

    `omegas[name]` names the frequency-dependent fields sampled at a
    chosen `omega` and the value chosen.  Those arrays carry a
    trailing axis of length 2 -- the real part and the imaginary part --
    since a sample is float64 throughout and a complex field is two real
    numbers per component, not a new dtype.  It is empty for a sample of
    static fields alone.
    """

    characters: dict[str, Character]
    dimensions: dict[str, Dimensions | None]
    frames: dict[str, str]
    domains: dict[str, tuple[int, ...]]
    scales: Scales
    skeleton: Skeleton
    _: KW_ONLY
    omegas: dict[str, float] = _dc_field(default_factory=dict)


def _trailing(character: Character) -> tuple[int, ...]:
    """The component shape a sampled array carries: Voigt where it exists."""
    shape = character.voigt_shape
    return character.component_shape if shape is None else shape


@dataclass(frozen=True)
class Sample:
    """A body's fields and mapping evaluated on radial times angular nodes.

    `fields[name]` has shape `(nnode, ntheta, nphi) + c` for a field that
    depends on direction and `(nnode,) + c` for one that does not, with
    `c` the character's Voigt shape where it has one and its component
    shape otherwise, and one more trailing axis of length 2 for a field
    of `metadata.omegas` -- one sampled at a chosen frequency, whose
    values are complex and are stored as (real, imaginary);
    `displacement`, when a mapping was sampled, has
    shape `(nnode, ntheta, nphi, 3)` in the spherical frame.  A field
    defined on part of the body -- `metadata.domains[name]` lists the
    layers -- is NaN on the nodes of every element outside it, and the
    writers turn those into the file's fill value.  `source`
    and `mapping` keep the objects the arrays were sampled from so that
    `planetmodel.testing.check_sample` can compare the two; a Sample read
    back from a file carries None there and is checked for shape and
    layout alone.
    """

    radial: RadialMesh
    angular: AngularGrid
    fields: dict[str, np.ndarray]
    displacement: np.ndarray | None
    metadata: SampleMetadata
    _: KW_ONLY
    source: dict[str, Field] | None = None
    mapping: Mapping | None = None

    @property
    def nnode(self) -> int:
        """Flat node count, nspec * ngll: element boundaries counted twice."""
        return self.radial.nspec * self.radial.ngll

    @property
    def radius(self) -> np.ndarray:
        """The flat `radius(node)` array the file format stores."""
        return self.radial.r.ravel()

    @property
    def element_start(self) -> np.ndarray:
        """First flat node of each element, plus the end: (nelement + 1,)."""
        return np.arange(self.radial.nspec + 1) * self.radial.ngll

    @property
    def element_layer(self) -> np.ndarray:
        """Skeleton layer index of each element."""
        return self.radial.layer

    def stored_shape(self, name: str) -> tuple[int, ...]:
        """The trailing axes a field's array carries beyond the nodes.

        The character's components, and the (real, imaginary) pair of a
        field sampled at a chosen frequency.
        """
        c = _trailing(self.metadata.characters[name])
        return c + ((2,) if name in self.metadata.omegas else ())

    def is_radial(self, name: str) -> bool:
        """Whether `name` was sampled on `(node,)` alone."""
        return self.fields[name].shape == (self.nnode,) + self.stored_shape(name)

    def __repr__(self) -> str:
        names = ", ".join(self.fields)
        disp = "" if self.displacement is None else ", displacement"
        return (f"Sample({self.nnode} nodes x {self.angular.ntheta} x "
                f"{self.angular.nphi}; fields: {names}{disp})")


# -- the sampler -----------------------------------------------------------

def _radial_mesh(body, grid: AngularGrid, radial, ngll: int, drmax):
    """The consumer's mesh, or one built from the grid's band."""
    if radial is not None:
        if not isinstance(radial, RadialMesh):
            raise TypeError(f"radial must be a RadialMesh, got "
                            f"{type(radial).__name__}")
        if radial.model.skeleton != body.skeleton:
            raise ValueError("radial is a mesh of a different skeleton; a "
                             "sample's nodes must be on the body it samples")
        if drmax is not None:
            raise ValueError(
                "give radial= or drmax=, not both: a mesh already fixes its "
                "element widths, so a second width would be two answers to "
                "one question")
        return radial
    if drmax is not None:
        return RadialMesh(body, ngll=ngll, drmax=drmax)
    if grid.lmax is None:
        raise ValueError(
            f"a {grid.kind!r} angular grid carries no band, so there is "
            "nothing to size the radial mesh by: pass radial= (a RadialMesh "
            "of this body) or drmax=")
    return RadialMesh(body, ngll=ngll, lmax=grid.lmax)


def _resolve_fields(body, fields, *, omega=None) -> dict[str, Field]:
    """Names, a dict of fields, or everything the body has.

    With no `omega` that is every *static* field: a frequency- or
    time-dependent one has no values until an argument is chosen.  With
    an `omega` it is the static fields and every frequency-dependent
    field the body holds, each sampled at that frequency; a
    time-dependent field is never included, since `omega` says nothing
    about it.
    """
    if fields is None:
        kinds = ("static",) if omega is None else ("static", "frequency")
        return {n: body[n] for n in body.field_names
                if getattr(body[n], "kind", "static") in kinds}
    if isinstance(fields, str):
        raise TypeError("fields must be a sequence of names or a dict of "
                        "name -> Field, not a single string")
    if isinstance(fields, _MappingABC):
        chosen = {}
        for name, fld in fields.items():
            if not isinstance(fld, Field):
                raise TypeError(f"fields[{name!r}] is not a Field: "
                                f"{type(fld).__name__}")
            if fld.skeleton != body.skeleton:
                raise ValueError(f"fields[{name!r}] lives on a different "
                                 "skeleton from the body being sampled")
            chosen[str(name)] = fld
        return chosen
    return {n: body[n] for n in fields}


def _expect(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    """A C-contiguous float64 array of exactly the promised shape.

    Complex values are refused rather than cast: dropping an imaginary
    part silently is the one thing a sample must never do.
    """
    if np.iscomplexobj(values):
        raise TypeError(
            f"field {name!r} evaluated to complex values, and a sample is "
            "float64 throughout: sample a frequency-dependent field with "
            "omega= so that both parts are stored, or freeze it with "
            "part=\"real\" or \"imag\"")
    values = np.asarray(values, dtype=float)
    if values.shape != shape:
        raise ValueError(
            f"field {name!r} evaluated to shape {values.shape}, but its "
            f"character promises {shape} on these nodes")
    return np.ascontiguousarray(values)


def _domain_of(fld: Field, nlayers: int) -> tuple[int, ...]:
    """The layers a field is defined on; every layer if it says nothing."""
    d = getattr(fld, "domain", None)
    if d is None:
        return tuple(range(nlayers))
    return tuple(int(i) for i in d)


def _parts(values) -> np.ndarray:
    """Complex values as (real, imaginary) on a trailing axis of length 2."""
    v = np.asarray(values)
    return np.stack((np.real(v), np.imag(v)), axis=-1)


def _sample_field(fld: Field, name: str, r, node_layer, theta, phi, domain, *,
                  omega=None):
    """One field on the nodes, layer by layer, both sides of every jump.

    Only the layers of `domain` are asked: a field defined on part of
    the body is evaluated where it is defined and left NaN elsewhere,
    since a value there would be an invention and a refusal would make
    a partial field unsamplable.

    With `omega` the field is frequency-dependent and is asked for its
    complex values there; they are stored as two real numbers per
    component on a trailing axis, real first, so that a sample stays
    float64 throughout and a consumer that knows nothing of complex
    numbers reads two arrays.
    """
    comp = _trailing(fld.character) + ((2,) if omega is not None else ())
    radial_only = bool(getattr(fld, "is_radial", False))
    nnode, ntheta, nphi = r.size, theta.size, phi.size

    def values(*points, layer: int):
        if omega is None:
            return fld.evaluate(*points, layer=layer, frame="spherical")
        return _parts(fld.evaluate(*points, omega=omega, layer=layer,
                                   frame="spherical", part="complex"))

    if radial_only:
        out = np.full((nnode,) + comp, np.nan)
        for L in np.unique(node_layer):
            if int(L) not in domain:
                continue
            m = node_layer == L
            v = values(r[m], layer=int(L))
            out[m] = _expect(v, (int(m.sum()),) + comp, name)
    else:
        out = np.full((nnode, ntheta, nphi) + comp, np.nan)
        for L in np.unique(node_layer):
            if int(L) not in domain:
                continue
            m = node_layer == L
            v = values(r[m][:, None, None], theta[None, :, None],
                       phi[None, None, :], layer=int(L))
            out[m] = _expect(v, (int(m.sum()), ntheta, nphi) + comp, name)
    return out


def _sample_displacement(mapping: Mapping, r, theta, phi) -> np.ndarray:
    """R^T (m(X) - X) at every node: spherical components of the shift."""
    R = spherical_frame(theta[:, None], phi[None, :])    # (nt, np, 3, 3)
    X = r[:, None, None, None] * R[None, ..., :, 0]      # X = r e_r
    disp = getattr(mapping, "displacement", None)
    u = disp(X) if disp is not None else np.asarray(mapping(X), dtype=float) - X
    u = np.asarray(u, dtype=float)
    if u.shape != X.shape:
        raise ValueError(f"mapping.displacement returned shape {u.shape} for "
                         f"points of shape {X.shape}")
    # R[..., j, i] is Cartesian component j of frame vector i, so the
    # spherical components are u . e_i = sum_j R[j, i] u_j.
    return np.ascontiguousarray(np.einsum("tpji,ntpj->ntpi", R, u))


def _check_omega(omega) -> float:
    """A real scalar frequency, in the body's own units."""
    w = np.asarray(omega)
    if w.ndim != 0:
        raise ValueError(f"omega must be a scalar, got shape {w.shape}: a "
                         "sample is taken at one frequency (loop over values)")
    if np.iscomplexobj(w) and np.imag(w) != 0:
        raise ValueError(
            f"omega must be real to sample at, got {omega!r}: the file format "
            "records a sample's omega as one number, "
            "so a field wanted off the real axis is frozen with "
            "at_frequency(field, omega, part='real') or part='imag' and "
            "passed by name")
    return float(np.real(w))


def sample_body(body, grid: AngularGrid, *, fields=None, mapping=None,
                radial=None, ngll: int = 5, drmax=None, omega=None) -> Sample:
    """Evaluate a body's fields, and optionally a mapping, on a grid.

    The entry point behind `ReferenceBody.sample`.  `grid`
    is the angular node set; the radial one is `radial` when the
    consumer has a `RadialMesh` of this body, and otherwise
    `RadialMesh(body, ngll, lmax=grid.lmax)` -- so a Gauss-Legendre band
    sizes both grids at once -- or `RadialMesh(body, ngll, drmax=drmax)`.
    A custom grid without a band needs one of the two said explicitly.

    `fields` is None for every field the body carries, a sequence of
    names, or a dict of `name -> Field` on the body's skeleton, which is
    how a field not attached to the body -- a pushed-forward one, say
    -- is sampled alongside the rest.  Each is evaluated once per layer
    of its own domain, on the nodes of the elements that layer owns, so
    the two one-sided values at an interface come from the two layers
    that own them, and a field the body holds on some layers only is
    NaN on the nodes of the rest rather than refusing or being filled.

    With `mapping=None` no displacement is sampled and
    `Sample.displacement` is None; an explicit `IdentityMapping` samples
    to zeros.

    `omega` is the frequency at which every frequency-dependent field
    named is sampled, in the body's own units and real: its
    complex values are stored as (real, imaginary) on a trailing axis of
    length 2 and the field is named in `metadata.omegas`.  With
    `fields=None` an `omega` adds every frequency-dependent field the
    body holds to the static ones; naming such a field without an
    `omega` is refused, since there is nothing to evaluate it at.
    """
    if not isinstance(grid, AngularGrid):
        raise TypeError(f"grid must be an AngularGrid, got "
                        f"{type(grid).__name__}")
    if omega is not None:
        omega = _check_omega(omega)
    mesh = _radial_mesh(body, grid, radial, ngll, drmax)
    chosen = _resolve_fields(body, fields, omega=omega)
    r = mesh.r.ravel()
    node_layer = np.repeat(mesh.layer, mesh.ngll)
    theta, phi = grid.colatitudes, grid.longitudes

    nlayers = body.skeleton.nlayers
    arrays, chars, dims, frames, doms, omegas = {}, {}, {}, {}, {}, {}
    for name, fld in chosen.items():
        kind = getattr(fld, "kind", "static")
        if kind == "time":
            raise ValueError(
                f"field {name!r} depends on time, and a sample is taken at "
                "one moment: freeze it with at_time(field, t) and pass the "
                "result by name")
        if kind == "frequency":
            if omega is None:
                raise ValueError(
                    f"field {name!r} depends on frequency and has no values "
                    "until one is chosen: pass omega= to sample it there")
            omegas[name] = omega
        elif kind != "static":
            raise ValueError(f"field {name!r} is of unknown kind {kind!r}")
        doms[name] = _domain_of(fld, nlayers)
        arrays[name] = _sample_field(fld, name, r, node_layer, theta, phi,
                                     doms[name],
                                     omega=omega if kind == "frequency" else None)
        chars[name] = fld.character
        dims[name] = getattr(fld, "dimensions", None)
        frames[name] = "spherical"
    disp = None if mapping is None else _sample_displacement(mapping, r,
                                                             theta, phi)
    meta = SampleMetadata(characters=chars, dimensions=dims, frames=frames,
                          domains=doms, scales=body.scales,
                          skeleton=body.skeleton, omegas=omegas)
    return Sample(radial=mesh, angular=grid, fields=arrays, displacement=disp,
                  metadata=meta, source=chosen, mapping=mapping)
