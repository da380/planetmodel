"""Elastic moduli: their conversions, the elastic tensor, and what a
layer's fields imply.

The moduli are the canonical elastic description, because velocities
have no transformation law (see character.py).  A transversely
isotropic medium with its symmetry axis along e_r is described by the
five moduli

    A = rho vph^2      C = rho vpv^2      L = rho vsv^2
    N = rho vsh^2      F = eta (A - 2 L)

which collapse to the isotropic case at vph = vpv, vsh = vsv, eta = 1,
where A = C = kappa + 4 mu / 3, L = N = mu and F = kappa - 2 mu / 3.
The Voigt average of a transversely isotropic medium is

    kappa = (4 A + C + 4 F - 4 N) / 9
    mu    = (A + C - 2 F + 5 N + 6 L) / 15

and reduces to (kappa, mu) exactly where the medium is isotropic.

The conversions are written on numbers and arrays with the arithmetic
operators alone, so the same code runs on rank-0 fields through the
field algebra: a product of polynomial layers is a polynomial layer,
and the moduli of a polynomial model are exact.  `velocities_from_moduli`
takes square roots and is numbers only.

`voigt_matrix` builds the Voigt form (the order and the absence of
engineering factors are stated in frames.py) in the spherical frame,
where (1, 2, 3) = (r, theta, phi) and the symmetry axis is index 1:

    [ C     F     F     .  .  . ]
    [ F     A     A-2N  .  .  . ]      rows and columns in the order
    [ F     A-2N  A     .  .  . ]      (rr, tt, pp, tp, rp, rt)
    [ .     .     .     N  .  . ]
    [ .     .     .     .  L  . ]
    [ .     .     .     .  .  L ]

`ElasticField` holds the independent moduli of its symmetry as rank-0
fields on one interval and builds the Voigt matrix when asked; its
Cartesian form is the Bond rotation the base class applies to every
Voigt rank-4 field.

The functions of a layer read whatever fields it holds, by name, and
refuse by name when the fields they need are absent: `is_fluid` asks
the shear-bearing fields whether they vanish, `moduli` builds the five
moduli, `elastic_moduli` the tensor, and `kappa_mu` the Voigt average.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .character import ELASTIC, Symmetry
from .fields import Field, FieldBase, RadialField
from .frames import voigt_to_tensor
from .layerfunction import same_interval

if TYPE_CHECKING:
    from .model import Layer

__all__ = ["moduli_from_velocities", "velocities_from_moduli",
           "kappa_mu_from_moduli", "voigt_matrix", "ElasticField",
           "is_fluid", "moduli", "elastic_moduli", "kappa_mu", "LayerLike",
           "Operand", "MODULI_NAMES", "SHEAR_NAMES", "ELASTIC_NAMES"]

#: What the functions of a layer accept: a model's `Layer`, or any mapping
#: of field name to field.
type LayerLike = Layer | Mapping[str, Field]

#: What the conversions take and return: a rank-0 field, or a number or
#: array.
type Operand = Field | ArrayLike

#: The independent moduli of each symmetry, in canonical order.
MODULI_NAMES = {Symmetry.ISOTROPIC: ("kappa", "mu"),
                Symmetry.VTI: ("A", "C", "F", "L", "N")}

#: The names whose vanishing makes a layer fluid.
SHEAR_NAMES = ("vs", "vsv", "vsh", "mu", "L", "N")

#: The five transversely isotropic velocities beside rho.
_TI_VELOCITIES = ("vpv", "vph", "vsv", "vsh", "eta")

#: Every name that describes the elastic medium beside rho: what
#: `moduli` reads from, and what an isotropic re-description replaces.
ELASTIC_NAMES = ("vp", "vs") + _TI_VELOCITIES + MODULI_NAMES[Symmetry.ISOTROPIC] \
    + MODULI_NAMES[Symmetry.VTI] + ("elastic_moduli",)


def _unknown_symmetry(symmetry: object) -> ValueError:
    names = [s.name for s in MODULI_NAMES]
    return ValueError(f"unknown symmetry {symmetry!r}: expected one of {names}")


def _operand(x: Operand) -> Field | np.ndarray:
    """x as it is when it is a field, else as a float64 array."""
    return x if isinstance(x, Field) else np.asarray(x, dtype=float)


def moduli_from_velocities(rho: Operand, vpv: Operand, vsv: Operand, *,
                           vph: Operand | None = None, vsh: Operand | None = None,
                           eta: Operand | None = None
                           ) -> dict[str, Field | np.ndarray]:
    """The moduli A, C, F, L, N from density and velocities.

    With vph, vsh and eta omitted the medium is isotropic: vph = vpv,
    vsh = vsv, eta = 1, giving A = C and F = A - 2 L.  Numbers and
    arrays broadcast; rank-0 fields on one interval combine through
    the field algebra.
    """
    rho, vpv, vsv = _operand(rho), _operand(vpv), _operand(vsv)
    vph = vpv if vph is None else _operand(vph)
    vsh = vsv if vsh is None else _operand(vsh)
    eta = 1.0 if eta is None else _operand(eta)
    A = rho * vph ** 2
    C = rho * vpv ** 2
    L = rho * vsv ** 2
    N = rho * vsh ** 2
    F = eta * (A - 2.0 * L)
    return {"A": A, "C": C, "F": F, "L": L, "N": N}


def velocities_from_moduli(rho: ArrayLike, A: ArrayLike, C: ArrayLike, F: ArrayLike,
                           L: ArrayLike, N: ArrayLike) -> dict[str, np.ndarray]:
    """The velocities vpv, vph, vsv, vsh and eta from density and moduli.

    The inverse of `moduli_from_velocities` on numbers and arrays.  A
    velocity is zero where rho is not positive or the modulus is
    negative, and eta is one where A = 2 L leaves it undefined.
    """
    rho = np.asarray(rho, dtype=float)
    A, C = np.asarray(A, dtype=float), np.asarray(C, dtype=float)
    F, L = np.asarray(F, dtype=float), np.asarray(L, dtype=float)
    N = np.asarray(N, dtype=float)
    safe = np.where(rho > 0.0, rho, 1.0)
    positive = rho > 0.0
    denom = A - 2.0 * L
    defined = denom != 0.0

    def speed(modulus: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(np.where(positive, modulus / safe, 0.0), 0.0))

    return {"vpv": speed(C), "vph": speed(A), "vsv": speed(L), "vsh": speed(N),
            "eta": np.where(defined, F / np.where(defined, denom, 1.0), 1.0)}


def kappa_mu_from_moduli(A: Operand, C: Operand, F: Operand, L: Operand, N: Operand
                         ) -> tuple[Field | np.ndarray, Field | np.ndarray]:
    """The Voigt average (kappa, mu) of the moduli, on numbers or fields."""
    A, C, F, L, N = (_operand(x) for x in (A, C, F, L, N))
    kappa = (4.0 * A + C + 4.0 * F - 4.0 * N) / 9.0
    mu = (A + C - 2.0 * F + 5.0 * N + 6.0 * L) / 15.0
    return kappa, mu


def _named(field: Field, name: str) -> Field:
    """The field under `name` where it can be renamed, else as it is."""
    return field.renamed(name) if hasattr(field, "renamed") else field


def _vti_from_isotropic(kappa: Field, mu: Field) -> dict[str, Field]:
    """A = C = kappa + 4 mu / 3, F = kappa - 2 mu / 3, L = N = mu, as fields
    named for what they are."""
    A = kappa + (4.0 / 3.0) * mu
    F = kappa - (2.0 / 3.0) * mu
    return {k: _named(f, k)
            for k, f in (("A", A), ("C", A), ("F", F), ("L", mu), ("N", mu))}


def voigt_matrix(symmetry: Symmetry, components: Mapping[str, ArrayLike]) -> np.ndarray:
    """The Voigt matrix of a second elasticity tensor in the spherical
    frame, shape broadcast + (6, 6), with the symmetry axis along e_r.

    `components` maps the independent moduli of `symmetry` to numbers
    or arrays that broadcast: kappa and mu for ISOTROPIC, A, C, F, L
    and N for VTI.  The dtype follows the inputs, so complex moduli
    give a complex matrix.
    """
    if symmetry is Symmetry.ISOTROPIC:
        kappa, mu = (np.asarray(components[k]) for k in MODULI_NAMES[symmetry])
        lam = kappa - 2.0 * mu / 3.0
        shape = np.broadcast(kappa, mu).shape
        out = np.zeros(shape + (6, 6), dtype=np.result_type(float, kappa, mu))
        for i in range(3):
            for j in range(3):
                out[..., i, j] = lam
            out[..., i, i] = lam + 2.0 * mu
            out[..., 3 + i, 3 + i] = mu
        return out
    if symmetry is Symmetry.VTI:
        A, C, F, L, N = (np.asarray(components[k]) for k in MODULI_NAMES[symmetry])
        shape = np.broadcast(A, C, F, L, N).shape
        out = np.zeros(shape + (6, 6), dtype=np.result_type(float, A, C, F, L, N))
        out[..., 0, 0] = C
        out[..., 1, 1] = out[..., 2, 2] = A
        out[..., 0, 1] = out[..., 1, 0] = F
        out[..., 0, 2] = out[..., 2, 0] = F
        out[..., 1, 2] = out[..., 2, 1] = A - 2.0 * N
        out[..., 3, 3] = N
        out[..., 4, 4] = out[..., 5, 5] = L
        return out
    raise _unknown_symmetry(symmetry)


class ElasticField(FieldBase):
    """A second elasticity tensor, character ELASTIC, stored by its moduli.

    `moduli` maps the independent moduli of `symmetry` to rank-0 fields
    on one interval: kappa and mu for ISOTROPIC, A, C, F, L and N for
    VTI.  The Voigt matrix in the spherical frame is built at each
    evaluation from the moduli there; `evaluate(voigt=False)` expands it
    to the full (3, 3, 3, 3) components.  The angles are always
    required: the Cartesian components depend on direction even where
    the moduli do not, because the frame carrying the symmetry axis does.
    """

    _character = ELASTIC

    def __init__(self, symmetry: Symmetry, moduli: Mapping[str, Field], *,
                 name: str | None = None) -> None:
        expected = MODULI_NAMES.get(symmetry)
        if expected is None:
            raise _unknown_symmetry(symmetry)
        missing = [k for k in expected if k not in moduli]
        if missing:
            raise ValueError(f"{symmetry} needs {list(expected)}; missing {missing}")
        extra = [k for k in moduli if k not in expected]
        if extra:
            raise ValueError(f"{symmetry} takes only {list(expected)}; got {extra}")
        fields = {k: moduli[k] for k in expected}
        for k, f in fields.items():
            if not isinstance(f, Field):
                raise TypeError(f"modulus {k!r} is not a Field: {f!r}")
            if f.character.rank != 0:
                raise ValueError(
                    f"modulus {k!r} has rank {f.character.rank}; a modulus is a "
                    "field of rank 0")
        first = fields[expected[0]]
        rtol = float(getattr(first, "rtol", 1e-9))
        for k, f in fields.items():
            if not same_interval(first.interval, f.interval, rtol=rtol):
                raise ValueError(
                    f"moduli on different intervals: {expected[0]!r} on "
                    f"{first.interval} and {k!r} on {f.interval}")
        self._symmetry = symmetry
        self._moduli = fields
        self._interval = tuple(float(x) for x in first.interval)
        self._name = name
        self._rtol = rtol

    @property
    def symmetry(self) -> Symmetry:
        return self._symmetry

    @property
    def moduli(self) -> dict[str, Field]:
        """The independent moduli, in canonical order."""
        return dict(self._moduli)

    @property
    def is_radial(self) -> bool:
        """Whether every modulus is a function of the radius alone."""
        return all(bool(getattr(f, "is_radial", False)) for f in self._moduli.values())

    def _values(self, r: np.ndarray, theta: np.ndarray | None,
                phi: np.ndarray | None) -> np.ndarray:
        vals = {k: f.evaluate(r, theta, phi) for k, f in self._moduli.items()}
        return voigt_matrix(self._symmetry, vals)

    def evaluate(self, r: ArrayLike, theta: ArrayLike | None, phi: ArrayLike | None,
                 *, frame: str = "spherical", voigt: bool = True) -> np.ndarray:
        """The tensor at (r, theta, phi) in `frame`: Voigt (..., 6, 6), or
        the full (..., 3, 3, 3, 3) components with `voigt=False`."""
        v = super().evaluate(r, theta, phi, frame=frame)
        return v if voigt else voigt_to_tensor(v, rank=4)

    def _with(self, moduli: Mapping[str, Field], *, name: str | None) -> "ElasticField":
        return ElasticField(self._symmetry, moduli, name=name)

    def on_interval(self, lo: float, hi: float) -> "ElasticField":
        """The same moduli re-stated on [lo, hi], each by its own rule."""
        return self._with({k: f.on_interval(lo, hi) for k, f in self._moduli.items()},
                          name=self._name)

    def rescaled(self, *, k: float, v: float) -> "ElasticField":
        """v C(r / k) on the interval scaled by k, through the moduli."""
        return self._with({n: f.rescaled(k=k, v=v) for n, f in self._moduli.items()},
                          name=self._name)

    def renamed(self, name: str | None) -> "ElasticField":
        return self._with(self._moduli, name=name)

    def as_symmetry(self, symmetry: Symmetry) -> "ElasticField":
        """The same medium described in `symmetry`.

        ISOTROPIC promotes to VTI through the field algebra, exactly on
        polynomial layers; a VTI medium is not isotropic and narrowing
        is refused.  The same symmetry returns the field itself.
        """
        if symmetry is self._symmetry:
            return self
        if self._symmetry is Symmetry.ISOTROPIC and symmetry is Symmetry.VTI:
            kappa, mu = self._moduli["kappa"], self._moduli["mu"]
            return ElasticField(Symmetry.VTI, _vti_from_isotropic(kappa, mu),
                                name=self._name)
        if symmetry in MODULI_NAMES:
            raise ValueError(
                f"cannot narrow {self._symmetry} to {symmetry}: that discards "
                "moduli rather than re-describing them")
        raise _unknown_symmetry(symmetry)

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return (f"ElasticField({nm}{self._symmetry.name.lower()} on "
                f"[{lo:g}, {hi:g}])")


# -- what a layer's fields imply -------------------------------------------

def _held(layer: LayerLike) -> tuple[str, ...]:
    """The names a layer holds: its `names`, else its keys."""
    names = getattr(layer, "names", None)
    return tuple(names if names is not None else layer.keys())


def _describe(layer: LayerLike) -> str:
    name = getattr(layer, "name", None)
    if name is not None:
        return f"layer {name!r}"
    index = getattr(layer, "index", None)
    if index is not None:
        return f"layer {index}"
    return f"a layer holding {list(_held(layer))}"


def _vanishes(field: Field) -> bool:
    """Whether a field is zero throughout its interval: by its polynomial
    coefficients where it has them, else at nine radii."""
    if isinstance(field, RadialField) and field.character.rank == 0:
        fn = field.function
        if hasattr(fn, "is_zero"):
            return bool(fn.is_zero())
    lo, hi = field.interval
    r = np.linspace(lo, hi, 9)
    radial_scalar = getattr(field, "is_radial", False) and field.character.rank == 0
    values = field(r) if radial_scalar else field(r, 1.0, 0.5)
    return not np.any(np.asarray(values))


def is_fluid(layer: LayerLike) -> bool:
    """Whether every shear-bearing field the layer holds, among
    `SHEAR_NAMES`, vanishes throughout it; a layer holding none of them
    is refused with KeyError."""
    held = [n for n in SHEAR_NAMES if n in layer]
    if not held:
        raise KeyError(
            f"{_describe(layer)} holds none of {list(SHEAR_NAMES)}, so its "
            "fluidity cannot be read")
    return all(_vanishes(layer[n]) for n in held)


def _independent_moduli(layer: LayerLike) -> tuple[Symmetry, dict[str, Field]]:
    """The symmetry a layer states and its independent moduli as fields.

    The five moduli when held; else kappa and mu; else rho with the five
    transversely isotropic velocities; else rho with vp and vs, which
    are isotropic; a layer holding none of these is refused with
    KeyError.
    """
    vti = MODULI_NAMES[Symmetry.VTI]
    iso = MODULI_NAMES[Symmetry.ISOTROPIC]
    if all(n in layer for n in vti):
        return Symmetry.VTI, {n: layer[n] for n in vti}
    if all(n in layer for n in iso):
        return Symmetry.ISOTROPIC, {n: layer[n] for n in iso}
    if "rho" in layer and all(n in layer for n in _TI_VELOCITIES):
        v = {n: layer[n] for n in _TI_VELOCITIES}
        m = moduli_from_velocities(layer["rho"], v["vpv"], v["vsv"], vph=v["vph"],
                                   vsh=v["vsh"], eta=v["eta"])
        return Symmetry.VTI, {k: _named(f, k) for k, f in m.items()}
    if all(n in layer for n in ("rho", "vp", "vs")):
        m = moduli_from_velocities(layer["rho"], layer["vp"], layer["vs"])
        kappa, mu = kappa_mu_from_moduli(**m)
        return Symmetry.ISOTROPIC, {"kappa": _named(kappa, "kappa"),
                                    "mu": _named(mu, "mu")}
    raise KeyError(
        f"{_describe(layer)} holds {list(_held(layer))}; the moduli need "
        f"{list(vti)}, or {list(iso)}, or rho with {list(_TI_VELOCITIES)}, "
        "or rho with vp and vs")


def moduli(layer: LayerLike) -> dict[str, Field]:
    """The moduli A, C, F, L, N of a layer as rank-0 fields, from
    whatever it holds (see `elastic_moduli`); exact on polynomial layers."""
    symmetry, fields = _independent_moduli(layer)
    if symmetry is Symmetry.ISOTROPIC:
        fields = _vti_from_isotropic(fields["kappa"], fields["mu"])
    return fields


def elastic_moduli(layer: LayerLike) -> ElasticField:
    """The elastic tensor of a layer: ISOTROPIC when it holds kappa and
    mu or rho with vp and vs, VTI when it holds the five moduli or rho
    with vpv, vph, vsv, vsh and eta; refused with KeyError otherwise."""
    symmetry, fields = _independent_moduli(layer)
    return ElasticField(symmetry, fields, name="elastic_moduli")


def kappa_mu(layer: LayerLike) -> tuple[Field, Field]:
    """The Voigt average (kappa, mu) of `moduli(layer)`, as fields."""
    kappa, mu = kappa_mu_from_moduli(**moduli(layer))
    return _named(kappa, "kappa"), _named(mu, "mu")
