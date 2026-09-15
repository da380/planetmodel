"""rheology.py -- laws: frequency-dependent moduli from a layer's fields.

A law is a function that builds a frequency-dependent field from a
layer's static fields and a few constants, by composition, and records
how it did so.  The layer does not know what a law is: the field a law
returns is attached like any other, and the law survives only as
*provenance* on that field, a `LawRecord` the writers copy so that a
consumer that cannot call the function can rebuild the field from the
static fields in its own code.  Each law is registered under the kind
"rheology" so a file can name it, and carries `from_record`, the
inverse of that provenance, so `rebuild` needs no knowledge of any
law's signature.

    constant_q(moduli, qkappa, qmu, *, reference_period)     B.9.1
    constant_q_scalar(modulus, q, *, reference_period)       one modulus
    maxwell(moduli, viscosity)                               B.9.2
    prony(moduli, relaxation_times, relaxation_strengths)    B.9.3

Frequency enters as `omega` with the time convention of `materials.py`:
`exp(+i omega t)`, the Laplace variable `s = i omega`, and a lossy
modulus has `Im M > 0` for `omega > 0`.

**B.9.1, constant Q** is the absorption band of Liu, Anderson & Kanamori
(1976) and Kanamori & Anderson (1977), PREM's rheology, defined on the
real frequency axis (`omega_domain = "real"`).  With the reference
angular frequency `omega_0 = 2 pi / T_0` and the dispersion factor

    f(omega) = (2 / pi) ln(omega / omega_0) + i,

a modulus M with quality factor Q holding at omega_0 becomes

    M(omega) = M_0 + f(omega) M_0 / Q.

For transversely isotropic moduli the convention `"voigt_average"`
applies the band to the isotropic-equivalent bulk and shear moduli of
Dahlen & Tromp (1998, §8.9), the Voigt averages

    kappa = (C + 4 (A - N + F)) / 9,
    mu    = (C + A + 6 L + 5 N - 2 F) / 15,

and leaves the anisotropic residual undispersed: each modulus gains
f(omega) times its attenuation-equivalent part,

    A_att = C_att = kappa / Q_kappa + (4/3) mu / Q_mu,
    F_att         = kappa / Q_kappa - (2/3) mu / Q_mu,
    L_att = N_att = mu / Q_mu,

with a **zero Q contributing nothing** (PREM's fluid core has mu = 0
and Q_mu = 0; a deck's Q = 0 elsewhere reads as "no attenuation").  On
an isotropic medium the three lines reduce to the one-modulus formula
on kappa and mu, which is the oracle.  Another convention is a new
registered name.

**B.9.2, Maxwell** and **B.9.3, the Prony series (generalised Maxwell)**
are entire in `s` (`omega_domain = "complex"`).  Maxwell relaxes the
shear modulus alone,

    mu(s) = mu_0 s tau / (1 + s tau),   tau = eta / mu_0,   kappa(s) = kappa_0,

and the Prony series, with relaxation times tau_k, strengths M_k and
the long-time modulus M_inf, is

    M(s) = M_inf + sum_k M_k s tau_k / (1 + s tau_k),

applied to mu with kappa unrelaxed.  For transversely isotropic moduli
everything but the isotropic-equivalent bulk part relaxes, with one
relaxation time, so that a stress left alone loses its deviatoric part
entirely while the bulk modulus never moves: with K and mu the Voigt
averages above, `tau = eta / mu` and `g(s) = s tau / (1 + s tau)`,

    C(s) = K I(x)I + g(s) (C - K I(x)I),   i.e.
    A(s) = K + g (A - K),   C(s) = K + g (C - K),   F(s) = K + g (F - K),
    L(s) = g L,   N(s) = g N.

At s -> 0 the tensor is K I(x)I, pure pressure for any strain; at
s -> inf it is the full tensor; on an isotropic medium the law is
exactly the isotropic one.  For the Prony series `g` is the series
relative to its unrelaxed sum, `g(s) = M(s) / (M_inf + sum_k M_k)`, so
one term with no long-time modulus is Maxwell.  The relaxation times
and strengths are sequences of scalar fields, one per term.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field as _dc_field, replace as _replace

import numpy as np

from ..registry import lookup, register, registered
from .character import ELASTIC, SCALAR
from .fields.dependent import LiftedField, kind_of
from .fields.frequency import (ComposedFrequencyField, LiftedFrequencyField,
                               lifted_to_frequency)
from .frames import spherical_frame
from .materials import (MODULI_NAMES, ElasticField, Symmetry, bond_matrix,
                        check_frame, kappa_mu_from_moduli, voigt_matrix)
from .units import Dimensions

__all__ = ["LawRecord", "LawField", "constant_q", "constant_q_scalar", "maxwell",
           "prony", "CONVENTIONS", "STATIC", "law_record_of", "rebuild",
           "constant_dimensions_of"]

#: The transversely isotropic conventions constant_q knows.
CONVENTIONS = ("voigt_average",)

#: The `LawRecord.law` of a *lift*: a static field standing at every
#: frequency.  A lift is not a law and nothing is registered under this
#: name; it is the spelling a file uses so that a consumer rebuilding a
#: layer's rheology knows to lift rather than to call something.
STATIC = "static"


@dataclass(frozen=True)
class LawRecord:
    """How a frequency-dependent field was built: provenance for the file.

    `law` is the registered name; `parameters` the names of the fields
    the law read, in the order it took them; `constants` its numbers, in
    the body's units at the time (a reference period in seconds for an
    SI body); `convention` the named variant where the law has one.
    Nothing in the model reads this: it is copied out by the writers,
    which may restate `parameters` under the names a layer files the
    fields by (`with_parameters`).  The dimensions of the constants are
    the law's to say (`constant_dimensions_of`).
    """

    law: str
    _: KW_ONLY
    parameters: tuple[str, ...] = ()
    constants: Mapping[str, float] = _dc_field(default_factory=dict)
    convention: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "constants",
                           {k: float(v) for k, v in dict(self.constants).items()})

    def with_parameters(self, names) -> "LawRecord":
        """The same record naming the parameter fields differently."""
        names = tuple(names)
        if len(names) != len(self.parameters):
            raise ValueError(
                f"the law {self.law!r} read {len(self.parameters)} fields, "
                f"and {len(names)} names were given")
        return _replace(self, parameters=names)

    def with_constants(self, constants: Mapping[str, float]) -> "LawRecord":
        """The same record with its constants restated (other units)."""
        return _replace(self, constants=dict(constants))


def constant_dimensions_of(record: LawRecord) -> dict[str, Dimensions]:
    """The physical dimensions of a record's constants, by name.

    A law declares them as `law.constant_dimensions`; a lift has none.
    A registered law that declares nothing is taken to have
    dimensionless constants.
    """
    if record.law == STATIC:
        return {}
    law = _registered_law(record.law)
    declared = getattr(law, "constant_dimensions", {})
    return {k: declared.get(k, Dimensions.DIMENSIONLESS)
            for k in record.constants}


# ---------------------------------------------------------------------------
# the field a law returns
# ---------------------------------------------------------------------------

class LawField(ComposedFrequencyField):
    """A law's field: composed over its operand fields, remade from constants.

    A law closes over constants with units -- a reference period, in
    seconds for an SI body -- so its composed function cannot simply be
    kept when the body changes scales.  The field therefore keeps the
    law's *maker*, `make_fn(constants) -> fn`, and the constants with
    their `Dimensions`, and `rescaled` converts the constants and the
    operands and makes the function again.

    With `symmetry` given the field is an ELASTIC tensor: the first
    `len(components)` operands are the component fields of an
    `ElasticField` (scalars, which know no frame), `fn(omega, comps,
    *params)` maps their values, a dict by name, to complex moduli by
    name, and the Voigt matrix is built here in the spherical frame the
    layout assumes and rotated to Cartesian on request by the Bond
    matrix, as `ElasticField.evaluate` does.  Without `symmetry`, `fn(omega,
    *values)` returns the values of a field of `character`.

    The operands are static fields: a law's parameters.  A law of a
    law's output is refused, since the loss of the inner law would be
    silently discarded.
    """

    def __init__(self, make_fn: Callable, sources, *, constants: Mapping,
                 constant_dimensions: Mapping[str, Dimensions], law: LawRecord,
                 symmetry: Symmetry | None = None, components=(),
                 character=None, dimensions=None, name: str | None = None,
                 omega_domain: str = "real") -> None:
        sources = tuple(_static_operand(f) for f in sources)
        self.symmetry = symmetry
        self._component_names = tuple(components)
        self._make_fn = make_fn
        self._constants = {k: float(v) for k, v in dict(constants).items()}
        self._constant_dimensions = dict(constant_dimensions)
        if symmetry is not None:
            character, dimensions = ELASTIC, Dimensions.MODULUS
        super().__init__(make_fn(self._constants), sources, character=character,
                         dimensions=dimensions, name=name,
                         omega_domain=omega_domain, law=law)

    @property
    def constants(self) -> dict:
        """The law's constants, in the body's units."""
        return dict(self._constants)

    def _remade(self, sources, *, constants, law, name):
        return LawField(self._make_fn, sources, constants=constants,
                        constant_dimensions=self._constant_dimensions, law=law,
                        symmetry=self.symmetry, components=self._component_names,
                        character=self.character, dimensions=self.dimensions,
                        name=name, omega_domain=self._omega_domain)

    def rebuilt_from(self, operands, *, name=None):
        return self._remade(operands, constants=self._constants, law=self.law,
                            name=name)

    def matches(self, other) -> bool:
        return (type(other) is LawField
                and other._make_fn is self._make_fn
                and other.symmetry is self.symmetry
                and other._component_names == self._component_names
                and other._constants == self._constants
                and len(other.sources) == len(self.sources)
                and other.law == self.law)

    def rescaled(self, convert, old, new):
        """This law on operands converted by `convert`, constants rescaled."""
        constants = {
            k: v * old.factor(self._constant_dimensions.get(
                k, Dimensions.DIMENSIONLESS)) / new.factor(
                    self._constant_dimensions.get(k, Dimensions.DIMENSIONLESS))
            for k, v in self._constants.items()}
        law = None if self.law is None else self.law.with_constants(constants)
        return self._remade([convert(f) for f in self.sources],
                            constants=constants, law=law, name=self.name)

    def evaluate_with(self, r, theta, phi, arg, *, layer, side, frame):
        check_frame(frame)
        vals = [np.asarray(f.evaluate_with(r, theta, phi, arg, layer=layer,
                                           side=side, frame="spherical")).real
                for f in self.sources]
        if self.symmetry is None:
            return np.asarray(self._fn(arg, *vals), dtype=complex)
        n = len(self._component_names)
        comps = dict(zip(self._component_names, vals[:n]))
        v = voigt_matrix(self.symmetry, self._fn(arg, comps, *vals[n:]))
        if frame == "cartesian" and self.symmetry is not Symmetry.ISOTROPIC:
            if theta is None or phi is None:
                raise ValueError(
                    f"frame='cartesian' needs theta and phi: the Cartesian "
                    f"components of a {self.symmetry.name.lower()} tensor "
                    "depend on direction, because the frame carrying its "
                    "symmetry axis does")
            R = spherical_frame(theta, phi)
            R = np.broadcast_to(R, v.shape[:-2] + (3, 3))
            M = bond_matrix(R)
            v = M @ v @ np.swapaxes(M, -1, -2)
        return v

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        what = (self.symmetry.name.lower() if self.symmetry is not None
                else "scalar")
        return f"LawField({self.law.law if self.law else '?'}, {what}{nm})"


def _static_operand(field):
    """A law's operand is static; its lift (which restriction hands back) is
    unwrapped, and any other dependent field is refused by name."""
    if isinstance(field, LiftedField):
        return field.source
    if kind_of(field) != "static":
        raise TypeError(
            f"a law takes static fields; {getattr(field, 'name', field)!r} is "
            f"{kind_of(field)}-dependent, and a law of a law's output would "
            "discard the inner law's loss")
    return field


def _elastic_law(make_fn, moduli: ElasticField, params, *, constants=None,
                 constant_dimensions=None, law: LawRecord, name: str,
                 omega_domain: str) -> LawField:
    """A LawField over an ElasticField's components and the parameter fields."""
    components = dict(moduli.components)
    return LawField(make_fn, list(components.values()) + list(params),
                    constants=constants or {},
                    constant_dimensions=constant_dimensions or {}, law=law,
                    symmetry=moduli.symmetry, components=tuple(components),
                    name=name, omega_domain=omega_domain)


def _iso_or_vti(moduli, entry: str) -> Symmetry:
    if not isinstance(moduli, ElasticField):
        raise TypeError(
            f"{entry} takes an ElasticField of moduli, got "
            f"{type(moduli).__name__}")
    if moduli.symmetry not in (Symmetry.ISOTROPIC, Symmetry.VTI):
        raise NotImplementedError(
            f"{entry} on {moduli.symmetry} moduli: the law is stated for "
            "ISOTROPIC and VTI symmetry only")
    return moduli.symmetry


def _name_of(field, default: str) -> str:
    return getattr(field, "name", None) or default


# ---------------------------------------------------------------------------
# B.9.1 constant Q
# ---------------------------------------------------------------------------

def _dispersion_factor(omega, omega0: float):
    """f(omega) = (2/pi) ln(omega/omega_0) + i, on the positive real axis."""
    if not omega > 0.0:
        raise ValueError(f"constant Q needs omega > 0 on the real axis, got "
                         f"{omega!r}")
    return (2.0 / np.pi) * np.log(omega / omega0) + 1j


def _ratio(M, Q):
    """M / Q where Q != 0, and 0 where Q == 0: a zero Q attenuates nothing."""
    M = np.asarray(M, dtype=float)
    Q = np.asarray(Q, dtype=float)
    live = Q != 0.0
    return np.where(live, M / np.where(live, Q, 1.0), 0.0)


def _check_period(reference_period) -> float:
    period = float(reference_period)
    if not period > 0.0:
        raise ValueError(f"reference_period must be positive, got {period}")
    return period


def _constant_q_moduli(omega, omega0, symmetry, values: dict, qk, qm) -> dict:
    """The complex moduli at omega, B.9.1 under the voigt_average convention."""
    f = _dispersion_factor(omega, omega0)
    if symmetry is Symmetry.ISOTROPIC:
        kappa, mu = values["kappa"], values["mu"]
        return {"kappa": kappa + f * _ratio(kappa, qk),
                "mu": mu + f * _ratio(mu, qm)}
    A, C, F, L, N = (values[k] for k in MODULI_NAMES)
    kappa, mu = kappa_mu_from_moduli(A, C, F, L, N)
    rk, rm = _ratio(kappa, qk), _ratio(mu, qm)
    return {"A": A + f * (rk + (4.0 / 3.0) * rm),
            "C": C + f * (rk + (4.0 / 3.0) * rm),
            "F": F + f * (rk - (2.0 / 3.0) * rm),
            "L": L + f * rm,
            "N": N + f * rm}


def _constant_q_maker(symmetry):
    def make_fn(constants):
        omega0 = 2.0 * np.pi / constants["reference_period"]

        def fn(omega, comps, qk, qm):
            return _constant_q_moduli(omega, omega0, symmetry, comps, qk, qm)
        return fn
    return make_fn


@register("rheology", "constant_q")
def constant_q(moduli: ElasticField, qkappa, qmu, *, reference_period,
               convention: str = "voigt_average", name: str | None = None):
    """B.9.1: the moduli under constant Q, a frequency-dependent ELASTIC field.

    `moduli` is the static `ElasticField` holding at `reference_period`
    (seconds in an SI body, the body's time unit otherwise; PREM's
    1 s); `qkappa` and `qmu` the quality-factor fields on the same
    skeleton.  ISOTROPIC moduli take the band on kappa and mu; VTI
    moduli take it under the `"voigt_average"` convention of the module
    docstring.  The result composes the component fields and the two Q
    fields, lives on the real frequency axis, and carries a `LawRecord`
    as `.law`.
    """
    symmetry = _iso_or_vti(moduli, "constant_q")
    if convention not in CONVENTIONS:
        raise ValueError(
            f"unknown constant-Q convention {convention!r}; registered: "
            f"{CONVENTIONS}")
    period = _check_period(reference_period)
    return _elastic_law(
        _constant_q_maker(symmetry), moduli, [qkappa, qmu],
        constants={"reference_period": period},
        constant_dimensions=constant_q.constant_dimensions,
        name=name if name is not None else "constant_q", omega_domain="real",
        law=LawRecord("constant_q",
                      parameters=(_name_of(moduli, "elastic_moduli"),
                                  _name_of(qkappa, "qkappa"),
                                  _name_of(qmu, "qmu")),
                      constants={"reference_period": period},
                      convention=convention))


constant_q.constant_dimensions = {"reference_period": Dimensions.TIME}


def _constant_q_from_record(record: LawRecord, fields: Mapping):
    moduli, qkappa, qmu = _operands(record, fields, 3)
    kw = {} if record.convention is None else {"convention": record.convention}
    return constant_q(moduli, qkappa, qmu,
                      reference_period=record.constants["reference_period"], **kw)


constant_q.from_record = _constant_q_from_record


@register("rheology", "constant_q_scalar")
def constant_q_scalar(modulus, q, *, reference_period, name: str | None = None):
    """B.9.1 for one scalar modulus with one Q: a frequency-dependent field.

    The building block of the law, exposed for a body that carries a
    single modulus -- a shear modulus and its Q for a scalar problem.
    Same band, same domain, a zero Q attenuating nothing; provenance
    `constant_q_scalar`.
    """
    period = _check_period(reference_period)

    def make_fn(constants):
        omega0 = 2.0 * np.pi / constants["reference_period"]

        def fn(omega, M0, Q):
            return M0 + _dispersion_factor(omega, omega0) * _ratio(M0, Q)
        return fn

    return LawField(
        make_fn, [modulus, q], constants={"reference_period": period},
        constant_dimensions=constant_q_scalar.constant_dimensions,
        character=SCALAR, dimensions=getattr(modulus, "dimensions", None),
        omega_domain="real",
        name=name if name is not None else f"constant_q({_name_of(modulus, '?')})",
        law=LawRecord("constant_q_scalar",
                      parameters=(_name_of(modulus, "modulus"), _name_of(q, "q")),
                      constants={"reference_period": period}))


constant_q_scalar.constant_dimensions = {"reference_period": Dimensions.TIME}


def _constant_q_scalar_from_record(record: LawRecord, fields: Mapping):
    modulus, q = _operands(record, fields, 2)
    return constant_q_scalar(modulus, q,
                             reference_period=record.constants["reference_period"])


constant_q_scalar.from_record = _constant_q_scalar_from_record


# ---------------------------------------------------------------------------
# B.9.2 Maxwell, B.9.3 Prony
# ---------------------------------------------------------------------------

def _shear_factor(s, tau):
    """g(s) = s tau / (1 + s tau); zero tau gives zero (nothing to relax)."""
    tau = np.asarray(tau, dtype=float)
    return np.where(tau != 0.0, s * tau / (1.0 + s * tau), 0.0 + 0.0j)


def _safe_ratio(num, den):
    """num / den where den != 0, and 0 where den == 0."""
    den = np.asarray(den, dtype=float)
    live = den != 0.0
    return np.where(live, np.asarray(num, dtype=float) / np.where(live, den, 1.0),
                    0.0)


def _maxwell_modulus(s, mu0, eta):
    """mu_0 s tau / (1 + s tau), tau = eta / mu_0; zero mu_0 stays zero."""
    return np.asarray(mu0, dtype=float) * _shear_factor(s, _safe_ratio(eta, mu0))


def _vti_relaxed(comps: dict, g) -> dict:
    """K I(x)I + g (C - K I(x)I): everything but the bulk part relaxes."""
    A, C, F, L, N = (comps[k] for k in MODULI_NAMES)
    K, _ = kappa_mu_from_moduli(A, C, F, L, N)
    return {"A": K + g * (A - K), "C": K + g * (C - K), "F": K + g * (F - K),
            "L": g * L, "N": g * N}


def _relaxed(symmetry, comps: dict, g_iso, g_vti) -> dict:
    """The moduli scaled by a relaxation factor, by symmetry.

    `g_iso(mu)` gives the relaxed shear modulus from the unrelaxed one;
    `g_vti(mu)` the factor `g` for the Voigt-average mu.
    """
    if symmetry is Symmetry.ISOTROPIC:
        return {"kappa": np.asarray(comps["kappa"], dtype=complex),
                "mu": g_iso(comps["mu"])}
    _, mu = kappa_mu_from_moduli(*(comps[k] for k in MODULI_NAMES))
    return _vti_relaxed(comps, g_vti(mu))


@register("rheology", "maxwell")
def maxwell(moduli: ElasticField, viscosity, *, name: str | None = None):
    """B.9.2: Maxwell shear relaxation, a frequency-dependent ELASTIC field.

    `moduli` are the unrelaxed moduli and `viscosity` the field eta on
    the same skeleton (dimensions of pressure times time).  With
    `s = i omega`: ISOTROPIC, `mu(s) = mu_0 s tau / (1 + s tau)` for
    `tau = eta / mu_0` and `kappa` unrelaxed; VTI, everything but the
    isotropic-equivalent bulk part scaled by `s tau / (1 + s tau)` with
    the one `tau = eta / mu` of the isotropic-equivalent mu, so the
    deviatoric stress relaxes completely (module docstring).
    Entire in omega, so a Laplace-domain code evaluates off the real
    axis.
    """
    symmetry = _iso_or_vti(moduli, "maxwell")

    def make_fn(constants):
        def fn(omega, comps, eta):
            s = 1j * omega
            return _relaxed(symmetry, comps,
                            lambda mu: _maxwell_modulus(s, mu, eta),
                            lambda mu: _shear_factor(s, _safe_ratio(eta, mu)))
        return fn

    return _elastic_law(
        make_fn, moduli, [viscosity],
        name=name if name is not None else "maxwell", omega_domain="complex",
        law=LawRecord("maxwell",
                      parameters=(_name_of(moduli, "elastic_moduli"),
                                  _name_of(viscosity, "viscosity"))))


maxwell.constant_dimensions = {}


def _maxwell_from_record(record: LawRecord, fields: Mapping):
    moduli, viscosity = _operands(record, fields, 2)
    return maxwell(moduli, viscosity)


maxwell.from_record = _maxwell_from_record


@register("rheology", "prony")
def prony(moduli: ElasticField, relaxation_times, relaxation_strengths, *,
          long_time_modulus=None, name: str | None = None):
    """B.9.3: a Prony series (generalised Maxwell) on the shear modulus.

    `relaxation_times` and `relaxation_strengths` are sequences of
    scalar fields of equal length, one per term: `tau_k` (time) and
    `M_k` (modulus).  `long_time_modulus` is the field `M_inf`, or None
    for zero.  Then `M(s) = M_inf + sum_k M_k s tau_k / (1 + s tau_k)`
    with `s = i omega`.  ISOTROPIC: `mu(s) = M(s)`, the series defining
    mu (the unrelaxed shear modulus of `moduli` is not read), and
    `kappa` is unrelaxed; one term with `M_inf = 0` and `M_1 = mu_0` is
    Maxwell with `tau_1 = eta / mu_0`.  VTI: everything but the
    isotropic-equivalent bulk part is scaled by the series relative to
    its unrelaxed sum, `M(s) / (M_inf + sum_k M_k)` (module docstring).
    The record's constant `terms` is the number of terms, which is how
    `from_record` knows where the times end and the strengths begin.
    """
    symmetry = _iso_or_vti(moduli, "prony")
    taus = tuple(relaxation_times)
    Ms = tuple(relaxation_strengths)
    if not taus or len(taus) != len(Ms):
        raise ValueError(
            f"prony needs equally many relaxation times and strengths, got "
            f"{len(taus)} and {len(Ms)}")
    n = len(taus)
    with_inf = long_time_modulus is not None
    params = list(taus) + list(Ms) + ([long_time_modulus] if with_inf else [])

    def make_fn(constants):
        n = int(constants["terms"])

        def fn(omega, comps, *vals):
            s = 1j * omega
            tau, M = vals[:n], vals[n:2 * n]
            inf = np.asarray(vals[2 * n], dtype=float) if with_inf else 0.0
            total = inf + 0.0j
            unrelaxed = inf + 0.0
            for t, m in zip(tau, M):
                total = total + m * _shear_factor(s, t)
                unrelaxed = unrelaxed + m
            return _relaxed(symmetry, comps,
                            lambda mu: total + 0.0 * np.asarray(mu),
                            lambda mu: _safe_ratio(1.0, unrelaxed) * total)
        return fn

    names = tuple(_name_of(f, f"tau_{k}") for k, f in enumerate(taus))
    names += tuple(_name_of(f, f"M_{k}") for k, f in enumerate(Ms))
    if with_inf:
        names += (_name_of(long_time_modulus, "M_inf"),)
    return _elastic_law(
        make_fn, moduli, params, constants={"terms": n},
        constant_dimensions=prony.constant_dimensions,
        name=name if name is not None else "prony", omega_domain="complex",
        law=LawRecord("prony",
                      parameters=(_name_of(moduli, "elastic_moduli"),) + names,
                      constants={"terms": n}))


prony.constant_dimensions = {"terms": Dimensions.DIMENSIONLESS}


def _prony_from_record(record: LawRecord, fields: Mapping):
    n = int(record.constants["terms"])
    got = _operands(record, fields, 1 + 2 * n)
    moduli, taus, Ms = got[0], got[1:1 + n], got[1 + n:1 + 2 * n]
    inf = got[1 + 2 * n] if len(got) > 1 + 2 * n else None
    return prony(moduli, taus, Ms, long_time_modulus=inf)


prony.from_record = _prony_from_record


# ---------------------------------------------------------------------------
# provenance out, and back in again
# ---------------------------------------------------------------------------

def law_record_of(field, *, source_name: str | None = None) -> LawRecord | None:
    """The provenance of a frequency-dependent field, or None.

    A law's field carries its `LawRecord` as `.law`, and that is returned
    as it stands.  A **lift** -- a static field standing at every
    frequency, which is how an elastic layer sits beside a viscoelastic
    one -- is no law at all, and is recorded as `LawRecord("static",
    parameters=(source,))`, naming the static field lifted;
    `source_name` is that name, since only the caller knows what its
    layer holds the field under, and the lifted field's own `name` is
    the fallback.  Anything else -- a composition written by hand, a
    sum -- has no provenance, returns None, and cannot be rebuilt from a
    file.
    """
    law = getattr(field, "law", None)
    if isinstance(law, LawRecord):
        return law
    if isinstance(field, LiftedFrequencyField):
        name = (source_name if source_name is not None
                else getattr(field.source, "name", None))
        return None if name is None else LawRecord(STATIC, parameters=(name,))
    return None


def _registered_law(name: str):
    try:
        return lookup("rheology", name)
    except KeyError:
        raise ValueError(
            f"no rheology law named {name!r}; registered: "
            f"{list(registered('rheology'))}") from None


def _operands(record: LawRecord, fields: Mapping, at_least: int) -> list:
    """The fields a record names, in its order, looked up by name."""
    p = record.parameters
    if len(p) < at_least:
        raise ValueError(
            f"the law {record.law!r} needs at least {at_least} parameters, "
            f"and its record names {list(p)}")
    out = []
    for name in p:
        try:
            out.append(fields[name])
        except KeyError:
            raise ValueError(
                f"rebuilding {record.law!r}: no field named {name!r} here; "
                f"what is available is {sorted(fields)}") from None
    return out


def rebuild(record: LawRecord, fields: Mapping):
    """The field a `LawRecord` describes, rebuilt from a layer's fields.

    The inverse of the provenance the writers copy.  `record.law` names
    a registered law, `record.parameters` the fields it read *in the
    order it took them*, and `record.constants` and `record.convention`
    the rest of its call; `fields` is the mapping those names are looked
    up in, which for a netCDF reader is one layer's own fields, so the
    field rebuilt is that layer's piece.  The lift of a static field is
    recorded as the law `"static"` (`law_record_of`) and is rebuilt as
    one, nothing being registered under that name.

    Every registered law knows its own signature through
    `law.from_record(record, fields)`; a law registered without one is
    refused by name rather than called wrongly.
    """
    if not isinstance(record, LawRecord):
        raise TypeError(f"rebuild takes a LawRecord, got {type(record).__name__}")
    if record.law == STATIC:
        return lifted_to_frequency(_operands(record, fields, 1)[0])
    law = _registered_law(record.law)
    from_record = getattr(law, "from_record", None)
    if from_record is None:
        raise ValueError(
            f"the law {record.law!r} is registered but carries no "
            "`from_record(record, fields)`, so it cannot be rebuilt from a "
            "file; give the law that attribute")
    return from_record(record, fields)
