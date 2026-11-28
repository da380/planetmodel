"""Linear viscoelastic rheologies, and a model frozen at one frequency.

A linear viscoelastic body at angular frequency omega is an elastic body
with complex moduli, so a solver written for the moduli A, C, F, L, N
serves any linear rheology once the model is frozen at a frequency.
What a layer's rheology is, is read from the fields it holds, in the
manner of `is_fluid` and `moduli`:

- `viscosity`: a Maxwell body in shear,
  mu(omega) = mu i omega tau / (1 + i omega tau) with tau = viscosity / mu,
  mu being the Voigt-averaged shear modulus;
- Prony branches, `relaxation_strength_k` and `relaxation_time_k` for k
  = 1, 2, ... (`vocabulary.FAMILIES`): a generalised Maxwell body in
  shear, mu(omega) = mu [1 - sum_k w_k / (1 + i omega tau_k)], the
  relaxed part (1 - sum w_k) mu, of which the viscosity is the one-branch
  case (w_1 = 1, tau_1 = viscosity / mu), so a layer holding both is
  refused, as is a branch without its partner or a gap in the indices
  (`branches`);
- `stress_exponent` n and `transition_stress` sigma_t: power-law creep
  beside the linear rheology, which a code that steps in time reads (a
  relaxation time divided by `1 + gamma (|dev sigma| / 2 mu_0)^(n - 1)`,
  gamma = (2 mu_0 / sigma_t)^(n - 1), so that power-law creep strains as
  fast as linear creep at sigma_t); being nonlinear they are no part of
  a frozen model, and freezing ignores them;
- `qmu`, `qkappa`: a constant-Q absorption band about the reference
  frequency omega_0,
  mu(omega) = mu [1 + (2 / pi Q_mu) ln(omega / omega_0) + i / Q_mu],
  and likewise kappa with Q_kappa; a non-positive Q means no loss.

A transversely isotropic layer is perturbed through its Voigt averages:
with delta kappa = (f_kappa - 1) kappa and delta mu = (f_mu - 1) mu,
A and C gain delta kappa + 4 delta mu / 3, F gains
delta kappa - 2 delta mu / 3, and L and N gain delta mu, which is the
isotropic rule when the layer is isotropic and leaves the anisotropy
real.  A layer holding both a viscosity (or branches) and Q_mu is a
Maxwell body in shear (the shear band is not applied on top) with the
bulk band; a layer holding none of them is elastic and left alone.
`is_rheology_name` says which names a layer's rheology is read from,
and `is_linear_rheology_name` which of them a linear rheology is, the
names a frozen model's moduli are derived from.

`frozen(model, omega, *, reference_omega=None)` returns the model with
the complex moduli of every viscoelastic layer stored under A, C, F, L,
N, which `moduli` reads before anything else, and the frequency
recorded as the constant `omega`.  Frequencies are in the model's
units; the reference frequency of the band is the model's constant
`omega_ref` where it declares one (`omega_ref_constant` makes it from
a reference period, as PREM and a mineos deck's model do), else
2 pi rad/s, a period of one second (`reference_omega`).
`is_frozen(model)` says whether a model carries the constant `omega`.
The velocities of a frozen layer, and kappa and mu where it holds them,
stay as the elastic description at the reference frequency; the five
are what a solver reads.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .character import DENSITY
from .fields import ComposedField, Field
from .materials import LayerLike, kappa_mu, moduli, named
from .units import FREQUENCY
from .vocabulary import Constant, member

if TYPE_CHECKING:
    from .model import Model

__all__ = ["is_viscoelastic", "frozen_moduli", "dispersive_moduli", "frozen",
           "is_frozen", "reference_omega", "omega_ref_constant", "RHEOLOGY_NAMES",
           "BRANCH_STEMS", "POWER_LAW_NAMES", "branches", "is_rheology_name",
           "is_linear_rheology_name"]

#: The fields a layer's rheology is read from, beside its Prony branches.
RHEOLOGY_NAMES = ("viscosity", "qmu", "qkappa")

#: The stems of a Prony branch's two names: its strength and its time.
BRANCH_STEMS = ("relaxation_strength", "relaxation_time")

#: The names of power-law creep, read beside the linear rheology.
POWER_LAW_NAMES = ("stress_exponent", "transition_stress")


def is_rheology_name(name: str) -> bool:
    """Whether a layer's rheology is read from fields of this name: one of
    `RHEOLOGY_NAMES`, a branch's, or one of `POWER_LAW_NAMES`."""
    return name in POWER_LAW_NAMES or is_linear_rheology_name(name)


def is_linear_rheology_name(name: str) -> bool:
    """Whether a layer's linear rheology is read from fields of this name:
    one of `RHEOLOGY_NAMES` or a Prony branch's.  These are the names
    `frozen` reads, so the ones a frozen model's moduli are derived from;
    those of power-law creep (`POWER_LAW_NAMES`) are not among them."""
    found = member(name)
    return name in RHEOLOGY_NAMES or (found is not None and found[0] in BRANCH_STEMS)


def branches(layer: LayerLike) -> tuple[tuple[Field, Field], ...]:
    """The layer's Prony branches, (strength, time) for k = 1, 2, ...;
    empty where it holds none, refused where a branch lacks its partner or
    the indices skip one."""
    held: dict[int, dict[str, Field]] = {}
    for name in layer:
        found = member(name)
        if found is not None and found[0] in BRANCH_STEMS:
            held.setdefault(found[1], {})[found[0]] = layer[name]
    if not held:
        return ()
    if sorted(held) != list(range(1, len(held) + 1)):
        raise ValueError(f"the Prony branches are numbered {sorted(held)}, not 1 to "
                         f"{len(held)}")
    out = []
    for k in range(1, len(held) + 1):
        if set(held[k]) != set(BRANCH_STEMS):
            raise ValueError(f"Prony branch {k} holds {sorted(held[k])}; a branch "
                             f"is {BRANCH_STEMS[0]}_{k} and {BRANCH_STEMS[1]}_{k}")
        out.append((held[k][BRANCH_STEMS[0]], held[k][BRANCH_STEMS[1]]))
    return tuple(out)


def is_viscoelastic(layer: LayerLike) -> bool:
    """Whether a layer holds any of `RHEOLOGY_NAMES` or a Prony branch."""
    return any(is_linear_rheology_name(n) for n in layer)


def _maxwell_shift(mu: Field, eta: Field, omega: float) -> ComposedField:
    """delta mu of a Maxwell body: mu (f - 1) with f = z / (1 + z),
    z = i omega eta / mu, zero where mu vanishes."""
    def fn(m: ArrayLike, e: ArrayLike) -> np.ndarray:
        m = np.asarray(m, dtype=complex)
        out = np.zeros(np.broadcast(m, e).shape, dtype=complex)
        solid = m != 0.0
        np.divide(-m * m, m + 1j * omega * np.asarray(e), out=out, where=solid)
        return out
    return ComposedField(fn, [mu, eta], character=DENSITY)


def _prony_shift(mu: Field, held: tuple[tuple[Field, Field], ...],
                 omega: float) -> ComposedField:
    """delta mu of a generalised Maxwell body: -mu sum_k w_k / (1 + i omega
    tau_k)."""
    def fn(m: ArrayLike, *pairs: ArrayLike) -> np.ndarray:
        total = np.zeros(np.shape(m), dtype=complex)
        for w, tau in zip(pairs[::2], pairs[1::2]):
            total = total + np.asarray(w) / (1.0 + 1j * omega * np.asarray(tau))
        return -np.asarray(m) * total
    return ComposedField(fn, [mu] + [f for pair in held for f in pair],
                         character=DENSITY)


def _band_shift(modulus: Field, q: Field, omega: float,
                reference_omega: float) -> ComposedField:
    """delta of a modulus in a constant-Q band: modulus times
    (2 / pi Q) ln(omega / omega_0) + i / Q, zero where Q <= 0."""
    log = np.log(omega / reference_omega)

    def fn(m: ArrayLike, Q: ArrayLike) -> np.ndarray:
        Q = np.asarray(Q, dtype=float)
        inv = np.zeros(Q.shape)
        np.divide(1.0, Q, out=inv, where=Q > 0.0)
        return np.asarray(m) * ((2.0 / np.pi) * inv * log + 1j * inv)
    return ComposedField(fn, [modulus, q], character=DENSITY)


def frozen_moduli(layer: LayerLike, omega: float, *,
                  reference_omega: float) -> dict[str, Field]:
    """The complex moduli A, C, F, L, N of a viscoelastic layer at `omega`
    as fields, from its rheology fields and its elastic moduli; an
    elastic layer's moduli come back unchanged.  A viscosity makes the
    shear a Maxwell body, and Prony branches a generalised one; otherwise
    the Q fields make the constant-Q band of `dispersive_moduli`."""
    return _shifted_moduli(layer, omega, reference_omega, maxwell=True)


def dispersive_moduli(layer: LayerLike, omega: float, *,
                      reference_omega: float) -> dict[str, Field]:
    """The complex moduli A, C, F, L, N of a layer at `omega` under the
    constant-Q absorption band alone: kappa dispersed by `qkappa` and mu
    by `qmu` about `reference_omega`, whichever of the two the layer
    holds, a viscosity ignored.  A layer holding neither Q comes back
    with its static moduli."""
    return _shifted_moduli(layer, omega, reference_omega, maxwell=False)


def _shifted_moduli(layer: LayerLike, omega: float, reference_omega: float, *,
                    maxwell: bool) -> dict[str, Field]:
    """The five at `omega`, kappa and mu shifted by their rheologies and
    recombined through the Voigt rule so that the anisotropy stays real."""
    base = moduli(layer)
    held = branches(layer) if maxwell else ()
    if held and "viscosity" in layer:
        raise ValueError("the layer holds a viscosity and Prony branches; the "
                         "viscosity is the one-branch case, so give one or the "
                         "other")
    names = RHEOLOGY_NAMES if maxwell else ("qmu", "qkappa")
    if not held and not any(n in layer for n in names):
        return base
    kappa, mu = kappa_mu(layer)
    dmu = dkappa = None
    if held:
        dmu = _prony_shift(mu, held, omega)
    elif maxwell and "viscosity" in layer:
        dmu = _maxwell_shift(mu, layer["viscosity"], omega)
    elif "qmu" in layer:
        dmu = _band_shift(mu, layer["qmu"], omega, reference_omega)
    if "qkappa" in layer:
        dkappa = _band_shift(kappa, layer["qkappa"], omega, reference_omega)
    zero = 0.0 * kappa
    dmu = zero if dmu is None else dmu
    dkappa = zero if dkappa is None else dkappa
    A = base["A"] + dkappa + (4.0 / 3.0) * dmu
    C = base["C"] + dkappa + (4.0 / 3.0) * dmu
    F = base["F"] + dkappa - (2.0 / 3.0) * dmu
    return {k: named(f, k) for k, f in (("A", A), ("C", C), ("F", F),
                                         ("L", base["L"] + dmu),
                                         ("N", base["N"] + dmu))}


def omega_ref_constant(period: float) -> Constant:
    """The constant `omega_ref` of a model whose elastic values are those
    at a reference period of `period` seconds: the angular frequency
    2 pi / period, which a constant-Q band disperses the values about."""
    period = float(period)
    if not period > 0.0:
        raise ValueError(f"a reference period is positive, got {period:g}")
    return Constant(2.0 * np.pi / period, FREQUENCY,
                    meaning="reference angular frequency of the model's elastic "
                            "values")


def reference_omega(model: Model) -> float:
    """The reference angular frequency of the constant-Q band in the
    model's units: the constant `omega_ref` where the model declares
    one (PREM's, a deck's header period), else 2 pi rad/s, a period of
    one second."""
    if "omega_ref" in model.constants:
        return float(model.constant("omega_ref"))
    return 2.0 * np.pi / model.scales.factor(FREQUENCY)


#: `reference_omega` under a name the parameter of `frozen` does not shadow.
_model_reference_omega = reference_omega


def is_frozen(model: Model) -> bool:
    """Whether the model is frozen at a frequency: it carries the constant
    `omega` that `frozen` records."""
    return "omega" in model.constants


def frozen(model: Model, omega: float, *,
           reference_omega: float | None = None) -> Model:
    """The model frozen at angular frequency `omega`: every viscoelastic
    layer carries its complex moduli under A, C, F, L, N, and the
    constant `omega` records the frequency.  A model with no
    viscoelastic layer comes back as a copy with the constant alone; a
    model that already carries `omega` is refused, since its moduli are
    no longer the elastic ones the dispersion starts from.
    `reference_omega` defaults to the model's, see `reference_omega`."""
    omega = float(omega)
    if not omega > 0.0:
        raise ValueError(f"omega must be positive, got {omega:g}")
    if is_frozen(model):
        raise ValueError("the model is already frozen, at omega = "
                         f"{float(model.constant('omega')):g}; freeze the elastic "
                         "model it came from")
    factor = model.scales.factor(FREQUENCY)
    if reference_omega is None:
        reference_omega = _model_reference_omega(model)
    reference_omega = float(reference_omega)
    if not reference_omega > 0.0:
        raise ValueError(f"reference_omega must be positive, got {reference_omega:g}")
    layers = []
    for layer in model.layers:
        fields = dict(layer.fields)
        if is_viscoelastic(layer):
            fields.update(frozen_moduli(layer, omega, reference_omega=reference_omega))
        layers.append(fields)
    constants = dict(model.constants)
    constants["omega"] = Constant(omega * factor, FREQUENCY,
                                  meaning="angular frequency the model is frozen at")
    return model.replaced(layers=layers, constants=constants)
