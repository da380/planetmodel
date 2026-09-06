"""Linear viscoelastic rheologies, and a model frozen at one frequency.

A linear viscoelastic body at angular frequency omega is an elastic body
with complex moduli, so a solver written for the moduli A, C, F, L, N
serves any linear rheology once the model is frozen at a frequency.
What a layer's rheology is, is read from the fields it holds, in the
manner of `is_fluid` and `moduli`:

- `viscosity`: a Maxwell body in shear,
  mu(omega) = mu i omega tau / (1 + i omega tau) with tau = viscosity / mu,
  mu being the Voigt-averaged shear modulus;
- `qmu`, `qkappa`: a constant-Q absorption band about the reference
  frequency omega_0,
  mu(omega) = mu [1 + (2 / pi Q_mu) ln(omega / omega_0) + i / Q_mu],
  and likewise kappa with Q_kappa; a non-positive Q means no loss.

A transversely isotropic layer is perturbed through its Voigt averages:
with delta kappa = (f_kappa - 1) kappa and delta mu = (f_mu - 1) mu,
A and C gain delta kappa + 4 delta mu / 3, F gains
delta kappa - 2 delta mu / 3, and L and N gain delta mu, which is the
isotropic rule when the layer is isotropic and leaves the anisotropy
real.  A layer holding both a viscosity and Q_mu is a Maxwell body in
shear (the shear band is not applied on top) with the bulk band; a
layer holding none of the three is elastic and left alone.

`frozen(model, omega, *, reference_omega=None)` returns the model with
the complex moduli of every viscoelastic layer stored under A, C, F, L,
N, which `moduli` reads before anything else, and the frequency
recorded as the constant `omega`.  Frequencies are in the model's
units; `reference_omega` defaults to 2 pi rad/s, a period of one
second, which is PREM's.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .character import DENSITY
from .fields import ComposedField, Field
from .materials import LayerLike, _named, kappa_mu, moduli
from .units import FREQUENCY
from .vocabulary import Constant

if TYPE_CHECKING:
    from .model import Model

__all__ = ["is_viscoelastic", "frozen_moduli", "frozen"]

RHEOLOGY_NAMES = ("viscosity", "qmu", "qkappa")


def is_viscoelastic(layer: LayerLike) -> bool:
    """Whether a layer holds any of `RHEOLOGY_NAMES`."""
    return any(n in layer for n in RHEOLOGY_NAMES)


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
    elastic layer's moduli come back unchanged."""
    base = moduli(layer)
    if not is_viscoelastic(layer):
        return base
    kappa, mu = kappa_mu(layer)
    dmu = dkappa = None
    if "viscosity" in layer:
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
    return {k: _named(f, k) for k, f in (("A", A), ("C", C), ("F", F),
                                         ("L", base["L"] + dmu),
                                         ("N", base["N"] + dmu))}


def frozen(model: Model, omega: float, *,
           reference_omega: float | None = None) -> Model:
    """The model frozen at angular frequency `omega`: every viscoelastic
    layer carries its complex moduli under A, C, F, L, N, and the
    constant `omega` records the frequency.  A model with no
    viscoelastic layer comes back as a copy with the constant alone."""
    omega = float(omega)
    if not omega > 0.0:
        raise ValueError(f"omega must be positive, got {omega:g}")
    factor = model.scales.factor(FREQUENCY)
    if reference_omega is None:
        reference_omega = 2.0 * np.pi / factor
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
