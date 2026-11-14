"""Prony branches: generalised Maxwell bodies, built and put on a model.

A generalised Maxwell body in shear is a sum of branches, each a
fraction w_k of the unrelaxed shear modulus mu_U relaxing with time
tau_k, its complex modulus at angular frequency omega

    mu(omega) / mu_U = 1 - sum_k w_k / (1 + i omega tau_k),

the relaxed part (1 - sum w_k) mu_U (`rheology` reads the branches;
`vocabulary.FAMILIES` names them).  A `PronySeries` holds a body's
strengths w_k and its times in units of the Maxwell time eta / mu
(c_k = tau_k mu / eta), so that one series serves every point of a
model whose viscosity varies: `with_branches(model, series)` writes
`relaxation_strength_k` = w_k and `relaxation_time_k` = c_k eta / mu on
every solid layer holding a viscosity (or on `layers`), and removes the
viscosity, which the branches replace.  A time that varies laterally is
written as a log perturbation of c_k eta_0 / mu_0, the layer's
unperturbed viscosity and shear modulus, as a viscosity is: it spans
orders of magnitude, so its values sampled into a file and read between
nodes would overshoot, where its logarithm does not.  Built:

- `maxwell_series()`: one branch, w = 1, c = 1, the viscosity itself;
- `burgers_series(*, strength, viscosity_ratio)`: a Maxwell element (mu_M,
  eta_M) in series with a Kelvin-Voigt one (mu_K, eta_K), strength Delta
  = mu_M / mu_K and viscosity_ratio = eta_M / eta_K, exactly two
  branches with no relaxed part: the compliance J(s) = 1/mu_M + 1/(eta_M
  s) + 1/(mu_K + eta_K s) vanishes at the two negative roots s_k of
  eta_M eta_K s^2 + (eta_M mu_K + mu_M eta_K + mu_M eta_M) s + mu_M mu_K,
  so tau_k = -1 / s_k, and the residues of 1/J there give mu_k, which sum
  to mu_M (Ivins et al. 2022 call Delta the relaxation strength; Caron
  et al. 2017 take Delta and the viscosity ratio from 1 to 10);
- `with_burgers(model, layers, *, kelvin_viscosity,
  kelvin_modulus_fraction)`: named layers made Burgers bodies over the
  model's own (Maxwell) viscosity eta, the Kelvin element given
  absolutely, its viscosity eta_K in the model's units (Pa s in SI) and
  its shear modulus mu_K as a fraction f of the layer's elastic mu, and
  written as their two Prony branches exactly (`burgers_terms`), the
  faster first, as `burgers_series` orders them.  With a = mu / eta,
  b = mu / eta_K and c = f mu / eta_K the rates 1 / tau_k are the roots
  of x^2 - (a + b + c) x + a c, and w_1 = (1/tau_1 - c) / g,
  w_2 = (c - 1/tau_2) / g with g = 1/tau_1 - 1/tau_2 the gap between
  the roots: c lies between the roots, so both strengths are positive,
  they sum to one (no relaxed part), and w_1 tau_1 + w_2 tau_2
  = eta / mu (the steady viscosity is eta).  The terms are formed
  without cancellation.  Since eta_K is absolute, the terms vary through
  a layer whose eta does, so they are fields of the layer's eta and mu
  rather than one `PronySeries`; with eta_K a fixed fraction of eta they
  are `burgers_series(strength=1 / f, viscosity_ratio=eta / eta_K)`.
  `burgers_branches` is the same pair of branches over fields of a
  layer, a Kelvin element that varies with radius among them;
- `burgers_parameters(modulus, strengths, times)`: the inverse, the
  viscosity eta and the Kelvin element (eta_K, f) of two branches whose
  strengths sum to one.  From a c = x_1 x_2 and a + b + c = x_1 + x_2,
  the rates x_k = 1 / tau_k, and w_1 x_2 + w_2 x_1 = c: with
  s = w_1 tau_1 + w_2 tau_2 and d = w_1 w_2 (tau_2 - tau_1)^2,
  eta = mu s, eta_K = mu s tau_1 tau_2 / d and f = s^2 / d, all three
  positive for any such pair with distinct times, so every two-branch
  body without a relaxed part is a Burgers body;
- `prony_deck(model)` and `burgers_deck(model)`: the model as the decks
  `deck.PRONY` and `deck.BURGERS` hold it, on its own knots where it
  was read from a deck.  A prony deck holds every layer's branches, a
  layer holding a viscosity as its one Maxwell branch (`with_branches`
  of `maxwell_series`), N the most a layer holds; a burgers deck holds
  each layer as the Maxwell or Burgers body it is, the Kelvin element
  recovered by `burgers_parameters`, and refuses a layer that is
  neither;
- `fitted_series(modulus, *, band, ...)`: the strengths of branches at
  fixed times, `per_decade` of them spread a decade beyond the band each
  way, fitted by non-negative least squares to a complex modulus
  mu(omega) / mu_U given as a function of omega times the Maxwell time,
  at 20 frequencies a decade over `band`; for a body that flows
  (`fluid=True`) the strengths sum to one and its long-time viscosity is
  eta (sum w_k c_k = 1), both held as heavily weighted rows; `misfit` is
  the largest relative misfit of the modulus over the band;
- `andrade_series(alpha, *, andrade_time, band, ...)`: such a fit to the
  Andrade body J(omega) / J_U = 1 + Gamma(1 + alpha) (i omega tau_A)^-alpha
  + 1 / (i omega tau_M), tau_A in Maxwell times (`andrade_time`, 1 by
  default), which has no finite Prony form.
"""
from __future__ import annotations

import numbers
from collections.abc import Callable, Iterable, Sequence
from dataclasses import KW_ONLY, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import nnls
from scipy.special import gamma

from ..character import SCALAR
from ..deck import (BURGERS, BURGERS_COLUMNS, MAXWELL_COLUMNS, PRONY, PRONY_COLUMNS,
                    SAMPLE_SPACING, STRENGTH_ROUNDING, Deck, prony_columns, sample_deck)
from ..fields import ComposedField, Field, constant_field
from ..materials import is_fluid, kappa_mu
from ..model import Layer, Model, cited
from ..perturbation import PerturbedField, unperturbed
from ..rheology import BRANCH_STEMS, branches

__all__ = ["PronySeries", "maxwell_series", "burgers_series", "fitted_series",
           "andrade_series", "with_branches", "burgers_terms", "burgers_branches",
           "burgers_parameters", "with_burgers", "prony_deck", "burgers_deck"]


@dataclass(frozen=True)
class PronySeries:
    """A generalised Maxwell body by the module docstring: `strengths` w_k
    and `times` c_k in Maxwell times, `name`, the `misfit` of a fitted one
    and the works it rests on."""

    strengths: Sequence[float]
    times: Sequence[float]
    _: KW_ONLY
    name: str
    misfit: float = 0.0
    citation: Iterable[str] = ()

    def __post_init__(self) -> None:
        w = tuple(float(x) for x in self.strengths)
        c = tuple(float(x) for x in self.times)
        if len(w) != len(c) or not w:
            raise ValueError("a series has as many strengths as times, at least one")
        if any(not x >= 0.0 for x in w) or any(not x > 0.0 for x in c):
            raise ValueError("strengths are non-negative and times positive")
        if sum(w) > 1.0 + 1e-12:
            raise ValueError(f"the strengths sum to {sum(w):.15g}, more than the "
                             "whole unrelaxed modulus")
        object.__setattr__(self, "strengths", w)
        object.__setattr__(self, "times", c)
        object.__setattr__(self, "citation", cited((), self.citation))

    def modulus(self, omega: ArrayLike) -> np.ndarray:
        """mu(omega) / mu_U at `omega` in units of the inverse Maxwell time."""
        x = np.asarray(omega, dtype=float)
        out = np.ones(x.shape, dtype=complex)
        for w, c in zip(self.strengths, self.times):
            out = out - w / (1.0 + 1j * x * c)
        return out


def maxwell_series() -> PronySeries:
    """The Maxwell body, one branch: w = 1, tau = eta / mu."""
    return PronySeries((1.0,), (1.0,), name="Maxwell")


def burgers_series(*, strength: float, viscosity_ratio: float) -> PronySeries:
    """The Burgers body of the module docstring, Delta = mu_M / mu_K and
    eta_M / eta_K, as its two branches."""
    if not (strength > 0.0 and viscosity_ratio > 0.0):
        raise ValueError("strength and viscosity_ratio are positive")
    mu_m, eta_m = 1.0, 1.0
    mu_k, eta_k = 1.0 / strength, 1.0 / viscosity_ratio
    a, b, c = eta_m * eta_k, eta_m * mu_k + mu_m * eta_k + mu_m * eta_m, mu_m * mu_k
    root = np.sqrt(b * b - 4.0 * a * c)
    s = np.array([(-b - root) / (2.0 * a), 2.0 * c / (-b - root)])
    mu = mu_m * eta_m * (mu_k + eta_k * s) / (2.0 * a * s + b)
    return PronySeries(tuple(mu / mu_m), tuple(-1.0 / s),
                       name=f"Burgers (strength {strength:g}, viscosity ratio "
                            f"{viscosity_ratio:g})")


def fitted_series(modulus: Callable[[np.ndarray], ArrayLike], *,
                  band: tuple[float, float], per_decade: int = 2, fluid: bool = True,
                  name: str = "fitted") -> PronySeries:
    """The series fitted to `modulus`, mu(omega) / mu_U at omega in inverse
    Maxwell times, by the rule of the module docstring."""
    lo, hi = (float(x) for x in band)
    if not 0.0 < lo < hi:
        raise ValueError(f"band is (low, high) with 0 < low < high, got {band}")
    if per_decade < 1:
        raise ValueError("per_decade is at least one")
    omega = np.logspace(np.log10(lo), np.log10(hi), int(20 * np.log10(hi / lo)) + 1)
    n = int(round(per_decade * (np.log10(hi / lo) + 2.0))) + 1
    times = np.logspace(-np.log10(hi) - 1.0, -np.log10(lo) + 1.0, n)
    target = np.asarray(modulus(omega), dtype=complex)
    if not np.all(np.isfinite(target)):
        raise ValueError("the modulus is not finite over the band")
    basis = 1.0 / (1.0 + 1j * omega[:, None] * times[None, :])
    scale = 1.0 / np.abs(target)
    rows = [np.real(basis) * scale[:, None], np.imag(basis) * scale[:, None]]
    rhs = [np.real(1.0 - target) * scale, np.imag(1.0 - target) * scale]
    if fluid:
        heavy = 1e3 * np.sqrt(omega.size)
        rows += [heavy * np.ones((1, n)), heavy * times[None, :]]
        rhs += [np.array([heavy]), np.array([heavy])]
    w, _ = nnls(np.vstack(rows), np.concatenate(rhs), maxiter=50 * n)
    keep = w > 1e-14 * w.max()
    w, c = w[keep], times[keep]
    if w.sum() > 1.0:
        w = w / w.sum()
    series = PronySeries(tuple(w), tuple(c), name=name)
    misfit = float(np.max(np.abs(series.modulus(omega) - target) / np.abs(target)))
    return PronySeries(series.strengths, series.times, name=name, misfit=misfit)


def andrade_series(alpha: float, *, andrade_time: float = 1.0,
                   band: tuple[float, float] = (1e-3, 1e3), per_decade: int = 2
                   ) -> PronySeries:
    """The Andrade body of the module docstring fitted over `band` (in
    inverse Maxwell times), `alpha` in (0, 1)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha lies in (0, 1), got {alpha}")
    if not andrade_time > 0.0:
        raise ValueError("andrade_time is positive")
    g = gamma(1.0 + alpha)

    def modulus(omega: np.ndarray) -> np.ndarray:
        j = 1.0 + g * (1j * omega * andrade_time) ** (-alpha) + 1.0 / (1j * omega)
        return 1.0 / j

    return fitted_series(modulus, band=band, per_decade=per_decade, fluid=True,
                         name=f"Andrade (alpha {alpha:g}, Andrade time "
                              f"{andrade_time:g} Maxwell times)")


def with_branches(model: Model, series: PronySeries, *,
                  layers: Iterable[int | str] | None = None,
                  name: str = "viscosity") -> Model:
    """The model with `series` on every solid layer holding `name` (or on
    `layers`), by the rule of the module docstring: the strengths constant,
    the times c_k eta / mu, the viscosity removed."""
    which = ([lay.index for lay in model.layers if name in lay
              and not is_fluid(lay)] if layers is None
             else sorted({model.layer(k).index for k in layers}))
    if not which:
        raise ValueError(f"no solid layer holds {name!r} to build branches from")
    fields = [dict(lay.fields) for lay in model.layers]
    plain = unperturbed(model)
    for i in which:
        lay = model.layer(i)
        if name not in lay:
            raise KeyError(f"layer {i} ({lay.name}) holds no {name!r}")
        if is_fluid(lay):
            raise ValueError(f"layer {i} ({lay.name}) is fluid: it has no shear to "
                             "relax")
        eta, eta0 = lay[name], plain.layer(i)[name]
        mu, mu0 = kappa_mu(lay)[1], kappa_mu(plain.layer(i))[1]
        for k, (w, c) in enumerate(zip(series.strengths, series.times), start=1):
            key = f"relaxation_time_{k}"
            fields[i][f"relaxation_strength_{k}"] = constant_field(
                lay.interval, w, name=f"relaxation_strength_{k}")
            fields[i][key] = _time(c, eta, mu, eta0, mu0, key)
        del fields[i][name]
    return model.replaced(layers=fields).with_citation(*series.citation)




def burgers_terms(modulus: ArrayLike, viscosity: ArrayLike, *,
                  kelvin_viscosity: ArrayLike, kelvin_modulus_fraction: ArrayLike
                  ) -> tuple[np.ndarray, np.ndarray]:
    """A Burgers body's two Prony branches at each point, (strengths,
    times), each of shape (2,) + the broadcast shape of the arguments, the
    faster branch first (module docstring): a Maxwell element of shear
    modulus `modulus` and viscosity `viscosity` in series with a Kelvin
    element of viscosity `kelvin_viscosity` and shear modulus
    `kelvin_modulus_fraction` times `modulus`.  Every argument is
    positive and finite; any consistent units."""
    mu, eta, eta_k, f = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (
        modulus, viscosity, kelvin_viscosity, kelvin_modulus_fraction)))
    for x, what in ((mu, "modulus"), (eta, "viscosity"),
                    (eta_k, "kelvin_viscosity"),
                    (f, "kelvin_modulus_fraction")):
        if not np.all(np.isfinite(x) & (x > 0.0)):
            raise ValueError(f"a Burgers body's {what} is positive and finite")
    a, b = mu / eta, mu / eta_k
    c = f * b
    gap = np.sqrt((a - c) ** 2 + b * b + 2.0 * b * (a + c))
    fast = 0.5 * (a + b + c + gap)
    slow = a * c / fast                    # the product of the roots is a c

    def plus(s: np.ndarray, product: np.ndarray) -> np.ndarray:
        """gap + s, formed without cancellation where s < 0 (gap^2 - s^2
        is 4 product, positive)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s >= 0.0, gap + s, 4.0 * product / (gap - s))

    above_slow = c * 0.5 * plus(b + c - a, a * b) / fast   # c - slow
    below_fast = 0.5 * plus(a + b - c, b * c)              # fast - c
    return (np.stack([below_fast / gap, above_slow / gap]),
            np.stack([1.0 / fast, 1.0 / slow]))


def burgers_parameters(modulus: ArrayLike, strengths: ArrayLike, times: ArrayLike
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Burgers body of two Prony branches, the inverse of
    `burgers_terms`: (viscosity, kelvin_viscosity, kelvin_modulus_fraction)
    at each point, by the module docstring, of the shear modulus
    `modulus` and the branches' `strengths` and `times`, each of shape
    (2,) + the points' shape, in either order.  The strengths are taken
    to sum to one, as a Burgers body's do; every value is positive and
    finite and the two times differ at every point; any consistent
    units."""
    mu = np.asarray(modulus, dtype=float)
    w = np.asarray(strengths, dtype=float)
    tau = np.asarray(times, dtype=float)
    if w.shape[:1] != (2,) or tau.shape[:1] != (2,):
        raise ValueError(f"strengths and times are two branches each, of shape "
                         f"(2, ...); got {w.shape} and {tau.shape}")
    for x, what in ((mu, "modulus"), (w, "strengths"), (tau, "times")):
        if not np.all(np.isfinite(x) & (x > 0.0)):
            raise ValueError(f"a Burgers body's {what} are positive and finite")
    if np.any(tau[0] == tau[1]):
        raise ValueError("the two branches have the same time at a point; a Burgers "
                         "body's two times differ")
    steady = w[0] * tau[0] + w[1] * tau[1]
    d = w[0] * w[1] * (tau[1] - tau[0]) ** 2
    return mu * steady, mu * steady * tau[0] * tau[1] / d, steady * steady / d


def burgers_branches(modulus: Field, viscosity: Field, *, kelvin_viscosity: Field,
                     kelvin_modulus_fraction: Field) -> dict[str, Field]:
    """A Burgers body's two Prony branches as fields of one layer, by
    `burgers_terms` at every point, the faster first:
    `relaxation_strength_k` and `relaxation_time_k` for k = 1, 2, of a
    Maxwell element of shear modulus `modulus` and viscosity `viscosity`
    in series with a Kelvin element of viscosity `kelvin_viscosity` and
    shear modulus `kelvin_modulus_fraction` times `modulus`, each of them
    a field on the layer."""
    sources = [modulus, viscosity, kelvin_viscosity, kelvin_modulus_fraction]
    out: dict[str, Field] = {}
    for k in range(2):
        for part, stem in enumerate(BRANCH_STEMS):
            key = f"{stem}_{k + 1}"
            out[key] = ComposedField(_burgers_term(part, k), sources, character=SCALAR,
                                     name=key)
    return out


def with_burgers(model: Model, layers: Iterable[int | str], *,
                 kelvin_viscosity: float, kelvin_modulus_fraction: float,
                 name: str = "viscosity") -> Model:
    """The model with each of `layers` (by index or name) a Burgers body
    (module docstring): its field `name` the Maxwell element's viscosity,
    in series with a Kelvin element of viscosity `kelvin_viscosity` (in
    the model's units) and shear modulus `kelvin_modulus_fraction` times
    the layer's elastic (Voigt) shear modulus, written as the two Prony
    branches of `burgers_branches` at every point, the viscosity
    removed."""
    for value, what in ((kelvin_viscosity, "kelvin_viscosity"),
                        (kelvin_modulus_fraction, "kelvin_modulus_fraction")):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) \
                or not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{what} is a positive number, got {value!r}")
    if isinstance(layers, (str, int)):
        raise TypeError("layers is a sequence of layers, by index or name")
    which = sorted({model.layer(k).index for k in layers})
    if not which:
        raise ValueError("no layer is named to make a Burgers body")
    fields = [dict(lay.fields) for lay in model.layers]
    for i in which:
        lay = model.layer(i)
        if name not in lay:
            raise KeyError(f"layer {i} ({lay.name}) holds no {name!r}: a Burgers "
                           "body is built over its Maxwell viscosity")
        if is_fluid(lay):
            raise ValueError(f"layer {i} ({lay.name}) is fluid: it has no shear to "
                             "relax")
        kelvin = {key: constant_field(lay.interval, float(value), name=key)
                  for key, value in (("kelvin_viscosity", kelvin_viscosity),
                                     ("kelvin_modulus_fraction",
                                      kelvin_modulus_fraction))}
        fields[i].update(burgers_branches(kappa_mu(lay)[1], lay[name], **kelvin))
        del fields[i][name]
    return model.replaced(layers=fields)


def _burgers_term(part: int, k: int
                  ) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                np.ndarray]:
    """Branch k's strength (`part` 0) or time (1) of `burgers_terms` as a
    function of the shear modulus, the viscosity and the Kelvin element's
    viscosity and modulus fraction."""
    def term(mu: np.ndarray, eta: np.ndarray, eta_k: np.ndarray,
             f: np.ndarray) -> np.ndarray:
        return burgers_terms(mu, eta, kelvin_viscosity=eta_k,
                             kelvin_modulus_fraction=f)[part][k]
    return term


def _deck_knots(model: Model, knots: Sequence[ArrayLike] | None
                ) -> Sequence[ArrayLike] | None:
    """`knots`, or else the model's own where it was read from a deck
    (`deck.Tabulated`), or else None for `sample_deck` to place them."""
    if knots is not None:
        return knots
    own = getattr(model, "knots", ())
    return own if len(own) else None


def _both(lay: Layer, held: tuple[tuple[Field, Field], ...]) -> None:
    """Refuse a layer holding a viscosity and Prony branches, as freezing
    does: the viscosity is the one-branch case."""
    if held and "viscosity" in lay:
        raise ValueError(f"layer {lay.index} ({lay.name}) holds a viscosity and Prony "
                         "branches; the viscosity is the one-branch case, so give one "
                         "or the other")


def prony_deck(model: Model, *, knots: Sequence[ArrayLike] | None = None,
               spacing: float = SAMPLE_SPACING) -> Deck:
    """The model as a prony deck (`deck.PRONY`), in SI, as it is written:
    the density, the elastic velocities, and the Prony branches of every
    layer by the module docstring, a solid layer holding a viscosity as
    its one Maxwell branch (strength one, time the viscosity over the
    layer's elastic (Voigt) shear modulus), a layer holding branches as
    it holds them, an elastic or fluid layer NaN in every branch column;
    N is the most branches a layer holds, and a layer holding fewer is
    completed as `deck.PRONY` writes it.  `knots` and `spacing` are
    `sample_deck`'s; without knots a model read from a deck is sampled
    on its own.  A layer holding a viscosity and branches is refused, as
    is a model whose every layer is elastic: a prony deck has at least
    one branch."""
    knots = _deck_knots(model, knots)
    viscous = []
    for lay in model.layers:
        _both(lay, branches(lay))
        if "viscosity" in lay and not is_fluid(lay):
            viscous.append(lay.index)
    if viscous:
        model = with_branches(model, maxwell_series(), layers=viscous)
    n = max((len(branches(lay)) for lay in model.layers), default=0)
    if n == 0:
        raise ValueError("no solid layer holds a viscosity or Prony branches; a prony "
                         "deck has at least one branch")
    sampled = sample_deck(model, prony_columns(n), knots=knots, spacing=spacing)
    deck = PRONY.complete(sampled)
    PRONY.check_table(deck)
    return deck


def burgers_deck(model: Model, *, knots: Sequence[ArrayLike] | None = None,
                 spacing: float = SAMPLE_SPACING) -> Deck:
    """The model as a burgers deck (`deck.BURGERS`), in SI: the density,
    the elastic velocities, and each layer as the body it is.  A layer
    holding a viscosity, or one Prony branch of strength one (to
    `deck.STRENGTH_ROUNDING`; its viscosity mu tau), is a Maxwell body; a
    layer holding two branches whose strengths sum to one (to twice that)
    is a Burgers body, its viscosity and Kelvin element by
    `burgers_parameters` at every knot, mu the layer's elastic (Voigt)
    shear modulus; an elastic or fluid layer holds none of the three.  A
    layer holding other branches, more than two or with a relaxed part, is
    no Burgers body and is refused, as is one holding a viscosity and
    branches.  `knots` and `spacing` are `prony_deck`'s."""
    knots = _deck_knots(model, knots)
    counts = []
    for lay in model.layers:
        held = branches(lay)
        _both(lay, held)
        if len(held) > 2:
            raise ValueError(f"layer {lay.index} ({lay.name}) holds {len(held)} Prony "
                             "branches; a Burgers body has two, a Maxwell body one")
        counts.append(len(held))
    n = max(counts, default=0)
    extra = prony_columns(n)[len(PRONY_COLUMNS):] if n else ()
    sampled = sample_deck(model, MAXWELL_COLUMNS + extra, knots=knots, spacing=spacing)
    columns = {name: np.array(sampled[name]) for name in MAXWELL_COLUMNS}
    for name in BURGERS_COLUMNS[len(MAXWELL_COLUMNS):]:
        columns[name] = np.full(sampled.nknots, np.nan)
    for lay, count, s in zip(model.layers, counts, sampled.layers()):
        if count == 0:
            continue
        r = sampled.radius[s]
        w, tau = (np.array([sampled[f"{stem}_{k}"][s] for k in range(1, count + 1)])
                  for stem in BRANCH_STEMS)
        total = np.zeros(r.size)
        for row in w:
            total = total + row
        off = np.abs(total - 1.0) > count * STRENGTH_ROUNDING
        if off.any():
            j = int(np.argmax(off))
            raise ValueError(f"layer {lay.index} ({lay.name}): the strengths of its "
                             f"branches sum to {total[j]:.9g} at r = {r[j]:g}, not "
                             "one; a body with a relaxed part is no Maxwell or Burgers "
                             "body")
        mu = np.asarray(kappa_mu(lay)[1](r), dtype=float)
        if count == 1:
            columns["viscosity"][s] = mu * tau[0]
            continue
        eta, eta_k, f = burgers_parameters(mu, w, tau)
        columns["viscosity"][s] = eta
        columns["kelvin_viscosity"][s] = eta_k
        columns["kelvin_modulus_fraction"][s] = f
    deck = Deck(sampled.radius, columns, header=sampled.header)
    BURGERS.check_table(deck)
    return deck


def _time(c: float, eta: Field, mu: Field, eta0: Field, mu0: Field,
          name: str) -> Field:
    """c eta / mu, as a log perturbation of c eta_0 / mu_0 where it varies
    laterally and the latter does not."""
    def ratio(e: np.ndarray, m: np.ndarray) -> np.ndarray:
        return c * np.asarray(e) / np.asarray(m)

    time = ComposedField(ratio, [eta, mu], character=SCALAR, name=name)
    ref = ComposedField(ratio, [eta0, mu0], character=SCALAR, name=name)
    if time.is_radial or not ref.is_radial:
        return time
    log = ComposedField(lambda t, t0: np.log(np.asarray(t) / np.asarray(t0)),
                        [time, ref], character=SCALAR, name=name)
    return PerturbedField(ref, log, kind="log", name=name)
