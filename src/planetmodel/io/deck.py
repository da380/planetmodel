"""deck.py -- reading tabulated radial models.

A deck is a table of knots with radius first, in which a repeated radius
marks a discontinuity.  Geometry comes from that structure, so a
boundary is recorded even where the tabulated values happen to agree
across it -- continuity is a per-field numerical accident, not a
property of the model.

read_deck() is the generic reader (skip n header lines, then numbers);
header *interpretation* belongs to concrete readers such as read_mineos_deck.

Decks tabulate velocities, but velocities have no transformation law, so
what a model stores is moduli (see model/character.py).  The conversion
therefore happens **on load**: every reader here attaches the five TI
moduli and an ElasticField alongside the tabulated columns.  The moduli
are products rho*v^2 formed on the layer functions themselves, so an
exact polynomial model stays exact; velocities remain available as the
columns that were read.

Symmetry is information, and load is where it would be easiest to lose.
A deck tabulating vp and vs alone describes an *isotropic* medium, and
its ElasticField says so -- Symmetry.ISOTROPIC, stored as (kappa, mu) --
rather than a VTI tensor whose five moduli happen to be degenerate.  The
push-forward fast paths dispatch on that symmetry, so discarding it
would quietly cost work later.  The five moduli A, C, F, L, N are
attached alongside in either case, as the derived five-moduli form of
the same medium: the 1D solver and the velocity views read them by name.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ..model.body import ReferenceBody
from ..model.character import SCALAR, Symmetry
from ..model.units import Dimensions
from ..model.vocabulary import character_of, dimensions_of
from ..model.fields.layer_function import (combine_layer_functions,
                                      multiply_layer_functions)
from ..model.fields.radial import RadialField, make_fitter
from ..model.materials import MODULI_NAMES, ElasticField
from ..model.skeleton import Skeleton
from ..registry import register

__all__ = ["read_deck", "read_mineos_deck", "read_isotropic_deck",
           "MINEOS_COLUMNS", "ISOTROPIC_COLUMNS",
           "attach_moduli", "attach_velocity_views"]


def _split_layers(radii) -> tuple[Skeleton, list[slice]]:
    """Boundaries and per-layer row slices from a knot-radius column.

    A repeated radius cuts a layer; geometry comes from structure, so a
    boundary is recorded even if tabulated values happen to be equal
    across it (continuity is a per-field numerical accident).
    """
    r = np.asarray(radii, dtype=float)
    d = np.diff(r)
    if np.any(d < 0):
        raise ValueError("knot radii must be non-decreasing")
    cuts = np.flatnonzero(d == 0.0)
    if cuts.size and np.any(np.diff(cuts) == 1):
        raise ValueError("a radius repeats more than twice (zero-thickness layer)")
    starts = np.concatenate(([0], cuts + 1))
    stops = np.concatenate((cuts + 1, [r.size]))
    slices = [slice(int(a), int(b)) for a, b in zip(starts, stops)]
    if any(s.stop - s.start < 2 for s in slices):
        raise ValueError("every layer needs at least two knots")
    boundaries = np.concatenate(([r[0]], r[cuts], [r[-1]]))
    return Skeleton(boundaries), slices


def _model_from_table(table, columns: Sequence[str], kind: str | Callable,
                      meta: dict | None) -> ReferenceBody:
    """A body from a (radius + fields) knot table, its moduli attached."""
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] != len(columns) + 1:
        raise ValueError(f"expected {len(columns) + 1} columns "
                         f"(radius + {len(columns)} fields), got {table.shape}")
    sk, slices = _split_layers(table[:, 0])
    fit = make_fitter(kind=kind)
    fields = {
        nm: RadialField(sk,
                        tuple(fit(table[s, 0], table[s, j + 1]) for s in slices),
                        name=nm, character=character_of(nm),
                        dimensions=dimensions_of(nm))
        for j, nm in enumerate(columns)
    }
    model = ReferenceBody.from_fields(sk, fields, meta=meta)
    attach_moduli(model)
    return model


def as_model_class(body, *, reference_period=None):
    """A read body as the model class its fields warrant.

    With `rho` and `elastic_moduli`, and `qkappa` and `qmu` and a
    reference period, a `ViscoelasticModel` whose every layer with
    moduli holds `viscoelastic_moduli` under constant Q at that period;
    with `rho` and `elastic_moduli` but no period, or no Q columns, an
    `ElasticModel`, the type refusing to pretend to a calibration it
    was not given; with neither, the body as it is.  A deck's header
    period, where a reader trusts it, is the reference period; it is
    also kept as `meta["tref"]` for provenance, and nothing reads it
    from there.
    """
    from ..model.rheology import constant_q
    from ..model.classes import ElasticModel, ViscoelasticModel
    if "rho" not in body or "elastic_moduli" not in body:
        return body
    if (reference_period is not None and "qkappa" in body and "qmu" in body):
        law = constant_q(body["elastic_moduli"], body["qkappa"], body["qmu"],
                         reference_period=reference_period)
        body.add_field("viscoelastic_moduli", law)
        return body.as_class(ViscoelasticModel)
    return body.as_class(ElasticModel)




def read_deck(source: str | os.PathLike | Iterable[str],
              columns: Sequence[str], *, header_lines: int = 0,
              kind: str | Callable = "cubic",
              meta: dict | None = None) -> ReferenceBody:
    """Generic deck reader: skip header_lines, then read numeric knots.

    Rows are whitespace-separated with radius first.  Header
    *interpretation* belongs to concrete readers (read_mineos_deck);
    this function only skips.  source may be a path or an iterable of
    lines.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source) as fh:
            lines = fh.read().splitlines()
    else:
        lines = list(source)
    body = [ln for ln in lines[header_lines:] if ln.strip()]
    table = np.loadtxt(body, dtype=float, ndmin=2)
    return _model_from_table(table, columns, kind, meta)


#: Columns of an isotropic deck: r rho vp vs qkappa qmu.
ISOTROPIC_COLUMNS = ("rho", "vp", "vs", "qkappa", "qmu")

MINEOS_COLUMNS = ("rho", "vpv", "vsv", "qkappa", "qmu", "vph", "vsh", "eta")


def _velocity_columns(model) -> dict[str, str] | None:
    """Which velocity naming a model uses, mapped to the TI names.

    Anisotropic decks give vpv/vsv/vph/vsh/eta; isotropic ones give
    vp/vs, which is the degenerate case vph = vpv, vsh = vsv, eta = 1.
    Returns None when the model tabulates no velocities at all.
    """
    if all(n in model for n in ("vpv", "vsv", "vph", "vsh")):
        return {"vpv": "vpv", "vsv": "vsv", "vph": "vph", "vsh": "vsh"}
    if "vp" in model and "vs" in model:
        return {"vpv": "vp", "vsv": "vs", "vph": "vp", "vsh": "vs"}
    return None


def attach_moduli(model, *, replace: bool = False):
    """Attach the TI moduli and an ElasticField built from the velocities.

    The moduli are formed layer function by layer function, so where the
    model is exact -- PREM, or any polynomial deck -- the moduli are
    exact too, rather than resampled.  A model with no velocity columns
    is left alone.

    Registers A, C, F, L, N individually and an ElasticField under
    "elastic_moduli"; the velocity columns stay exactly as they were read.

    A deck with one P and one S velocity and no eta column is isotropic,
    and the ElasticField reports Symmetry.ISOTROPIC, stored as the pair
    kappa = A - 4L/3 and mu = L, which are also registered by name.  The
    five moduli are attached in that case too, as the derived
    five-moduli description: as_symmetry(Symmetry.VTI) recovers them.
    """
    cols = _velocity_columns(model)
    if cols is None:
        return None
    if "elastic_moduli" in model and not replace:
        return model["elastic_moduli"]

    sk = model.skeleton
    rho = model["rho"]
    eta = model["eta"] if "eta" in model else None
    # One P and one S velocity, and no eta: the deck says isotropic, and
    # that is what the ElasticField will report.
    isotropic = (eta is None and cols["vph"] == cols["vpv"]
                 and cols["vsh"] == cols["vsv"])

    def each(fn, *columns):
        """fn over the layer functions, None wherever a column has none."""
        out = []
        for i in range(sk.nlayers):
            parts = [c.functions[i] if hasattr(c, "functions") else c[i]
                     for c in columns]
            out.append(None if any(p is None for p in parts) else fn(*parts))
        return out

    def squared(name):
        """v^2, layer by layer, exactly where the layer functions allow."""
        v = model[cols[name]]
        return each(lambda f: multiply_layer_functions(f, f, names=(name, name)),
                    v)

    def scaled_by_rho(sq, out):
        """rho * v^2, layer by layer."""
        return each(lambda d, q: multiply_layer_functions(d, q, names=("rho", out)),
                    rho, sq)

    A = scaled_by_rho(squared("vph"), "A")
    C = scaled_by_rho(squared("vpv"), "C")
    L = scaled_by_rho(squared("vsv"), "L")
    N = scaled_by_rho(squared("vsh"), "N")

    # F = eta (A - 2L); with no eta column the medium is isotropic and
    # eta = 1, which keeps the combination exact instead of multiplying
    # by a resampled constant field.
    Fl = each(lambda a, l: combine_layer_functions([(1.0, a), (-2.0, l)]), A, L)
    if eta is not None:
        Fl = each(lambda e, f: multiply_layer_functions(e, f, names=("eta", "F")),
                  eta, Fl)

    # A modulus is a scalar-valued component of an elastic tensor, not a
    # tensor: it carries the dimensions of a modulus and the character of
    # a number.  The tensor's rank-4 ELASTIC character belongs to the
    # ElasticField assembled from them, below.
    moduli = {}
    for nm, funcs in zip(MODULI_NAMES, (A, C, Fl, L, N)):
        moduli[nm] = model.add_field(
            nm, RadialField(sk, funcs, name=nm, character=SCALAR,
                            dimensions=Dimensions.MODULUS),
            replace=replace)

    if isotropic:
        # kappa = rho (vp^2 - 4 vs^2 / 3) = A - 4L/3 and mu = rho vs^2 = L,
        # formed on the same layer functions the moduli were, so an exact
        # deck stays exact and as_symmetry(VTI) returns to A..N exactly.
        kappa_funcs = each(
            lambda a, l: combine_layer_functions([(1.0, a), (-4.0 / 3.0, l)]),
            A, L)
        pair = {}
        for nm, funcs in (("kappa", kappa_funcs), ("mu", list(L))):
            pair[nm] = model.add_field(
                nm, RadialField(sk, funcs, name=nm, character=SCALAR,
                                dimensions=Dimensions.MODULUS),
                replace=replace)
        elastic = ElasticField(Symmetry.ISOTROPIC, pair, name="elastic_moduli")
    else:
        elastic = ElasticField(Symmetry.VTI, moduli, name="elastic_moduli")
    model.add_field("elastic_moduli", elastic, replace=replace)
    return elastic


def attach_velocity_views(model, *, replace: bool = False):
    """Attach velocities as lazy views over (rho, moduli).

    The inverse direction of attach_moduli, and deliberately *not*
    symmetric with it.  Moduli are products, so they stay exact
    polynomials; velocities need a square root, so they cannot.  They
    are therefore ComposedFields -- exact at any point they are asked
    about, approximate under integration, and honest about it -- rather
    than splines refitted to sampled values, which would look exact and
    would not be.

    A model that tabulates its velocities already has better ones: this
    only fills in names that are missing.
    """
    from ..model.fields.composite import ComposedField

    need = {"rho", *MODULI_NAMES}
    if not need <= set(model.field_names):
        raise ValueError(
            f"velocity views need rho and the moduli; model has "
            f"{list(model.field_names)}")

    rho = model["rho"]
    src = {k: model[k] for k in MODULI_NAMES}

    def root():
        """sqrt(modulus / rho), guarded at rho = 0."""
        def fn(m, d):
            safe = np.where(d > 0.0, d, 1.0)
            return np.sqrt(np.maximum(np.where(d > 0.0, m / safe, 0.0), 0.0))
        return fn

    views = {
        "vph": (root(), (src["A"], rho)),
        "vpv": (root(), (src["C"], rho)),
        "vsv": (root(), (src["L"], rho)),
        "vsh": (root(), (src["N"], rho)),
    }
    out = {}
    for name, (fn, sources) in views.items():
        if name in model and not replace:
            continue
        out[name] = model.add_field(
            name, ComposedField(fn, sources, name=name, character=SCALAR,
                                dimensions=Dimensions.VELOCITY),
            replace=replace)

    if "eta" not in model or replace:
        def eta_fn(F, A, L):
            denom = A - 2.0 * L
            return np.where(denom != 0.0,
                            F / np.where(denom != 0.0, denom, 1.0), 1.0)
        out["eta"] = model.add_field(
            "eta", ComposedField(eta_fn, (src["F"], src["A"], src["L"]),
                                 name="eta", character=SCALAR,
                                 dimensions=Dimensions.DIMENSIONLESS),
            replace=replace)
    return out


def _read_mineos_style(path, columns, *, kind, reference_period,
                       period_from_header):
    """Read a three-header-line deck and return the model class it warrants.

    Header lines: a title; `ifanis tref ifdeck`; `nknot nic noc`.
    Trailing tokens on the numeric lines are tolerated and kept in
    meta["header_extras"].  The knot table is ground truth: mismatches
    with nknot, nic and noc only warn.  Layers are classified on read
    by the default state rule, so a layer with no shear velocity is
    fluid.  `reference_period` (seconds) says at what period the moduli
    hold; where the header's `tref` is trusted it supplies the default.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    if len(lines) < 5:
        raise ValueError(f"{path}: too short to be a mineos-style deck")
    h2, h3 = lines[1].split(), lines[2].split()
    meta = {
        "name": lines[0].strip(),
        "ifanis": int(h2[0]), "tref": float(h2[1]), "ifdeck": int(h2[2]),
        "nknot": int(h3[0]), "nic": int(h3[1]), "noc": int(h3[2]),
        "header_extras": (h2[3:], h3[3:]),   # tolerated, kept for reference
        "source": str(path),
    }
    body = [ln for ln in lines[3:] if ln.strip()]
    table = np.loadtxt(body, dtype=float, ndmin=2)
    model = _model_from_table(table, columns, kind, meta)
    if reference_period is None and period_from_header:
        reference_period = meta["tref"]
    if reference_period is not None:
        model.meta["tref"] = float(reference_period)
    else:
        model.meta.pop("tref", None)
    model = model.classify_states()
    model = as_model_class(model, reference_period=reference_period)

    # Header cross-checks: the table is ground truth, so warn only.
    if table.shape[0] != meta["nknot"]:
        warnings.warn(f"header says {meta['nknot']} knots, "
                      f"file has {table.shape[0]}")
    r = table[:, 0]
    for key in ("nic", "noc"):
        i = meta[key]          # 1-based index of the knot below the boundary
        if not (1 <= i < r.size and r[i - 1] == r[i]):
            warnings.warn(f"header {key}={i} does not sit on a repeated radius")
    return model


@register("deck_reader", "mineos_deck")
def read_mineos_deck(path: str | os.PathLike, *, kind: str | Callable = "cubic",
                     reference_period=None):
    """Read a mineos/PREM deck: three header lines, then knots with columns
    r rho vpv vsv qkappa qmu vph vsh eta (SI, radius in metres).

    The five transversely isotropic moduli and the elastic tensor are
    attached on read.  Returns a `ViscoelasticModel` under constant Q at
    the reference period, which is the header's `tref` unless
    `reference_period` is given.
    """
    return _read_mineos_style(path, MINEOS_COLUMNS, kind=kind,
                              reference_period=reference_period,
                              period_from_header=True)


read_mineos_deck.columns = MINEOS_COLUMNS


@register("deck_reader", "isotropic_deck")
def read_isotropic_deck(path: str | os.PathLike, *,
                        kind: str | Callable = "cubic", reference_period=None):
    """Read an isotropic deck: three header lines, then r rho vp vs qkappa qmu.

    Returns an `ElasticModel` unless `reference_period` is given, in
    which case a `ViscoelasticModel` under constant Q: the deck has Q
    columns but its header period is not relied on, and the type does
    not pretend to a calibration it was not given.

    The layout of mfemElasticity's prem.nocrust.  Isotropy is the
    degenerate TI case vph = vpv = vp, vsh = vsv = vs, eta = 1, so the
    five moduli attached on read satisfy A = C, L = N and F = A - 2L
    exactly.  The elastic tensor, though, reports Symmetry.ISOTROPIC and
    stores (kappa, mu): what the deck says is what the model knows, and
    as_symmetry(Symmetry.VTI) widens it back to the five moduli on
    request.
    """
    return _read_mineos_style(path, ISOTROPIC_COLUMNS, kind=kind,
                              reference_period=reference_period,
                              period_from_header=False)


read_isotropic_deck.columns = ISOTROPIC_COLUMNS
