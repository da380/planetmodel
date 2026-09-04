"""radial.py -- fields that are functions of radius alone, layer by layer.

A layer function is any object callable on arrays over its interval
that also provides .derivative(nu=1) and .integrate(a, b).  scipy's
piecewise-polynomial families (CubicSpline, PchipInterpolator,
make_interp_spline results) all qualify, as do the exact PREM-style
polynomials built by polynomial_layer().

RadialField binds one such function per layer of a shared Skeleton.
Evaluation is deliberately layer-indexed -- field[i](r) -- because at a
discontinuity the two one-sided values are both meaningful; use
Skeleton.locate to pick a side explicitly.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import PchipInterpolator, PPoly, make_interp_spline

from ..character import SCALAR, Character
from ..skeleton import Skeleton
from ..units import Dimensions
from .composite import FieldBase
from .layer_function import as_layer_function

__all__ = ["constant_field", "make_fitter", "polynomial_layer", "RadialField",
           "derived_field"]


def make_fitter(*, kind: str | Callable = "cubic") -> Callable:
    """Return fit(x, y) -> layer function for tabulated knots.

    kind is "cubic", "pchip", "linear", or any callable(x, y) returning
    an object satisfying the layer-function protocol (callable on
    arrays, .derivative(nu), .integrate(a, b)).  Spline degree degrades
    automatically in layers with few knots (k = min(kmax, nknots - 1)).
    """
    if callable(kind):
        return kind
    key = str(kind).lower()
    if key == "pchip":
        return lambda x, y: PchipInterpolator(x, y)
    try:
        kmax = {"linear": 1, "cubic": 3}[key]
    except KeyError:
        raise ValueError(f"unknown interpolant kind {kind!r}") from None

    def fit(x, y):
        """Interpolate one layer's knots with a degree-degrading B-spline."""
        deg = min(kmax, len(x) - 1)
        if deg < 1:
            raise ValueError("a layer needs at least two knots")
        return make_interp_spline(x, y, k=deg)

    return fit


def polynomial_layer(coeffs: Sequence[float] | np.ndarray,
                     interval: tuple[float, float],
                     *, scale: float = 1.0) -> PPoly:
    """Exact PREM-style polynomial layer: sum_k c_k * (r/scale)**k.

    Returned as a one-piece PPoly in r, so evaluation, .derivative and
    .integrate are exact and taken with respect to r (the chain rule
    for the normalized coordinate x = r/scale is absorbed here).
    """
    rl, rr = map(float, interval)
    if not rr > rl:
        raise ValueError("interval must have positive width")
    p = Polynomial(np.asarray(coeffs, dtype=float))
    q = p(Polynomial([0.0, 1.0 / scale]))   # coefficients in r
    s = q(Polynomial([rl, 1.0]))            # coefficients in (r - rl)
    c = np.asarray(s.coef, dtype=float)[::-1].reshape(-1, 1)
    return PPoly(c, np.array([rl, rr]))


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

class RadialField(FieldBase):
    """Layer functions on a Skeleton: the 1D workhorse.

    A field belongs to one layer, and a RadialField on a many-layer
    skeleton is the *view* a body assembles from the single-layer
    fields its layers hold.  Both are this one class: a single-layer
    RadialField holds one function on one interval, and a many-layer
    one holds one function per layer, `None` where a layer has no such
    field.  The layers with a function are the `domain`.

    field[i]             -> the layer-i piece, a single-layer RadialField
    field[i](r)          -> its values (a single-layer field is callable)
    field[i].function    -> the bare layer function
    field.domain         -> the layers that have a function
    field.derivative()   -> new RadialField of d/dr, layer by layer
    field.integrate(a,b) -> definite integral, stitched across layers
    field.plot(...)      -> per-layer segments, radius vertical by
                            default, one-sided values linked at
                            discontinuities
    """

    def __init__(self, skeleton: Skeleton, functions: Iterable,
                 *, name: str | None = None,
                 character: Character = SCALAR,
                 dimensions=None) -> None:
        """Bind one layer function per skeleton layer, in order.

        An entry may be None, meaning the field is not defined on that
        layer: evaluation there is refused by name.  Functions are
        adapted to the layer-function protocol as they come in, so a
        plain callable gains numerical `derivative` and `integrate`
        while an exact polynomial keeps its own.

        character is the (rank, weight) transformation law of the field;
        it defaults to SCALAR, which is right for the invariant
        rheological parameters and wrong for density -- pass DENSITY
        explicitly there, as the deck readers do.

        dimensions is the field's physical dimensions (a
        units.Dimensions), independent of character: Q_kappa and a
        relaxation time share Character(0, 0) and differ in dimensions.
        None means undeclared, which rescaled() refuses by name --
        declare Dimensions.DIMENSIONLESS when that is what you mean.
        """
        functions = tuple(None if f is None else as_layer_function(f)
                          for f in functions)
        if len(functions) != skeleton.nlayers:
            raise ValueError(
                f"got {len(functions)} functions for {skeleton.nlayers} layers")
        if not any(f is not None for f in functions):
            raise ValueError("a RadialField needs a function on at least "
                             "one layer")
        self._sk = skeleton
        self._fs = functions
        self._pieces: dict[int, RadialField] = {}
        self.name = name
        self.character = character
        self.dimensions = dimensions

    @property
    def skeleton(self) -> Skeleton:
        """The Skeleton this field lives on."""
        return self._sk

    @property
    def functions(self) -> tuple:
        """One layer function per layer, None where the field is absent."""
        return self._fs

    @property
    def function(self):
        """The layer function of a single-layer field."""
        if self._sk.nlayers != 1:
            raise ValueError(
                f"{self!r} spans {self._sk.nlayers} layers; index a layer "
                "first: field[i].function")
        return self._fs[0]

    @property
    def domain(self) -> tuple[int, ...]:
        """The layers on which the field is defined."""
        return tuple(i for i, f in enumerate(self._fs) if f is not None)

    def __len__(self) -> int:
        """The number of layers of the skeleton."""
        return len(self._fs)

    def __getitem__(self, i: int) -> RadialField:
        """The layer-i piece: a single-layer RadialField (negatives allowed).

        A single-layer field is its own piece.  A layer outside the
        domain raises ValueError naming the field and the layer.
        """
        i = self._sk.layer_index(i)
        if self._sk.nlayers == 1:
            return self
        if i not in self._pieces:
            if self._fs[i] is None:
                raise ValueError(
                    f"{self!r} is not defined on layer {i}: its domain is "
                    f"{self.domain}")
            self._pieces[i] = RadialField(
                Skeleton(self._sk.interval(i)), (self._fs[i],),
                name=self.name, character=self.character,
                dimensions=self.dimensions)
        return self._pieces[i]

    def __iter__(self):
        """Iterate over the pieces on the domain, centre outwards."""
        return (self[i] for i in self.domain)

    def restricted(self, layer) -> RadialField:
        """The piece on one layer: for a RadialField, `field[layer]`."""
        return self[layer]

    def on_interval(self, lo: float, hi: float) -> RadialField:
        """A single-layer field's function, re-stated on another interval.

        Clipping (the interval inside this one) and extrapolation (the
        interval beyond it) are the same operation: the layer function
        is what it is, and the skeleton says where it is asked.  Used
        by truncation, refinement and `extended(fields="extrapolate")`.
        """
        return RadialField(Skeleton([lo, hi]), (self.function,),
                           name=self.name, character=self.character,
                           dimensions=self.dimensions)

    def rescaled(self, convert, old, new):
        """The layer functions re-expressed in the new scales, exactly.

        A piecewise polynomial converts by one multiply per coefficient,
        so an exact model stays exact through non-dimensionalisation and
        back; anything else is wrapped pointwise.  A field with no
        declared dimensions is refused by name, since silently leaving
        a modulus unscaled produces a wrong answer that looks plausible.
        """
        from .layer_function import rescale_layer_function
        dims = self.dimensions
        if dims is None:
            raise ValueError(
                f"cannot rescale field {self.name!r}: it declares no "
                "dimensions. Set dimensions=Dimensions.DIMENSIONLESS if it "
                "genuinely has none, or its actual Dimensions otherwise.")
        k = old.length / new.length
        vr = old.factor(dims) / new.factor(dims)
        funcs = [None if f is None else rescale_layer_function(f, k, vr)
                 for f in self._fs]
        return RadialField(Skeleton(self._sk.boundaries * k), funcs,
                           name=self.name, character=self.character,
                           dimensions=dims)

    @classmethod
    def assembled(cls, skeleton: Skeleton, pieces, *, name=None):
        """One RadialField on `skeleton` from single-layer RadialFields.

        A piece matching a layer supplies its function; several pieces
        abutting inside one layer are merged into a MergedLayerFunction;
        a layer with no piece gets None.  Anything else is declined
        (NotImplemented) and assembled generically.
        """
        from .layer_function import MergedLayerFunction
        funcs: list = []
        for i in range(skeleton.nlayers):
            lo, hi = skeleton.interval(i)
            tol = skeleton.tolerance
            inside = sorted((p for p in pieces
                             if _within(p.skeleton, lo, hi, tol)),
                            key=lambda p: p.skeleton.boundaries[0])
            if not inside:
                funcs.append(None)
            elif len(inside) == 1 and skeleton.spans(
                    *inside[0].skeleton.boundaries, layer=i):
                funcs.append(inside[0].function)
            elif _abut(inside, lo, hi, tol):
                edges = [p.skeleton.boundaries[0] for p in inside] + [hi]
                funcs.append(MergedLayerFunction(
                    [p.function for p in inside], edges))
            else:
                return NotImplemented
        first = pieces[0]
        out = cls(skeleton, funcs, name=name if name is not None else first.name,
                  character=first.character,
                  dimensions=getattr(first, "dimensions", None))
        # A piece that matched a layer whole is that layer's piece: hand
        # the same object back from out[i], so a body's view indexes to
        # what its layer holds.
        for i in range(skeleton.nlayers):
            for p in pieces:
                if (skeleton.spans(*p.skeleton.boundaries, layer=i)
                        and p.name == out.name):
                    out._pieces[i] = p
        return out

    _assembled = assembled

    @property
    def is_radial(self) -> bool:
        """A function of radius alone: always true here, by construction."""
        return True

    def _refuse_gap(self, idx: np.ndarray) -> None:
        """Raise if any resolved layer index has no function."""
        for i in np.unique(idx):
            if self._fs[int(i)] is None:
                raise ValueError(
                    f"{self!r} is not defined on layer {int(i)}: its domain "
                    f"is {self.domain}, and a radius in another layer has no "
                    "value here")

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical"):
        """Values at radius r; angles are accepted and ignored.

        With layer=None each radius is resolved against the skeleton and
        `side` decides which layer owns a radius sitting exactly on an
        interior boundary: "upper" takes the layer above, "lower" the one
        below.  Both are correct answers there -- the field is genuinely
        two-valued -- so pass layer= explicitly when it matters.

        r broadcasts against theta and phi; the result has the broadcast
        shape.  Radii outside the skeleton, or in a layer the field is
        not defined on, raise ValueError.

        The values are scalars, so the two frames coincide and `frame`
        changes nothing -- but it is still validated rather than
        ignored, since a caller asking for a frame this field has never
        heard of has made a mistake worth hearing about.
        """
        if side not in ("upper", "lower"):
            raise ValueError("side must be 'upper' or 'lower'")
        if frame not in ("spherical", "cartesian"):
            raise ValueError(
                f"unknown frame {frame!r}: a RadialField is scalar-valued, so "
                "its 'spherical' and 'cartesian' components are the same "
                "numbers, and no other frame is defined")
        r = np.asarray(r, dtype=float)
        if theta is not None or phi is not None:
            shape = np.broadcast(r, np.asarray(0.0 if theta is None else theta),
                                 np.asarray(0.0 if phi is None else phi)).shape
            r = np.broadcast_to(r, shape)
        flat = np.atleast_1d(r).ravel()

        b = self._sk.boundaries
        if flat.size and (flat.min() < b[0] or flat.max() > b[-1]):
            bad = flat[(flat < b[0]) | (flat > b[-1])][0]
            raise ValueError(f"radius {bad} outside [{b[0]}, {b[-1]}]")

        out = np.empty(flat.shape, dtype=float)
        if layer is not None:
            i = self._sk.layer_index(layer)
            lo, hi = self._sk.interval(i)
            tol = self._sk.tolerance
            if flat.size and (flat.min() < lo - tol or flat.max() > hi + tol):
                bad = flat[(flat < lo - tol) | (flat > hi + tol)][0]
                raise ValueError(
                    f"radius {bad:.6g} is not in layer {i} [{lo:.6g}, "
                    f"{hi:.6g}]: layer= names the side at a boundary, it "
                    "does not extrapolate that layer's function")
            self._refuse_gap(np.array([i]))
            out[:] = self._fs[i](flat)
        else:
            # searchsorted - 1 is the containing layer.  On a boundary the
            # two sides differ: 'right' counts the boundary as already
            # passed, giving the layer ABOVE it, and 'left' the one below.
            idx = np.searchsorted(b, flat,
                                  side="right" if side == "upper" else "left") - 1
            np.clip(idx, 0, self._sk.nlayers - 1, out=idx)
            self._refuse_gap(idx)
            for i in np.unique(idx):
                m = idx == i
                out[m] = self._fs[i](flat[m])
        return out.reshape(r.shape) if r.shape else out[0]

    def derivative(self, nu: int = 1) -> RadialField:
        """The nu-th radial derivative, taken layer by layer.

        d/dr divides the physical dimensions by length once per order;
        the character is unchanged, since differentiating in the
        reference coordinate does not alter how a field transforms.
        """
        dims = self.dimensions
        if dims is not None:
            from ..units import Dimensions
            dims = dims / Dimensions.LENGTH ** nu
        return RadialField(self._sk,
                           tuple(None if f is None else f.derivative(nu)
                                 for f in self._fs),
                           name=None if self.name is None
                           else self.name + "'" * nu,
                           character=self.character, dimensions=dims)

    def integrate(self, a: float, b: float) -> float:
        """Integral of the field over [a, b], crossing layers as needed.

        An interval reaching into a layer the field is not defined on is
        refused: there is nothing there to integrate.
        """
        a, b = float(a), float(b)
        if a == b:
            return 0.0
        sign = 1.0
        if a > b:
            a, b, sign = b, a, -1.0
        bnd = self._sk.boundaries
        if a < bnd[0] or b > bnd[-1]:
            raise ValueError(f"[{a}, {b}] outside model [{bnd[0]}, {bnd[-1]}]")
        total = 0.0
        for i in range(self._sk.nlayers):
            lo, hi = max(a, bnd[i]), min(b, bnd[i + 1])
            if hi > lo:
                if self._fs[i] is None:
                    raise ValueError(
                        f"[{a}, {b}] reaches into layer {i}, where {self!r} "
                        "is not defined")
                total += float(self._fs[i].integrate(lo, hi))
        return sign * total

    def plot(self, *, ax=None, n: int = 200, radial_axis: str = "y",
             connect: bool = True, show_boundaries: bool = False,
             label: str | None = None, **kwargs):
        """Draw the field, one curve segment per layer of its domain.

        By default radius runs up the vertical axis (radial_axis="y",
        the usual convention for Earth-model profiles); pass
        radial_axis="x" for the transposed layout.  Segments share one
        colour and a single legend entry.  With connect=True (default)
        the two one-sided values at each interior discontinuity are
        linked by a straight segment perpendicular to the radial axis
        (horizontal in the default orientation); continuous crossings
        yield degenerate, invisible links.  Returns the axes.
        """
        import matplotlib.pyplot as plt
        if radial_axis not in ("x", "y"):
            raise ValueError("radial_axis must be 'x' or 'y'")
        if ax is None:
            _, ax = plt.subplots()

        def draw(r, v, **kw):
            """Plot r against v in the orientation chosen by radial_axis."""
            xy = (r, v) if radial_axis == "x" else (v, r)
            return ax.plot(*xy, **kw)

        kw = dict(kwargs)
        lbl = self.name if label is None else label
        first = True
        for i in self.domain:
            lo, hi = self._sk.interval(i)
            r = np.linspace(lo, hi, n)
            (line,) = draw(r, self._fs[i](r), label=lbl if first else None, **kw)
            kw["color"] = line.get_color()
            first = False
        if connect:
            for j in range(1, self._sk.nlayers):
                if self._fs[j - 1] is None or self._fs[j] is None:
                    continue
                b = float(self._sk.boundaries[j])
                v = [float(self._fs[j - 1](b)), float(self._fs[j](b))]
                draw([b, b], v, **kw)
        if show_boundaries:
            rule = ax.axvline if radial_axis == "x" else ax.axhline
            for bb in self._sk.inner_boundaries:
                rule(bb, color="0.85", lw=0.7, ls=":", zorder=0)
        return ax

    def __repr__(self) -> str:
        """Compact summary: layer count, gaps if any, and optional name."""
        nm = f" {self.name!r}" if self.name else ""
        n = len(self)
        gaps = n - len(self.domain)
        g = f", {gaps} without" if gaps else ""
        return f"RadialField({n} layers{g}{nm})"


def _within(sk: Skeleton, lo: float, hi: float, tol: float) -> bool:
    """Whether a one-layer skeleton lies inside [lo, hi]."""
    b = sk.boundaries
    return sk.nlayers == 1 and b[0] >= lo - tol and b[-1] <= hi + tol


def _abut(pieces, lo: float, hi: float, tol: float) -> bool:
    """Whether sorted one-layer pieces tile [lo, hi] exactly."""
    edge = lo
    for p in pieces:
        b = p.skeleton.boundaries
        if abs(b[0] - edge) > tol:
            return False
        edge = b[-1]
    return abs(edge - hi) <= tol


def constant_field(skeleton: Skeleton, value: float, *, name: str | None = None,
                   character: Character = SCALAR,
                   dimensions: Dimensions | None = None) -> RadialField:
    """A field equal to `value` on every layer of `skeleton`.

    The one-line way to give a layer a uniform property -- a viscosity, a
    porosity, a modulus to compare against -- with the character and
    dimensions it should carry; `constant_field(Skeleton(layer.interval),
    v)` is a single-layer field ready for `Layer.with_field`.
    """
    v = float(value)
    return RadialField(skeleton, [lambda r, v=v: np.full_like(
        np.asarray(r, dtype=float), v)] * skeleton.nlayers,
        name=name, character=character, dimensions=dimensions)


def derived_field(skeleton: Skeleton, fn: Callable,
                  sources: Sequence[RadialField], *, n: int = 65,
                  kind: str | Callable = "cubic",
                  name: str | None = None, dimensions=None) -> RadialField:
    """Pointwise-derived field: fn(r, *source_values), sampled and refit.

    Each layer is sampled on n points and re-interpolated with the
    given fitter, so the result is an honest (re-interpolated)
    approximation of the composition rather than an exact algebraic
    combination.  All sources must share the skeleton; the result is
    defined on the layers where every source is.
    """
    sources = tuple(sources)
    for f in sources:
        if f.skeleton != skeleton:
            raise ValueError("all source fields must share the model skeleton")
    fit = make_fitter(kind=kind)
    funcs = []
    for i in range(skeleton.nlayers):
        parts = [f.functions[i] for f in sources]
        if any(p is None for p in parts):
            funcs.append(None)
            continue
        lo, hi = skeleton.interval(i)
        r = np.linspace(lo, hi, n)
        funcs.append(fit(r, fn(r, *(p(r) for p in parts))))
    if all(f is None for f in funcs):
        raise ValueError("the sources share no layer, so there is nothing "
                         "to derive")
    return RadialField(skeleton, tuple(funcs), name=name, dimensions=dimensions)
