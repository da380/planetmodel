"""Fields belong to layers; a body-wide field is a view with a domain.

A RadialField on a many-layer skeleton is one layer function per layer,
and an entry may be None where a layer has no function.  Attached to a
body it is split into single-layer pieces, one per layer of its domain,
and `body["rho"]` is the view assembled back from those pieces.  The
view is defined on exactly the layers that hold a piece -- its
`domain` -- and refuses any other radius by name rather than filling
it.  `restricted(i)` and `field[i]` give the piece; `on_interval`
restates a single-layer field on another interval where its rule of
evaluation extends.

This script builds a density with a gap, attaches it, and shows the
view, the refusal, the pieces and the two sides of a discontinuity.
"""
import numpy as np

from planetmodel import DENSITY, Dimensions, RadialField, ReferenceBody, Skeleton

sk = Skeleton([0.0, 1.0, 2.0, 3.0])

# One function per layer; None where the middle layer has no density.
rho = RadialField(sk, [lambda r: 10.0 + 0.0 * r, None, lambda r: 4.0 + r],
                  name="rho", character=DENSITY, dimensions=Dimensions.DENSITY)
assert rho.domain == (0, 2)
assert rho.is_radial

# -- pieces ------------------------------------------------------------------
piece = rho[2]                                    # a single-layer RadialField
assert piece.skeleton == Skeleton([2.0, 3.0])
assert piece.domain == (0,)                       # its one layer
assert np.isclose(piece(2.5), 6.5)                # a single-layer field is callable
assert rho.restricted(2).skeleton == piece.skeleton

# -- the body's view ---------------------------------------------------------
body = ReferenceBody.from_fields(sk, {"rho": rho})
assert body.layers_with("rho") == (0, 2)
assert "rho" not in body.layers[1]                # the gap is a layer with no field

view = body["rho"]
assert view.domain == (0, 2)
assert np.isclose(view.evaluate(0.5), 10.0) and np.isclose(view.evaluate(2.5), 6.5)
try:
    view.evaluate(1.5)                            # inside the gap
except ValueError as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("a radius outside the domain must be refused")

# The view's piece on a layer is the layer's own field object.
assert view[2] is body.layers[2]["rho"]

# -- sides of a discontinuity -------------------------------------------------
# At r = 2 the density jumps from nothing to 6.  `layer=` chooses the side;
# `side=` breaks the tie when only the radius is given.
assert np.isclose(view.evaluate(2.0, layer=2), 6.0)
assert np.isclose(view.evaluate(2.0, side="upper"), 6.0)
try:
    view.evaluate(2.5, layer=0)                   # a radius outside layer 0
except ValueError as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("layer= must not extrapolate another layer's rule")

# -- on_interval -------------------------------------------------------------
# A layer function is a rule of evaluation, so it can be restated on a
# neighbouring interval; the polynomial 4 + r continues.
wider = piece.on_interval(1.5, 3.0)
assert wider.skeleton == Skeleton([1.5, 3.0]) and np.isclose(wider(1.5), 5.5)

# -- derivatives and integrals, where the layer functions supply them --------
assert np.isclose(view.derivative().evaluate(2.5), 1.0)
assert np.isclose(view.integrate(2.0, 3.0), 4.0 + 2.5)   # int_2^3 (4 + r) dr
try:
    view.integrate(0.5, 2.5)                      # across the gap
except ValueError as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("an integral across a gap must be refused")

print("ok: views carry a domain and refuse outside it by name")
