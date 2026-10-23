"""Executable contracts for the protocols of planetmodel.

Each `check_*` takes a candidate object, exercises what its protocol
promises, and raises AssertionError naming the first violation.  The
shipped implementations are held to these functions in the test suite;
an implementation written elsewhere is checked by the same call.  The
imports of the rest of the package happen inside the functions.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .datasets import Dataset, RegionalGrid, Strata, Surface
    from .displacement import RadialDisplacement
    from .fields import Field
    from .geometry import Geometry
    from .layerfunction import LayerFunction
    from .mapping import Mapping
    from .mesh3d.spec import Refinement
    from .model import Model
    from .modelfile import Surfaces
    from .randomfield.basis import SpectralBasis, SphericalBasis
    from .sampling import AngularGrid, Sample
    from .skeleton import Skeleton
    from .surfacefile import SurfaceVariable
    from .units import Scales

__all__ = ["check_displacement", "check_mapping", "check_geometry",
           "check_layer_function", "check_field", "check_model",
           "check_refinement", "check_sample", "check_spectral_basis",
           "check_spherical_basis", "check_model_file", "check_dataset",
           "check_surface", "check_strata", "check_regional",
           "check_surface_file"]


def _rel(a: ArrayLike, b: ArrayLike, *, rtol: float, floor: float = 1.0) -> bool:
    """Whether a and b agree to rtol relative to max(floor, |b|)."""
    return np.allclose(a, b, rtol=0.0, atol=rtol * np.maximum(floor, np.abs(b)))


def check_displacement(h: RadialDisplacement, skeleton: Skeleton, *,
                       n: int = 33) -> None:
    """Hold a RadialDisplacement to its protocol over a skeleton's span.

    Checks finite values of the broadcast shape; single-valuedness on the
    polar axis; continuity across every declared knot inside the
    skeleton; `radial_derivative`, when present,
    against a central difference away from the knots and free of
    undeclared kinks; the shape of `angular_gradient`; and, when present,
    `scaled(k)` against k h(r / k, theta, phi), with its knots and
    derivatives.  Every tolerance is relative to the skeleton's span or
    to the value.
    """
    lo, hi = float(skeleton.boundaries[0]), float(skeleton.boundaries[-1])
    span = hi - lo
    r = np.linspace(lo + 1e-3 * span, hi, n)[:, None, None]
    theta = np.linspace(0.05, np.pi - 0.05, 5)[None, :, None]
    phi = np.linspace(-np.pi, np.pi, 5, endpoint=False)[None, None, :]
    values = np.asarray(h(r, theta, phi), dtype=float)
    shape = np.broadcast(r, theta, phi).shape
    assert values.shape == shape, f"h returned shape {values.shape}, expected {shape}"
    assert np.all(np.isfinite(values)), "h returned non-finite values"
    scale = max(float(np.max(np.abs(values))), 1e-12 * span)

    # single-valued on the polar axis: a ring of longitudes just off each
    # pole must agree
    ring = np.linspace(-np.pi, np.pi, 8, endpoint=False)[None, None, :]
    for pole in (1e-7, np.pi - 1e-7):
        on_ring = np.asarray(h(r, pole, ring), dtype=float)
        spread = np.max(on_ring, axis=-1) - np.min(on_ring, axis=-1)
        assert np.all(spread <= 1e-4 * scale), (
            f"h is not single-valued at the pole theta = {pole:.3g}: it varies "
            "with phi there")

    knots = tuple(sorted(float(k) for k in getattr(h, "knots", ())))
    eps = 1e-6 * span
    for k in knots:
        if not lo < k < hi:
            continue
        below = np.asarray(h(k - eps, theta, phi), dtype=float)
        above = np.asarray(h(k + eps, theta, phi), dtype=float)
        assert np.allclose(below, above, atol=1e-4 * scale, rtol=0.0), (
            f"h is discontinuous across its knot at r = {k:g}")

    if hasattr(h, "radial_derivative"):
        probes = np.linspace(lo + 1e-3 * span, hi - 1e-3 * span, 4 * n + 1)
        away = np.ones(probes.shape, dtype=bool)
        for k in knots:
            away &= np.abs(probes - k) > 1e-3 * span
        rp = probes[away][:, None, None]
        step = 1e-5 * span
        want = (np.asarray(h(rp + step, theta, phi), dtype=float)
                - np.asarray(h(rp - step, theta, phi), dtype=float)) / (2.0 * step)
        got = np.asarray(h.radial_derivative(rp, theta, phi), dtype=float)
        assert got.shape == want.shape, "radial_derivative has the wrong shape"
        # dh/dr is dimensionless, so a floor of 1 is a floor of 100 % strain
        dscale = max(1.0, float(np.max(np.abs(want))))
        assert np.allclose(got, want, atol=1e-3 * dscale, rtol=0.0), (
            "radial_derivative disagrees with a central difference of h")
        # between two probes in one knot span the derivative may change by
        # what a smooth function changes over that spacing, judged by the
        # typical change; an outlier is an undeclared kink
        flat = got.reshape(len(rp), -1)
        edges = np.searchsorted(np.asarray(knots, dtype=float), probes[away])
        same = edges[1:] == edges[:-1]
        jumps = np.max(np.abs(flat[1:] - flat[:-1]), axis=1)[same]
        if jumps.size:
            typical = float(np.median(jumps))
            limit = 0.05 * float(np.max(np.abs(want))) + 10.0 * typical + 1e-12
            assert np.all(jumps <= limit), (
                "radial_derivative jumps between neighbouring probes at a "
                "radius not declared as a knot")

    if hasattr(h, "angular_gradient"):
        gt, gp = h.angular_gradient(r, theta, phi)
        assert np.shape(gt) == shape and np.shape(gp) == shape, (
            "angular_gradient components have the wrong shape")

    if callable(getattr(h, "scaled", None)):
        _check_scaled_displacement(h, r, theta, phi, knots, span=span, scale=scale)


def _check_scaled_displacement(h: RadialDisplacement, r: np.ndarray,
                               theta: np.ndarray, phi: np.ndarray,
                               knots: tuple[float, ...], *, span: float,
                               scale: float) -> None:
    """`h.scaled(k)` is k h(r / k, theta, phi), its knots k times h's and
    its derivatives h's read at r / k, the angular ones times k; radii
    are kept clear of the knots, where dh/dr is one-sided."""
    k = 2.5
    hk = h.scaled(k)
    away = np.all([np.abs(r - x) > 1e-3 * span for x in knots], axis=0) \
        if knots else np.ones(r.shape, dtype=bool)
    rk = r[away.ravel()].reshape(-1, 1, 1)
    want = np.asarray(h(rk, theta, phi), dtype=float)
    got = np.asarray(hk(k * rk, theta, phi), dtype=float)
    assert np.allclose(got, k * want, rtol=0.0, atol=1e-12 * k * scale), (
        "scaled(k) is not k h(r / k, theta, phi)")
    got_knots = sorted(float(x) for x in getattr(hk, "knots", ()))
    assert np.allclose(got_knots, [k * x for x in knots], rtol=1e-12, atol=0.0), (
        "scaled(k) does not carry the knots multiplied by k")
    if hasattr(h, "radial_derivative") and hasattr(hk, "radial_derivative"):
        want = np.asarray(h.radial_derivative(rk, theta, phi), dtype=float)
        got = np.asarray(hk.radial_derivative(k * rk, theta, phi), dtype=float)
        assert _rel(got, want, rtol=1e-9), (
            "scaled(k).radial_derivative is not dh/dr at r / k")
    if hasattr(h, "angular_gradient") and hasattr(hk, "angular_gradient"):
        for got, want in zip(hk.angular_gradient(k * rk, theta, phi),
                             h.angular_gradient(rk, theta, phi)):
            want = k * np.asarray(want, dtype=float)
            assert _rel(np.asarray(got, dtype=float), want, rtol=1e-9,
                        floor=1e-12 * k * scale), (
                "scaled(k).angular_gradient is not k times h's at r / k")


def check_mapping(m: Mapping, points: ArrayLike, *, rtol: float = 1e-6,
                  step: float | None = None) -> None:
    """Hold a Mapping to its protocol on the given Cartesian points.

    Checks the shapes; `deformation_gradient` against a central
    difference of `m` with `F[i, j] = d m_i / d X_j`; `jacobian` against
    det F; and, when present, `displacement` against m(X) - X,
    `inverse` as a round trip, and `scaled(k)` against k m(X / k) with
    the same F and J.  The difference step is 1e-6 of each
    point's radius unless `step` gives one step for every point; with the
    default, a point that disagrees is differenced again at a hundredth
    of its step, since a mapping once but not twice differentiable there
    (a bicubic relief at a pole) has a difference converging only
    linearly in the step.
    """
    X = np.asarray(points, dtype=float)
    assert X.shape[-1] == 3, "points must have shape (..., 3)"
    scale = max(float(np.max(np.abs(X))), 1.0)
    x = np.asarray(m(X), dtype=float)
    assert x.shape == X.shape, f"m(X) has shape {x.shape}, expected {X.shape}"
    assert np.all(np.isfinite(x)), "m(X) is not finite"

    F = np.asarray(m.deformation_gradient(X), dtype=float)
    assert F.shape == X.shape[:-1] + (3, 3), f"F has shape {F.shape}"
    assert np.all(np.isfinite(F)), "F is not finite"
    if step is not None:
        hs = np.full(X.shape[:-1], float(step))
    else:
        # relative to each point's own radius: a mapping whose gradient
        # depends on direction at the origin is smooth on every ray, and a
        # step much smaller than the radius stays on that ray
        hs = 1e-6 * np.maximum(np.linalg.norm(X, axis=-1), 1e-12 * scale)

    def central(X: np.ndarray, hs: np.ndarray) -> np.ndarray:
        Fd = np.empty(X.shape + (3,))
        for j in range(3):
            e = np.zeros_like(X)
            e[..., j] = hs
            Fd[..., :, j] = ((np.asarray(m(X + e), dtype=float)
                              - np.asarray(m(X - e), dtype=float))
                             / (2.0 * hs[..., None]))
        return Fd

    Fd = central(X, hs)
    off = ~np.all(np.abs(F - Fd) <= rtol * np.maximum(1.0, np.abs(Fd)), axis=(-2, -1))
    if step is None and np.any(off):
        # once but not twice differentiable at a point (a bicubic relief at
        # a pole): the difference converges there only linearly in the step
        Fd[off] = central(X[off], hs[off] / 100.0)
    assert _rel(F, Fd, rtol=rtol), (
        "deformation_gradient disagrees with a central difference of m")

    J = np.asarray(m.jacobian(X), dtype=float)
    assert J.shape == X.shape[:-1], f"J has shape {J.shape}"
    assert np.allclose(J, np.linalg.det(F), rtol=1e-9, atol=1e-9), (
        "jacobian is not det F")

    if hasattr(m, "displacement"):
        u = np.asarray(m.displacement(X), dtype=float)
        assert np.allclose(u, x - X, atol=1e-12 * scale, rtol=0.0), (
            "displacement is not m(X) - X")

    if hasattr(m, "inverse"):
        try:
            back = np.asarray(m.inverse(x), dtype=float)
        except NotImplementedError:
            back = None
        if back is not None:
            assert np.allclose(back, X, atol=1e-6 * scale, rtol=0.0), (
                "inverse(m(X)) does not return X")

    if callable(getattr(m, "scaled", None)):
        k = 2.5
        mk = m.scaled(k)
        assert np.allclose(np.asarray(mk(k * X), dtype=float), k * x, rtol=0.0,
                           atol=1e-12 * k * scale), "scaled(k) is not k m(X / k)"
        assert _rel(np.asarray(mk.deformation_gradient(k * X), dtype=float), F,
                    rtol=1e-9), "scaled(k) does not keep F"
        assert np.allclose(np.asarray(mk.jacobian(k * X), dtype=float), J,
                           rtol=1e-9, atol=1e-9), "scaled(k) does not keep J"


def check_geometry(geometry: Geometry) -> None:
    """Hold a Geometry to its invariants and its numbering.

    Checks that the invariants hold (by reconstructing the geometry with
    checks on), that names are unique, that the interfaces sit on the
    skeleton's boundaries with consistent `between`, that `scaled(k)`
    round-trips, and that the mapping passes `check_mapping` on the
    validity lattice's angles, at radii inside every span between the
    boundaries and the declared knots.
    """
    from .frames import cartesian_points
    from .geometry import Geometry
    from .mapping import validity_lattice

    sk = geometry.skeleton
    Geometry(sk, mapping=geometry.mapping,
             layer_names=[lay.name for lay in geometry.layers],
             interface_names=[f.name for f in geometry.interfaces],
             rtol=geometry.rtol, check=True)

    faces = geometry.interfaces
    b = sk.boundaries
    first = 0 if sk.is_hollow else 1
    assert len(faces) == b.size - first, "wrong number of interfaces"
    for k, f in enumerate(faces):
        assert f.index == k
        assert f.radius == float(b[first + k]), "interface off a boundary"
        below, above = f.between
        assert below == first + k - 1, "between[0] inconsistent"
        assert above == (first + k if first + k < sk.nlayers else -1), (
            "between[1] inconsistent")
    for i, lay in enumerate(geometry.layers):
        assert lay.index == i and lay.interval == sk.interval(i)

    k = 2.5
    twice = geometry.scaled(k).scaled(1.0 / k)
    assert np.allclose(twice.skeleton.boundaries, b, rtol=1e-12), (
        "scaled(k).scaled(1/k) does not restore the skeleton")
    # points well inside every span between the boundaries and the
    # declared knots, where the mapping's gradient may jump
    _, theta, phi = validity_lattice(sk, n_theta=5, n_phi=4)
    tol = geometry.rtol * sk.span
    cuts = np.unique(np.concatenate([b, np.asarray(geometry.knots(), dtype=float)]))
    cuts = cuts[np.concatenate([[True], np.diff(cuts) > tol])]
    r = np.concatenate([lo + (hi - lo) * np.array([0.25, 0.5, 0.75])
                        for lo, hi in zip(cuts[:-1], cuts[1:])])[:, None, None]
    X = cartesian_points(r, theta, phi)
    assert np.allclose(twice.mapping(X), geometry.mapping(X), rtol=1e-10,
                       atol=1e-10 * max(1.0, float(sk.span))), (
        "scaled(k).scaled(1/k) does not restore the mapping")

    check_mapping(geometry.mapping, X.reshape(-1, 3))


def check_layer_function(fn: LayerFunction, *, n: int = 17,
                         rtol: float = 1e-6) -> None:
    """Hold a LayerFunction to its protocol on its own interval.

    Checks finite float64 (or complex128) values of the argument's shape; `derivative`
    against a central difference; `integrate` against a fine
    trapezoid rule and its sign convention; `on_interval` agreeing on
    the overlap; and `rescaled` against v f(r / k) and its round trip.
    Every tolerance is relative to the value or the interval's width.
    """
    lo, hi = (float(x) for x in fn.interval)
    assert hi > lo, "the interval does not increase"
    width = hi - lo
    r = np.linspace(lo, hi, n)
    y = fn(r)
    assert isinstance(y, np.ndarray) and y.dtype in (np.float64, np.complex128), (
        f"values are {type(y).__name__} of dtype {getattr(y, 'dtype', None)}")
    assert y.shape == r.shape, f"values have shape {y.shape} for {r.shape} radii"
    assert np.all(np.isfinite(y)), "values are not finite"
    assert np.asarray(fn(r.reshape(-1, 1))).shape == (n, 1), "no broadcasting"
    assert np.asarray(fn(float(r[n // 2]))).shape == (), "a scalar radius fails"
    scale = max(float(np.max(np.abs(y))), 1e-300)

    inner = np.linspace(lo + 0.05 * width, hi - 0.05 * width, n)
    h = 1e-4 * width
    d = fn.derivative()
    fd = (fn(inner + h) - fn(inner - h)) / (2.0 * h)
    assert _rel(d(inner), fd, rtol=1e-4, floor=scale / width), (
        "derivative disagrees with a central difference")
    assert fn.derivative(nu=0) is fn or np.allclose(fn.derivative(nu=0)(r), y), (
        "derivative(nu=0) is not the function")

    fine = np.linspace(lo, hi, 20001)
    want = np.trapezoid(fn(fine), fine)
    got = fn.integrate(lo, hi)
    assert isinstance(got, complex if y.dtype.kind == "c" else float), (
        f"integrate returns {type(got).__name__}")
    assert abs(got - want) <= 1e-6 * max(abs(want), scale * width), (
        "integrate disagrees with a trapezoid rule")
    assert abs(fn.integrate(hi, lo) + got) <= 1e-12 * max(abs(got), scale * width), (
        "integrate is not signed")

    wider = fn.on_interval(lo - 0.5 * width, hi + 0.5 * width)
    assert wider.interval == (lo - 0.5 * width, hi + 0.5 * width), (
        "on_interval does not set the interval")
    assert np.allclose(wider(r), y, rtol=rtol, atol=rtol * scale), (
        "on_interval changes the values on the overlap")
    assert np.all(np.isfinite(wider(np.array([lo - 0.4 * width, hi + 0.4 * width])))), (
        "on_interval does not extend")

    k, v = 2.5, 0.3
    g = fn.rescaled(k=k, v=v)
    assert np.allclose(g.interval, (k * lo, k * hi), rtol=1e-12), (
        "rescaled does not scale the interval")
    assert np.allclose(g(k * r), v * y, rtol=1e-12, atol=1e-12 * scale), (
        "rescaled is not v f(r / k)")
    back = g.rescaled(k=1.0 / k, v=1.0 / v)
    assert np.allclose(back(r), y, rtol=1e-12, atol=1e-12 * scale), (
        "rescaled does not round-trip")


def check_field(field: Field, *, rng: np.random.Generator | None = None,
                n: int = 64) -> None:
    """Hold a Field to its protocol.

    Checks the three attributes; that a call with the radius alone is
    accepted exactly when the field is radial and of rank 0; float64 (or
    complex128) values of the broadcast shape plus the stored shape,
    agreeing with `dtype`; both ends of the
    interval; refusal outside it and of an unknown frame; the two
    frames related by the rotation the character implies (a factor of
    R on every slot, expanded from Voigt where the field is Voigt);
    `evaluate_at` against `evaluate`; `on_interval` on the overlap;
    and the `rescaled` round trip.
    """
    from .fields import stored_shape
    from .frames import cartesian_points, rotate_slots, spherical_frame
    from .frames import tensor_to_voigt, voigt_to_tensor

    rng = np.random.default_rng(0) if rng is None else rng
    lo, hi = (float(x) for x in field.interval)
    assert hi > lo, "the interval does not increase"
    width = hi - lo
    char = field.character
    assert hasattr(field, "name"), "no name attribute"
    shape = stored_shape(char)
    radial_scalar = bool(getattr(field, "is_radial", False)) and char.rank == 0

    r = lo + width * rng.uniform(0.0, 1.0, n)
    r[0], r[-1] = lo, hi
    theta = rng.uniform(0.05, np.pi - 0.05, n)
    phi = rng.uniform(-np.pi, np.pi, n)

    values = field.evaluate(r, theta, phi)
    assert isinstance(values, np.ndarray) and values.dtype in (np.float64,
                                                                np.complex128), (
        f"values are dtype {getattr(values, 'dtype', None)}, not float64")
    assert field.dtype == values.dtype, "dtype disagrees with the values"
    assert values.shape == (n,) + shape, (
        f"values have shape {values.shape}, expected {(n,) + shape}")
    assert np.all(np.isfinite(values)), "values are not finite"
    scale = max(float(np.max(np.abs(values))), 1e-300)

    grid = field.evaluate(r[:, None, None], theta[None, :4, None], phi[None, None, :3])
    assert grid.shape == (n, 4, 3) + shape, "coordinates do not broadcast"
    single = field.evaluate(float(r[1]), float(theta[1]), float(phi[1]))
    assert single.shape == shape, "a single point does not give the stored shape"
    assert np.allclose(single, values[1], rtol=1e-12, atol=1e-12 * scale), (
        "a single point disagrees with the batch")

    assert np.allclose(field(r, theta, phi), values), "__call__ disagrees"
    if radial_scalar:
        alone = field(r)
        assert np.allclose(alone, values, rtol=1e-12, atol=1e-12 * scale), (
            "the radius alone gives different values")
    else:
        try:
            field(r)
        except ValueError:
            pass
        else:
            raise AssertionError("the radius alone was accepted by a field that "
                                 "is not radial and of rank 0")

    for bad in (lo - 1e-3 * width, hi + 1e-3 * width):
        try:
            field.evaluate(bad, 0.5, 0.5)
        except ValueError:
            pass
        else:
            raise AssertionError(f"radius {bad:g} outside the interval was accepted")
    try:
        field.evaluate(r, theta, phi, frame="nowhere")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown frame was accepted")

    cart = field.evaluate(r, theta, phi, frame="cartesian")
    assert cart.shape == values.shape, "the Cartesian frame changes the shape"
    if char.rank:
        R = spherical_frame(theta, phi)
        full = (voigt_to_tensor(values, rank=char.rank) if char.voigt_shape
                else values)
        want = rotate_slots(full, R, char.rank)
        if char.voigt_shape:
            want = tensor_to_voigt(want, rank=char.rank)
        assert np.allclose(cart, want, rtol=1e-10, atol=1e-10 * scale), (
            "the Cartesian components are not the rotated spherical ones")
    else:
        assert np.allclose(cart, values), "a rank-0 field changes with the frame"

    X = cartesian_points(r, theta, phi)
    at = field.evaluate_at(X)
    assert np.allclose(at, cart, rtol=1e-10, atol=1e-10 * scale), (
        "evaluate_at disagrees with evaluate in the Cartesian frame")
    at_s = field.evaluate_at(X, frame="spherical")
    assert np.allclose(at_s, values, rtol=1e-10, atol=1e-10 * scale), (
        "evaluate_at in the spherical frame disagrees with evaluate")

    wider = field.on_interval(lo - 0.25 * width, hi + 0.25 * width)
    assert np.allclose(wider.interval, (lo - 0.25 * width, hi + 0.25 * width)), (
        "on_interval does not set the interval")
    assert wider.character == char, "on_interval changes the character"
    assert np.allclose(wider.evaluate(r, theta, phi), values,
                       rtol=1e-10, atol=1e-10 * scale), (
        "on_interval changes the values on the overlap")

    k, v = 2.5, 0.3
    g = field.rescaled(k=k, v=v)
    assert np.allclose(g.interval, (k * lo, k * hi), rtol=1e-12), (
        "rescaled does not scale the interval")
    assert g.character == char, "rescaled changes the character"
    assert np.allclose(g.evaluate(k * r, theta, phi), v * values,
                       rtol=1e-10, atol=1e-10 * scale), "rescaled is not v f(r / k)"
    back = g.rescaled(k=1.0 / k, v=1.0 / v)
    assert np.allclose(back.evaluate(r, theta, phi), values,
                       rtol=1e-10, atol=1e-10 * scale), "rescaled does not round-trip"


def check_refinement(refinement: Refinement, points: ArrayLike) -> None:
    """Hold a refinement to its protocol on the given points.

    Checks `size` and `far_size` as a positive floor and ceiling; that
    the (n, d) `points` give float64 sizes of shape (n,), finite and
    within [size, far_size]; and that a batch agrees with its points
    taken one at a time, so vectorisation is a promise rather than an
    optimisation.  The points are the caller's probes and should span
    the domain the refinement will be asked about, the refined region
    included.
    """
    X = np.asarray(points, dtype=float)
    assert X.ndim == 2 and X.shape[1] in (2, 3), (
        f"probe points must be (n, 2) or (n, 3), got {X.shape}")
    assert callable(refinement), f"{type(refinement).__name__} is not callable"
    size = getattr(refinement, "size", None)
    far = getattr(refinement, "far_size", None)
    assert isinstance(size, (int, float)) and size > 0.0, (
        f"size must be a positive number, got {size!r}")
    assert isinstance(far, (int, float)) and far >= size, (
        f"far_size must be at least size ({size!r}), got {far!r}")

    h = refinement(X)
    assert isinstance(h, np.ndarray) and h.dtype == np.float64, (
        f"sizes are {type(h).__name__} of dtype {getattr(h, 'dtype', None)}, "
        "not a float64 array")
    assert h.shape == (X.shape[0],), (
        f"sizes have shape {h.shape} for {X.shape[0]} points")
    assert np.all(np.isfinite(h)), "sizes are not finite"
    tol = 1e-12 * far
    assert np.all(h >= size - tol), "a size falls below the declared floor"
    assert np.all(h <= far + tol), "a size exceeds the declared ceiling"

    for k in (0, X.shape[0] - 1):
        one = refinement(X[k:k + 1])
        assert np.shape(one) == (1,) and np.isclose(one[0], h[k], rtol=1e-12), (
            "a batch disagrees with its points taken one at a time")


def check_model(model: Model, *, rng: np.random.Generator | None = None) -> None:
    """Hold a Model to its invariants.

    Checks that `replaced` with checks on accepts it and keeps its class,
    fields and citation, which `cite` gives a work a line; that every
    layer's fields pass `check_field`, sit on the layer's interval, and
    have their spec's character; that `layer(name)` and `layer(i)`
    agree; that every constant is its SI value over the scales' factor;
    that conversion to other scales and back restores every field to
    1e-12; and that every surgery keeps the class and validates again.
    """
    from .model import Model
    from .units import Scales

    rng = np.random.default_rng(0) if rng is None else rng
    assert isinstance(model, Model), f"{type(model).__name__} is not a Model"
    cls = type(model)
    again = model.replaced(check=True)
    assert type(again) is cls, "replaced does not keep the class"
    assert again.citation == model.citation, "replaced does not keep the citation"
    assert all(isinstance(w, str) and w and "\n" not in w for w in model.citation), (
        "a work cited is not one line of text")
    assert model.cite().splitlines() == list(model.citation), (
        "cite does not give the citation a work a line")
    for a, b in zip(again.layers, model.layers):
        assert a.fields == b.fields, "replaced changes the fields"
    rtol = model.geometry.rtol
    for i, layer in enumerate(model.layers):
        assert layer.index == i, f"layer {i} reports index {layer.index}"
        assert layer.interval == model.skeleton.interval(i), (
            f"layer {i} has interval {layer.interval}")
        if layer.name is not None:
            assert model.layer(layer.name) is layer, (
                f"layer({layer.name!r}) is not layer {i}")
        assert model.layer(i) is layer
        for name, field in layer.fields.items():
            width = layer.interval[1] - layer.interval[0]
            assert np.allclose(field.interval, layer.interval, rtol=0.0,
                               atol=rtol * width), (
                f"{name!r} on layer {i} is on {field.interval}")
            spec = model.spec(name)
            if spec is not None:
                assert field.character == spec.character, (
                    f"{name!r} on layer {i} has character {field.character}, "
                    f"its spec says {spec.character}")
            check_field(field, rng=rng)

    for name, c in model.constants.items():
        want = c.value_si / model.scales.factor(c.dimensions)
        assert np.isclose(model.constant(name), want, rtol=1e-15), (
            f"constant {name!r} is not its SI value in the model's units")

    names = model.field_names()
    convertible = all(model.spec(n) is not None
                      and model.spec(n).dimensions is not None for n in names)
    if convertible:
        other = Scales(length=3.0 * model.scales.length,
                       mass=0.5 * model.scales.mass, time=2.0 * model.scales.time,
                       temperature=1.5 * model.scales.temperature)
        back = model.converted(other).converted(model.scales)
        assert np.allclose(back.skeleton.boundaries, model.skeleton.boundaries,
                           rtol=1e-12), "conversion does not restore the skeleton"
        for layer, layer2 in zip(model.layers, back.layers):
            lo, hi = layer.interval
            r = lo + (hi - lo) * rng.uniform(0.0, 1.0, 8)
            theta, phi = rng.uniform(0.1, 3.0, 8), rng.uniform(-3.0, 3.0, 8)
            for name, f in layer.fields.items():
                a, b = f.evaluate(r, theta, phi), layer2[name].evaluate(r, theta, phi)
                scale = max(float(np.max(np.abs(a))), 1e-300)
                assert np.allclose(a, b, rtol=1e-12, atol=1e-12 * scale), (
                    f"conversion does not round-trip {name!r} on layer "
                    f"{layer.index}")

    b = model.skeleton.boundaries
    lo, hi = float(b[0]), float(b[-1])
    mid = 0.5 * (lo + hi)
    inner = b[(b > lo) & (b < hi)]
    cut = float(inner[0]) if inner.size else mid
    for verb, made in (("refined", lambda: model.refined([mid])),
                       ("truncated", lambda: model.truncated(cut)),
                       ("hollowed", lambda: model.hollowed(cut)),
                       ("extended", lambda: model.extended([hi * 1.1],
                                                           fields="extrapolate")),
                       ("renamed", lambda: model.renamed())):
        if verb in ("refined",) and np.min(np.abs(b - mid)) <= rtol * (hi - lo):
            continue
        if verb in ("truncated", "hollowed") and cut in (lo, hi):
            continue
        try:
            out = made()
        except ValueError as exc:
            if verb == "extended" and not model.geometry.is_identity:
                continue
            raise AssertionError(f"{verb} refused: {exc}") from exc
        assert type(out) is cls, f"{verb} does not keep the class"
        out.validate()


def check_sample(sample: Sample, model: Model, *,
                 rng: np.random.Generator | None = None, n: int = 64,
                 rtol: float = 1e-12) -> None:
    """Hold a Sample to its layout and to the model it was taken from.

    Layout: the radial mesh lies over the model's skeleton and the nodes
    are its per-element GLL nodes flattened; the angular nodes increase,
    colatitudes in [0, pi] and longitudes in [0, 2 pi); every field is
    float64 and read-only, shaped `(nnode,) + c` exactly when every
    layer holding it declares itself radial and `(nnode, ntheta, nphi)
    + c` otherwise, with `c` the stored shape of its character.
    Metadata: the characters and
    dimensions are those of the model's fields and specs, the scales
    and layer names the model's.  Values: NaN on every node of a layer
    lacking the field and finite everywhere else; `n` random (node,
    colatitude, longitude) entries re-evaluated from the node's own
    layer, so the two nodes at an interface are two questions.  The
    displacement is None for an identity geometry and otherwise
    `(nnode, ntheta, nphi, 3)`, read-only, finite, and `R^T (m(X) - X)`
    at `n` random entries.  Tolerances are `rtol` of the field's
    largest value, or of the outer radius for the displacement.
    """
    from .fields import stored_shape
    from .frames import spherical_frame
    from .mesh1d.mesh import RadialMesh
    from .sampling import AngularGrid, Sample

    rng = np.random.default_rng(0) if rng is None else rng
    assert isinstance(sample, Sample), f"not a Sample: {type(sample).__name__}"
    mesh, grid = sample.radial, sample.angular
    assert isinstance(mesh, RadialMesh), "radial is not a RadialMesh"
    assert isinstance(grid, AngularGrid), "angular is not an AngularGrid"
    assert mesh.skeleton == model.skeleton, (
        "the radial mesh lies over another skeleton than the model's")

    # -- layout -------------------------------------------------------------
    nnode = mesh.nspec * mesh.ngll
    assert sample.nnode == nnode, "nnode is not nspec * ngll"
    r = sample.radius
    assert np.array_equal(r, mesh.r.ravel()), (
        "radius is not the mesh's per-element nodes flattened")
    assert np.array_equal(sample.element_layer, mesh.layer), (
        "element_layer differs from the mesh's layer array")
    node_layer = np.repeat(np.asarray(mesh.layer, dtype=int), mesh.ngll)
    assert np.array_equal(sample.node_layer, node_layer), (
        "node_layer is not element_layer repeated over the nodes")
    theta, phi = grid.colatitudes, grid.longitudes
    assert np.all(np.diff(theta) > 0) and 0 <= theta[0] and theta[-1] <= np.pi, (
        "colatitudes are not increasing in [0, pi]")
    assert np.all(np.diff(phi) > 0) and 0 <= phi[0] and phi[-1] < 2 * np.pi, (
        "longitudes are not increasing inside [0, 2 pi)")
    ntheta, nphi = theta.size, phi.size

    # -- metadata -----------------------------------------------------------
    names = set(sample.fields)
    for what in ("characters", "dimensions"):
        d = getattr(sample, what)
        assert set(d) == names, (
            f"{what} names {sorted(d)} but the fields are {sorted(names)}")
    assert sample.scales == model.scales, (
        f"scales are {sample.scales!r}, the model's are {model.scales!r}")
    layer_names = tuple(layer.name for layer in model.layers)
    assert tuple(sample.layer_names) == layer_names, (
        f"layer_names are {sample.layer_names}, the model's are {layer_names}")

    # -- the fields ---------------------------------------------------------
    nodes = rng.integers(nnode, size=n)
    it = rng.integers(ntheta, size=n)
    ip = rng.integers(nphi, size=n)
    for name, arr in sample.fields.items():
        char = sample.characters[name]
        holders = model.layers_with(name)
        assert holders, f"no layer of the model holds {name!r}"
        for L in holders:
            fc = model.layer(L)[name].character
            assert fc == char, (
                f"{name!r} is recorded with character {char}; on layer {L} "
                f"the model's field has {fc}")
        spec = model.spec(name)
        want_dims = None if spec is None else spec.dimensions
        assert sample.dimensions[name] == want_dims, (
            f"{name!r} is recorded with dimensions {sample.dimensions[name]}; "
            f"the model's spec says {want_dims}")
        c = stored_shape(char)
        assert sample.stored_shape(name) == c, (
            f"stored_shape({name!r}) is {sample.stored_shape(name)}, not {c}")
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64, (
            f"{name!r} is not a float64 array")
        assert not arr.flags.writeable, f"{name!r} is writable"
        radial = all(getattr(model.layer(L)[name], "is_radial", False)
                     for L in holders)
        want_shape = (nnode,) + c if radial else (nnode, ntheta, nphi) + c
        assert arr.shape == want_shape, (
            f"{name!r} has shape {arr.shape}; expected {want_shape} for a field "
            f"that {'does not depend' if radial else 'depends'} on direction")
        assert sample.is_radial(name) == radial, (
            f"is_radial({name!r}) is {sample.is_radial(name)}")

        keep = np.isin(node_layer, np.asarray(holders, dtype=int))
        axes = tuple(range(1, arr.ndim))
        finite = np.all(np.isfinite(arr), axis=axes) if axes else np.isfinite(arr)
        nan = np.all(np.isnan(arr), axis=axes) if axes else np.isnan(arr)
        assert np.all(finite[keep]), (
            f"{name!r} has non-finite values on nodes of layers {holders}, "
            "which hold it")
        assert np.all(nan[~keep]), (
            f"{name!r} is not NaN on the nodes of the layers lacking it")

        here = np.flatnonzero(keep)
        if here.size == 0:
            continue
        scale = float(np.max(np.abs(arr[keep])))
        sel = here[nodes % here.size]
        for k in range(n):
            j, L = int(sel[k]), int(node_layer[sel[k]])
            field = model.layer(L)[name]
            point = (r[j], theta[it[k]], phi[ip[k]])
            got = arr[j] if radial else arr[j, it[k], ip[k]]
            want = field.evaluate(*point)
            assert np.allclose(got, want, rtol=rtol, atol=rtol * scale), (
                f"{name!r} at node {j} (layer {L}, r = {r[j]:.6g}), colatitude "
                f"{theta[it[k]]:.4g}, longitude {phi[ip[k]]:.4g} differs from "
                "the model's field")

    # -- the displacement ---------------------------------------------------
    u = sample.displacement
    geometry = model.geometry
    if geometry.is_identity:
        assert u is None, "the geometry is the identity but displacement is not None"
        return
    assert u is not None, "the geometry moves points but displacement is None"
    assert isinstance(u, np.ndarray) and u.dtype == np.float64, (
        "displacement is not a float64 array")
    assert not u.flags.writeable, "displacement is writable"
    assert u.shape == (nnode, ntheta, nphi, 3), (
        f"displacement has shape {u.shape}, expected {(nnode, ntheta, nphi, 3)}")
    assert np.all(np.isfinite(u)), "displacement has non-finite values"
    R = spherical_frame(theta[it], phi[ip])                # (n, 3, 3)
    X = r[nodes, None] * R[..., :, 0]
    m = geometry.mapping
    cart = np.asarray(m(X), dtype=float) - X
    want = np.einsum("kji,kj->ki", R, cart)
    scale = float(np.max(np.abs(r)))
    assert np.allclose(u[nodes, it, ip], want, rtol=rtol, atol=rtol * scale), (
        "displacement differs from R^T (m(X) - X) at the sample points")


def _full_mass(basis: SpectralBasis) -> np.ndarray:
    """The family's diagonal mass at every node of the mesh, the massless
    centre of a ball included."""
    mass = basis.family.mass(basis.degree)
    full = np.zeros(basis.family.mesh.nglob)
    full[full.size - mass.size:] = mass
    return full


def check_spectral_basis(basis: SpectralBasis, *,
                         rng: np.random.Generator | None = None,
                         rtol: float = 1e-10) -> None:
    """Hold a SpectralBasis to what it promises of one degree's eigenbasis.

    The spectrum: `theta` read-only, ascending, at least 1, `nmodes` of
    them.  The modes: read-only, of shape (nglob, nmodes), M-orthonormal
    on the full mesh, each an eigenfunction of the family's operator with
    its theta, and zero at the centre of a ball when l >= 1.  The
    transforms: the physical synthesis is the full one read on the
    restriction's nodes; `analyse` inverts the full synthesis and is
    idempotent on arbitrary nodal values, for one vector and for a stack.
    Evaluation: `evaluate` at the physical nodes reproduces the modes
    there, keeps the shape of its argument and refuses a radius outside
    the interval.  Integration: `weights` are non-negative, the family's
    mass inside the physical interval and no more than it at the two
    ends (where the mass counts the pad elements too), and, on a mesh of
    at least three GLL nodes, sum to the measure of the physical
    interval, (r2^3 - r1^3) / 3 or r2 - r1.  `pointwise_variance` is the
    diagonal of the synthesised covariance.  `white_noise` has the shapes
    it promises and unit covariance on the leading modes to six standard
    errors.  `rtol` is relative to the largest kept eigenvalue where an
    operator is applied and to one elsewhere.
    """
    from .randomfield.basis import SpectralBasis

    rng = np.random.default_rng(0) if rng is None else rng
    assert isinstance(basis, SpectralBasis), (
        f"not a SpectralBasis: {type(basis).__name__}")
    fam, l, R = basis.family, basis.degree, basis.restriction
    mesh = fam.mesh
    theta, k = basis.theta, basis.nmodes

    assert k >= 1 and theta.shape == (k,), "theta is not (nmodes,) with a mode kept"
    assert not theta.flags.writeable, "theta is writeable"
    assert np.all(np.diff(theta) >= 0.0), "theta is not ascending"
    assert theta[0] >= 1.0 - rtol, f"an eigenvalue below 1: {theta[0]!r}"

    Phi = basis.modes()
    assert Phi.shape == (mesh.nglob, k), (
        f"modes have shape {Phi.shape}, expected {(mesh.nglob, k)}")
    assert not Phi.flags.writeable, "the modes are writeable"
    M = _full_mass(basis)
    gram = Phi.T @ (M[:, None] * Phi)
    assert _rel(gram, np.eye(k), rtol=rtol), "the modes are not M-orthonormal"
    n0 = mesh.nglob - fam.ndof(l)
    resid = fam.apply(l, Phi[n0:]) - Phi[n0:] * theta
    scale = theta[-1] * np.max(np.abs(Phi))
    assert np.max(np.abs(resid)) <= 1e3 * rtol * scale, (
        "a mode is not an eigenfunction of the family's operator")
    if fam.is_ball and l >= 1:
        assert np.all(Phi[0] == 0.0), "a mode of degree l >= 1 is non-zero at r = 0"

    assert np.array_equal(basis.r, mesh.rglob[R.nodes]), (
        "r is not the mesh's nodes on the restriction")
    for shape in ((k,), (k, 3)):
        c = rng.standard_normal(shape)
        full = basis.synthesise(c, physical=False)
        assert full.shape == (mesh.nglob,) + shape[1:], (
            "full synthesis has the wrong shape")
        assert _rel(basis.synthesise(c), full[R.nodes], rtol=rtol,
                    floor=np.max(np.abs(full))), (
            "the physical synthesis is not the full one on the restriction")
        assert _rel(basis.analyse(full), c, rtol=rtol), (
            "analyse does not invert the full synthesis")
        v = rng.standard_normal((mesh.nglob,) + shape[1:])
        cv = basis.analyse(v)
        assert _rel(basis.analyse(basis.synthesise(cv, physical=False)), cv,
                    rtol=rtol), "analyse is not idempotent"

    at_nodes = basis.evaluate(basis.r)
    assert at_nodes.shape == basis.r.shape + (k,), "evaluate has the wrong shape"
    assert _rel(at_nodes, Phi[R.nodes], rtol=rtol, floor=np.max(np.abs(Phi))), (
        "evaluate at the physical nodes is not the modes there")
    r1, r2 = R.interval
    grid = rng.uniform(r1, r2, size=(2, 3))
    assert basis.evaluate(grid).shape == (2, 3, k), (
        "evaluate does not keep the shape of its argument")
    try:
        basis.evaluate(r2 + 0.1 * (r2 - r1))
    except ValueError:
        pass
    else:
        raise AssertionError("evaluate accepted a radius outside the interval")

    w = basis.weights()
    assert w.shape == basis.r.shape and np.all(w >= 0.0), (
        "weights are not non-negative values at the physical nodes")
    inside = slice(1, -1)
    assert _rel(w[inside], M[R.nodes][inside], rtol=rtol, floor=np.max(M)), (
        "weights differ from the family's mass inside the physical interval")
    assert np.all(w <= M[R.nodes] * (1.0 + rtol)), (
        "a weight exceeds the family's mass at its node")
    if mesh.ngll >= 3:
        measure = (r2 ** 3 - r1 ** 3) / 3.0 if fam.weight == "r2" else r2 - r1
        assert abs(w.sum() - measure) <= rtol * abs(measure), (
            f"weights sum to {w.sum()!r}, the interval measures {measure!r}")

    for power in (1.0, 2.5):
        S = basis.synthesise(np.eye(k))
        want = np.einsum("ij,j,ij->i", S, theta ** -power, S)
        assert _rel(basis.pointwise_variance(power), want, rtol=rtol,
                    floor=np.max(want)), (
            "pointwise_variance is not the diagonal of the covariance")

    assert basis.white_noise(rng=rng).shape == (k,), "white noise is not (nmodes,)"
    n, lead = 4000, min(k, 8)
    z = basis.white_noise(rng=rng, size=n)
    assert z.shape == (k, n), "white noise with size is not (nmodes, size)"
    cov = z[:lead] @ z[:lead].T / n
    assert np.max(np.abs(cov - np.eye(lead))) < 6.0 * np.sqrt(2.0 / n), (
        "white noise does not have unit covariance")


def check_spherical_basis(basis: SphericalBasis, *,
                          rng: np.random.Generator | None = None,
                          rtol: float = 1e-10) -> None:
    """Hold a SphericalBasis to the order it fixes and to its degrees.

    The bases are those of degrees 0 ... lmax on one family and one
    restriction, with `nmodes` and `size` counting them.  The blocks of
    the harmonics, taken in the order of `harmonics.packing`, tile
    `range(size)` without gap or overlap, and `theta` on each block is
    its degree's.  `synthesise` fills each harmonic of the coefficient
    array with its degree's synthesis of that block and leaves the
    entries of no harmonic zero, physical values being the full ones on
    the restriction's nodes; `analyse` inverts the full synthesis, and
    ignores the entries of no harmonic.
    """
    from .harmonics import packing
    from .randomfield.basis import SphericalBasis

    rng = np.random.default_rng(0) if rng is None else rng
    assert isinstance(basis, SphericalBasis), (
        f"not a SphericalBasis: {type(basis).__name__}")
    L = basis.lmax
    assert len(basis) == L + 1, "len is not lmax + 1"
    for l in range(L + 1):
        b = basis[l]
        assert b.degree == l, f"basis[{l}] has degree {b.degree}"
        assert b.family is basis.family, "the bases do not share one family"
        assert b.restriction.nodes == basis.restriction.nodes, (
            "the bases do not share one restriction")
        assert basis.nmodes[l] == b.nmodes, f"nmodes[{l}] is not the basis's"
    assert basis.size == int(np.sum((2 * np.arange(L + 1) + 1) * basis.nmodes)), (
        "size is not sum (2 l + 1) nmodes[l]")
    assert basis.theta.shape == (basis.size,) and not basis.theta.flags.writeable, (
        "theta is not a read-only vector of length size")

    at = 0
    S, Ls, Ms = packing(L)
    for s, l, m in zip(S, Ls, Ms):
        blk = basis.block(int(s), int(l), int(m))
        assert (blk.start, blk.stop) == (at, at + basis[l].nmodes), (
            f"the block of (s, l, m) = ({s}, {l}, {m}) does not follow its "
            f"predecessor in the packing order")
        assert np.array_equal(basis.theta[blk], basis[l].theta), (
            f"theta on the block of ({s}, {l}, {m}) is not its degree's")
        at = blk.stop
    assert at == basis.size, "the blocks do not tile the vector"
    for bad in ((1, 0, 0), (0, 1, 2), (0, L + 1, 0), (2, 0, 0)):
        try:
            basis.block(*bad)
        except IndexError:
            continue
        raise AssertionError(f"block accepted (s, l, m) = {bad}")

    v = rng.standard_normal(basis.size)
    full = basis.synthesise(v, physical=False)
    nglob = basis.family.mesh.nglob
    assert full.shape == (2, L + 1, L + 1, nglob), "full synthesis has the wrong shape"
    phys = basis.synthesise(v)
    assert _rel(phys, full[..., basis.restriction.nodes], rtol=rtol,
                floor=np.max(np.abs(full))), (
        "the physical synthesis is not the full one on the restriction")
    seen = np.zeros(full.shape[:3], dtype=bool)
    for s, l, m in zip(S, Ls, Ms):
        seen[s, l, m] = True
        want = basis[l].synthesise(v[basis.block(int(s), int(l), int(m))],
                                   physical=False)
        assert _rel(full[s, l, m], want, rtol=rtol, floor=np.max(np.abs(want))), (
            f"harmonic ({s}, {l}, {m}) is not its degree's synthesis of its block")
    assert np.all(full[~seen] == 0.0), "an entry of no harmonic is non-zero"
    assert _rel(basis.analyse(full), v, rtol=rtol), (
        "analyse does not invert the full synthesis")
    junk = full.copy()
    junk[~seen] = 1.0
    assert np.array_equal(basis.analyse(junk), basis.analyse(full)), (
        "analyse reads entries that belong to no harmonic")


def check_model_file(path: str | os.PathLike[str], model: Model, *,
                     surfaces: Surfaces | None = None,
                     rng: np.random.Generator | None = None, n: int = 48,
                     rtol: float = 1e-12) -> None:
    """Hold a model file to the model it was written from.

    Checks the structure (`validate_structure`); the skeleton, the layer
    and interface names, the scales and the citation; that the file names
    its source and carries a parseable created stamp; that every variable is held on
    the layers its `layers` attribute names, NaN elsewhere, by fields of
    the model of the stated character; that a lateral variable names in
    `radial_layers` exactly the layers where the model's field is radial,
    and reads back radial there; that its values at `n` random
    entries of the file's own nodes are the model's there, the
    reference and the perturbation of a perturbed field apart, and so
    are its polar values; that the geometry group holds the mapping's
    reliefs at its nodes, and a displacement variable the mapping's
    displacement; and that the model read back from the file takes the
    stored values at those nodes, for every variable read by a local
    rule.  Given the `surfaces` the file was written with, that its
    surfaces group holds exactly those names on exactly those
    interfaces, NaN on the others, their values at its nodes and poles,
    and that `read_surfaces` gives functions taking the stored values at
    the nodes, for a local rule.  Tolerances are `rtol` of a variable's
    largest value.  netCDF4 is imported here.
    """
    from .frames import cartesian_points, spherical_frame
    from .modelfile import (read_model, read_model_file, read_surfaces,
                            validate_structure)

    rng = np.random.default_rng(0) if rng is None else rng
    validate_structure(path)
    data = read_model_file(path)
    assert np.allclose(data.skeleton.boundaries, model.skeleton.boundaries,
                       rtol=1e-15), "the file's skeleton is not the model's"
    assert data.layer_names == tuple(lay.name for lay in model.geometry.layers), (
        "the file's layer names are not the model's")
    assert data.interface_names == tuple(f.name for f in model.geometry.interfaces), (
        "the file's interface names are not the model's")
    assert data.scales == model.scales, (
        f"the file is in {data.scales!r}, the model in {model.scales!r}")
    assert data.citation == model.citation, "the file does not cite what the model does"
    assert data.source.strip(), "the file names no source"
    from datetime import datetime
    datetime.fromisoformat(data.created)
    assert data.rtol == model.geometry.rtol, (
        f"the file states rtol {data.rtol!r}, the model's geometry has "
        f"{model.geometry.rtol!r}")
    back = read_model(path)

    def stored_against(values: np.ndarray, want: np.ndarray, what: str) -> None:
        scale = max(float(np.nanmax(np.abs(values))) if values.size else 0.0, 1e-300)
        assert np.allclose(values, want, rtol=0.0, atol=rtol * scale), (
            f"{what}: the file's values are not the model's at its nodes")

    for k, part in enumerate(data.parts):
        mesh, grid = part.mesh, part.grid
        r_all = mesh.r.ravel()
        node_layer = np.repeat(mesh.layer, mesh.ngll)
        for name, var in part.variables.items():
            if name.endswith(("_poles", "_perturbation")) or name == "displacement":
                continue
            holders = [int(i) for i in np.atleast_1d(var.attributes["layers"])]
            values = var.values
            for i in part.layers:
                on = node_layer == i
                if i not in holders:
                    assert np.all(np.isnan(values[on])), (
                        f"part_{k}/{name} is not NaN on layer {i}, which lacks it")
                    continue
                field = model.layer(i)[name]
                c = field.character
                stated = (int(var.attributes["rank"]), int(var.attributes["weight"]),
                          bool(var.attributes["voigt"]))
                assert stated == (c.rank, c.weight, bool(c.voigt)), (
                    f"part_{k}/{name} states the character {stated}, the model's "
                    f"field on layer {i} is {c}")
                pert_name = var.attributes.get("perturbation")
                reference = getattr(field, "reference", None)
                source = reference if (pert_name and reference is not None) else field
                idx = rng.choice(np.flatnonzero(on), size=min(n, int(on.sum())),
                                 replace=False)
                r = r_all[idx]
                if values.ndim == 1 + len(source.stored_shape):
                    th = np.full(r.shape, 0.5 * np.pi)
                    want = source.evaluate(r, th, np.zeros_like(r))
                    stored_against(values[idx], want, f"part_{k}/{name}")
                    got = back.layer(i)[name]
                    got = getattr(got, "reference", got)
                    stored_against(values[idx], got.evaluate(r, th, np.zeros_like(r)),
                                   f"part_{k}/{name} read back")
                else:
                    it = rng.integers(grid.ntheta, size=idx.size)
                    ip = rng.integers(grid.nphi, size=idx.size)
                    t, p = grid.colatitudes[it], grid.longitudes[ip]
                    stored_against(values[idx, it, ip], source.evaluate(r, t, p),
                                   f"part_{k}/{name}")
                    if var.attributes.get("lateral_interpolation") != "harmonic":
                        stored_against(values[idx, it, ip],
                                       back.layer(i)[name].evaluate(r, t, p),
                                       f"part_{k}/{name} read back")
                    listed = i in {int(j) for j in np.atleast_1d(
                        var.attributes.get("radial_layers", []))}
                    radial = bool(getattr(source, "is_radial", False))
                    assert listed == radial, (
                        f"part_{k}/{name}: layer {i} is "
                        f"{'' if listed else 'not '}in radial_layers, but the "
                        f"model's field there is {'' if radial else 'not '}radial")
                    assert not listed or getattr(back.layer(i)[name], "is_radial",
                                                 False), (
                        f"part_{k}/{name}: layer {i} is in radial_layers but does "
                        "not read back radial")
                poles = part.variables.get(f"{name}_poles")
                if poles is not None:
                    want = source.evaluate(r[:, None], np.array([0.0, np.pi]),
                                           np.zeros(2), frame="cartesian")
                    stored_against(poles.values[idx], want, f"part_{k}/{name}_poles")
                if pert_name and reference is not None:
                    pv = part.variables[pert_name]
                    d = field.perturbation
                    if pv.values.ndim == 1:
                        want = d.evaluate(r, np.full(r.shape, 0.5 * np.pi),
                                          np.zeros_like(r))
                        stored_against(pv.values[idx], want, f"part_{k}/{pert_name}")
                    else:
                        it = rng.integers(grid.ntheta, size=idx.size)
                        ip = rng.integers(grid.nphi, size=idx.size)
                        t, p = grid.colatitudes[it], grid.longitudes[ip]
                        stored_against(pv.values[idx, it, ip], d.evaluate(r, t, p),
                                       f"part_{k}/{pert_name}")
        disp = part.variables.get("displacement")
        if disp is not None:
            it = rng.integers(grid.ntheta, size=n)
            ip = rng.integers(grid.nphi, size=n)
            idx = rng.integers(r_all.size, size=n)
            X = cartesian_points(r_all[idx], grid.colatitudes[it], grid.longitudes[ip])
            R = spherical_frame(grid.colatitudes[it], grid.longitudes[ip])
            u = np.einsum("pji,pj->pi", R, model.geometry.mapping(X) - X)
            stored_against(disp.values[idx, it, ip], u, f"part_{k}/displacement")
    if data.geometry is not None:
        g = data.geometry
        h = model.geometry.mapping.h
        theta = g.grid.colatitudes[:, None]
        phi = g.grid.longitudes[None, :]
        for j, radius in enumerate(g.radii):
            want = np.broadcast_to(h(np.full(theta.shape, radius), theta, phi),
                                   g.values[j].shape)
            stored_against(g.values[j], want, f"geometry/boundary_displacement[{j}]")
    if surfaces is None:
        return
    held = data.surfaces
    assert held is not None, "the file has no surfaces group"
    given = {name: {model.geometry.interface(w).index: fn for w, fn in on.items()}
             for name, on in surfaces.items()}
    names = {name for name in held.variables if not name.endswith("_poles")}
    assert names == set(given), (
        f"the surfaces group holds {sorted(names)}, written with {sorted(given)}")
    faces = sorted({k for on in given.values() for k in on})
    assert list(held.interfaces) == faces, (
        f"the surfaces are interfaces {list(held.interfaces)}, the functions are on "
        f"{faces}")
    read = read_surfaces(path)
    theta, phi = held.grid.colatitudes[:, None], held.grid.longitudes[None, :]
    ends = (np.array([0.0, np.pi]), np.zeros(2))
    for name, on in given.items():
        var, poles = held.variables[name], held.variables.get(f"{name}_poles")
        listed = {int(k) for k in np.atleast_1d(var.attributes["interfaces"])}
        assert listed == set(on), (
            f"surfaces/{name} names interfaces {sorted(listed)}, held on {sorted(on)}")
        for row, k in enumerate(held.interfaces):
            where = f"surfaces/{name} on interface {k}"
            if k not in on:
                assert np.all(np.isnan(var.values[row])) and (
                    poles is None or np.all(np.isnan(poles.values[row]))), (
                    f"{where} is not NaN, and the interface lacks it")
                continue
            values = var.values[row]
            want = np.broadcast_to(np.asarray(on[k](theta, phi), dtype=float),
                                   values.shape)
            stored_against(values, want, where)
            if poles is not None:
                stored_against(poles.values[row],
                               np.broadcast_to(np.asarray(on[k](*ends), dtype=float),
                                               (2,)), f"{where}, its poles")
            if var.attributes["lateral_interpolation"] != "harmonic":
                key = data.interface_names[k]
                fn = read[name][k if key is None else key]
                stored_against(values, fn(theta, phi), f"{where} read back")


def check_dataset(dataset: Dataset, *,
                  paths: Iterable[str | os.PathLike[str]] | None = None,
                  options: Mapping[str, Any] | None = None) -> None:
    """Hold a data set's entry, and the object its reader makes, to their
    protocols.

    Checks that the entry's sources would be cached under distinct names
    and its reader has `read`; that `describe` and `cite` name it and its
    works; and, given the `paths` `fetch` gave, that there is one for each
    file and member in the sources' order, each a file, a plain source's
    with the entry's sha256, and that the reader reads them, with
    `options`, into an object that holds to its own protocol: a `Model`
    to `check_model`, a `RegionalGrid` to `check_regional`, a `Surface`
    to `check_surface`,
    `Strata` to `check_strata`.
    Nothing here touches the network.
    """
    from pathlib import Path

    from .datasets import RegionalGrid, Strata, Surface, cite, describe, sha256_of
    from .model import Model

    names = [s.name for s in dataset.sources]
    assert len(set(names)) == len(names), f"{dataset.name}: sources share a name"
    assert callable(getattr(dataset.reader, "read", None)), (
        f"{dataset.name}: the reader has no read method")
    text = describe(dataset)
    assert dataset.name in text and dataset.licence in text, (
        f"{dataset.name}: describe does not name the data set and its licence")
    assert cite(dataset).splitlines() == list(dataset.citation), (
        f"{dataset.name}: cite does not give the entry's citation a work a line")
    if paths is None:
        return
    paths = tuple(os.fspath(p) for p in paths)
    want = sum(1 if s.members is None else len(s.members) for s in dataset.sources)
    assert len(paths) == want, (
        f"{dataset.name}: {len(paths)} paths for {want} files and members")
    i = 0
    for source in dataset.sources:
        count = 1 if source.members is None else len(source.members)
        for p in paths[i:i + count]:
            assert os.path.isfile(p), f"{dataset.name}: {p} is not a file"
        if source.members is None:
            assert sha256_of(paths[i]) == source.sha256, (
                f"{dataset.name}: {paths[i]} does not have the entry's sha256")
        i += count
    product = dataset.reader.read(tuple(Path(p) for p in paths), dataset,
                                  **dict(options or {}))
    assert product is not None, f"{dataset.name}: the reader gave nothing"
    held = getattr(product, "citation", None)
    assert held is not None and set(dataset.citation) <= set(held), (
        f"{dataset.name}: what the reader gives does not cite the entry")
    assert product.cite().splitlines() == list(held), (
        f"{dataset.name}: what the reader gives does not cite a work a line")
    if isinstance(product, Model):
        check_model(product)
    if isinstance(product, Surface):
        check_surface(product)
    if isinstance(product, Strata):
        check_strata(product)
    if isinstance(product, RegionalGrid):
        check_regional(product)


def check_surface(surface: Surface, *, rng: np.random.Generator | None = None,
                  n: int = 64) -> None:
    """Hold a `datasets.Surface` to its protocol at `n` random points: its
    values finite and of the points' shape; its radius its values, or its
    datum's radius plus an elevation or less a depth or a thickness, and
    positive, or refused by name where it has no datum; held on a grid,
    the same values at the grid's nodes; and its mean, where its function
    has one of its own and a band, the mean of its values by quadrature
    on a grid of that band (at most 512)."""
    from .datasets import Surface
    from .sampling import gauss_legendre

    rng = np.random.default_rng(0) if rng is None else rng
    theta = np.arccos(rng.uniform(-1.0, 1.0, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    values = surface(theta, phi)
    assert values.shape == theta.shape and np.all(np.isfinite(values)), (
        f"{surface!r}: its values are not finite numbers of the points' shape")
    if surface.measures != "radius" and surface.datum is None:
        try:
            surface.radius(theta, phi)
        except ValueError as error:
            assert "with_datum" in str(error), f"{surface!r}: {error}"
        else:
            raise AssertionError(f"{surface!r} gives a radius without its datum")
    else:
        radius = surface.radius(theta, phi)
        if surface.measures == "radius":
            want = values
        else:
            base = (surface.datum.radius(theta, phi)
                    if isinstance(surface.datum, Surface) else surface.datum)
            want = base + values if surface.measures == "elevation" else base - values
        assert np.allclose(radius, want, rtol=1e-12, atol=0), (
            f"{surface!r}: its radius is not its datum and values by its measure")
        assert np.all(radius > 0.0), f"{surface!r}: a radius that is not positive"
    band = getattr(surface.function, "lmax", None)
    grid = gauss_legendre(32 if band is None else min(int(band), 512))
    held = surface.sampled(grid, lateral="linear")
    t, p = grid.colatitudes[:, None], grid.longitudes[None, :]
    on = surface(t, p)
    scale = max(float(np.max(np.abs(on))), 1e-300)
    assert np.allclose(held(t, p), on, rtol=0, atol=1e-9 * scale), (
        f"{surface!r}: held on a grid, it does not keep its values at the nodes")
    if (getattr(surface.function, "mean", None) is not None and band is not None
            and int(band) <= 512):
        quadrature = float(grid.mean(on))
        assert np.isclose(surface.mean(), quadrature, rtol=0, atol=1e-9 * scale), (
            f"{surface!r}: its mean {surface.mean()} is not its values' "
            f"{quadrature}")


def check_strata(strata: Strata, *, rng: np.random.Generator | None = None,
                 n: int = 64) -> None:
    """Hold `datasets.Strata` to their protocol at `n` random points: each
    stratum's bottom the next one's top; every surface to `check_surface`;
    no stratum of negative thickness (beyond a part in 1e9 of the radius);
    every property, and those below, finite, and read by `value` at the
    top and the bottom of each stratum as its two ends, and between them
    between those ends."""
    rng = np.random.default_rng(0) if rng is None else rng
    for upper, lower in zip(strata.strata[:-1], strata.strata[1:]):
        assert upper.bottom is lower.top, (
            f"{strata!r}: {upper.name}'s bottom is not {lower.name}'s top")
    for surface in strata.surfaces:
        check_surface(surface, rng=rng, n=n)
    theta = np.arccos(rng.uniform(-1.0, 1.0, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    for s in strata:
        top, bottom = s.top.radius(theta, phi), s.bottom.radius(theta, phi)
        assert np.all(top - bottom >= -1e-9 * np.abs(top)), (
            f"{strata!r}: {s.name} is of negative thickness somewhere")
        for name, (at_top, at_bottom) in s.properties.items():
            a = np.asarray(at_top(theta, phi), dtype=float)
            b = np.asarray(at_bottom(theta, phi), dtype=float)
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), (
                f"{strata!r}: {s.name}'s {name} is not finite")
            scale = max(float(np.max(np.abs(np.concatenate([a, b])))), 1e-300)
            assert np.allclose(s.value(name, top, theta, phi), a, rtol=0,
                               atol=1e-12 * scale), (
                f"{strata!r}: {s.name}'s {name} at its top is not its top value")
            thick = top - bottom > 0.0
            assert np.allclose(s.value(name, bottom, theta, phi)[thick], b[thick],
                               rtol=0, atol=1e-12 * scale), (
                f"{strata!r}: {s.name}'s {name} at its bottom is not its bottom "
                "value")
            mid = s.value(name, 0.5 * (top + bottom), theta, phi)
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            assert np.all((mid >= lo - 1e-12 * scale) & (mid <= hi + 1e-12 * scale)), (
                f"{strata!r}: {s.name}'s {name} leaves its ends between them")
    for name, fn in strata.below.items():
        assert np.all(np.isfinite(np.asarray(fn(theta, phi), dtype=float))), (
            f"{strata!r}: the {name} beneath is not finite")


def check_regional(regional: RegionalGrid, *, rng: np.random.Generator | None = None,
                   n: int = 400) -> None:
    """Hold a `datasets.RegionalGrid` to its protocol: at `n` random points
    and the centre of its box, its values of the points' shape and NaN
    outside its region; its edge distance zero outside, positive inside;
    its own coordinates of the box's centre inside its box; and a value
    at that centre, half way down its radii, for some name."""
    from .datasets import RegionalGrid

    assert isinstance(regional, RegionalGrid)
    rng = np.random.default_rng(0) if rng is None else rng
    (r0, r1), (a0, a1), (b0, b1) = regional.extent
    assert r0 < r1 and a0 < a1 and b0 < b1, f"{regional!r}: an empty box"
    a, b = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
    if regional.projection is None:
        centre = (np.array([a]), np.array([b]))
    else:
        centre = regional.projection.to_angles(np.array([b]), np.array([a]))
    back = regional.coordinates(*centre)
    assert np.allclose([back[0][0], back[1][0]], [a, b], rtol=1e-9,
                       atol=1e-9 * max(abs(a1 - a0), abs(b1 - b0))), (
        f"{regional!r}: its coordinates of its box's centre are not the centre")
    theta = np.concatenate([np.arccos(rng.uniform(-1.0, 1.0, n)), centre[0]])
    phi = np.concatenate([rng.uniform(0.0, 2.0 * np.pi, n), centre[1]])
    inside = regional.contains(theta, phi)
    d = regional.edge_distance(theta, phi)
    assert d.shape == theta.shape and np.all(d[~inside] == 0.0), (
        f"{regional!r}: an edge distance outside its region")
    assert np.all(d[inside] >= 0.0), f"{regional!r}: a negative edge distance"
    r = 0.5 * (r0 + r1)
    held = False
    for name in regional.names:
        v = regional(name, r, theta, phi)
        assert v.shape == theta.shape, f"{regional!r}: {name} of the wrong shape"
        assert np.all(np.isnan(v[~inside])), (
            f"{regional!r}: {name} has values outside its region")
        held = held or bool(np.isfinite(v[-1]))
    assert held, f"{regional!r}: nothing at its centre half way down"


def check_surface_file(path: str | os.PathLike[str], grid: AngularGrid, *,
                       fields: Mapping[str, SurfaceVariable],
                       time: ArrayLike | None = None,
                       instant: ArrayLike | None = None,
                       radius: float | None = None,
                       scales: Scales | None = None,
                       meta: Mapping[str, Any] | None = None,
                       citation: Iterable[str] = ()) -> None:
    """Hold a surface file to what it was written from.

    Checks the structure (`surfacefile.validate_structure`); that the
    grid, the time and instant axes, the radius, the scales, the meta
    attributes and the citation read back as given; that `source` and
    `created` are present and `created` parses as ISO 8601; that the
    file holds exactly the given fields (their polar values apart), each
    of its stated support, rules and dimensions, its values the given
    ones with NaN preserved; and that a field's polar values are held
    exactly when they were given.  netCDF4 is imported here.
    """
    from datetime import datetime

    from .surfacefile import SCHEMA, read_surface_file, validate_structure
    from .units import BASE, Scales

    validate_structure(path)
    data = read_surface_file(path)
    assert data.schema == SCHEMA, f"the file's schema is {data.schema!r}"
    assert data.grid.kind == grid.kind and data.grid.lmax == grid.lmax, (
        "the file's grid is not of the given kind and band")
    assert np.allclose(data.grid.colatitudes, grid.colatitudes, rtol=0,
                       atol=1e-12), "the file's colatitudes are not the grid's"
    assert np.allclose(data.grid.longitudes, grid.longitudes, rtol=0,
                       atol=1e-12), "the file's longitudes are not the grid's"
    assert (grid.weights is None) == (data.grid.weights is None), (
        "the file does not hold the grid's weights")
    if grid.weights is not None:
        assert np.allclose(data.grid.weights, grid.weights, rtol=1e-15), (
            "the file's weights are not the grid's")
    for name, want in (("time", time), ("instant", instant)):
        stored = getattr(data, name)
        assert (want is None) == (stored is None), (
            f"the file's {name} axis is not as given")
        if want is not None:
            assert np.array_equal(stored, np.asarray(want, dtype=float)), (
                f"the file's {name} values are not as given")
    if radius is None:
        assert data.radius is None, "the file states a radius it was not given"
    else:
        assert data.radius == float(radius), (
            f"the file's radius is {data.radius!r}, not {radius!r}")
    assert data.scales == (Scales.SI if scales is None else scales), (
        f"the file is in {data.scales!r}")
    assert data.source.strip(), "the file names no source"
    datetime.fromisoformat(data.created)
    for key, value in (meta or {}).items():
        assert data.meta.get(key) == value, (
            f"meta_{key} is {data.meta.get(key)!r}, not {value!r}")
    assert data.citation == tuple(citation), (
        "the file does not cite what it was given")
    plain = {n for n in data.variables if not n.endswith("_poles")}
    assert plain == set(fields), (
        f"the file holds {sorted(plain)}, not {sorted(fields)}")
    sizes = {"static": None, "node": None if time is None else len(np.atleast_1d(
        np.asarray(time))), "instant": None if instant is None else len(
        np.atleast_1d(np.asarray(instant)))}
    for name, field in fields.items():
        var = data.variables[name]
        a = var.attributes
        stated = (a.get("time_support"), a.get("time_interpolation"),
                  a.get("lateral_interpolation"))
        given = (field.support, field.time_interpolation,
                 field.lateral_interpolation)
        assert stated == given, f"{name}: states {stated}, was given {given}"
        assert a.get("long_name") == (field.meaning or name), (
            f"{name}: long_name is {a.get('long_name')!r}")
        if field.dimensions is not None:
            exps = [int(e) for e in np.atleast_1d(a["dimensions"])]
            want_exps = [getattr(field.dimensions, b) for b in BASE]
            assert exps == want_exps, f"{name}: dimensions {exps} != {want_exps}"
        else:
            assert "dimensions" not in a, f"{name}: states undeclared dimensions"
        leading = {"static": (), "node": (sizes["node"],),
                   "interval": (None if sizes["node"] is None
                                else sizes["node"] - 1,),
                   "instant": (sizes["instant"],)}[field.support]
        shape = tuple(leading) + (grid.ntheta, grid.nphi)
        want = np.broadcast_to(np.asarray(field.values, dtype=float), shape)
        assert np.array_equal(var.values, want, equal_nan=True), (
            f"{name}: the file's values are not as given")
        if field.poles is None:
            assert f"{name}_poles" not in data.variables, (
                f"{name}: polar values it was not given")
        else:
            want = np.broadcast_to(np.asarray(field.poles, dtype=float),
                                   tuple(leading) + (2,))
            assert np.array_equal(data.variables[f"{name}_poles"].values, want,
                                  equal_nan=True), (
                f"{name}: the file's polar values are not as given")
