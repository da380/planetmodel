"""Executable contracts for the protocols of planetmodel.

Each `check_*` takes a candidate object, exercises what its protocol
promises, and raises AssertionError naming the first violation.  The
shipped implementations are held to these functions in the test suite;
an implementation written elsewhere is checked by the same call.  The
imports of the rest of the package happen inside the functions.
"""
from __future__ import annotations

import numpy as np

__all__ = ["check_displacement", "check_mapping", "check_geometry"]


def _rel(a, b, *, rtol: float, floor: float = 1.0) -> bool:
    """Whether a and b agree to rtol relative to max(floor, |b|)."""
    return np.allclose(a, b, rtol=0.0, atol=rtol * np.maximum(floor, np.abs(b)))


def check_displacement(h, skeleton, *, n: int = 33) -> None:
    """Hold a RadialDisplacement to its protocol over a skeleton's span.

    Checks finite values of the broadcast shape; single-valuedness on the
    polar axis; continuity across every declared knot inside the
    skeleton; `radial_derivative`, when present,
    against a central difference away from the knots and free of
    undeclared kinks; and the shape of `angular_gradient`.  Every
    tolerance is relative to the skeleton's span or to the value.
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

    knots = tuple(getattr(h, "knots", ()))
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


def check_mapping(m, points, *, rtol: float = 1e-6,
                  step: float | None = None) -> None:
    """Hold a Mapping to its protocol on the given Cartesian points.

    Checks the shapes; `deformation_gradient` against a central
    difference of `m` with `F[i, j] = d m_i / d X_j`; `jacobian` against
    det F; and, when present, `displacement` against m(X) - X and
    `inverse` as a round trip.  The difference step is 1e-6 of each
    point's radius unless `step` gives one step for every point.
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
    Fd = np.empty_like(F)
    for j in range(3):
        e = np.zeros_like(X)
        e[..., j] = hs
        Fd[..., :, j] = (np.asarray(m(X + e), dtype=float)
                         - np.asarray(m(X - e), dtype=float)) / (2.0 * hs[..., None])
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


def check_geometry(geometry) -> None:
    """Hold a Geometry to its invariants and its numbering.

    Checks that the invariants hold (by reconstructing the geometry with
    checks on), that names are unique, that the interfaces sit on the
    skeleton's boundaries with consistent `between`, that `scaled(k)`
    round-trips, and that the mapping passes `check_mapping` on the
    validity lattice.
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
    # points well inside every layer, away from the boundaries where the
    # mapping's gradient may jump
    _, theta, phi = validity_lattice(sk, n_theta=5, n_phi=4)
    r = np.concatenate([lo + (hi - lo) * np.array([0.25, 0.5, 0.75])
                        for lo, hi in zip(b[:-1], b[1:])])[:, None, None]
    X = cartesian_points(r, theta, phi)
    assert np.allclose(twice.mapping(X), geometry.mapping(X), rtol=1e-10,
                       atol=1e-10 * max(1.0, float(sk.span))), (
        "scaled(k).scaled(1/k) does not restore the mapping")

    check_mapping(geometry.mapping, X.reshape(-1, 3))
