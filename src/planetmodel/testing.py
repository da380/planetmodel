"""Executable contracts for the protocols of planetmodel.

Each `check_*` takes a candidate object, exercises what its protocol
promises, and raises AssertionError naming the first violation.  The
shipped implementations are held to these functions in the test suite;
an implementation written elsewhere is checked by the same call.  The
imports of the rest of the package happen inside the functions.
"""
from __future__ import annotations

import numpy as np

__all__ = ["check_displacement", "check_mapping", "check_geometry",
           "check_layer_function", "check_field", "check_model",
           "check_sample"]


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


def check_layer_function(fn, *, n: int = 17, rtol: float = 1e-6) -> None:
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


def check_field(field, *, rng=None, n: int = 64) -> None:
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


def check_model(model, *, rng=None) -> None:
    """Hold a Model to its invariants.

    Checks that construction with checks on accepts it; that every
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
    cls(model.geometry, [layer.fields for layer in model.layers],
        scales=model.scales, specs=model.specs, constants=model.constants,
        check=True)
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
                       mass=0.5 * model.scales.mass, time=2.0 * model.scales.time)
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


def check_sample(sample, model, *, rng=None, n: int = 64,
                 rtol: float = 1e-12) -> None:
    """Hold a Sample to its layout and to the model it was taken from.

    Layout: the radial mesh lies over the model's skeleton and the nodes
    are its per-element GLL nodes flattened; the angular nodes increase
    inside their open ranges; every field is float64 and read-only,
    shaped `(nnode,) + c` exactly when every layer holding it declares
    itself radial and `(nnode, ntheta, nphi) + c` otherwise, with `c`
    the stored shape of its character.  Metadata: the characters and
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
    assert np.all(np.diff(theta) > 0) and 0 < theta[0] and theta[-1] < np.pi, (
        "colatitudes are not increasing inside (0, pi)")
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
