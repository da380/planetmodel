"""testing.py -- executable contracts for planetmodel's protocols.

A contract check is a function that takes an object claiming to
implement one of the library's protocols and exercises everything the
protocol promises, raising AssertionError at the first violation with a
message naming what failed.  Write your own Field, Topography,
RadialDisplacement, Mapping, law or model class, run the matching check
on it, and you know whether planetmodel can use it without reading the
source:

    from planetmodel.testing import check_field
    check_field(my_field)

planetmodel's own implementations are held to exactly these functions in
CI, so the contracts cannot drift from what is checked.  The checks:

    check_layer_function             a callable on one layer's interval
    check_field                      a static Field on a Skeleton
    check_frequency_dependent_field  a Field taking omega
    check_time_dependent_field       a Field taking t
    check_law                        a rheological law's field and record
    check_model                      a model class and its guarantees
    check_topography                 a shape on the sphere
    check_displacement               a radial displacement and its knots
    check_mapping                    a mapping, its gradient and Jacobian
    check_sample                     a Sample's layout and its source

Each returns None on success.  Where a protocol member is optional the
check says so and exercises it only where the object provides it.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "check_layer_function", "check_field", "check_frequency_dependent_field",
    "check_time_dependent_field", "check_law", "check_model",
    "check_topography", "check_displacement", "check_mapping", "check_sample",
]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _fd(f, x, h):
    """Central difference of a scalar function of one variable."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _frame_matrix(theta, phi):
    """[e_r, e_theta, e_phi] as columns, the library's own definition.

    Taken from `model.frames` rather than rewritten here: a contract that
    checks a rotation against its own second copy of the rotation checks
    nothing.
    """
    from .model.frames import spherical_frame
    return spherical_frame(theta, phi)


def _trailing_shape(character) -> tuple[int, ...]:
    """The component shape a value carries: Voigt where the character has one."""
    shape = character.voigt_shape
    return character.component_shape if shape is None else shape


def _domain_of(field) -> tuple[int, ...]:
    """The layers a field is defined on; every layer where it does not say."""
    domain = tuple(getattr(field, "domain", range(field.skeleton.nlayers)))
    assert domain, "a field must be defined on at least one layer"
    return domain


def _domain_sample(field, rng, n: int, *, margin: float = 0.0) -> np.ndarray:
    """`n` radii drawn uniformly from the layers of a field's domain.

    `margin` keeps the draw away from the layer boundaries by that
    fraction of each layer's thickness.
    """
    sk = field.skeleton
    domain = _domain_of(field)
    lo = np.array([sk.interval(i)[0] for i in domain])
    hi = np.array([sk.interval(i)[1] for i in domain])
    which = rng.integers(0, len(domain), size=n)
    u = rng.uniform(margin, 1.0 - margin, size=n)
    return lo[which] + u * (hi[which] - lo[which])


def _midpoints(field) -> np.ndarray:
    """One radius per layer of the domain: the layer midpoints."""
    sk = field.skeleton
    return np.array([0.5 * sum(sk.interval(i)) for i in _domain_of(field)])


def _angles_for(field, r) -> tuple:
    """The angles a field needs: none if it accepts r alone, else a
    generic direction away from the poles and the seam.

    A field that refuses r alone must say so by naming the angles, and
    must not claim `is_radial`.
    """
    try:
        field.evaluate(r)
    except ValueError as exc:
        msg = str(exc).lower()
        assert "theta" in msg or "phi" in msg or "angle" in msg, (
            "a field that will not be called with r alone must say that the "
            f"angles are what it wants; it raised {exc!r}")
        assert not getattr(field, "is_radial", False), (
            "the field says is_radial, so it must accept r alone")
        return (0.7, 0.4)
    return ()


def _is_lift(field) -> bool:
    """Whether a dependent field is a static field lifted unchanged.

    Recognised by attribute: a lift exposes its one static operand as
    `source` and applies neither a function nor a factor to it.
    """
    from .model.fields.dependent import kind_of
    src = getattr(field, "source", None)
    if src is None or hasattr(field, "fn") or hasattr(field, "factor"):
        return False
    operands = getattr(field, "operands", None)
    if not callable(operands) or tuple(operands()) != (src,):
        return False
    return kind_of(src) == "static"


# ---------------------------------------------------------------------------
# layer functions
# ---------------------------------------------------------------------------

def check_layer_function(fn, interval, *, n: int = 17, rtol: float = 1e-6) -> None:
    """A layer function is callable on arrays over its interval.

    It returns finite float values of the argument's shape, and an array
    call agrees with a single-point one.  Where it provides
    `derivative(nu)` and `integrate(a, b)` they are checked against a
    finite difference and a quadrature: an exact-looking derivative that
    disagrees with the function it claims to differentiate is worse than
    none.
    """
    lo, hi = float(interval[0]), float(interval[1])
    assert hi > lo, f"interval {interval} is not increasing"
    r = np.linspace(lo, hi, n)

    v = np.asarray(fn(r), dtype=float)
    assert v.shape == r.shape, f"expected shape {r.shape}, got {v.shape}"
    assert np.all(np.isfinite(v)), "layer function returned non-finite values"

    scalar = np.asarray(fn(np.array([r[n // 2]])))
    assert np.allclose(scalar, v[n // 2]), "array and single-point calls disagree"

    if hasattr(fn, "derivative"):
        d = fn.derivative()
        h = (hi - lo) * 1e-5
        mid = r[2:-2]
        got = np.asarray(d(mid), dtype=float)
        want = np.array([_fd(lambda x: float(fn(np.array([x]))[0]), x, h)
                         for x in mid])
        scale = max(1.0, float(np.max(np.abs(want))))
        assert np.allclose(got, want, rtol=rtol, atol=rtol * scale), (
            "derivative() disagrees with a finite difference of the function")

    if hasattr(fn, "integrate"):
        fine = np.linspace(lo, hi, 4001)
        want = np.trapezoid(np.asarray(fn(fine), dtype=float), fine)
        got = float(fn.integrate(lo, hi))
        scale = max(1.0, abs(want))
        assert abs(got - want) < 1e-4 * scale, (
            f"integrate() = {got} disagrees with quadrature {want}")


# ---------------------------------------------------------------------------
# static fields
# ---------------------------------------------------------------------------

def check_field(field, *, rng=None, n: int = 64) -> None:
    """The contract of a static Field.

    A Field carries `skeleton`, `character`, `name` and `evaluate`, and
    `dimensions` where it has any.  `evaluate(r, theta, phi)` broadcasts
    its three arguments as independent axes and returns values of the
    broadcast shape followed by the character's component shape, Voigt
    where the character has one; a scalar radius drops the leading axis
    and agrees with the array call.  A field that accepts `r` alone is
    radial and must say so with `is_radial`; one that depends on angle
    refuses `r` alone by naming the angles.  Both sides of every
    interior boundary are reachable with `side=`, and where the field is
    layer-indexed they equal the pieces.  A single-layer field's
    `__call__` is `evaluate` with the defaults.  Radii in a layer outside
    the field's `domain`, and radii outside the skeleton, are refused by
    name rather than extrapolated.  `frame="cartesian"` is honoured or
    refused by name: a rank-1 value rotates by R, a rank-2 Voigt vector
    by the Bond matrix once, a rank-4 Voigt matrix by the Bond matrix
    twice.  Where the field provides them, `evaluate_at(X)` equals
    `evaluate(..., frame="cartesian")` at the same points,
    `restricted(i)` lives on layer i alone and agrees with the field
    there, and `rescaled(convert, old, new)` round-trips through a
    change of scales to 1e-12 or refuses by name.

    Not covered: derivatives and integrals (see `check_layer_function`),
    and the argument of a frequency- or time-dependent field (see the
    two dependent-field checks, which call this one at fixed arguments).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    _check_field_attributes(field)
    r = _domain_sample(field, rng, n)
    angles = _angles_for(field, r)
    v = _check_broadcasting(field, r, angles, n)
    _check_boundaries(field, angles)
    _check_call(field, r, angles, v)
    _check_domain(field, angles)
    values = _check_angular(field, rng, v)
    _check_evaluate_at(field, values)
    _check_restricted(field)
    _check_rescaled(field, r, angles)


def _check_field_attributes(field) -> None:
    for attr in ("skeleton", "character", "name", "evaluate"):
        assert hasattr(field, attr), f"a Field needs {attr!r}"
    dims = getattr(field, "dimensions", None)
    if dims is not None:
        from .model.units import Dimensions
        assert isinstance(dims, Dimensions), (
            f"dimensions must be a units.Dimensions, got {type(dims).__name__}")


def _check_broadcasting(field, r, angles, n: int) -> np.ndarray:
    """Shape, finiteness, scalar-versus-array, and input shape preserved."""
    v = np.asarray(field.evaluate(r, *angles), dtype=float)
    assert v.shape[:1] == r.shape, f"expected leading shape {r.shape}, got {v.shape}"
    assert np.all(np.isfinite(v)), "evaluate returned non-finite values"

    one = field.evaluate(float(r[0]), *angles)
    assert np.shape(one) == v.shape[1:], "scalar radius should drop the leading axis"
    assert np.allclose(one, v[0]), "scalar and array evaluation disagree"

    shaped = np.asarray(field.evaluate(r.reshape(-1, 1), *angles))
    assert shaped.shape[:2] == (n, 1), "evaluate does not preserve input shape"

    trailing = _trailing_shape(field.character)
    assert v.shape == r.shape + trailing, (
        f"a rank-{field.character.rank} field should return "
        f"{r.shape + trailing}, got {v.shape}")
    return v


def _check_boundaries(field, angles) -> None:
    """Both sides of every interior boundary inside the domain."""
    sk = field.skeleton
    b = np.asarray(sk.boundaries, dtype=float)
    domain = _domain_of(field)
    for j in range(1, sk.nlayers):
        if j not in domain or j - 1 not in domain:
            continue
        x = float(b[j])
        up = field.evaluate(x, *angles)
        lo = field.evaluate(x, *angles, side="lower")
        assert np.all(np.isfinite(np.asarray(up, dtype=float)))
        assert np.all(np.isfinite(np.asarray(lo, dtype=float)))
        if hasattr(field, "__getitem__"):
            assert np.allclose(up, field[j](x)), (
                f"side='upper' at boundary {j} is not the layer above")
            assert np.allclose(lo, field[j - 1](x)), (
                f"side='lower' at boundary {j} is not the layer below")


def _check_call(field, r, angles, v) -> None:
    """Calling the field is evaluate with the defaults."""
    if callable(field):
        assert np.allclose(np.asarray(field(r, *angles), dtype=float), v), (
            "calling the field is not evaluate with the defaults")


def _check_domain(field, angles) -> None:
    """Refusal by name outside the domain, and outside the skeleton."""
    sk = field.skeleton
    b = np.asarray(sk.boundaries, dtype=float)
    domain = _domain_of(field)
    for j in range(sk.nlayers):
        if j in domain:
            continue
        probe = np.full(3, 0.5 * sum(sk.interval(j)))
        try:
            field.evaluate(probe, *angles)
        except ValueError as exc:
            assert str(j) in str(exc) or "domain" in str(exc).lower(), (
                f"a radius in layer {j}, outside the domain {domain}, must "
                f"be refused by name; it raised {exc!r}")
        else:
            raise AssertionError(
                f"layer {j} is outside the domain {domain} and a radius in "
                "it was answered")

    for bad in (b[0] - 1.0, b[-1] + 1.0):
        try:
            field.evaluate(bad, *angles)
        except ValueError:
            pass
        else:
            raise AssertionError(f"radius {bad} outside the skeleton was accepted")


def _check_angular(field, rng, v) -> dict:
    """Three broadcast axes, `is_radial`, the frames and their rotations.

    Returns the values in each frame the field supplies, on the grid
    used, for the checks that follow.
    """
    char = field.character
    trailing = _trailing_shape(char)
    # One radius per layer rather than a random handful: a field may be
    # uniform over most of the body, and a check that samples only where
    # nothing varies cannot fail.
    mid = _midpoints(field)
    na, nt, np_ = mid.size, 3, 2
    ra = mid.reshape(na, 1, 1)
    th = rng.uniform(0.0, np.pi, size=(1, nt, 1))
    ph = rng.uniform(-np.pi, np.pi, size=(1, 1, np_))
    grid = np.asarray(field.evaluate(ra, th, ph), dtype=float)
    assert grid.shape == (na, nt, np_) + trailing, (
        f"r, theta and phi must broadcast as three axes: expected "
        f"{(na, nt, np_) + trailing}, got {grid.shape}")
    assert np.all(np.isfinite(grid)), "evaluate returned non-finite values"

    if getattr(field, "is_radial", False):
        alone = np.asarray(field.evaluate(ra), dtype=float)
        assert np.allclose(np.broadcast_to(alone, grid.shape), grid), (
            "the field says is_radial, so calling it with r alone must give "
            "what calling it with r, theta and phi gives")

    values = {}
    for frame in ("spherical", "cartesian"):
        try:
            values[frame] = np.asarray(
                field.evaluate(ra, th, ph, frame=frame), dtype=float)
        except ValueError as exc:
            msg = str(exc).lower()
            assert frame in msg or "frame" in msg, (
                f"a field that will not supply the {frame!r} frame must say "
                f"so by name; it raised {exc!r}")
        else:
            assert values[frame].shape == grid.shape, (
                f"frame={frame!r} changed the shape to {values[frame].shape}")
    assert values, "a Field must support at least one frame"
    assert np.allclose(values.get("spherical", grid), grid), (
        "frame='spherical' is the default and must give the default values")

    if len(values) == 2:
        _check_frame_rotation(char, values, th, ph)
    values["_grid"] = (ra, th, ph)
    return values


def _check_frame_rotation(char, values, th, ph) -> None:
    """Cartesian components are the rotation of the spherical ones.

    Rank 1 rotates by R, rank 2 by the Bond matrix once and rank 4 by
    the Bond matrix twice; a rank-4 field with no Voigt form has no Bond
    matrix to cast and is not asked to be the rotation of anything.
    """
    from .model.materials import bond_matrix
    R = _frame_matrix(th, ph)
    if char.rank == 4 and char.voigt_shape is not None:
        M = bond_matrix(R)
        want = M @ values["spherical"] @ np.swapaxes(M, -1, -2)
        how = "the Bond rotation"
    elif char.rank == 1:
        want = np.einsum("...ij,...j->...i", R, values["spherical"])
        how = "R v"
    elif char.rank == 2:
        want = np.einsum("...ab,...b->...a", bond_matrix(R), values["spherical"])
        how = "M v, for M the Bond matrix of R"
    else:
        return
    scale = max(1.0, float(np.max(np.abs(want))))
    assert np.allclose(values["cartesian"], want, rtol=1e-12,
                       atol=1e-12 * scale), (
        f"the Cartesian components are not {how} of the spherical ones: the "
        "largest discrepancy is "
        f"{float(np.max(np.abs(values['cartesian'] - want))):.3e}")


def _check_evaluate_at(field, values) -> None:
    """evaluate_at(X) is evaluate(r, theta, phi, frame='cartesian')."""
    if not (hasattr(field, "evaluate_at") and "cartesian" in values):
        return
    ra, th, ph = values["_grid"]
    rb, tb, pb = np.broadcast_arrays(ra, th, ph)
    st = np.sin(tb)
    X = np.stack([rb * st * np.cos(pb), rb * st * np.sin(pb), rb * np.cos(tb)],
                 axis=-1)
    got = np.asarray(field.evaluate_at(X), dtype=float)
    assert got.shape == values["cartesian"].shape, (
        f"evaluate_at gave {got.shape}, expected {values['cartesian'].shape}")
    scale = max(1.0, float(np.max(np.abs(values["cartesian"]))))
    assert np.allclose(got, values["cartesian"], rtol=1e-9, atol=1e-9 * scale), (
        "evaluate_at(X) is not evaluate(r, theta, phi, frame='cartesian') at "
        "the same points: the frame follows the coordinates, so Cartesian "
        "points must give Cartesian components of the same field")


def _check_restricted(field) -> None:
    """restricted(i) lives on layer i alone, agrees with the field there,
    is the piece where the field has pieces, and refuses other radii."""
    sk = field.skeleton
    if not (hasattr(field, "restricted") and sk.nlayers > 1):
        return
    b = np.asarray(sk.boundaries, dtype=float)
    probe = (0.7, 0.4)
    for j in _domain_of(field):
        lo, hi = sk.interval(j)
        rest = field.restricted(j)
        assert rest.skeleton.nlayers == 1, (
            f"restricted({j}) should live on one layer, it lives on "
            f"{rest.skeleton!r}")
        assert np.allclose(np.asarray(rest.skeleton.boundaries), [lo, hi]), (
            f"restricted({j}) lives on {rest.skeleton!r}, not on layer "
            f"{j} [{lo:.6g}, {hi:.6g}]")
        inside = np.full(3, 0.5 * (lo + hi))
        assert np.allclose(
            np.asarray(rest.evaluate(inside, *probe), dtype=float),
            np.asarray(field.evaluate(inside, *probe, layer=j), dtype=float)), (
            f"restricted({j}) disagrees with the field on the layer it restricts")
        assert rest.character == field.character, (
            f"restricted({j}) changed the character")
        if hasattr(field, "__getitem__"):
            assert rest is field[j] or np.allclose(
                np.asarray(rest.evaluate(inside, *probe), dtype=float),
                np.asarray(field[j].evaluate(inside, *probe), dtype=float)), (
                f"restricted({j}) is not the field's piece on layer {j}")
        if callable(rest):
            assert np.allclose(
                np.asarray(rest(inside, *probe), dtype=float),
                np.asarray(rest.evaluate(inside, *probe), dtype=float)), (
                f"restricted({j})(r) is not restricted({j}).evaluate(r)")
        outside = lo - 0.5 * (hi - lo) if lo > b[0] else hi + 0.5 * (hi - lo)
        try:
            rest.evaluate(np.full(3, outside), *probe)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"restricted({j}) accepted a radius outside its layer: a "
                "restriction that still answers outside its span is not one")


def _check_rescaled(field, r, angles, *, rtol: float = 1e-12) -> None:
    """rescaled(convert, old, new) round-trips through a change of scales,
    or refuses by name.

    The field is taken from SI to a geophysical set of scales and back;
    the result must evaluate to the original values at the original
    radii.  A field that cannot be re-expressed raises a TypeError or a
    ValueError that names its type or its name.
    """
    if not hasattr(field, "rescaled"):
        return
    from .model.units import Scales
    old = Scales.SI
    new = Scales.geophysical(float(field.skeleton.boundaries[-1]))

    def converter(a, b):
        seen: dict = {}

        def convert(f):
            if id(f) not in seen:
                seen[id(f)] = f.rescaled(convert, a, b)
            return seen[id(f)]
        return convert

    try:
        there = converter(old, new)(field)
    except (TypeError, ValueError) as exc:
        msg = str(exc)
        assert type(field).__name__ in msg or repr(field.name) in msg, (
            "a field that will not rescale must refuse by naming its type or "
            f"its name; it raised {exc!r}")
        return
    assert there.character == field.character, "rescaled changed the character"
    back = converter(new, old)(there)
    want = np.asarray(field.evaluate(r, *angles), dtype=float)
    got = np.asarray(back.evaluate(r, *angles), dtype=float)
    scale = max(1.0, float(np.max(np.abs(want))))
    assert got.shape == want.shape, (
        f"rescaling there and back changed the shape to {got.shape}")
    assert np.allclose(got, want, rtol=rtol, atol=rtol * scale), (
        "rescaling to geophysical scales and back does not return the field: "
        f"the largest discrepancy is {float(np.max(np.abs(got - want))):.3e}")


# ---------------------------------------------------------------------------
# frequency- and time-dependent fields
# ---------------------------------------------------------------------------

def _check_dependent_field(field, *, kind: str, args, rng=None, n: int = 64,
                           rtol: float = 1e-12) -> None:
    """The contract shared by the two dependent-field kinds."""
    from .model.fields.frequency import at_frequency
    from .model.fields.time import at_time
    arg_name = {"frequency": "omega", "time": "t"}[kind]
    for attr in ("skeleton", "character", "name", "evaluate", "kind"):
        assert hasattr(field, attr), f"a {kind}-dependent field needs {attr!r}"
    assert field.kind == kind, (
        f"the field says kind {field.kind!r}, the check is for {kind!r}")
    args = list(args)
    assert args, f"give at least one {arg_name} to check at"
    rng = np.random.default_rng(0) if rng is None else rng

    # At every value of the argument the field is a static Field and the
    # whole static contract holds of it; the real part is what a static
    # consumer sees, so that is what is frozen here.
    freeze = (lambda a: at_frequency(field, a, part="real")) \
        if kind == "frequency" else (lambda a: at_time(field, a))
    for a in args:
        frozen = freeze(a)
        assert getattr(frozen, "kind", "static") == "static", (
            f"the field frozen at {arg_name}={a} is not a static field")
        check_field(frozen, rng=rng, n=n)

    r = _domain_sample(field, rng, n)
    angles = () if getattr(field, "is_radial", False) else (0.7, 0.4)

    def ev(a, **kw):
        return field.evaluate(r, *angles, **{arg_name: a}, **kw)

    _check_scalar_argument(ev, args, arg_name)
    if kind == "frequency":
        _check_parts(field, ev, args, rtol)
        _check_omega_domain(field, ev, args)
    else:
        for a in args:
            v = np.asarray(ev(a))
            assert v.dtype == np.float64, f"a time field gave {v.dtype}"
        try:
            ev(1j)
        except ValueError as exc:
            assert "t" in str(exc), f"a complex t must be refused; got {exc!r}"
        else:
            raise AssertionError("a complex t was accepted by a time field")

    if hasattr(field, "evaluate_with"):
        for a in args:
            th, ph = (angles if angles else (None, None))
            got = np.asarray(field.evaluate_with(r, th, ph, a, layer=None,
                                                 side="upper", frame="spherical"))
            want = np.asarray(ev(a))
            assert np.allclose(got, want, rtol=rtol, atol=rtol), (
                f"evaluate_with disagrees with evaluate at {arg_name}={a}")

    if _is_lift(field):
        src = field.source
        for a in args:
            want = np.asarray(src.evaluate(r, *angles), dtype=float)
            got = np.asarray(ev(a, part="real") if kind == "frequency"
                             else ev(a), dtype=float)
            assert np.allclose(got, want, rtol=rtol, atol=rtol), (
                f"the lift differs from its source at {arg_name}={a}")
            if kind == "frequency":
                assert np.all(np.asarray(ev(a, part="imag")) == 0.0), (
                    "a lifted field has no imaginary part")

    _check_dependent_restricted(field, kind, arg_name, args[0], angles, rtol)


def _check_scalar_argument(ev, args, arg_name) -> None:
    """An array argument is refused as non-scalar."""
    try:
        ev(np.array(args[:2] if len(args) > 1 else [args[0], args[0]]))
    except ValueError as exc:
        assert "scalar" in str(exc).lower(), (
            f"an array {arg_name} must be refused as non-scalar, not with "
            f"{exc!r}")
    else:
        raise AssertionError(f"an array {arg_name} was accepted; the "
                             "protocol takes a scalar")


def _check_parts(field, ev, args, rtol) -> None:
    """The three parts agree, and complex is the default."""
    for a in args:
        c = np.asarray(ev(a, part="complex"))
        re = np.asarray(ev(a, part="real"))
        im = np.asarray(ev(a, part="imag"))
        assert c.dtype == np.complex128, f"part='complex' gave {c.dtype}"
        assert re.dtype == np.float64 and im.dtype == np.float64, (
            "part='real' and 'imag' must give float64")
        scale = max(1.0, float(np.max(np.abs(c))))
        assert np.allclose(re + 1j * im, c, rtol=rtol, atol=rtol * scale), (
            f"real + i imag != complex at omega={a}")
        default = np.asarray(ev(a))
        assert default.dtype == np.complex128 and np.array_equal(default, c), (
            "the default part is 'complex': evaluate without part= must give "
            "the complex values")


def _check_omega_domain(field, ev, args) -> None:
    """A real-axis field refuses a complex omega by name; a complex one
    answers finitely there."""
    assert field.omega_domain in ("real", "complex"), (
        f"omega_domain must be 'real' or 'complex', got {field.omega_domain!r}")
    probe = complex(abs(args[0]) or 1.0, abs(args[0]) or 1.0)
    if field.omega_domain == "real":
        try:
            ev(probe)
        except ValueError as exc:
            assert "omega" in str(exc), (
                "a real-axis field refusing a complex omega must say 'omega'; "
                f"it raised {exc!r}")
        else:
            raise AssertionError(
                "omega_domain is 'real' but a complex omega was accepted")
        on_axis = np.asarray(ev(complex(args[0], 0.0)))
        assert np.allclose(on_axis, np.asarray(ev(args[0]))), (
            "a complex omega with zero imaginary part is a real omega and "
            "must give the same values")
    else:
        c = np.asarray(ev(probe, part="complex"))
        assert np.all(np.isfinite(c)), "non-finite values at complex omega"


def _check_dependent_restricted(field, kind, arg_name, a, angles, rtol) -> None:
    """restricted(j) is a field of the same kind agreeing on its layer."""
    sk = field.skeleton
    if not (hasattr(field, "restricted") and sk.nlayers > 1):
        return
    for j in _domain_of(field):
        lo, hi = sk.interval(j)
        piece = field.restricted(j)
        assert getattr(piece, "kind", None) == kind, (
            f"restricted({j}) is not a {kind}-dependent field")
        inside = np.full(3, 0.5 * (lo + hi))
        got = piece.evaluate(inside, *angles, **{arg_name: a})
        want = field.evaluate(inside, *angles, layer=j, **{arg_name: a})
        assert np.allclose(np.asarray(got), np.asarray(want), rtol=rtol,
                           atol=rtol), (
            f"restricted({j}) disagrees with the field on its layer")


def check_frequency_dependent_field(field, *, omegas, rng=None, n: int = 64,
                                    rtol: float = 1e-12) -> None:
    """The contract of a frequency-dependent field.

    The field carries `kind == "frequency"` and `omega_domain`.  At every
    omega in `omegas` the field frozen there by `at_frequency` is a
    static Field satisfying `check_field`.  `part` is "complex" by
    default; "real" and "imag" are float64 and add up to it.  An array
    omega is refused as non-scalar.  A field with `omega_domain ==
    "real"` refuses a complex omega by name and treats one with zero
    imaginary part as real; one with "complex" answers finitely off the
    axis.  `evaluate_with` agrees with `evaluate`.  A lifted field
    equals its static source at every omega with no imaginary part.
    `restricted(j)` is a frequency-dependent field agreeing with the
    field on its layer.
    """
    _check_dependent_field(field, kind="frequency", args=omegas, rng=rng, n=n,
                           rtol=rtol)


def check_time_dependent_field(field, *, ts, rng=None, n: int = 64,
                               rtol: float = 1e-12) -> None:
    """The contract of a time-dependent field: as the frequency one with
    a real scalar `t`, float64 values, and a complex `t` refused."""
    _check_dependent_field(field, kind="time", args=ts, rng=rng, n=n, rtol=rtol)


# ---------------------------------------------------------------------------
# laws
# ---------------------------------------------------------------------------

def check_law(field, *, omegas, oracle, rng=None, n: int = 32,
              rtol: float = 1e-12) -> None:
    """A law's field is its formula, pointwise, and carries its record.

    `field` is what a law returned; `oracle(omega, r, theta, phi)` is a
    naive per-point evaluation of the formula the law claims to
    implement, written independently of it.  The field passes the
    frequency-dependent contract at every omega, agrees with the oracle
    at random points of its domain and every omega to `rtol`, and
    carries a `LawRecord` as `law` whose `law` is registered under
    "rheology" and whose `parameters` name at least one field.
    """
    from .model.rheology import LawRecord
    from .registry import registered
    check_frequency_dependent_field(field, omegas=omegas, rng=rng, n=n,
                                    rtol=rtol)
    record = getattr(field, "law", None)
    assert isinstance(record, LawRecord), (
        f"a law's field carries a LawRecord as .law; got {record!r}")
    assert record.law in registered("rheology"), (
        f"law {record.law!r} is not registered under 'rheology'")
    assert record.parameters, "a LawRecord names the fields the law read"

    rng = np.random.default_rng(1) if rng is None else rng
    r = _domain_sample(field, rng, n, margin=0.05)
    theta = rng.uniform(0.2, np.pi - 0.2, size=n)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    for omega in omegas:
        got = np.asarray(field.evaluate(r, theta, phi, omega=omega))
        want = np.asarray(oracle(omega, r, theta, phi))
        scale = max(1.0, float(np.max(np.abs(want))))
        assert got.shape == want.shape, (
            f"the law gives shape {got.shape}, the oracle {want.shape}")
        assert np.allclose(got, want, rtol=rtol, atol=rtol * scale), (
            f"the law differs from its formula at omega={omega}: worst "
            f"{float(np.max(np.abs(got - want))):.3e} against a scale of "
            f"{scale:.3e}")


# ---------------------------------------------------------------------------
# model classes
# ---------------------------------------------------------------------------

def check_model(model, *, rng=None) -> None:
    """A model class guarantees its fields, exposes them, and survives
    surgery and conversion.

    `validate()` holds and is idempotent: every layer holding any
    required field holds them all, and some layer holds them all.  Each
    aspect's properties that return a Field return the stitched view of
    a field the aspect requires.  `guaranteed_layers` is what it says.
    `truncated`, `extended`, `refined` and `coarsened` return the same
    class, still valid; so does `as_class` through a plain
    `ReferenceBody` and back.  For a `ViscoelasticModel`,
    `viscoelastic_moduli` is a frequency-dependent rank-4 field on
    exactly the layers with moduli, `moduli_at(omega)` is a static field
    equal to it there, and a layer holding only `elastic_moduli` is
    lifted at view time with nothing stored on the layer.
    """
    from .model.body import Field, ReferenceBody
    from .model.classes import ModelBase, ViscoelasticModel
    assert isinstance(model, ModelBase), (
        f"check_model checks a model class; got {type(model).__name__}")
    cls = type(model)
    required = cls.required_fields()
    assert required, f"{cls.__name__} requires no fields"
    model.validate()
    model.validate()
    for aspect in cls.ASPECTS:
        for name in aspect.REQUIRES:
            assert name in model, f"{cls.__name__} lacks {name!r}"
        for prop_name, prop in vars(aspect).items():
            if not isinstance(prop, property):
                continue
            value = getattr(model, prop_name)
            if not isinstance(value, Field):
                continue
            assert any(value is model[n] for n in aspect.REQUIRES), (
                f"{cls.__name__}.{prop_name} returns a field that is not the "
                f"view of any of {aspect.__name__}'s required "
                f"{aspect.REQUIRES}")
    assert model.guaranteed_layers, "no layer holds every required field"
    for i in model.guaranteed_layers:
        assert all(n in model.layer(i).fields for n in required)

    _check_model_surgery(model, cls)

    plain = model.as_class(ReferenceBody)
    assert type(plain) is ReferenceBody, "as_class(ReferenceBody) is not plain"
    again = plain.as_class(cls)
    assert type(again) is cls, f"as_class back gave {type(again).__name__}"
    again.validate()
    assert again.field_names == model.field_names, (
        "as_class there and back changed the stored fields")

    if isinstance(model, ViscoelasticModel):
        _check_viscoelastic(model, cls, rng)


def _check_model_surgery(model, cls) -> None:
    sk = model.skeleton
    b = np.asarray(sk.boundaries, dtype=float)
    outer = float(b[-1])
    cut = model.truncated(0.5 * (b[-2] + b[-1]))
    grown = model.extended([outer * 1.1])
    fine = model.refined([0.5 * (b[0] + b[1])])
    for name, other in (("truncated", cut), ("extended", grown),
                        ("refined", fine)):
        assert type(other) is cls, (
            f"{name} returned {type(other).__name__}, not {cls.__name__}")
        other.validate()
    if sk.nlayers > 1:
        with warnings.catch_warnings():
            # A field held on one of the merged layers alone is dropped by
            # rule, and the check wants the class, not the field.
            warnings.filterwarnings("ignore", message="coarsening layers")
            coarse, _ = model.coarsened(drop=[0], state=model.layer(1).state)
        assert type(coarse) is cls, (
            f"coarsened returned {type(coarse).__name__}, not {cls.__name__}")
        coarse.validate()


def _check_viscoelastic(model, cls, rng) -> None:
    from .model.classes import ViscoelasticModel
    sk = model.skeleton
    dyn = model.viscoelastic_moduli
    assert getattr(dyn, "kind", None) == "frequency", (
        "viscoelastic_moduli is not a frequency-dependent field")
    assert dyn.character.rank == 4, "viscoelastic_moduli is not rank 4"
    assert tuple(dyn.domain) == tuple(model.layers_with("elastic_moduli")), (
        "viscoelastic_moduli is not defined on exactly the layers with moduli")
    rng = np.random.default_rng(0) if rng is None else rng
    i = model.guaranteed_layers[0]
    lo, hi = sk.interval(i)
    r = np.full(3, 0.5 * (lo + hi))
    omega = 1.0
    frozen = model.moduli_at(omega)
    assert frozen.kind == "static", "moduli_at(omega) is not a static field"
    assert np.allclose(frozen.evaluate(r), dyn.evaluate(r, omega=omega)), (
        "moduli_at(omega) is not viscoelastic_moduli at that omega")
    real = model.moduli_at(omega, part="real")
    assert np.allclose(real.evaluate(r), np.real(dyn.evaluate(r, omega=omega))), (
        "moduli_at(omega, part='real') is not the real part of "
        "viscoelastic_moduli there")
    check_field(real, rng=rng)

    # An elastic layer is lifted at view time, with nothing stored.
    name = ViscoelasticModel.VISCOELASTIC
    elastic_only = model.without_field(name) if name in model else model
    lifted = elastic_only.as_class(cls)
    assert all(name not in lay.fields for lay in lifted.layers), (
        f"a layer holding only elastic_moduli must not have {name!r} stored "
        "on it: the class lifts at view time")
    view = lifted.viscoelastic_moduli
    assert tuple(view.domain) == tuple(lifted.layers_with("elastic_moduli")), (
        "the lifted view is not defined on exactly the layers with moduli")
    want = np.asarray(lifted["elastic_moduli"].evaluate(r), dtype=float)
    got = np.asarray(view.evaluate(r, omega=omega))
    assert np.allclose(got, want) and np.all(np.imag(got) == 0.0), (
        "the view of an elastic layer is not its static moduli lifted")
    assert all(name not in lay.fields for lay in lifted.layers), (
        "reading the view stored a lifted field on a layer")


# ---------------------------------------------------------------------------
# topography, displacement, mapping
# ---------------------------------------------------------------------------

def check_topography(topo, *, n: int = 64, rng=None, atol: float = 1e-9) -> None:
    """A Topography is continuous on the sphere and broadcasts.

    It returns finite float values of its arguments' broadcast shape,
    scalar and array calls agree, the longitude seam is continuous
    (phi = pi and phi = -pi give the same values), and each pole is a
    limit independent of the meridian of approach.  Where it provides
    `gradient` the two components match finite differences; where it
    provides `mean` the value is finite and near the sampled range.
    """
    rng = np.random.default_rng(0) if rng is None else rng

    theta = rng.uniform(0.05, np.pi - 0.05, size=n)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    v = np.asarray(topo(theta, phi), dtype=float)
    assert v.shape == theta.shape, f"expected shape {theta.shape}, got {v.shape}"
    assert np.all(np.isfinite(v)), "topography returned non-finite values"

    one = np.asarray(topo(float(theta[0]), float(phi[0])), dtype=float)
    assert np.allclose(one, v[0]), "scalar and array evaluation disagree"

    grid = np.asarray(topo(theta.reshape(-1, 1), phi.reshape(1, -1)),
                      dtype=float)
    assert grid.shape == (n, n), f"broadcasting gave {grid.shape}"

    # The seam: phi = pi and phi = -pi are the same meridian.
    th = np.linspace(0.2, np.pi - 0.2, 17)
    left = np.asarray(topo(th, np.full_like(th, np.pi)), dtype=float)
    right = np.asarray(topo(th, np.full_like(th, -np.pi)), dtype=float)
    scale = max(1.0, float(np.max(np.abs(left))))
    assert np.allclose(left, right, atol=atol * scale), (
        "the longitude seam is discontinuous: phi = pi and phi = -pi differ")

    # The poles: the limit must not depend on the longitude of approach.
    for pole, name in ((0.0, "north"), (np.pi, "south")):
        eps = 1e-7
        t = pole + (eps if pole == 0.0 else -eps)
        ring = np.asarray(topo(np.full(8, t), np.linspace(-np.pi, np.pi, 8,
                                                          endpoint=False)),
                          dtype=float)
        scale = max(1.0, float(np.max(np.abs(ring))))
        assert np.ptp(ring) <= 1e-4 * scale, (
            f"the {name} pole is discontinuous: approaching along different "
            f"meridians gives values spanning {np.ptp(ring):.3e}")

    if hasattr(topo, "gradient"):
        gt, gp = topo.gradient(theta, phi)
        gt, gp = np.asarray(gt, dtype=float), np.asarray(gp, dtype=float)
        assert gt.shape == theta.shape and gp.shape == theta.shape, (
            "gradient components must have the shape of their arguments")
        h = 1e-5
        fd = (np.asarray(topo(theta + h, phi), dtype=float)
              - np.asarray(topo(theta - h, phi), dtype=float)) / (2.0 * h)
        scale = max(1.0, float(np.max(np.abs(fd))))
        assert np.allclose(gt, fd, rtol=1e-3, atol=1e-3 * scale), (
            "d/dtheta disagrees with a finite difference of the values")
        fd = (np.asarray(topo(theta, phi + h), dtype=float)
              - np.asarray(topo(theta, phi - h), dtype=float)) / (2.0 * h)
        scale = max(1.0, float(np.max(np.abs(fd))))
        assert np.allclose(gp, fd, rtol=1e-3, atol=1e-3 * scale), (
            "d/dphi disagrees with a finite difference of the values")

    if hasattr(topo, "mean"):
        m = float(topo.mean())
        assert np.isfinite(m), "mean() is not finite"
        lo, hi = float(np.min(v)), float(np.max(v))
        pad = 0.5 * (hi - lo) + 1.0
        assert lo - pad <= m <= hi + pad, (
            f"mean() = {m} is nowhere near the sampled range [{lo}, {hi}]")


def check_displacement(h, skeleton, *, n: int = 33, atol: float = 1e-9) -> None:
    """A RadialDisplacement is C0, smooth between its knots, and honest
    about them.

    `h(r, theta, phi)` returns finite values of the broadcast shape.
    The `knots` say where dh/dr may jump, so the mesher can align
    elements with them: h is continuous across every declared knot, and
    where `radial_derivative` is provided it matches a finite difference
    away from the knots and is continuous within each span, which is
    the half that catches a stale or optimistic knot list.  Where
    `angular_gradient` is provided its two components have the argument
    shape.
    """
    b = np.asarray(skeleton.boundaries, dtype=float)
    lo, hi = float(b[0]), float(b[-1])
    theta = np.linspace(0.3, np.pi - 0.3, 5)
    phi = np.linspace(-2.5, 2.5, 5)

    r = np.linspace(lo + 1e-6 * (hi - lo), hi, n)
    R, T = np.meshgrid(r, theta, indexing="ij")
    P = np.broadcast_to(phi, T.shape)
    v = np.asarray(h(R, T, P), dtype=float)
    assert v.shape == R.shape, f"expected shape {R.shape}, got {v.shape}"
    assert np.all(np.isfinite(v)), "displacement returned non-finite values"

    knots = tuple(getattr(h, "knots", ()))
    scale = max(1.0, float(np.max(np.abs(v))))

    for k in knots:
        if not lo < k < hi:
            continue
        eps = 1e-7 * max(1.0, abs(k))
        below = np.asarray(h(k - eps, theta, phi), dtype=float)
        above = np.asarray(h(k + eps, theta, phi), dtype=float)
        assert np.allclose(below, above, atol=1e-4 * scale), (
            f"h is discontinuous at the declared knot r = {k}")

    if hasattr(h, "radial_derivative"):
        probe = np.array([x for x in np.linspace(lo, hi, 4 * n + 1)[1:-1]
                          if all(abs(x - k) > 1e-3 * (hi - lo) for k in knots)])
        if probe.size:
            R, T = np.meshgrid(probe, theta, indexing="ij")
            P = np.broadcast_to(phi, T.shape)
            d = 1e-5 * (hi - lo)
            fd = (np.asarray(h(R + d, T, P), dtype=float)
                  - np.asarray(h(R - d, T, P), dtype=float)) / (2.0 * d)
            got = np.asarray(h.radial_derivative(R, T, P), dtype=float)
            dscale = max(1e-12, float(np.max(np.abs(fd))))
            assert np.allclose(got, fd, rtol=1e-3, atol=1e-3 * dscale), (
                "radial_derivative disagrees with a finite difference of h")

            # Consecutive probes straddling a declared knot may differ;
            # only pairs inside one span are compared.
            edges = np.array(sorted({lo, hi, *knots}), dtype=float)
            same_span = (np.searchsorted(edges, probe[:-1], side="right")
                         == np.searchsorted(edges, probe[1:], side="right"))
            if same_span.any():
                jump = np.abs(np.diff(got, axis=0))[same_span]
                worst = float(np.max(jump)) if jump.size else 0.0
                assert worst <= 0.05 * dscale + 1e-9, (
                    f"dh/dr jumps by {worst:.3e} inside a span, away from "
                    f"every declared knot: the knot list {list(knots)} is "
                    "incomplete, so a kink would fall inside an element")

    if hasattr(h, "angular_gradient"):
        gt, gp = h.angular_gradient(R, T, P)
        assert np.asarray(gt).shape == R.shape
        assert np.asarray(gp).shape == R.shape


def check_mapping(m, points, *, rtol: float = 1e-6, step: float = None) -> None:
    """F is the gradient of m, J is its determinant, and rho pushes to rho/J.

    The convention is F = (grad m)^T, `F[i, j] = d m_i / d X_j`, checked
    against a central difference of `__call__` so that no closed form
    is compared with itself; `J = det F`; a weight-1 scalar pushes
    forward as rho/J and a weight-0 scalar is unchanged.  Where the
    mapping provides them: `displacement(X)` is m(X) - X and nothing
    else; `linearise` is the derivative in the amplitude of a
    perturbation (see `_check_linearisation`, which covers radial
    mappings and skips the rest); `inverse` undoes `__call__`.
    """
    from .model.character import DENSITY, SCALAR
    from .model.pushforward import push_forward

    X = np.asarray(points, dtype=float)
    assert X.shape[-1] == 3, f"points must be (..., 3), got {X.shape}"

    x = np.asarray(m(X), dtype=float)
    assert x.shape == X.shape, f"m maps {X.shape} to {x.shape}"
    assert np.all(np.isfinite(x)), "the mapping returned non-finite points"

    F = np.asarray(m.deformation_gradient(X), dtype=float)
    assert F.shape == X.shape[:-1] + (3, 3), (
        f"deformation_gradient gave {F.shape}, expected {X.shape[:-1] + (3, 3)}")
    assert np.all(np.isfinite(F)), "F has non-finite entries"

    scale = float(np.max(np.abs(X))) or 1.0
    h = step if step is not None else 1e-6 * scale
    fd = np.empty_like(F)
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        fd[..., :, j] = (np.asarray(m(X + e), dtype=float)
                         - np.asarray(m(X - e), dtype=float)) / (2.0 * h)
    fscale = max(1.0, float(np.max(np.abs(fd))))
    assert np.allclose(F, fd, rtol=rtol, atol=rtol * fscale), (
        "F disagrees with a central difference of the mapping; the largest "
        f"discrepancy is {float(np.max(np.abs(F - fd))):.3e}")

    J = np.asarray(m.jacobian(X), dtype=float)
    assert J.shape == X.shape[:-1], f"jacobian gave {J.shape}"
    assert np.allclose(J, np.linalg.det(F), rtol=1e-9, atol=1e-12), (
        "J is not the determinant of F")

    rho = np.full(X.shape[:-1], 3300.0)
    assert np.allclose(push_forward(rho, F, J, DENSITY), rho / J, rtol=1e-14), (
        "a weight-1 scalar does not push forward as rho/J")
    assert np.allclose(push_forward(rho, F, J, SCALAR), rho), (
        "a weight-0 scalar should be unchanged by push-forward")

    if hasattr(m, "displacement"):
        u = np.asarray(m.displacement(X), dtype=float)
        assert u.shape == X.shape, f"displacement gave {u.shape}"
        assert np.allclose(u, x - X, rtol=0.0, atol=1e-12 * scale), (
            "displacement(X) is not m(X) - X, which is all it is allowed "
            "to be: every export writes this array and a mapping that "
            "computes it a second way has two answers to be wrong in")

    _check_linearisation(m, X, rtol=rtol)

    if hasattr(m, "inverse"):
        try:
            back = np.asarray(m.inverse(x), dtype=float)
        except NotImplementedError:
            pass
        else:
            assert np.allclose(back, X, rtol=1e-9, atol=1e-6 * scale), (
                "inverse(m(X)) does not return X")


def _perturbation(scale: float):
    """A smooth displacement to linearise about, with exact derivatives.

    Analytic in all three coordinates, so the only numerical step in the
    comparison below is the amplitude difference itself, and small
    enough (1e-3 of the geometry) that it perturbs rather than reshapes.
    """
    from .model.displacement import CallableDisplacement

    a, k = 1e-3 * scale, np.pi / scale
    return CallableDisplacement(
        lambda r, t, p: a * np.sin(k * r) * np.cos(t),
        radial_derivative=lambda r, t, p: a * k * np.cos(k * r) * np.cos(t),
        angular_gradient=lambda r, t, p: (
            -a * np.sin(k * r) * np.sin(t), np.zeros(np.shape(r))),
        name="check_mapping perturbation")


def _check_linearisation(m, X, *, rtol: float, eps: float = 1e-3) -> None:
    """linearise(delta, X) against a central difference in the amplitude.

    dF and dJ are the derivatives at s = 0 of the mapping generated by
    h + s delta, so the test builds that mapping at +eps and -eps and
    differences it: the closed forms are checked against the mapping
    they claim to describe, not against themselves.

    Only radial mappings are covered, because forming "h + s delta" is a
    statement about how the mapping is generated and `Mapping` does not
    commit to one; anything else is skipped.  dF is exactly linear in s
    and dJ cubic, so a central difference at eps = 1e-3 is exact to
    roundoff for the first and to eps^2 times a tiny coefficient for
    the second.
    """
    from .model.displacement import SumDisplacement, ZeroDisplacement
    from .model.mapping import IdentityMapping, RadialStretch

    if not hasattr(m, "linearise"):
        return
    if isinstance(m, RadialStretch):
        base = m.h
    elif isinstance(m, IdentityMapping):
        base = ZeroDisplacement()
    else:
        return

    X = np.asarray(X, dtype=float)
    scale = float(np.max(np.abs(X))) or 1.0
    delta = _perturbation(scale)
    try:
        lin = m.linearise(delta, X=X)
    except NotImplementedError:
        return

    def stretched(s):
        """The mapping generated by h + s delta, as a SumDisplacement:
        the protocol asks a displacement for values and derivatives, not
        for an algebra."""
        return RadialStretch(SumDisplacement((base,), scale_of=((s, delta),)))

    dF = (stretched(eps).deformation_gradient(X)
          - stretched(-eps).deformation_gradient(X)) / (2.0 * eps)
    dJ = (np.asarray(stretched(eps).jacobian(X), dtype=float)
          - np.asarray(stretched(-eps).jacobian(X), dtype=float)) / (2.0 * eps)

    assert np.asarray(lin.dF).shape == dF.shape, (
        f"linearise gave dF of shape {np.shape(lin.dF)}, expected {dF.shape}")
    for name, got, want in (("dF", np.asarray(lin.dF), dF),
                            ("dJ", np.asarray(lin.dJ), dJ)):
        wscale = max(1e-30, float(np.max(np.abs(want))))
        assert np.allclose(got, want, rtol=rtol, atol=rtol * wscale), (
            f"linearise's {name} disagrees with a central difference of the "
            "mapping in the amplitude of the perturbation; the largest "
            f"discrepancy is {float(np.max(np.abs(got - want))):.3e}")


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------

def check_sample(sample, *, rng=None, n: int = 64, rtol: float = 1e-12,
                 atol: float = 0.0) -> None:
    """A Sample is laid out as promised and says what its source says.

    Layout: the radial nodes are the mesh's per-element GLL nodes
    flattened, so every element boundary is a repeated radius; the
    angular nodes increase inside their open ranges; every field is
    float64 and C-contiguous, node outermost and longitude fastest,
    shaped `(nnode,) + c` or `(nnode, ntheta, nphi) + c` with `c` the
    character's component shape, Voigt where it has one, plus a
    trailing axis of length 2 (real, imaginary) for a field sampled at
    a chosen omega; the metadata names exactly the sampled fields, their
    spherical frame and their domains; a field is finite at every node
    of the layers in its domain and NaN at every node outside them; the
    displacement, if any, is `(nnode, ntheta, nphi, 3)` and finite.

    Identity, when the sample still carries `source` and `mapping`: a
    field stored on `(node,)` declares itself radial and is re-evaluated
    at every node of its domain with `layer=` the element's layer, which
    makes the two nodes at an interface distinct questions; a field on
    the full product is re-evaluated at `n` random nodes of its domain;
    a field sampled at an omega is compared as the complex number its
    pair encodes; the displacement is `R^T (m(X) - X)` at every sample
    point, formed afresh from the mapping.
    """
    from .mesh1d.mesh import RadialMesh
    from .model.frames import spherical_frame
    from .sampling import AngularGrid, Sample, SampleMetadata

    rng = np.random.default_rng(0) if rng is None else rng
    assert isinstance(sample, Sample), f"not a Sample: {type(sample).__name__}"
    mesh, grid, meta = sample.radial, sample.angular, sample.metadata
    assert isinstance(mesh, RadialMesh), "radial is not a RadialMesh"
    assert isinstance(grid, AngularGrid), "angular is not an AngularGrid"
    assert isinstance(meta, SampleMetadata), "metadata is not SampleMetadata"

    # -- layout -------------------------------------------------------------
    nnode = mesh.nspec * mesh.ngll
    assert sample.nnode == nnode, "nnode is not nspec * ngll"
    assert np.array_equal(sample.radius, mesh.r.ravel()), (
        "radius is not the mesh's per-element nodes flattened")
    assert np.array_equal(sample.element_start,
                          np.arange(mesh.nspec + 1) * mesh.ngll), (
        "element_start does not step by ngll")
    assert np.array_equal(sample.element_layer, mesh.layer), (
        "element_layer differs from the mesh's layer array")
    theta, phi = grid.colatitudes, grid.longitudes
    assert np.all(np.diff(theta) > 0) and 0 < theta[0] and theta[-1] < np.pi, (
        "colatitudes are not increasing inside (0, pi)")
    assert np.all(np.diff(phi) > 0) and 0 <= phi[0] and phi[-1] < 2 * np.pi, (
        "longitudes are not increasing inside [0, 2 pi)")
    ntheta, nphi = theta.size, phi.size

    names = set(sample.fields)
    for what, d in (("characters", meta.characters),
                    ("dimensions", meta.dimensions), ("frames", meta.frames),
                    ("domains", meta.domains)):
        assert set(d) == names, (
            f"metadata.{what} names {sorted(d)} but the fields are "
            f"{sorted(names)}")
    assert set(meta.omegas) <= names, (
        f"metadata.omegas names {sorted(set(meta.omegas) - names)}, which are "
        "not sampled fields")
    nlayers = meta.skeleton.nlayers
    element_layer = np.asarray(mesh.layer, dtype=int)
    node_layer = np.repeat(element_layer, mesh.ngll)
    inside = {}
    for name, arr in sample.fields.items():
        c = _trailing_shape(meta.characters[name]) + (
            (2,) if name in meta.omegas else ())
        domain = tuple(meta.domains[name])
        assert domain, f"field {name!r} has an empty domain"
        assert all(0 <= i < nlayers for i in domain), (
            f"field {name!r} has domain {domain}, outside the "
            f"{nlayers} layers of the sample's skeleton")
        assert arr.dtype == np.float64, f"field {name!r} is not float64"
        assert arr.flags.c_contiguous, f"field {name!r} is not C-contiguous"
        assert arr.shape in ((nnode,) + c, (nnode, ntheta, nphi) + c), (
            f"field {name!r} has shape {arr.shape}; expected "
            f"{(nnode,) + c} or {(nnode, ntheta, nphi) + c}")
        assert meta.frames[name] == "spherical", (
            f"field {name!r} is recorded in frame {meta.frames[name]!r}, "
            "not the spherical frame a sample uses")
        keep = np.isin(node_layer, np.asarray(domain, dtype=int))
        inside[name] = keep
        axes = tuple(range(1, arr.ndim))
        finite = (np.all(np.isfinite(arr), axis=axes) if axes
                  else np.isfinite(arr))
        nan = (np.all(np.isnan(arr), axis=axes) if axes else np.isnan(arr))
        assert np.all(finite[keep]), (
            f"field {name!r} has non-finite values on layers {domain}, "
            "where it is defined")
        assert np.all(nan[~keep]), (
            f"field {name!r} is not NaN on the nodes outside its domain "
            f"{domain}: a hole in a field is carried, never filled")
    u = sample.displacement
    if u is not None:
        assert u.dtype == np.float64, "displacement is not float64"
        assert u.shape == (nnode, ntheta, nphi, 3), (
            f"displacement has shape {u.shape}, expected "
            f"{(nnode, ntheta, nphi, 3)}")
        assert np.all(np.isfinite(u)), "displacement has non-finite values"
        assert u.flags.c_contiguous, "displacement is not C-contiguous"

    # -- identity with the source --------------------------------------------
    nodes = rng.integers(nnode, size=n)
    it = rng.integers(ntheta, size=n)
    ip = rng.integers(nphi, size=n)
    r = sample.radius
    layers = node_layer
    if sample.source is not None:
        assert set(sample.source) == names, (
            "source names differ from the sampled fields")
        for name, fld in sample.source.items():
            arr = sample.fields[name]
            omega = meta.omegas.get(name)
            if omega is not None:
                arr = arr[..., 0] + 1j * arr[..., 1]
            keep = inside[name]
            here = np.flatnonzero(keep)
            if here.size == 0:                 # the mesh misses the domain
                continue
            if sample.is_radial(name):
                assert getattr(fld, "is_radial", False), (
                    f"field {name!r} is stored on (node,) but does not "
                    "declare itself radial")
                for L in np.unique(layers[keep]):
                    m = keep & (layers == L)
                    want = (fld.evaluate(r[m], layer=int(L)) if omega is None
                            else fld.evaluate(r[m], layer=int(L), omega=omega,
                                              part="complex"))
                    assert np.allclose(arr[m], want, rtol=rtol, atol=atol), (
                        f"field {name!r} on layer {int(L)} differs from its "
                        "source at the nodes")
            else:
                sel = here[nodes % here.size]
                for k in range(n):
                    point = (r[sel[k]], theta[it[k]], phi[ip[k]])
                    want = (fld.evaluate(*point, layer=int(layers[sel[k]]),
                                         frame="spherical") if omega is None
                            else fld.evaluate(*point, omega=omega,
                                              layer=int(layers[sel[k]]),
                                              frame="spherical",
                                              part="complex"))
                    assert np.allclose(arr[sel[k], it[k], ip[k]], want,
                                       rtol=rtol, atol=atol), (
                        f"field {name!r} at node {sel[k]}, colatitude "
                        f"{theta[it[k]]:.4g}, longitude {phi[ip[k]]:.4g} "
                        "differs from its source")
    if sample.mapping is not None:
        assert u is not None, "a mapping was sampled but displacement is None"
        R = spherical_frame(theta[:, None], phi[None, :])   # (nt, np, 3, 3)
        X = r[:, None, None, None] * R[None, ..., :, 0]
        disp = getattr(sample.mapping, "displacement", None)
        cart = (disp(X) if disp is not None
                else np.asarray(sample.mapping(X), dtype=float) - X)
        want = np.einsum("tpji,ntpj->ntpi", R, np.asarray(cart, dtype=float))
        scale = max(float(np.max(np.abs(want))), 1.0)
        assert np.allclose(u, want, rtol=rtol, atol=atol + rtol * scale), (
            "displacement differs from R^T (m(X) - X) at the sample points")
