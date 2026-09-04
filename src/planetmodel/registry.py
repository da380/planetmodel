"""registry.py -- short names for the components recipes refer to.

A recipe file cannot hold a Python object, so it names one:
`rule = "layer_linear"`, `policy = "angular_resolution"`. This is the
table those names resolve through, and the table a manifest records so a
mesh says what built it.

Registration is never required. A component that is not registered is
fully usable from Python; it simply cannot be named in a file, and a
manifest records it as custom with a repr. Keeping the registry optional
is what stops it becoming a second, competing way to construct things.

    @register("displacement_rule", "layer_linear")
    class LayerLinear: ...

    lookup("displacement_rule", "layer_linear")
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

__all__ = ["register", "lookup", "name_of", "registered", "KINDS"]

KINDS = (
    "topography",
    "displacement_rule",
    "sizing",
    "deck_reader",
    "state_rule",
    "rheology",
    "model_class",
)

_REGISTRY: dict[str, dict[str, Any]] = {k: {} for k in KINDS}

T = TypeVar("T")


def _check_kind(kind: str) -> None:
    """Reject unknown kinds, listing the ones that exist."""
    if kind not in _REGISTRY:
        raise KeyError(
            f"unknown component kind {kind!r}; known kinds are "
            + ", ".join(sorted(_REGISTRY)))


def register(kind: str, name: str, obj: T | None = None) -> Callable[[T], T] | T:
    """Register obj under (kind, name); usable as a decorator.

    Re-registering a name raises rather than silently shadowing, since
    two components answering to one name in a recipe is a mistake worth
    hearing about at import time.
    """
    _check_kind(kind)

    def do(target: T) -> T:
        if name in _REGISTRY[kind]:
            raise KeyError(
                f"{kind} {name!r} is already registered to "
                f"{_REGISTRY[kind][name]!r}")
        _REGISTRY[kind][name] = target
        return target

    return do if obj is None else do(obj)


def lookup(kind: str, name: str) -> Any:
    """The component registered under (kind, name)."""
    _check_kind(kind)
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY[kind])) or "(none registered)"
        raise KeyError(
            f"no {kind} named {name!r}; registered: {known}") from None


def name_of(kind: str, obj: Any) -> str | None:
    """The registered name of a component, or None if it has none.

    Matches the object itself or its type, so a registered dataclass is
    found from any of its instances.  This is what lets a manifest record
    the same name a recipe would use, rather than a Python class name
    that no recipe file could say.
    """
    _check_kind(kind)
    for name, target in _REGISTRY[kind].items():
        if target is obj or target is type(obj):
            return name
    return None


def registered(kind: str) -> tuple[str, ...]:
    """The names registered for a kind, sorted."""
    _check_kind(kind)
    return tuple(sorted(_REGISTRY[kind]))
