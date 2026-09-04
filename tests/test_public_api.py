"""The public surface: exports resolve, and optional arguments are keyword-only."""
import importlib
import inspect
import tomllib
from pathlib import Path

import planetmodel
import planetmodel.frames
import planetmodel.testing

ROOT = Path(__file__).resolve().parent.parent


def test_every_export_resolves():
    missing = [n for n in planetmodel.__all__ if not hasattr(planetmodel, n)]
    assert not missing, missing
    assert len(planetmodel.__all__) == len(set(planetmodel.__all__))


def test_version_matches_pyproject():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert planetmodel.__version__ == declared


# Every parameter with a default is keyword-only; the allowlist below is
# the agreed set of exceptions and is not extended solo.
KEYWORD_ONLY_EXCEPTIONS = set()


def _positional_defaults(fn):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    return [p.name for p in sig.parameters.values()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and p.default is not inspect.Parameter.empty]


def _modules():
    mods = [planetmodel, planetmodel.frames, planetmodel.testing]
    for name in ("planetmodel.mesh1d", "planetmodel.mesh3d"):
        try:
            mods.append(importlib.import_module(name))
        except ImportError:
            pass
    return mods


def _public_callables():
    seen = set()
    for mod in _modules():
        for name in getattr(mod, "__all__", ()):
            obj = getattr(mod, name)
            if inspect.isclass(obj):
                for cls in obj.__mro__:
                    if not getattr(cls, "__module__", "").startswith("planetmodel"):
                        continue
                    if cls in seen:
                        continue
                    seen.add(cls)
                    yield cls.__name__, cls
                    for attr, member in vars(cls).items():
                        if attr.startswith("_") and attr != "__call__":
                            continue
                        if isinstance(member, (staticmethod, classmethod)):
                            member = member.__func__
                        if inspect.isfunction(member):
                            yield f"{cls.__name__}.{attr}", member
            elif inspect.isfunction(obj):
                yield name, obj


def test_optional_arguments_are_keyword_only():
    offenders = {}
    for qualname, fn in _public_callables():
        short = qualname.split(".")[-1]
        if qualname in KEYWORD_ONLY_EXCEPTIONS or short in KEYWORD_ONLY_EXCEPTIONS:
            continue
        bad = _positional_defaults(fn)
        if bad:
            offenders[qualname] = bad
    assert not offenders, f"positional optionals: {offenders}"
