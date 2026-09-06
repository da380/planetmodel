"""The public surface: exports resolve, and optional arguments are keyword-only."""
import ast
import dataclasses
import importlib
import inspect
import pkgutil
import tomllib
from pathlib import Path

import planetmodel

ROOT = Path(__file__).resolve().parent.parent


def test_every_export_resolves():
    missing = [n for n in planetmodel.__all__ if not hasattr(planetmodel, n)]
    assert not missing, missing
    assert len(planetmodel.__all__) == len(set(planetmodel.__all__))


def test_version_matches_pyproject():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert planetmodel.__version__ == declared


# Every parameter with a default is keyword-only, in every function, method
# and constructor of the package; the allowlist below is the agreed set of
# exceptions and is not extended solo.
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
    """Every module of the package that imports without optional dependencies."""
    mods = [planetmodel]
    for info in pkgutil.walk_packages(planetmodel.__path__, "planetmodel."):
        try:
            mods.append(importlib.import_module(info.name))
        except ImportError:
            pass
    return mods


def _public_callables():
    """Every function and class defined in the package, and every method."""
    seen = set()
    for mod in _modules():
        for name, obj in vars(mod).items():
            if getattr(obj, "__module__", None) != mod.__name__ or obj in seen:
                continue
            seen.add(obj)
            if inspect.isclass(obj):
                yield obj.__name__, obj
                for attr, member in vars(obj).items():
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if inspect.isfunction(member):
                        yield f"{obj.__name__}.{attr}", member
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


# Every dataclass field with a default sits after a `_: KW_ONLY` sentinel, so
# a constructor reads like a signature: required fields, then `*`, then the
# optionals by name.
def test_dataclass_optional_fields_are_keyword_only():
    offenders = [qualname for qualname, obj in _public_callables()
                 if inspect.isclass(obj) and dataclasses.is_dataclass(obj)
                 and any(f.default is not dataclasses.MISSING
                         or f.default_factory is not dataclasses.MISSING
                         for f in dataclasses.fields(obj) if not f.kw_only)]
    assert not offenders, f"positional optional fields: {offenders}"


# Every parameter and every return is annotated, in every function of the
# package, nested functions and dunders included: a signature says what it
# takes and gives.  Read from the source so that modules needing gmsh are
# held to it too.
def _unannotated(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        params = a.posonlyargs + a.args + a.kwonlyargs
        missing = [p.arg for p in params
                   if p.annotation is None and p.arg not in ("self", "cls")]
        if a.vararg is not None and a.vararg.annotation is None:
            missing.append("*" + a.vararg.arg)
        if a.kwarg is not None and a.kwarg.annotation is None:
            missing.append("**" + a.kwarg.arg)
        if node.returns is None:
            missing.append("->")
        if missing:
            yield node.lineno, node.name, missing


def test_every_signature_is_annotated():
    root = Path(planetmodel.__file__).parent
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        rows = list(_unannotated(tree))
        if rows:
            offenders[str(path.relative_to(root))] = rows
    assert not offenders, f"unannotated signatures: {offenders}"
