"""The curated public surface of the package."""
import tomllib
from pathlib import Path

import pytest

import planetmodel

ROOT = Path(__file__).resolve().parent.parent


def test_every_export_resolves():
    missing = [n for n in planetmodel.__all__ if not hasattr(planetmodel, n)]
    assert not missing, f"__all__ names nothing: {missing}"


def test_no_duplicate_exports():
    assert len(planetmodel.__all__) == len(set(planetmodel.__all__))


def test_version_matches_pyproject():
    """__version__ and the packaging metadata are written separately."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert planetmodel.__version__ == declared


def test_the_new_names_are_exported():
    for name in ("ReferenceBody", "RadialField", "Field", "Skeleton",
                 "register", "lookup", "registered"):
        assert name in planetmodel.__all__


def test_importing_planetmodel_is_light():
    """The packaging promise: no plotting stack, no gmsh, no netCDF."""
    import subprocess
    import sys
    code = (
        "import sys, planetmodel\n"
        "heavy = [m for m in ('matplotlib', 'gmsh', 'netCDF4')"
        " if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


_EXAMPLES = sorted(str(p.relative_to(ROOT))
                   for sub in ("tutorials", "reference")
                   for p in (ROOT / "examples" / sub).glob("*.py"))


def test_there_are_examples():
    assert _EXAMPLES, "no example scripts found under examples/"


@pytest.mark.parametrize("path", _EXAMPLES)
def test_examples_still_compile(path):
    """Smoke check only -- the examples plot and mesh, so CI runs them."""
    import py_compile
    py_compile.compile(str(ROOT / path), doraise=True)


# ------------------------------------------------- keyword-only optional arguments

# House rule: every parameter with a default is keyword-only.  The
# exceptions below are the agreed allowlist and are not extended solo.
KEYWORD_ONLY_EXCEPTIONS_BY_NAME = {
    # evaluate(r, theta, phi): the natural call, angles optional only for
    # radial fields; apply and equilibrium_form share its shape, and so
    # does __call__, which is evaluate with the defaults
    "evaluate", "evaluate_full", "apply", "equilibrium_form", "__call__",
    # f.derivative(2) reads as numpy's
    "derivative",
    # the decorator form of register (the mesher's command-line main(argv)
    # is the other idiom kept, but it is not public and so not listed)
    "register",
}
KEYWORD_ONLY_EXCEPTIONS_QUALIFIED = {
    # bound positionally by the unchanged first-stage tests
    "solve_degree", "write_love_numbers",
    "RadialGRF.sample", "SphericalGRF.sample", "LayeredGRF.sample",
}


def _positional_defaults(fn):
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    return [p.name for p in sig.parameters.values()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and p.default is not inspect.Parameter.empty]


def _public_callables():
    """(qualified name, callable) for every public function and method."""
    import inspect
    import planetmodel.catalogue
    import planetmodel.io.manifest
    import planetmodel.io.netcdf
    import planetmodel.io.recipe
    import planetmodel.loading
    import planetmodel.mesh3d
    import planetmodel.randomfield
    import planetmodel.model.rheology
    import planetmodel.sampling
    import planetmodel.testing
    seen = set()
    roots = [getattr(planetmodel, n) for n in planetmodel.__all__]
    for mod in (planetmodel.sampling, planetmodel.testing, planetmodel.mesh1d,
                planetmodel.io, planetmodel.model, planetmodel.catalogue,
                planetmodel.mesh3d, planetmodel.io.netcdf,
                planetmodel.io.manifest, planetmodel.io.recipe,
                planetmodel.model.rheology, planetmodel.loading,
                planetmodel.randomfield):
        roots += [getattr(mod, n) for n in getattr(mod, "__all__", ())]
    for obj in roots:
        if inspect.isclass(obj):
            for cls in obj.__mro__:
                if not getattr(cls, "__module__", "").startswith("planetmodel"):
                    continue
                if cls in seen:
                    continue
                seen.add(cls)
                yield cls.__name__, cls                    # the constructor
                for name, member in vars(cls).items():
                    if name.startswith("_") and name != "__call__":
                        continue
                    fn = member
                    if isinstance(member, (classmethod, staticmethod)):
                        fn = member.__func__
                    elif isinstance(member, property):
                        continue
                    if inspect.isfunction(fn):
                        yield f"{cls.__name__}.{name}", fn
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            if obj not in seen and getattr(obj, "__module__",
                                           "").startswith("planetmodel"):
                seen.add(obj)
                yield obj.__name__, obj


def test_optional_arguments_are_keyword_only():
    offenders = []
    for qual, fn in _public_callables():
        bare = qual.rsplit(".", 1)[-1]
        if bare in KEYWORD_ONLY_EXCEPTIONS_BY_NAME:
            continue
        if qual in KEYWORD_ONLY_EXCEPTIONS_QUALIFIED:
            continue
        bad = _positional_defaults(fn)
        if bad:
            offenders.append(f"{qual}({', '.join(bad)})")
    assert not offenders, (
        "optional parameters that can be passed positionally "
        "(house rule: keyword-only):\n  " + "\n  ".join(sorted(offenders)))


def test_the_exception_list_is_still_needed():
    """An exception that no longer matches anything should be removed."""
    names = {q for q, _ in _public_callables()}
    bare = {q.rsplit(".", 1)[-1] for q in names}
    unused = ({n for n in KEYWORD_ONLY_EXCEPTIONS_BY_NAME if n not in bare}
              | {q for q in KEYWORD_ONLY_EXCEPTIONS_QUALIFIED if q not in names})
    assert not unused, f"exceptions matching nothing public: {sorted(unused)}"
