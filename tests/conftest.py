"""Fixtures shared by the whole suite."""
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def extra_modules() -> tuple[str, ...]:
    """The importable names of every optional dependency, read from the
    extras of `pyproject.toml`, which is their one declaration: each
    distribution there imports under its own name."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        extras = tomllib.load(fh)["project"]["optional-dependencies"]
    names = {re.match(r"[A-Za-z0-9_.-]+", req).group(0)
             for reqs in extras.values() for req in reqs}
    return tuple(sorted(names))


@pytest.fixture(scope="session")
def same_netcdf():
    """A function of two paths holding the netCDF files there to the same
    groups, dimensions, attributes and variables, to the bit, the root's
    `created` stamp apart."""
    netCDF4 = pytest.importorskip("netCDF4",
                                  reason="needs the planetmodel[netcdf] extra")

    def same_attributes(x, y, where):
        ax = {k: x.getncattr(k) for k in x.ncattrs() if k != "created"}
        ay = {k: y.getncattr(k) for k in y.ncattrs() if k != "created"}
        assert list(ax) == list(ay), where
        for key, value in ax.items():
            if isinstance(value, str):
                assert value == ay[key], (where, key)
            else:
                assert np.array_equal(value, ay[key], equal_nan=True), (where, key)

    def same(x, y, where):
        same_attributes(x, y, where)
        assert ({k: len(d) for k, d in x.dimensions.items()}
                == {k: len(d) for k, d in y.dimensions.items()}), where
        assert list(x.variables) == list(y.variables), where
        assert list(x.groups) == list(y.groups), where
        for name, var in x.variables.items():
            other = y.variables[name]
            assert var.dimensions == other.dimensions, (where, name)
            assert var.dtype == other.dtype, (where, name)
            same_attributes(var, other, f"{where}{name}")
            if var.dtype is str:
                assert list(var[...]) == list(other[...]), (where, name)
                continue
            var.set_auto_mask(False)
            other.set_auto_mask(False)
            assert np.array_equal(var[...], other[...], equal_nan=True), (where, name)
        for name, group in x.groups.items():
            same(group, y.groups[name], f"{where}{name}/")

    def same_files(a, b):
        with netCDF4.Dataset(a) as x, netCDF4.Dataset(b) as y:
            same(x, y, "/")

    return same_files
