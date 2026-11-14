"""Files written whole, with the permissions a plain open would give them.

A new file gets 0o666 less the umask, under several umasks; a file
already at the path keeps its own mode; a failed write leaves no
partial file and the old file standing.  The model, surface and fault
writers each obey both rules.
"""
import os
import stat
from contextlib import contextmanager

import numpy as np
import pytest

from planetmodel.files import partial_file, written_whole

posix = pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")


@contextmanager
def umask(mask):
    """The process's umask set to `mask` for the block."""
    old = os.umask(mask)
    try:
        yield
    finally:
        os.umask(old)


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@posix
@pytest.mark.parametrize("mask", [0o022, 0o077, 0o002, 0o027])
def test_a_new_file_has_the_mode_the_umask_gives(tmp_path, mask):
    target = tmp_path / "new.txt"
    with umask(mask):
        with written_whole(target) as partial:
            assert partial.parent == tmp_path and partial.name.startswith(".new.txt.")
            partial.write_text("whole")
        plain = tmp_path / "plain.txt"
        with open(plain, "w") as fh:
            fh.write("plain")
    assert target.read_text() == "whole"
    assert mode(target) == 0o666 & ~mask == mode(plain)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["new.txt", "plain.txt"]


@posix
@pytest.mark.parametrize("kept", [0o640, 0o604, 0o600, 0o664])
def test_an_existing_file_keeps_its_mode(tmp_path, kept):
    target = tmp_path / "old.txt"
    target.write_text("old")
    os.chmod(target, kept)
    with umask(0o077):
        assert mode(partial_file(target)) == kept
        with written_whole(target) as partial:
            partial.write_text("new")
    assert target.read_text() == "new" and mode(target) == kept


@posix
def test_a_writer_that_makes_the_file_anew_keeps_the_mode(tmp_path):
    target = tmp_path / "old.txt"
    target.write_text("old")
    os.chmod(target, 0o640)
    with written_whole(target) as partial:
        os.remove(partial)
        with umask(0o077):
            partial.write_text("new")
    assert target.read_text() == "new" and mode(target) == 0o640


def test_a_failed_write_leaves_no_partial_file_and_the_old_one_standing(tmp_path):
    target = tmp_path / "old.txt"
    target.write_text("old")
    with pytest.raises(RuntimeError, match="broke"):
        with written_whole(target) as partial:
            partial.write_text("half")
            raise RuntimeError("broke")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["old.txt"]
    with pytest.raises(FileNotFoundError):
        partial_file(tmp_path / "missing" / "new.txt")


def _surface(path):
    from planetmodel import gauss_legendre
    from planetmodel.surfacefile import SurfaceVariable, write_surface
    grid = gauss_legendre(2)
    write_surface(path, grid, source="test",
                  fields={"h": SurfaceVariable(np.ones((grid.ntheta, grid.nphi)))})


def _model(path):
    from planetmodel import PREM, gauss_legendre
    from planetmodel.modelfile import write_model
    write_model(path, PREM(), grid=gauss_legendre(2), fields=["rho"])


def _fault(path):
    from planetmodel.fault import CartesianFrame, Fault, Plane, Slip
    from planetmodel.faultfile import write_fault
    plane = Plane(CartesianFrame(), [0.0, 0.0, -2.0], strike=0.0, dip=45.0,
                  length=1.0, width=1.0, slip=Slip(np.zeros((1, 1, 3))))
    write_fault(path, Fault([plane]), source="test")


@posix
@pytest.mark.netcdf
@pytest.mark.parametrize("write", [_surface, _model, _fault],
                         ids=["surface", "model", "fault"])
def test_the_writers_give_a_new_file_the_umask_s_mode_and_keep_an_old_one_s(
        tmp_path, write):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    path = tmp_path / "file.nc"
    with umask(0o022):
        write(path)
    assert mode(path) == 0o644
    os.chmod(path, 0o640)
    with umask(0o077):
        write(path)
    assert mode(path) == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["file.nc"]
