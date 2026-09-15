"""gmsh is process-global state, so it is owned explicitly.

gmsh keeps one model registry per process.  A leaked session leaves
entities, options and physical groups behind for whatever runs next,
turning an unrelated later failure into the visible symptom.  Every
entry into gmsh therefore goes through this context manager, which
initialises, routes gmsh's own logging into the planetmodel logger,
and finalizes even when the body raises.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Mapping
from types import ModuleType

import gmsh

log = logging.getLogger("planetmodel.mesh3d")

__all__ = ["session", "is_active", "set_options"]


def is_active() -> bool:
    """Whether a gmsh session is currently initialised."""
    return bool(gmsh.isInitialized())


@contextlib.contextmanager
def session(*, name: str = "planetmodel", verbose: bool = False,
            terminal: bool = False) -> Iterator[ModuleType]:
    """A gmsh session that always finalizes; yields `gmsh.model`.

    Nesting is refused rather than silently reusing the outer session:
    two callers sharing one global model is the confusion this exists
    to prevent.
    """
    if is_active():
        raise RuntimeError(
            "a gmsh session is already active; planetmodel.mesh3d does not nest "
            "sessions, since gmsh has one global model per process")
    gmsh.initialize()
    try:
        gmsh.logger.start()                  # without this, get() is empty
        gmsh.option.setNumber("General.Terminal",
                              1 if (terminal or verbose) else 0)
        gmsh.option.setNumber("General.Verbosity", 5 if verbose else 2)
        gmsh.model.add(name)
        yield gmsh.model
    finally:
        _drain_log()
        try:
            gmsh.logger.stop()
        except Exception:
            pass
        gmsh.finalize()


def _drain_log() -> None:
    """Forward gmsh's accumulated log lines to the planetmodel logger."""
    try:
        for line in gmsh.logger.get():
            level = logging.WARNING if line.startswith(("Warning", "Error")) \
                else logging.DEBUG
            log.log(level, "gmsh: %s", line)
    except Exception:            # logger unavailable after an early failure
        pass


def set_options(options: Mapping[str, float | str]) -> None:
    """Apply gmsh options, choosing setNumber or setString by value type."""
    for key, value in options.items():
        if isinstance(value, str):
            gmsh.option.setString(key, value)
        else:
            gmsh.option.setNumber(key, float(value))
