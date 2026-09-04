"""The physical delivery: displace the mesh nodes by the mapping.

The reference mesh is built on concentric spheres and then every node
is moved by m, high-order nodes like any other.  A curved element can
fold between corners that are individually fine, which is why the
discrete Jacobian check afterwards is not optional.

Every guard runs on the whole node set first, and only then is a single
node written: a mesh half-displaced by a mapping that turns out to be
invalid looks finished.  gmsh has no bulk node setter (`addNodes` with
existing tags appends, `relocateNodes` re-places nodes from their
parametric coordinates and undoes a displacement), so `setNode` runs in
a loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np

__all__ = ["PerturbationReport", "apply_mapping"]


@dataclass(frozen=True)
class PerturbationReport:
    """What the displacement did, and how close it came to failing."""

    nodes: int
    max_displacement: float
    min_radius_before: float
    min_radius_after: float
    validity_margin: float

    def __repr__(self) -> str:
        return (f"PerturbationReport({self.nodes} nodes, max |dx| = "
                f"{self.max_displacement:.4g}, margin {self.validity_margin:.4g})")


def apply_mapping(mapping, *, check: bool = True) -> PerturbationReport:
    """Move every node of the current mesh by `mapping`.

    The mapping acts on the node coordinates as they are: the mesh is
    in the reference geometry's own numbers, and so is the mapping.
    """
    tags, coords, _ = gmsh.model.mesh.getNodes()
    if tags.size == 0:
        raise RuntimeError("no nodes to displace: generate the mesh first")

    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    x = np.asarray(mapping(X), dtype=float)
    if x.shape != X.shape:
        raise RuntimeError(
            f"the mapping returned {x.shape} for {X.shape} points")

    r_before = np.linalg.norm(X, axis=1)
    r_after = np.linalg.norm(x, axis=1)

    if check:
        if not np.all(np.isfinite(x)):
            bad = int(np.argmax(~np.isfinite(x).all(axis=1)))
            raise ValueError(
                f"the mapping is not finite at node {int(tags[bad])}, "
                f"reference position {X[bad]}")

        # A node that reaches or crosses the origin has folded the domain
        # through itself; the centre node legitimately stays at zero.
        interior = r_before > 0.0
        if np.any(r_after[interior] <= 0.0):
            bad = int(np.flatnonzero(interior & (r_after <= 0.0))[0])
            raise ValueError(
                f"node {int(tags[bad])} is displaced to radius "
                f"{r_after[bad]:.6g} from {r_before[bad]:.6g}: the mapping "
                "moves points to or through the origin")

        if hasattr(mapping, "is_valid"):
            verdict = mapping.is_valid(X=X)
            if not verdict:
                raise ValueError(
                    f"the mapping is not orientation-preserving on the mesh "
                    f"nodes: {verdict!r}")
            margin = float(verdict.margin)
        else:
            margin = float("nan")
    else:
        margin = float("nan")

    for i, tag in enumerate(tags):
        gmsh.model.mesh.setNode(int(tag), x[i].tolist(), [])

    return PerturbationReport(
        nodes=int(tags.size),
        max_displacement=float(np.max(np.linalg.norm(x - X, axis=1))),
        min_radius_before=float(r_before.min()),
        min_radius_after=float(r_after.min()),
        validity_margin=margin)
