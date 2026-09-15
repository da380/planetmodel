"""The Earth-scale half of the WP5.4 cross-check: PREM through PyMFEM.

`tests/mesh3d/test_mfem.py` asks MFEM what it makes of a unit body a few
elements across, because that is what a test may cost.  The question it
cannot ask is whether the same code survives the real thing: six
thousand kilometres of radius, twelve interfaces whose spans differ by
two orders of magnitude, and a mesh large enough that a single badly
oriented tetrahedron would be one in millions.  This script asks that,
manually, and prints what MFEM says.

It is not a test and is never collected as one.  Run it, read the
numbers, record them in a note.

    python scripts/mfem_cross_check.py --h-ref-km 400 --out /tmp/prem

The one piece of judgement in it is the coarsening.  PREM's outermost
spans are 9.4 and 12 km thick, and `check_sizing_resolves_spans` refuses
-- correctly -- any element size more than ten times a span it has to
fill, so a 400 km mesh cannot see the crust at all.  The script
therefore drops interior boundaries from the top down until the thinnest
remaining span is at least a tenth of `h_ref`, and says how many it
dropped.  That is a coarsening of the *geometry*: the model's fields are
unchanged, and a mesh meant to resolve the crust asks for a smaller
`h_ref` and keeps the boundaries it can afford.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from planetmodel import layer_linear
from planetmodel.io import manifest as sc
from planetmodel.mesh3d import (AngularResolution, BufferSpec, MeshSpec,
                           build_layered_mesh)
from planetmodel.model.topography import AnalyticTopography
from planetmodel.catalogue.prem import PREM

#: PREM's normalization radius, and the mesh's length unit.
RREF = 6371.0e3

#: Relief amplitude on the surface, in metres: Earth's own scale, a few
#: kilometres, so the displacement is a part in two thousand and the
#: validity margin is not the thing under test.
RELIEF_M = 3.0e3

#: The vacuum shell outside the body, as a fraction of its radius.
BUFFER_RATIO = 0.2


def relief(amplitude: float):
    """Degree two, zonal plus sectoral, with exactly zero mean.

    The same shape the acceptance tests use: `P_2(cos t)` and
    `sin^2 t cos 2p` both integrate to zero over the sphere, so the
    boundary keeps the mean radius PREM gives it.
    """
    return AnalyticTopography(
        lambda t, p: amplitude * (0.5 * (3.0 * np.cos(t) ** 2 - 1.0)
                                  + np.sin(t) ** 2 * np.cos(2.0 * p)))


def drop_for_sizing(body, h_min: float) -> list[int]:
    """Interior boundaries to drop so every span is at least `h_min`.

    Dropped from the outside in, one at a time, because that is where
    PREM's thin spans are and because merging the top two spans is the
    only move that cannot disturb a boundary below the one it merges.
    Returns interior-boundary indices, which is what
    `MeshSpec.drop_interfaces` takes.
    """
    b = np.asarray(body.skeleton.boundaries, dtype=float)
    kept = list(range(b.size - 2))
    dropped: list[int] = []
    while kept:
        radii = np.array([b[0], *(b[j + 1] for j in kept), b[-1]])
        if np.diff(radii).min() >= h_min:
            break
        dropped.append(kept.pop())
    return sorted(dropped)


def build(h_ref: float, h_far: float, out: Path):
    """Mesh PREM with relief on its surface, and report what was dropped."""
    body = (PREM(ocean=False)
            .name_interface(-1, "surface"))
    b = np.asarray(body.skeleton.boundaries, dtype=float)
    dropped = drop_for_sizing(body, 0.1 * h_ref)
    kept = [j for j in range(b.size - 2) if j not in set(dropped)]
    spans = np.diff(np.array([b[0], *(b[j + 1] for j in kept), b[-1]]))

    print(f"model      PREM (oceanless), outer radius {b[-1] / 1e3:.1f} km, "
          f"{b.size - 1} interfaces")
    print(f"coarsened  {len(dropped)} interior boundaries dropped "
          f"({', '.join(f'{b[j + 1] / 1e3:.1f}' for j in dropped) or 'none'}"
          " km); thinnest remaining span "
          f"{spans.min() / 1e3:.1f} km, against h_ref/10 = "
          f"{h_ref / 1e4:.1f} km")
    print(f"sizing     h_ref {h_ref / 1e3:.0f} km at r_ref "
          f"{RREF / 1e3:.0f} km, h_far {h_far / 1e3:.0f} km")

    spec = MeshSpec(
        body=body, rref=RREF, order=2, dimension=3,
        sizing=AngularResolution(h_ref=h_ref, r_ref=RREF, h_far=h_far),
        drop_interfaces=dropped,
        buffers=[BufferSpec(ratio=BUFFER_RATIO)],
        surfaces={"surface": relief(RELIEF_M)},
        mapping_rule=layer_linear(), mode="physical")

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    built = build_layered_mesh(spec, out / "prem")
    return built, time.perf_counter() - t0


def report(built, wall: float) -> None:
    """Load the written mesh with PyMFEM and print MFEM's own opinion."""
    import mfem.ser as mfem

    card = sc.read(built.manifest_path)
    print(f"\nwrote      {built.msh_path} "
          f"({built.msh_path.stat().st_size / 1e6:.1f} MB)")
    print("timings    " + ", ".join(f"{k} {v:.2f}s"
                                    for k, v in built.timings.items())
          + f", total {wall:.2f}s")

    t0 = time.perf_counter()
    mesh = mfem.Mesh(str(built.msh_path))
    print(f"loaded     in {time.perf_counter() - t0:.2f}s by MFEM")

    print(f"elements   {mesh.GetNE()} cells, {mesh.GetNBE()} boundary "
          f"elements, {mesh.GetNV()} vertices")
    print(f"           manifest says {card.mesh['n_elements']} elements "
          f"(cells + faces) and {card.mesh['n_nodes']} nodes; "
          f"cells + boundary = {mesh.GetNE() + mesh.GetNBE()}")
    print(f"orientation  CheckElementOrientation "
          f"{mesh.CheckElementOrientation(False)}, "
          f"CheckBdrElementOrientation "
          f"{mesh.CheckBdrElementOrientation(False)}")
    print(f"attributes   {list(mesh.attributes.ToList())}")
    print(f"bdr attrs    {list(mesh.bdr_attributes.ToList())}")
    for lay in card.layers:
        print(f"  layer {lay['attribute']:>2}  {lay['name']:<12} "
              f"[{lay['r_inner_nd']:.4f}, {lay['r_outer_nd']:.4f}] nd"
              + ("  (buffer)" if lay["is_buffer"] else ""))
    for face in card.interfaces:
        print(f"  face  {face['attribute']:>2}  {face['name']:<12} "
              f"r = {face['mean_radius_nd']:.4f} nd")

    r = card.interfaces[-1]["mean_radius_nd"]
    analytic = 4.0 / 3.0 * math.pi * r ** 3
    volume = sum(mesh.GetElementVolume(i) for i in range(mesh.GetNE()))
    print(f"volume     MFEM {volume:.8f}, analytic ball of r = {r:.6f} "
          f"is {analytic:.8f} (relative error "
          f"{abs(volume - analytic) / analytic:.2e})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mfem_cross_check.py",
        description="Mesh PREM with surface relief and cross-check it "
                    "against PyMFEM. A manual run, not a test.")
    parser.add_argument("--h-ref-km", type=float, default=400.0,
                        help="element size at r_ref = 6371 km (default 400)")
    parser.add_argument("--h-far-km", type=float, default=None,
                        help="element size away from every interface "
                             "(default: twice h_ref)")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for prem.msh and prem.json")
    args = parser.parse_args(argv)

    import mfem.ser  # noqa: F401  -- fail now, not after the mesh

    h_ref = args.h_ref_km * 1e3
    h_far = (2.0 * h_ref if args.h_far_km is None else args.h_far_km * 1e3)
    built, wall = build(h_ref, h_far, args.out)
    report(built, wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
