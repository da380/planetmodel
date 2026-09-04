"""An Earth-scale build of a PREM-shaped geometry, read back by PyMFEM.

The test suite meshes unit geometries a few elements across.  This script
asks whether the same code survives a planet-sized case: a radius of
6371 km, the boundaries of PREM whose spans differ by two orders of
magnitude, and a mesh large enough that one badly oriented tetrahedron
would be one in millions.  It is not a test and is never collected as
one.  Run it, read the numbers, record them in a note.

    python scripts/mfem_cross_check.py --h-ref-km 400 --out /tmp/prem

The coarsening is the one judgement in it.  PREM's outermost spans are
9.4 and 12 km thick, and the mesher refuses any element size more than
ten times a span it has to fill, so a 400 km mesh cannot see the crust.
The script drops interior boundaries from the top down until the
thinnest remaining span is at least a tenth of `h_ref`, and says how
many it dropped.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton
from planetmodel.mesh3d import (AngularResolution, MeshSpec, Shell,
                                build_layered_mesh, export_mfem_mesh, manifest)

#: PREM's boundary radii in metres, centre outward.
PREM_BOUNDARIES = [0.0, 1221.5e3, 3480.0e3, 3630.0e3, 5600.0e3, 5701.0e3,
                   5771.0e3, 5971.0e3, 6151.0e3, 6291.0e3, 6346.6e3, 6356.0e3,
                   6368.0e3, 6371.0e3]

#: Relief amplitude on the surface, in metres.
RELIEF_M = 3.0e3

#: The buffer shell outside the body, as a fraction of its radius.
BUFFER_RATIO = 0.2


def coarsened_for(h_ref: float) -> tuple[Skeleton, int]:
    """PREM's skeleton with thin outer spans merged away until h_ref fits."""
    b = list(PREM_BOUNDARIES)
    dropped = 0
    while len(b) > 2 and min(np.diff(b)) < 0.1 * h_ref:
        spans = np.diff(b)
        i = int(np.argmin(spans))
        # remove the boundary above the thinnest span, unless it is the surface
        j = i + 1 if i + 1 < len(b) - 1 else i
        del b[j]
        dropped += 1
    return Skeleton(b), dropped


def surface_relief(amplitude: float, a: float, b: float) -> RadialStretch:
    """Degree-two relief on the surface at radius `a`, growing linearly from
    0.9 a and decaying linearly to zero at the buffer's outer radius `b`,
    where the mapping must be the identity."""
    def h(r, theta, phi):
        up = np.clip((r - 0.9 * a) / (0.1 * a), 0.0, 1.0)
        down = np.clip((b - r) / (b - a), 0.0, 1.0)
        zonal = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
        shape = zonal + np.sin(theta) ** 2 * np.cos(2 * phi)
        return amplitude * up * down * shape

    return RadialStretch(CallableDisplacement(h, knots=[0.9 * a, a]), rmax=b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h-ref-km", type=float, default=400.0,
                    help="target element size at the surface, km")
    ap.add_argument("--out", type=Path, default=Path("/tmp/prem"),
                    help="basename of the files to write")
    ap.add_argument("--order", type=int, default=2)
    args = ap.parse_args(argv)

    h_ref = args.h_ref_km * 1e3
    sk, dropped = coarsened_for(h_ref)
    a = float(sk.boundaries[-1])
    sk = sk.refined([0.9 * a]) if not np.any(np.isclose(sk.boundaries, 0.9 * a)) else sk
    print(f"skeleton: {sk.nlayers} layers after dropping {dropped} boundaries")
    b = a * (1.0 + BUFFER_RATIO)
    geometry = Geometry(sk, mapping=surface_relief(RELIEF_M, a, b))
    print("validity:", geometry.validity())

    spec = MeshSpec(geometry, AngularResolution(h_ref, 3.0 * h_ref, fraction=0.25),
                    order=args.order, shells=[Shell(ratio=BUFFER_RATIO)],
                    delivery="referential",
                    meta={"script": "mfem_cross_check", "h_ref_km": args.h_ref_km})
    t0 = time.perf_counter()
    result = build_layered_mesh(spec, args.out, verbose=False)
    print(f"built {result} in {time.perf_counter() - t0:.1f} s")
    print("timings:", {k: round(v, 1) for k, v in result.timings.items()})
    print("validation:", result.validation)

    card = manifest.read(result.manifest_path)
    print("divisor:", card.geometry["divisor"], "| layers:", len(card.layers))

    try:
        import mfem.ser as mfem
    except ImportError:
        print("PyMFEM not installed; stopping after the build")
        return 0
    t0 = time.perf_counter()
    exported = export_mfem_mesh(result, args.out.with_name(args.out.name + "_mfem"),
                                delivery="referential")
    opts = exported.files["mesh_read_options"]
    mesh = mfem.Mesh(str(exported.mesh_path), opts["generate_edges"], opts["refine"],
                     opts["fix_orientation"])
    print(f"MFEM read {mesh.GetNE()} elements and {mesh.GetNBE()} boundary elements "
          f"in {time.perf_counter() - t0:.1f} s; attributes "
          f"{list(mesh.attributes.ToList())}")
    wrong = mesh.CheckElementOrientation(False)
    print("elements MFEM would reorient:", wrong)
    return 0 if math.isfinite(result.validation.min_sicn) else 1


if __name__ == "__main__":
    raise SystemExit(main())
