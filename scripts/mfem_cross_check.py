"""An Earth-scale build of a PREM-shaped geometry, read back by PyMFEM.

The test suite meshes unit geometries a few elements across.  This script
asks whether the same code survives a planet-sized case: a radius of
6371 km, the boundaries of PREM whose spans differ by two orders of
magnitude, and a mesh large enough that one badly oriented tetrahedron
would be one in millions.  It is not a test and is never collected as
one.  Run it, read the numbers, record them in a note.

    python scripts/mfem_cross_check.py --h-ref-km 400 --out /tmp/prem

The build is done on the non-dimensional model, radii in units of the
outer radius and densities in units of the mean density, with G equal
to one: gmsh's kernel works in absolute tolerances and leaves the outer
surfaces of a ball six million metres across without elements, and the
mesher normalises nothing on purpose.  With PyMFEM present the script
also exports PREM's density on the coarsened layers and integrates it
over the MFEM mesh: the referential density over the reference volume
is the mass whatever the mapping, so the number is compared with the
exact mass of PREM, and the mesh's own volume with that of the sphere,
to show what the discretisation costs.

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

from planetmodel import (DENSITY, CallableDisplacement, Geometry, Model,
                         RadialField, RadialStretch, Skeleton, mass, prem)
from planetmodel.units import LENGTH, MASS
from planetmodel.mesh3d import (AngularResolution, MeshSpec, Shell,
                                build_layered_mesh, export_mfem, manifest)

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


def prem_density_on(geometry: Geometry, fine: Model) -> Model:
    """The density of `fine` on the coarsened layers of `geometry`, as
    numeric layer functions that look the exact model up radius by radius."""
    b = fine.skeleton.boundaries

    def rho_fine(r):
        r = np.asarray(r, dtype=float)
        out = np.empty(r.shape)
        idx = np.clip(np.searchsorted(b, r, side="right") - 1, 0, fine.nlayers - 1)
        for i in np.unique(idx):
            m = idx == i
            out[m] = fine.layer(int(i))["rho"](r[m])
        return out

    layers = [{"rho": RadialField(geometry.skeleton.interval(i), rho_fine,
                                  character=DENSITY, name="rho")}
              for i in range(geometry.nlayers)]
    return Model(geometry, layers, scales=fine.scales)


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

    fine = prem().nondimensionalised()          # radii in units of a = 6371 km
    length = fine.scales.factor(LENGTH)         # metres per unit
    h_ref = args.h_ref_km * 1e3 / length
    sk_m, dropped = coarsened_for(args.h_ref_km * 1e3)
    sk = Skeleton(sk_m.boundaries / length)
    a = float(sk.boundaries[-1])
    sk = sk.refined([0.9 * a]) if not np.any(np.isclose(sk.boundaries, 0.9 * a)) else sk
    print(f"skeleton: {sk.nlayers} layers after dropping {dropped} boundaries; "
          f"one unit is {length / 1e3:.0f} km, G = {fine.G:.3g}")
    b = a * (1.0 + BUFFER_RATIO)
    geometry = Geometry(sk, mapping=surface_relief(RELIEF_M / length, a, b))
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
    print("outer radius:", card.geometry["outer_radius"], "| layers:", len(card.layers))

    try:
        import mfem.ser as mfem
    except ImportError:
        print("PyMFEM not installed; stopping after the build")
        return 0
    t0 = time.perf_counter()
    model = prem_density_on(geometry, fine)
    exported = export_mfem(result, args.out.with_name(args.out.name + "_mfem"),
                           model=model, delivery="referential")
    opts = exported.files["mesh_read_options"]
    mesh = mfem.Mesh(str(exported.mesh_path), opts["generate_edges"], opts["refine"],
                     opts["fix_orientation"])
    print(f"MFEM read {mesh.GetNE()} elements and {mesh.GetNBE()} boundary elements "
          f"in {time.perf_counter() - t0:.1f} s; attributes "
          f"{list(mesh.attributes.ToList())}")
    wrong = mesh.CheckElementOrientation(False)
    print("elements MFEM would reorient:", wrong)

    rho = mfem.GridFunction(mesh, str(exported.field_paths["rho"]))
    one = mfem.LinearForm(rho.FESpace())
    unity = mfem.ConstantCoefficient(1.0)      # kept alive while the form is
    one.AddDomainIntegrator(mfem.DomainLFIntegrator(unity))
    one.Assemble()
    integrated = one * rho
    volume = sum(mesh.GetElementVolume(e) for e in range(mesh.GetNE())
                 if mesh.GetAttribute(e) <= geometry.nlayers)
    exact_volume = 4.0 * math.pi * a ** 3 / 3.0
    exact_mass = mass(fine)
    kg = fine.scales.factor(MASS)
    print(f"mesh volume {volume:.6f} vs sphere {exact_volume:.6f}: "
          f"relative error {volume / exact_volume - 1.0:+.3e}")
    print(f"integrated density {integrated:.6f} vs PREM mass {exact_mass:.6f} "
          f"({exact_mass * kg:.4e} kg): relative error "
          f"{integrated / exact_mass - 1.0:+.3e}")
    return 0 if math.isfinite(result.validation.min_sicn) else 1


if __name__ == "__main__":
    raise SystemExit(main())
