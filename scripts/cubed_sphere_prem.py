"""An Earth-scale cubed sphere of PREM, exported for MFEM and read back.

The test suite builds cubed spheres a few cells along an edge.  This
script builds one at the resolution a calculation would use, of PREM
with every one of its thirteen layers kept, a buffer shell outside and
relief on the surface, and says what it costs: the cells and where they
are, the time of each stage, the size of the files, and what the checks
measured.  It is not a test and is never collected as one.  Run it,
read the numbers, record them in a note.

    python scripts/cubed_sphere_prem.py --edge 32 --out /tmp/prem_cubed

The radial sizes grow with depth: `--surface-km` at the surface and
`--deep-km` at the centre, linearly in between, which is the rule a
loading calculation wants and the reverse of the lateral size, fixed by
`--edge` and narrowing towards the centre.  No layer is merged away: a
layer far thinner than a cell is wide is one cell thick, and the report
says how flat that leaves the flattest cell.  The model is the
non-dimensional one, radii in units of the outer radius, as every mesh
of a model here is built.

With the density exported, the script integrates it over the mesh: the
referential density over the reference volume is the mass whatever the
mapping, so the number is compared with the mass of PREM.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from planetmodel import PREM, CallableDisplacement, RadialStretch, mass
from planetmodel.geometry import InterfaceInfo
from planetmodel.mesh3d import (CubedSphereSpec, InterfaceSizing, Shell,
                                build_cubed_sphere, export_mfem)
from planetmodel.mesh3d.cubed import cell_volumes
from planetmodel.mesh3d.export import load_mesh
from planetmodel.units import LENGTH

#: The buffer shell outside the body, as a fraction of its radius.
BUFFER_RATIO = 0.2


def sizes_growing_with_depth(surface: float, deep: float, outer: float
                             ) -> object:
    """A sizing rule: `surface` at the radius `outer` and above it, `deep`
    at the centre, linear in radius between, each interface's size
    relaxing to `deep` over half its radius."""
    def rule(interfaces: Sequence[InterfaceInfo], _: float
             ) -> dict[int, InterfaceSizing]:
        out = {}
        for face in interfaces:
            depth = max(0.0, 1.0 - face.radius / outer)
            size = surface + (deep - surface) * depth
            out[face.index] = InterfaceSizing(size, max(deep, size),
                                              0.5 * face.radius)
        return out
    return rule


def surface_relief(amplitude: float, a: float, b: float) -> RadialStretch:
    """Relief of degree two on the surface at radius `a`, growing linearly
    from 0.9 a and decaying linearly to zero at the buffer's outer radius
    `b`."""
    def h(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        up = np.clip((r - 0.9 * a) / (0.1 * a), 0.0, 1.0)
        down = np.clip((b - r) / (b - a), 0.0, 1.0)
        zonal = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
        shape = zonal + np.sin(theta) ** 2 * np.cos(2 * phi)
        return amplitude * up * down * shape

    return RadialStretch(CallableDisplacement(h, knots=[0.9 * a, a]), rmax=b)


def megabytes(path: Path) -> float:
    """The size of a file in megabytes."""
    return path.stat().st_size / 1e6


def main(argv: Sequence[str] | None = None) -> int:
    """Build, export and report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--edge", type=int, default=32,
                    help="cells along an edge: a quarter of a great circle")
    ap.add_argument("--surface-km", type=float, default=50.0,
                    help="radial size at the surface, km")
    ap.add_argument("--deep-km", type=float, default=600.0,
                    help="radial size at the centre, km")
    ap.add_argument("--order", type=int, default=2, help="order of the geometry")
    ap.add_argument("--aspect", type=float, default=4.0,
                    help="cap on the blend's cells; 0 for none")
    ap.add_argument("--relief-km", type=float, default=3.0,
                    help="amplitude of the surface relief, km; 0 for none")
    ap.add_argument("--out", type=Path, default=Path("/tmp/prem_cubed"),
                    help="basename of the files to write")
    args = ap.parse_args(argv)

    model = PREM().nondimensionalised()
    length = model.scales.factor(LENGTH)          # metres to a unit
    km = 1e3 / length
    a = float(model.skeleton.boundaries[-1])
    b = a * (1.0 + BUFFER_RATIO)
    geometry = model.geometry
    if args.relief_km:
        geometry = geometry.with_mapping(surface_relief(args.relief_km * km, a, b))
    spec = CubedSphereSpec(
        geometry, args.edge, order=args.order, aspect=args.aspect or None,
        sizing=sizes_growing_with_depth(args.surface_km * km, args.deep_km * km, a),
        shells=(Shell(ratio=BUFFER_RATIO),))

    blend = spec.blend.size - 1
    cube = spec.edge ** 3
    block = 6 * spec.edge ** 2
    print(f"edge {spec.edge}: {0.5 * np.pi * a / spec.edge / km:.0f} km wide at the "
          f"surface, {0.5 * np.pi * spec.first_sphere / spec.edge / km:.0f} km at "
          f"the first sphere, {2 * spec.half_width / spec.edge / km:.0f} km in "
          "the cube")
    print(f"levels: {blend} in the blend, {spec.spheres.size - 1} between "
          f"{spec.spheres.size} spheres")
    print(f"cells: {spec.cells} = {cube} in the cube ({cube / spec.cells:.0%}) + "
          f"{block * blend} in the blend ({block * blend / spec.cells:.0%}) + "
          f"{block * (spec.spheres.size - 1)} above")
    inside = spec.spheres[spec.spheres <= a]
    steps = np.diff(inside) / km
    print(f"radial sizes in the body: {steps.min():.1f} to {steps.max():.0f} km")

    t0 = time.perf_counter()
    result = build_cubed_sphere(spec, args.out)
    built = time.perf_counter() - t0
    v = result.validation
    print(f"built in {built:.1f} s: {result.summary()}")
    print(f"  nodes {result.counts['nodes']}, {megabytes(result.mesh_path):.0f} MB; "
          f"worst layer volume off by {v.max_layer_volume_error:.2e}, worst aspect "
          f"ratio {v.max_aspect_ratio:.0f}")
    for warning in v.warnings:
        print(f"  warning: {warning}")

    t0 = time.perf_counter()
    export = export_mfem(result, str(args.out) + "_delivery", model=model,
                         fields=["rho"], delivery="physical")
    print(f"exported in {time.perf_counter() - t0:.1f} s: "
          f"{megabytes(export.mesh_path):.0f} MB of mesh, "
          f"{megabytes(export.field_paths['rho']):.0f} MB of density; "
          f"Jacobian ratio {export.quality['min_ratio']:.3f}")

    import mfem.ser as mfem
    t0 = time.perf_counter()
    reference = load_mesh(result.mesh_path)
    rho = mfem.GridFunction(reference, str(export.field_paths["rho"]))
    form = mfem.LinearForm(rho.FESpace())
    form.AddDomainIntegrator(mfem.DomainLFIntegrator(
        mfem.GridFunctionCoefficient(rho), 2, 3 * args.order + 2))
    form.Assemble()
    got = float(np.sum(form.GetDataArray()))
    exact = mass(model)
    volume = float(cell_volumes(reference).sum())
    sphere = 4.0 / 3.0 * np.pi * b ** 3
    print(f"read back in {time.perf_counter() - t0:.1f} s: mass off by "
          f"{got / exact - 1.0:.2e}, volume off by {volume / sphere - 1.0:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
