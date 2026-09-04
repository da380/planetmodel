"""python -m planetmodel.mesh3d recipe.toml -- build the mesh a recipe describes.

The whole command line is one recipe file, because everything a build
needs is in the recipe: a flag that changed the mesh without changing
the file would be a mesh nothing could reproduce.  The two flags there
are do not change what is built -- `--check` parses and reports without
meshing, and `--verbose` lets gmsh talk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _describe(recipe) -> str:
    """What the recipe resolved to, in the units the mesh will use."""
    spec = recipe.spec
    body = spec.body
    lines = [f"{recipe.source.name} -> {recipe.output}.msh",
             f"  model      {body.meta.get('name', 'unnamed')}, "
             f"{len(body.interfaces)} interfaces, "
             f"outer radius {float(body.skeleton.boundaries[-1]):.6g} m",
             f"  reference  rref = {spec.rref!r} m"]
    if spec.outer_radius is not None:
        lines.append(f"  outer      boundary at {spec.outer_radius:.6g} m"
                     + (f", named {spec.outer_name!r}"
                        if spec.outer_name else ""))
    if spec.drop_interfaces:
        lines.append(f"  drop       interfaces {list(spec.drop_interfaces)}")
    for key in ("insert", "extend"):
        radii = getattr(spec, f"{key}_radii")
        if len(radii):
            names = list(getattr(spec, f"{key}_names"))
            lines.append(f"  {key:<10} "
                         + ", ".join(f"{n or '?'} at {r:.6g} m"
                                     for r, n in zip(radii, names)))
    for buf in spec.buffers:
        lines.append(f"  buffer     ratio={buf.ratio} radius={buf.radius}")
    for name, surf in spec.surfaces.items():
        lines.append(f"  surface    {name} at {surf.reference_radius:.6g} m")
    lines.append(f"  sizing     {spec.sizing}")
    lines.append(f"  mesh       {spec.dimension}D, order {spec.order}, "
                 f"{spec.delivery} delivery")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m planetmodel.mesh3d",
        description="Build the mesh described by a TOML recipe.")
    parser.add_argument("recipe", type=Path, help="the recipe file")
    parser.add_argument("--check", action="store_true",
                        help="parse and report what would be built, without "
                             "meshing")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="let gmsh report its own progress")
    args = parser.parse_args(argv)

    from ..io import recipe as _recipe

    try:
        card = _recipe.read(args.recipe)
    except (OSError, ValueError, KeyError) as exc:
        # A recipe is a file a person wrote: a traceback through the
        # parser tells them nothing they can act on.
        print(f"{args.recipe}: {exc}", file=sys.stderr)
        return 2

    print(_describe(card))
    if args.check:
        return 0

    try:
        result = card.build(verbose=args.verbose)
    except (ValueError, RuntimeError) as exc:
        # A refusal -- sizing, validity, validation -- is an answer, not
        # a crash; it names what to change.
        print(f"{args.recipe}: {exc}", file=sys.stderr)
        return 1
    print(f"\n{result.counts['elements']} elements, "
          f"{result.counts['nodes']} nodes")
    for warning in result.validation.warnings:
        print(f"warning: {warning}")
    print(f"wrote {result.msh_path}\n      {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
