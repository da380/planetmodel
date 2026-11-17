"""A flat model evaluated on depth nodes times a horizontal grid's nodes.

The counterpart of `sampling.sample` for a model of
`planetmodel.cartesian`: a consumer's unknowns live on a product of a
node set in depth and a horizontal grid, and the model is evaluated
there once, in one vectorised call per layer and field
(`sampling.sample_fields`, which the two samplers share); the result is
a flat `Sample`.

**Layout.**  Every sampled array is `(node,) + grid.shape +
components`: `(node, x, y, ...)` over a box, y fastest, and `(node, x,
...)` over a rectangle, whose body is the same at every y (its fields are
read at y = 0).  The nodes in depth are the per-element GLL nodes of a
`RadialMesh` over the model's skeleton of depths, flattened element by
element from the surface down, so an interface is two nodes at one
depth, one on each side, each with its own layer's one-sided value.  A
field whose layers are all of depth alone (`is_radial`) is sampled on
`(node,)` alone.  The mesh is the caller's, or sized by `drmax`, or by
default to the grid's finest node spacing, so that the depth is sampled
as finely as the lateral directions.

**Components** are in the local frame (e_up, e_x, e_y), the same at
every node, with ranks 2 and 4 in Voigt form where the character has
one.  The displacement of the geometry's mapping, m(X) - X, is sampled
the same way, (w, 0, 0) for a vertical stretch of upward displacement
w; it is None when the geometry is the identity.  A name a layer does
not hold is refused, naming the layer and the name, unless
`missing="nan"` asks for NaN on that layer's nodes.
"""
from __future__ import annotations

from collections import abc
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..character import Character
from ..mapping import Mapping
from ..mesh1d.mesh import RadialMesh
from ..sampling import MISSING, SampleBase, sample_fields
from ..units import Dimensions, Scales
from .coordinates import FLAT, LOCAL_FRAME
from .grid import Grid
from .model import Model

__all__ = ["Sample", "sample"]


@dataclass(frozen=True)
class Sample(SampleBase):
    """A flat model's fields and mapping on depth nodes times a grid's
    nodes; see the module docstring.

    `fields[name]` has shape `(nnode,) + grid.shape + c` for a field that
    varies laterally and `(nnode,) + c` for one of depth alone, with `c`
    the stored shape of its character, the components in the local
    frame; a field a layer lacks is NaN on the nodes of that layer's
    elements.  `displacement` is m(X) - X in the local frame,
    `(nnode,) + grid.shape + (3,)`, or None for an identity geometry.
    `characters` and `dimensions` say what each array is, `scales` what
    its numbers are in, and `layer_names` names the layers the nodes
    refer to.  `radius` is the depth of every node.  Every array and
    mapping the sample holds or hands out is read-only.
    """

    radial: RadialMesh
    grid: Grid
    fields: abc.Mapping[str, np.ndarray]
    displacement: np.ndarray | None
    characters: abc.Mapping[str, Character]
    dimensions: abc.Mapping[str, Dimensions | None]
    scales: Scales
    layer_names: tuple[str | None, ...]

    def __post_init__(self) -> None:
        for attr in ("fields", "characters", "dimensions"):
            object.__setattr__(self, attr, MappingProxyType(dict(getattr(self, attr))))
        object.__setattr__(self, "layer_names", tuple(self.layer_names))

    @property
    def depth(self) -> np.ndarray:
        """The depth of every flat node: `radius`, as a box names it."""
        return self.radius

    def __repr__(self) -> str:
        names = ", ".join(self.fields)
        disp = "" if self.displacement is None else ", displacement"
        return (f"cartesian.Sample({self.nnode} nodes x "
                f"{' x '.join(map(str, self.grid.shape))}; fields: {names}{disp})")


def _depth_mesh(model: Model, grid: Grid, radial: RadialMesh | None, ngll: int,
                drmax: float | None) -> RadialMesh:
    """The caller's mesh, or one sized by `drmax`, or by the grid's finest
    spacing."""
    if radial is not None:
        if not isinstance(radial, RadialMesh):
            raise TypeError(f"radial must be a RadialMesh, got "
                            f"{type(radial).__name__}")
        if radial.skeleton != model.skeleton:
            raise ValueError("radial is a mesh over another skeleton; a "
                             "sample's nodes lie on the model it samples")
        if drmax is not None:
            raise ValueError("give radial= or drmax=, not both: a mesh already "
                             "fixes its element widths")
        return radial
    step = min(grid.spacing) if drmax is None else float(drmax)
    return RadialMesh(model.geometry, ngll=ngll, drmax=step)


def _displacement(mapping: Mapping, depth: np.ndarray, grid: Grid) -> np.ndarray:
    """R^T (m(X) - X) at every node, R the local frame: the shift's local
    components."""
    x, y = grid.nodes
    X = FLAT.points(depth.reshape((-1,) + (1,) * grid.dimension), x[None], y[None])
    disp = getattr(mapping, "displacement", None)
    u = disp(X) if disp is not None else np.asarray(mapping(X), dtype=float) - X
    u = np.asarray(u, dtype=float)
    if u.shape != X.shape:
        raise ValueError(f"the mapping's displacement has shape {u.shape} for "
                         f"points of shape {X.shape}")
    out = np.ascontiguousarray(u @ LOCAL_FRAME)
    out.setflags(write=False)
    return out


def sample(model: Model, grid: Grid, *, fields: Iterable[str] | None = None,
           radial: RadialMesh | None = None, ngll: int = 5,
           drmax: float | None = None, missing: str = "refuse") -> Sample:
    """Evaluate a flat model's fields and its mapping's displacement on
    depth nodes times the nodes of `grid`, by the rules of the module
    docstring.

    `grid` is a `cartesian.Grid`, over the model's domain or any other.
    The nodes in depth are those of `radial` when the caller has a
    `RadialMesh` over the model's skeleton, else of
    `RadialMesh(model.geometry, ngll=ngll, drmax=drmax)`, `drmax` the
    grid's finest spacing unless given.  `fields` is None for the names
    every layer holds, or a sequence of names; `missing` is
    `sampling.MISSING`'s, a layer lacking a name refused unless "nan".
    """
    if not isinstance(model, Model):
        raise TypeError(f"expected a cartesian.Model, got {type(model).__name__}")
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a cartesian.Grid, got {type(grid).__name__}")
    if missing not in MISSING:
        raise ValueError(f"missing must be one of {MISSING}, got {missing!r}")
    mesh = _depth_mesh(model, grid, radial, ngll, drmax)
    arrays, chars, dims = sample_fields(model, mesh, grid, fields=fields,
                                        missing=missing)
    geometry = model.geometry
    disp = (None if geometry.is_identity
            else _displacement(geometry.mapping, mesh.r.ravel(), grid))
    return Sample(radial=mesh, grid=grid, fields=arrays, displacement=disp,
                  characters=chars, dimensions=dims, scales=model.scales,
                  layer_names=tuple(layer.name for layer in model.layers))
