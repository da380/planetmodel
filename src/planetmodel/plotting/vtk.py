"""VTK files of a model, for ParaView to slice, clip and contour.

`write_vtk` writes a model as VTK's XML: one structured grid (`.vts`) per
layer, in a directory beside the multiblock file (`.vtm`) that gathers
them, which ParaView opens.  A layer's grid is its radial nodes (`n` of
them, both ends included) by the angular grid, the pole rows a grid
lacks added and the longitudes closed by a repeated column, so that the
surface has no hole and no seam; each layer keeps its own ends, so that
a jump at an interface stays one.  The points are Cartesian, at their
physical positions by default and on the reference sphere with
`physical=False`.  The point data are the fields as they are: a scalar
as it is, a vector in Cartesian components, a rank-2 tensor as its nine
Cartesian components (which ParaView takes as a tensor), a rank-4 one
as its 36 Cartesian Voigt components; a `layer` array for thresholding;
and, asked for with `departures={"vs": "mean"}`, a field's percent
departure as an array of its own (`vs_percent_from_mean`), by
`slices.evaluate`.  The numbers are written in binary, each array
compressed by zlib in blocks and encoded in base64, all with the
standard library; the files are written under a name of their own and
moved into place when whole, so that a failed write leaves nothing and
an earlier write stands until it is replaced.  Nothing here needs
matplotlib.

`write_flat_vtk` writes a flat model of `planetmodel.cartesian` the same
way: a layer's grid is its depth nodes by the nodes of a `cartesian.Grid`
(48 along each axis of the model's domain by default), each lateral axis
closed by the far end of its period, where the fields are evaluated as
they are, so that the box has no gap; over a `Rectangle`, whose body is
the same at every y, the one row y = 0, a sheet in the (x, z) plane.
The points are z up, at their physical positions by default, the
components Cartesian, and a departure is from the horizontal mean
(`cartesian.slices.evaluate`), taken on the grid by default.
"""
from __future__ import annotations

import base64
import os
import shutil
import tempfile
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..fields import Field, stored_shape
from ..frames import cartesian_points, voigt_to_tensor
from ..slices import evaluate

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..cartesian.grid import Grid
    from ..cartesian.model import Model as FlatModel
    from ..model import Model, ModelBase
    from ..sampling import AngularGrid

__all__ = ["write_vtk", "write_flat_vtk", "BLOCK_BYTES", "FLAT_NODES"]

#: The bytes of each block an array is compressed in, as VTK's own writer
#: blocks them.
BLOCK_BYTES = 32768

#: The nodes along each axis of a flat model's grid when none is given.
FLAT_NODES = 48

_TYPES = {np.dtype("<f8"): "Float64", np.dtype("<i4"): "Int32"}


def _encoded(values: np.ndarray) -> str:
    """An array as VTK's compressed binary: a header of UInt64 (the block
    count, the block size, the last block's size, each compressed block's
    size) and the blocks, each encoded in base64."""
    raw = np.ascontiguousarray(values).tobytes()
    blocks = [raw[i:i + BLOCK_BYTES] for i in range(0, len(raw), BLOCK_BYTES)]
    packed = [zlib.compress(block) for block in blocks]
    header = np.array([len(blocks), BLOCK_BYTES, len(blocks[-1]) if blocks else 0,
                       *(len(p) for p in packed)], dtype="<u8")
    return (base64.b64encode(header.tobytes()).decode()
            + base64.b64encode(b"".join(packed)).decode())


def _array(name: str | None, values: np.ndarray, components: int) -> str:
    """One DataArray element."""
    named = "" if name is None else f' Name="{name}"'
    return (f'<DataArray type="{_TYPES[values.dtype]}"{named} '
            f'NumberOfComponents="{components}" format="binary">'
            f"{_encoded(values)}</DataArray>")


def _point_values(field: Field, r: np.ndarray, theta: np.ndarray,
                  phi: np.ndarray) -> tuple[np.ndarray, int]:
    """A field at the points in Cartesian components, flattened to (points,
    components), and the number of components, by the module docstring."""
    v = np.asarray(field.evaluate(r, theta, phi, frame="cartesian"))
    if np.iscomplexobj(v):
        raise TypeError(f"{field!r} is complex; write its real or imaginary part as a "
                        "field of its own")
    character = field.character
    shape = stored_shape(character)
    if character.rank == 2 and shape == (6,):
        v = voigt_to_tensor(v, rank=2)
        shape = (3, 3)
    size = int(np.prod(shape))
    return v.reshape(-1, size).astype("<f8"), size


def _layer_grid(model: Model, layer: int, grid: AngularGrid, n: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The radii, the colatitudes with the pole rows a grid lacks, and the
    longitudes closed by a repeated column, of a layer's block."""
    lo, hi = model.skeleton.interval(layer)
    theta = grid.colatitudes
    north, south = grid.pole_nodes
    theta = np.concatenate([[] if north else [0.0], theta, [] if south else [np.pi]])
    phi = np.append(grid.longitudes, grid.longitudes[0] + 2.0 * np.pi)
    return np.linspace(lo, hi, n), theta, phi


def _block(model: Model, layer: int, grid: AngularGrid, n: int, physical: bool,
           names: Iterable[str], departures: Mapping[str, str],
           mean_grid: AngularGrid | None) -> str:
    """One layer's structured grid, as a `.vts` document."""
    r, theta, phi = _layer_grid(model, layer, grid, n)
    R, T, P = np.meshgrid(r, theta, phi, indexing="ij")
    X = cartesian_points(R, T, P)
    if physical and not model.geometry.is_identity:
        X = np.asarray(model.geometry.mapping(X), dtype=float)
    held = model.layer(layer)
    arrays = []
    for name in names:
        if name not in held:
            continue
        values, size = _point_values(held[name], R, T, P)
        values = values.reshape(R.shape + (size,))
        values[:, :, -1] = values[:, :, 0]
        arrays.append(_array(name, values.reshape(-1, size), size))
    for name, what in departures.items():
        if name not in held:
            continue
        values = evaluate(model, name, layer, R, T, P, relative_to=what,
                          departure="percent", mean_grid=mean_grid)
        values[:, :, -1] = values[:, :, 0]
        arrays.append(_array(f"{name}_percent_from_{what}",
                             np.asarray(values, dtype="<f8").reshape(-1, 1), 1))
    arrays.append(_array("layer", np.full(R.size, layer, dtype="<i4"), 1))
    extent = f"0 {phi.size - 1} 0 {theta.size - 1} 0 {r.size - 1}"
    return _document(extent, arrays, X)


def _document(extent: str, arrays: list[str], X: np.ndarray) -> str:
    """A structured grid of `extent`, its point data `arrays` and its points
    X, as a `.vts` document."""
    points = _array(None, np.asarray(X, dtype="<f8").reshape(-1, 3), 3)
    return ('<?xml version="1.0"?>\n'
            '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" '
            'header_type="UInt64" compressor="vtkZLibDataCompressor">\n'
            f'<StructuredGrid WholeExtent="{extent}">\n<Piece Extent="{extent}">\n'
            f'<PointData>\n{chr(10).join(arrays)}\n</PointData>\n'
            f'<Points>\n{points}\n</Points>\n'
            '</Piece>\n</StructuredGrid>\n</VTKFile>\n')


def write_vtk(path: str | os.PathLike[str], model: Model, *,
              grid: AngularGrid | None = None, fields: Iterable[str] | None = None,
              n: int = 8, physical: bool = True,
              departures: Mapping[str, str] | None = None,
              mean_grid: AngularGrid | None = None) -> Path:
    """Write a model as VTK files for ParaView, by the rules of the module
    docstring, and return the multiblock file's path.

    `path` names the `.vtm` (the suffix added where it lacks it, its
    directory made where it is missing), the layers' `.vts` files going
    into a directory of its stem beside it;
    `grid` the angular grid, equiangular at three degrees by default;
    `fields` the names to write, every name a layer holds by default; `n`
    the points per layer; `physical` the points' positions; `departures`
    a field's percent departure from `"mean"` or `"reference"`, by name,
    a field without a mean of its own taking it on `mean_grid`.
    """
    from ..sampling import equiangular
    grid = equiangular(60, 120) if grid is None else grid
    names, departures = _checked(model, n, fields, departures)
    return _written(path, model, lambda layer: _block(
        model, layer, grid, int(n), physical, names, departures, mean_grid))


def _checked(model: ModelBase, n: int, fields: Iterable[str] | None,
             departures: Mapping[str, str] | None
             ) -> tuple[tuple[str, ...], dict[str, str]]:
    """The names to write and the departures, or the refusal."""
    if int(n) < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    names = model.field_names() if fields is None else tuple(fields)
    if isinstance(names, str):
        raise TypeError("fields is a sequence of names, not one string")
    for name in names:
        if not model.layers_with(name):
            raise KeyError(f"no layer holds a field {name!r}")
    departures = dict(departures or {})
    for name, what in departures.items():
        if what not in ("mean", "reference"):
            raise ValueError(f"a departure of {name!r} is from 'mean' or 'reference', "
                             f"got {what!r}")
    return names, departures


def _written(path: str | os.PathLike[str], model: ModelBase,
             document: Callable[[int], str]) -> Path:
    """The multiblock file at `path` and its layers' documents, each
    `document(layer)`, written whole under names of their own and moved
    into place, an earlier write standing until it is replaced; the
    multiblock file's path."""
    path = Path(path)
    if path.suffix != ".vtm":
        path = path.with_name(path.name + ".vtm")
    stem, parent = path.stem, path.parent
    parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{stem}.", dir=parent))
    try:
        (partial / stem).mkdir()
        entries = []
        for info in model.geometry.layers:
            label = info.name or "layer"
            file = f"{info.index:02d}_{label}.vts"
            (partial / stem / file).write_text(document(info.index))
            entries.append(f'<DataSet index="{info.index}" name="{label}" '
                           f'file="{stem}/{file}"/>')
        (partial / f"{stem}.vtm").write_text(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="vtkMultiBlockDataSet" version="1.0" '
            'byte_order="LittleEndian" header_type="UInt64">\n'
            f'<vtkMultiBlockDataSet>\n{chr(10).join(entries)}\n'
            '</vtkMultiBlockDataSet>\n</VTKFile>\n')
        # in place: the old directory moved aside, the new one and the file
        # moved in, the old one removed
        directory = parent / stem
        aside = None
        if directory.exists():
            aside = Path(tempfile.mkdtemp(prefix=f".{stem}.old.", dir=parent)) / stem
            os.replace(directory, aside)
        os.replace(partial / stem, directory)
        os.replace(partial / f"{stem}.vtm", path)
        if aside is not None:
            shutil.rmtree(aside.parent, ignore_errors=True)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return path


def _flat_block(model: FlatModel, layer: int, grid: Grid, n: int, physical: bool,
                names: Iterable[str], departures: Mapping[str, str],
                mean_grid: Grid) -> str:
    """One layer of a flat model's structured grid, as a `.vts` document."""
    from ..cartesian.coordinates import FLAT
    from ..cartesian.slices import evaluate as flat_evaluate
    lo, hi = model.skeleton.interval(layer)
    depth = np.linspace(lo, hi, n)
    x = np.append(grid.x, grid.x[0] + grid.lengths[0])
    y = (np.append(grid.y, grid.y[0] + grid.lengths[1]) if grid.dimension == 2
         else np.zeros(1))
    D, A, B = np.meshgrid(depth, x, y, indexing="ij")
    X = FLAT.points(D, A, B)
    if physical and not model.geometry.is_identity:
        X = np.asarray(model.geometry.mapping(X), dtype=float)
    held = model.layer(layer)
    arrays = []
    for name in names:
        if name not in held:
            continue
        values, size = _point_values(held[name], D, A, B)
        arrays.append(_array(name, values, size))
    for name, what in departures.items():
        if name not in held:
            continue
        values = flat_evaluate(model, name, layer, D, A, B, relative_to=what,
                               departure="percent", mean_grid=mean_grid)
        arrays.append(_array(f"{name}_percent_from_{what}",
                             np.asarray(values, dtype="<f8").reshape(-1, 1), 1))
    arrays.append(_array("layer", np.full(D.size, layer, dtype="<i4"), 1))
    extent = f"0 {y.size - 1} 0 {x.size - 1} 0 {depth.size - 1}"
    return _document(extent, arrays, X)


def write_flat_vtk(path: str | os.PathLike[str], model: FlatModel, *,
                   grid: Grid | None = None, fields: Iterable[str] | None = None,
                   n: int = 8, physical: bool = True,
                   departures: Mapping[str, str] | None = None,
                   mean_grid: Grid | None = None) -> Path:
    """Write a flat model as VTK files for ParaView, by the rules of the
    module docstring, and return the multiblock file's path.

    `path` names the `.vtm` as `write_vtk` takes it; `grid` a
    `cartesian.Grid`, `FLAT_NODES` along each axis of the model's domain
    by default; `fields`, `n`, `physical` and `departures` as
    `write_vtk` takes them, a departure from the mean of a field without
    one of its own taken on `mean_grid`, `grid` itself by default.
    """
    from ..cartesian.grid import Grid
    from ..cartesian.model import Model as FlatModel
    if not isinstance(model, FlatModel):
        raise TypeError(f"expected a cartesian.Model, got {type(model).__name__}: a "
                        "spherical model is written by write_vtk")
    grid = Grid(model.geometry.domain, FLAT_NODES) if grid is None else grid
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a cartesian.Grid, got {type(grid).__name__}")
    names, departures = _checked(model, n, fields, departures)
    mean_grid = grid if mean_grid is None else mean_grid
    return _written(path, model, lambda layer: _flat_block(
        model, layer, grid, int(n), physical, names, departures, mean_grid))
