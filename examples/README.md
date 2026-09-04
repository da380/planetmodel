# Examples

`tutorials/` walks through the ideas of the library in order, one script
per idea. Each is a plain Python file with `# %%` cell markers, so it runs
as a script from any directory and cell by cell in an editor. Every
tutorial runs headless in a few seconds and writes any files it makes to
a temporary directory.

| tutorial | what it shows |
|---|---|
| `01_skeleton_and_geometry.py` | a skeleton, its surgery, a geometry with names, hollow geometries, scaling |
| `02_an_analytic_mapping.py` | a radial stretch from an analytic displacement, `F` and `J`, validity, kinks, a non-radial mapping |
| `03_a_mesh_for_mfem.py` | a 2D and a 3D mesh of a geometry with a buffer shell, the manifest, export to MFEM (needs the `meshing` and `mfem` extras) |

```
poetry run python examples/tutorials/01_skeleton_and_geometry.py
```

To run a tutorial cell by cell as a notebook, install the `notebook` extra
(`poetry install --extras notebook`), which provides the IPython kernel
an editor's interactive window uses.
