Manual runs, not tests. Scripts here build meshes at Earth scale, which
takes minutes and memory, and are never collected by `pytest`.

- `mfem_cross_check.py`: a PREM-shaped geometry with surface relief and a
  buffer shell, built at a chosen resolution and read back by PyMFEM.
