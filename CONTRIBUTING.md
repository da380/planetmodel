# Contributing

## Development setup

Install the package with its development dependencies and the extras you
have the libraries for, then install the git hooks:

```bash
poetry install --with dev --extras "plot netcdf maps"     # the fast suite
poetry install --with dev --all-extras                     # with gmsh and PyMFEM
poetry run pre-commit install --hook-type pre-commit --hook-type pre-push
```

Both hook types are needed. On `git commit`, `pre-commit` runs ruff's
linter and a few checks on the YAML and TOML files; on `git push` it runs
the fast test suite. It also refuses commits made directly on `main`:
work happens on `develop`.

The same hooks run in CI (the `Lint (pre-commit)` job reads this very
`.pre-commit-config.yaml`), so anything that passes locally passes there.
CI runs on pull requests to `main` and on `main` once they merge, not on
pushes to `develop`: a branch is tested once, on its pull request, and
the pre-push hook covers the pushes before it.
To check the whole repository by hand:

```bash
poetry run pre-commit run --all-files
```

`ruff format` is not part of the hooks or of CI. The house style uses
hanging indents that the formatter would rewrite throughout; the linter
alone is the contract (`poetry run ruff check .`).

## Tests

```bash
poetry run pytest                                 # the fast suite, which is the default
poetry run pytest -m "not slow and not network"   # plus the gmsh unit geometries and MFEM exports
poetry run pytest -m "not network"                # everything off the network
poetry run pytest -m network                      # the registry check: fetches every data set
```

The default run is `-m "not gmsh and not mfem and not slow and not
network"`, set in `pyproject.toml`: it needs the dependencies (numpy,
scipy and ducc0), and uses the `plot`, `netcdf` and `maps` extras where
installed (the tests that need them skip otherwise). The `gmsh` and
`mfem` tests need the `meshing` and `mfem` extras and are CI's `meshing`
job, on Linux; `full-tests` runs everything off the network on demand
from the Actions tab (Actions -> CI -> Run workflow), and is worth
running before a release. The `network` tests are the `Registry check`
workflow, monthly and on demand; nothing else touches the network.

CI installs the lock. Two jobs move the dependencies off it: CI's
`lowest-pins` steps numpy, scipy and ducc0 down to the floors in
`pyproject.toml`, and the `Newest dependencies` workflow raises every
dependency to the newest the pins admit, on pull requests to `main`, on
`main` and weekly, so that an upstream release that breaks the library
is seen before a fresh `pip install` meets it. A floor that moves in
`pyproject.toml` moves in `lowest-pins` too.

No test builds a large mesh. Meshing is tested on unit geometries a few
elements across; Earth-scale runs are the scripts under `scripts/`, run by
hand.

## Tutorials

The tutorials under `examples/tutorials/` are plain Python files with
`# %%` cell markers. The notebooks under `examples/tutorials/notebooks/`
are generated from them by `examples/tutorials/make_notebooks.py`, which
is run again after a tutorial is edited (a figure is saved by the three
lines `out = FIGURES / "..."`, `fig.savefig(out, dpi=...)`,
`print("figure written to", out)`, which the notebook leaves out to show
the figure instead); `tests/test_tutorials.py` checks
that they are current, and the nbstripout hook keeps them free of outputs
should one be opened and saved. The badges in `examples/README.md` open
them in Google Colab, where their first cell installs the library from
GitHub's `main`.
Every one of them runs headless in the test suite (`tests/test_tutorials.py`),
the core ones with warnings as errors; a tutorial that starts failing is a
test failure.

## Documentation

The API reference is built by Sphinx from the docstrings and published
on Read the Docs, which builds every push from `.readthedocs.yaml`. To
build it locally:

```bash
poetry install --with docs
make -C docs html                        # the pages land in docs/build/html
```

`sphinx-apidoc` regenerates the per-module `.rst` stubs on every build
(wired up in `docs/source/conf.py`), so the sources under `docs/source/`
are `conf.py`, `index.rst`, `cpp.rst` and the file-format pages under
`formats/`; the stubs and the built pages are gitignored. The optional extras are not needed: `conf.py`
mocks them for autodoc.

## Releasing

1. Set the version in `pyproject.toml` and `src/planetmodel/__init__.py`
   (`test_version_matches_pyproject` holds the two together).
2. Merge `develop` into `main` with CI green.
3. Publish a GitHub release whose tag is `v<version>`. The `Publish to
   PyPI` workflow checks that the tag and `pyproject.toml` agree, builds
   the wheel and the sdist, and uploads them with the `PYPI_API_TOKEN`
   repository secret.

PyPI does not allow a version to be re-uploaded, so a release that goes
wrong is fixed by a new version, never by force-pushing the tag.
