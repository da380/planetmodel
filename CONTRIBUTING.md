# Contributing

## Development setup

Install the package with its development dependencies and the extras you
have the libraries for, then install the git hooks:

```bash
poetry install --with dev --extras "plot harmonics"        # the fast suite
poetry install --with dev --all-extras                     # with gmsh and PyMFEM
poetry run pre-commit install --hook-type pre-commit --hook-type pre-push
```

Both hook types are needed. On `git commit`, `pre-commit` runs ruff's
linter and a few checks on the YAML and TOML files; on `git push` it runs
the fast test suite. It also refuses commits made directly on `main`:
work happens on `develop`.

The same hooks run in CI (the `Lint (pre-commit)` job reads this very
`.pre-commit-config.yaml`), so anything that passes locally passes there.
To check the whole repository by hand:

```bash
poetry run pre-commit run --all-files
```

`ruff format` is not part of the hooks or of CI. The house style uses
hanging indents that the formatter would rewrite throughout; the linter
alone is the contract (`poetry run ruff check .`).

## Tests

```bash
poetry run pytest                        # the fast suite, which is the default
poetry run pytest -m "not slow"          # plus the gmsh unit geometries and MFEM exports
poetry run pytest -m ""                  # everything
```

The default run is `-m "not gmsh and not mfem and not slow"`, set in
`pyproject.toml`: it needs numpy and scipy, and uses the `plot` and
`harmonics` extras where installed (the tests that need them skip
otherwise). The `gmsh` and `mfem` tests need the `meshing` and `mfem`
extras and are CI's `meshing` job, on Linux; `full-tests` runs everything
on demand from the Actions tab (Actions -> CI -> Run workflow), and is
worth running before a release.

No test builds a large mesh. Meshing is tested on unit geometries a few
elements across; Earth-scale runs are the scripts under `scripts/` and are
recorded under `docs/notes/`.

## Tutorials

The tutorials under `examples/tutorials/` are plain Python files with
`# %%` cell markers, not notebooks, so there are no outputs to strip.
Every one of them runs headless in the test suite (`tests/test_tutorials.py`),
the core ones with warnings as errors; a tutorial that starts failing is a
test failure.

## Releasing

1. Set the version in `pyproject.toml` and `src/planetmodel/__init__.py`
   (`test_version_matches_pyproject` holds the two together).
2. Merge `develop` into `main` with CI green.
3. Publish a GitHub release whose tag is `v<version>`. The `Publish to
   PyPI` workflow checks that the tag and `pyproject.toml` agree, builds
   the wheel and the sdist, and uploads them with the `PYPI_API_TOKEN`
   repository secret. The workflow's release trigger is switched off
   until the PyPI project and the token exist; the comment at the top
   of `.github/workflows/publish.yml` says how to enable it.

PyPI does not allow a version to be re-uploaded, so a release that goes
wrong is fixed by a new version, never by force-pushing the tag.

## Setting up PyPI for the first release

Done once, before the first release. The name `planetmodel` was free on
PyPI on 15 September 2026; a PyPI project comes into existence with its
first upload, so there is nothing to create on the site beforehand.

1. **Account.** Sign in at https://pypi.org (two-factor authentication
   is required for uploading; the account settings page walks through a
   TOTP app or a security key). Do the same at https://test.pypi.org if
   a rehearsal is wanted; it is a separate account.
2. **A token for the first upload.** Account settings -> API tokens ->
   Add API token. The scope must be "Entire account" for the first
   upload, because a project-scoped token can only be made for a project
   that already exists. Copy the `pypi-...` string at once; PyPI shows it
   only once.
3. **The repository secret.** On GitHub, the repository's Settings ->
   Secrets and variables -> Actions -> New repository secret, named
   `PYPI_API_TOKEN`, with the token as its value.
4. **Rehearsal, optional.** With a token from test.pypi.org:

   ```bash
   poetry config repositories.testpypi https://test.pypi.org/legacy/
   poetry build
   poetry publish -r testpypi --username __token__ --password pypi-...
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ planetmodel
   ```

   The extra index is there so that numpy and scipy come from the real
   PyPI. A version uploaded to TestPyPI cannot be re-uploaded there
   either, so rehearse with a `1.0.0rc1` rather than the release number.
5. **Switch the workflow on.** In `.github/workflows/publish.yml`,
   replace the `workflow_dispatch` trigger with the `release` trigger
   given in the comment above it.
6. **Release.** Set the version to `1.0.0` (step 1 above), merge to
   `main`, then on GitHub: Releases -> Draft a new release -> tag
   `v1.0.0` on `main`, a title and notes, Publish. The workflow runs;
   its log ends with the upload, and https://pypi.org/project/planetmodel
   shows the version within a minute.
7. **Narrow the token.** Once the project exists, make a new token
   scoped to `planetmodel` alone (Account settings -> API tokens -> Add
   API token -> scope: Project: planetmodel), put it in the
   `PYPI_API_TOKEN` secret in place of the account-wide one, and delete
   the account-wide token on PyPI.

The alternative to a token is PyPI's trusted publishing: on PyPI,
Account settings -> Publishing -> Add a new pending publisher, with
owner `da380`, repository `planetmodel`, workflow `publish.yml`, and no
environment; the workflow then needs `permissions: id-token: write` and
the upload step becomes `uses: pypa/gh-action-pypi-publish@release/v1`
with no username or password. No secret to rotate, but a different
workflow from pyslfp's; the token route above is the one the workflow
is written for.
