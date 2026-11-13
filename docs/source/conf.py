# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# The package lives in a src layout; point autodoc at it.
sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "planetmodel"
copyright = "2026, David Al-Attar"
author = "David Al-Attar"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Generate docs from docstrings.
    "sphinx.ext.napoleon",  # Understand numpy-style docstrings.
    "sphinx.ext.viewcode",  # Link the documentation to the source.
]

templates_path = ["_templates"]
exclude_patterns = []

# The docstrings use single backticks for inline code (`Model`, `Skeleton`).
# Left to itself reStructuredText reads those as "title reference" and
# renders them as italics; this makes them render as code, which is what
# they mean everywhere in this project.
default_role = "code"

# The dataclasses describe their fields in an `Attributes:` docstring
# section *and* declare them as annotated fields.  Napoleon's default turns
# that section into standalone `.. attribute::` directives, which autodoc
# then documents a second time from the annotations.  Rendering them as
# info-field entries instead leaves each field defined exactly once.
napoleon_use_ivar = True

# The optional extras are not installed where the docs are built; their
# importers (gmsh at the top of the mesh3d modules that build meshes, the
# rest inside the functions that need them) document against these
# stand-ins.
autodoc_mock_imports = ["gmsh", "mfem", "matplotlib"]


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]


# -- Generated API reference -------------------------------------------------
# sphinx-apidoc runs from here rather than from .readthedocs.yaml, so that a
# local build and a Read the Docs build produce the same pages from the same
# settings.  The .rst files it writes are build output, not source, and are
# gitignored.

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SOURCE_DIR, "..", ".."))


def _run_apidoc(_app: object) -> None:
    """Regenerate the per-module .rst stubs before the build reads them."""
    from sphinx.ext.apidoc import main

    main(
        [
            "--force",
            "--templatedir",
            os.path.join(_REPO_ROOT, "docs", "apidoc_templates"),
            "--output-dir",
            _SOURCE_DIR,
            os.path.join(_REPO_ROOT, "src", "planetmodel"),
            # Private modules stay out of the API reference.  The class
            # [!_] keeps the pattern off the __init__.py files, which
            # apidoc needs to see to know a directory is a package.
            os.path.join(_REPO_ROOT, "src", "planetmodel", "*", "_[!_]*.py"),
            os.path.join(_REPO_ROOT, "src", "planetmodel", "_[!_]*.py"),
        ]
    )


# -- The C++ reference -------------------------------------------------------
# Doxygen writes the reference of cpp/ by cpp/docs/Doxyfile, the file CI
# builds it by, into docs/_doxygen/cpp-reference; html_extra_path copies it
# beside these pages, where docs/source/cpp.rst links to it.  Without
# Doxygen (a local build, say) the reference is left out with a warning.

_DOXYGEN_OUT = os.path.join(_REPO_ROOT, "docs", "_doxygen")
os.makedirs(_DOXYGEN_OUT, exist_ok=True)
html_extra_path = [_DOXYGEN_OUT]


def _run_doxygen(_app: object) -> None:
    """Build the C++ reference, every warning a failure."""
    import shutil
    import subprocess
    import tomllib

    from sphinx.util import logging

    doxygen = shutil.which("doxygen")
    if doxygen is None:
        logging.getLogger(__name__).warning(
            "doxygen is not installed, so the C++ reference is left out")
        return
    with open(os.path.join(_REPO_ROOT, "pyproject.toml"), "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    env = dict(os.environ, PLANETMODEL_DOCS_OUTPUT=_DOXYGEN_OUT,
               PLANETMODEL_VERSION=version)
    subprocess.run([doxygen, os.path.join("docs", "Doxyfile")],
                   cwd=os.path.join(_REPO_ROOT, "cpp"), env=env, check=True)


def setup(app: object) -> None:
    app.connect("builder-inited", _run_apidoc)
    app.connect("builder-inited", _run_doxygen)
