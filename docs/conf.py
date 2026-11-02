"""Sphinx configuration."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "SynCCFD"
copyright = "2026, Alexandre Clem"
author = "Alexandre Clem"
from synccfd import __version__  # noqa: E402

release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "plan-*.md"]

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_custom_sections = [
    ("Injection parameters", "params_style"),
    ("Merchant parameters", "params_style"),
]
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
