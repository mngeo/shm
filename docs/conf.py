"""Sphinx configuration for shgeomag documentation."""

project = "shgeomag"
author = "shgeomag contributors"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
html_theme = "pydata_sphinx_theme"
