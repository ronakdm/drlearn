# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'drlearn'
copyright = '2025, Ronak Mehta'
author = 'Ronak Mehta'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#   'sphinx.ext.doctest',
#   'sphinx.ext.duration',
#   'sphinx.ext.autodoc',
#   'sphinx.ext.autosummary',
#   'sphinx.ext.intersphinx',
#   'sphinx.ext.inheritance_diagram',
#   'sphinx.ext.napoleon',
#   'sphinx.ext.viewcode',
#   "myst_parser",
#   'myst_nb',  # This is used for the .ipynb notebooks
# ]
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
]
nbsphinx_execute = "never" 
autosummary_generate = True
nb_execution_mode = "off"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Nice behavior for sklearn-style docs:
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
