# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DRLearn'
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
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "numpy": ("https://numpy.org/doc/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
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

# Make all autoclass/automodule pages include members by default:
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
# (optional) If your class docstrings live on the class instead of __init__:
autoclass_content = "class"  # or "both" to merge class + __init__ docstrings
suppress_warnings = ["autodoc"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    # optional tweaks
    "sidebar_hide_name": True,
    # "light_logo": "logo.png",  
    # "dark_logo": "logo-dark.png",
    "sidebar_hide_name": False,
}
html_static_path = ['_static']

import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
