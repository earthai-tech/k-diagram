# -*- coding: utf-8 -*-

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime
# Adjust the path depending on where your conf.py is relative to the package
# Assuming conf.py is in docs/source and kdiagram is in the root project dir
sys.path.insert(0, os.path.abspath('../../'))

# =================
# pip install furo sphinx-copybutton
# =================
# Try to import the package version
try:
    import kdiagram
    pkg_version = kdiagram.__version__
except ImportError:
    print("Warning: kdiagram package not found. Version set to 'dev'.")
    pkg_version = 'dev'


# -- Project information -----------------------------------------------------

project = 'k-diagram'
author = 'Laurent Kouadio' # From your setup.py
current_year = datetime.datetime.now().year
copyright = f'{current_year}, {author}'

# The short X.Y version
version = '.'.join(pkg_version.split('.')[:2])
# The full version, including alpha/beta/rc tags
release = pkg_version

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',         # Include documentation from docstrings
    'sphinx.ext.autosummary',     # Generate summary tables for API docs
    'sphinx.ext.napoleon',        # Support NumPy and Google style docstrings
    'sphinx.ext.intersphinx',     # Link to other projects' documentation
    'sphinx.ext.viewcode',        # Add links to source code
    'sphinx.ext.githubpages',     # Help create .nojekyll file for GitHub Pages
    'sphinx.ext.mathjax',         # Render math equations (via MathJax)
    'sphinx_copybutton',          # Add a "copy" button to code blocks
    'myst_parser',                # Allow parsing Markdown files (like README.md)
    'sphinx_design',              # Enable design elements like cards, buttons, grids
]

# Configure Napoleon settings (for parsing NumPy/Google docstrings)
napoleon_google_docstring = False # We are using NumPy style
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False # Doc __init__ under the class docstring
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True # Include special methods like __len__
napoleon_use_admonition_for_examples = True # Use .. admonition:: Example
napoleon_use_admonition_for_notes = True    # Use .. admonition:: Note
napoleon_use_admonition_for_references = True # Use .. admonition:: Reference
napoleon_use_ivar = True          # Use :ivar: for instance variables
napoleon_use_param = True         # Use :param: for parameters
napoleon_use_rtype = True         # Use :rtype: for return types
napoleon_preprocess_types = True    # Process type strings into links
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Configure Autodoc settings
autodoc_default_options = {
    'members': True,              # Document members (methods, attributes)
    'member-order': 'bysource',   # Order members by source order
    'special-members': '__init__',# Include __init__ docstring if present
    'undoc-members': False,       # DO NOT include members without docstrings
    'show-inheritance': True,     # Show base classes
    # 'exclude-members': '__weakref__' # Exclude specific members
}
autodoc_typehints = "description" # Show typehints in description, not signature
autoclass_content = 'class'       # Include docstrings from class and __init__

# Configure Autosummary settings
autosummary_generate = True       # Enable automatic generation of stub files
autosummary_imported_members = False # Don't list imported members in summary

# MyST Parser Settings (if using Markdown includes)
myst_enable_extensions = [
    "colon_fence",  # Allow ``` fenced code blocks
    "deflist",      # Allow definition lists
    # "smartquotes", # Use smart quotes (optional)
    # "replacements", # Apply replacements (optional)
]
# myst_heading_anchors = 3 # Automatically add anchors to headings up to level 3


# Configure Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
intersphinx_cache_limit = 5 # Days to cache remote inventories
intersphinx_timeout = 10    # Seconds to wait for network access

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = 'furo' # Use the Furo theme
# extensions.append("sphinx_wagtail_theme")
# html_theme = 'sphinx_wagtail_theme'
# Theme options are theme-specific and customize the look and feel.
# Consult the Furo documentation: https://pradyunsg.me/furo/customisation/
html_theme_options = {
    # Example options:
    "light_css_variables": {
        "color_brand_primary": "#007ACC", # Example blue
        "color_brand_content": "#005FAB",
    },
    "dark_css_variables": {
        "color_brand_primary": "#52C0FF", # Example light blue for dark mode
        "color_brand_content": "#7BCBFF",
    },
    # Add Furo specific options here, e.g., for sidebar, fonts, etc.
    "sidebar_hide_name": False, # Show project name in sidebar
    "navigation_with_keys": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Optional: Add custom CSS files (relative to html_static_path)
html_css_files = [
    'css/custom.css',
]

# Optional: Add path to your logo file (relative to source directory)
html_logo = "_static/logo_k_diagram.png"

# Optional: Add path to your favicon (relative to source directory)
html_favicon = "_static/logo.ico"

# HTML title
html_title = f"{project} v{release} Documentation"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True # Set to False if you don't want source links

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Options for sphinx-copybutton ------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True