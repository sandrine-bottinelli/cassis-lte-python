# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Package in non-standard location
import sys
from pathlib import Path
import sphinx_toolbox

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# autodoc_mock_imports = ['astropy', 'pandas', 'lmfit', 'spectral_cube', 'regions', 'scipy']
# import mock
#
# MOCK_MODULES = ['numpy', 'scipy', 'sklearn', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate', 'scipy.special', 'math', '__future__', 'toolboxutilities']
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()
source_encoding = 'utf-8-sig'

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CASSIS LTE Python'
copyright = '2022-2025, CASSIS Team'
author = 'CASSIS Team'
release = '0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_toolbox.confval',
    # 'sphinx.ext.duration',
    # 'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # 'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- sphinx_togglebutton options ---------------------------------------------
# togglebutton_hint = str(_("Click to expand"))
# togglebutton_hint_hide = str(_("Click to collapse"))


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_last_updated_fmt = ""

# html_theme = 'classic'
# html_theme = 'pydata_sphinx_theme'
html_theme = "sphinx_book_theme"  # https://sphinx-book-theme.readthedocs.io
html_title = "CASSIS LTE Python"
html_theme_options = {
    "home_page_in_toc": True,  # book theme
    "show_navbar_depth": 3,  # book theme
    "toc_title": "On this page",  # book theme
    "repository_url": "https://gitlab.in2p3.fr/sandrine.bottinelli/cassis-lte-python",  # book theme
    "use_repository_button": True,  # book theme
    # "use_source_button": True,  # book theme ; source button
    # "repository_branch": "master",  # book theme ; source button
    # "path_to_docs": "docs/source",  # book theme ; source button
    # "header_links_before_dropdown": 4,
    # "show_toc_level": 3,
    # # [left, content, right] For testing that the navbar items align properly
    # "navbar_align": "left",
    # "show_nav_level": 2,
    # "secondary_sidebar_items": {
    #     "**/*": ["page-toc", "edit-this-page", "sourcelink"],
    #     "examples/no-sidebar": [],
    # },
}

# html_theme_options = {
#     "collapsiblesidebar": "false",
#     "rightsidebar": "false",
#     "sidebarbgcolor": "gainsboro",
#     "sidebartextcolor": "indigo",
#     "sidebarlinkcolor": "darkmagenta",
#     "relbarbgcolor": "indigo",
#     "headbgcolor": "lavender",
#     "headtextcolor": "indigo"
# }

# html_sidebars = {
#     '**': [
#         'localtoc.html',
#         'globaltoc.html',
#         'searchbox.html'
#     ],
#     'collapse_navigation': True
# }

# html_context = {
#     "github_user": "sandrine.bottinelli",  # Yes, it's called "github_user" but works for GitLab too
#     "github_repo": "cassis-lte-python",
#     "github_version": "main",
#     "doc_path": "docs",  # Path to your docs in the repo
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["cassis.css"]
