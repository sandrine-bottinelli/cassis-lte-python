# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Package in non-standard location
import sys
import os
import shutil
from pathlib import Path
import sphinx_toolbox

package_dir = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, package_dir)

# Remove generated stubs with autosummary
# gen_dir = os.path.join(os.path.dirname(__file__), "api/generated")
# if os.path.isdir(gen_dir):
#     shutil.rmtree(gen_dir)

# Diagnostic output
# print("\n" + "="*50)
# print("SPHINX DIAGNOSTIC INFO")
# print("="*50)
# print(f"Current working directory: {os.getcwd()}")
# print(f"conf.py location: {__file__}")
# print(f"Absolute path being added: {os.path.abspath('..')}")
# print(f"\nPython path (sys.path):")
# for i, path in enumerate(sys.path[:5]):
#     print(f"  [{i}] {path}")
#
# print("\n" + "-"*50)
# print("ATTEMPTING IMPORT")
# print("-"*50)
#
# try:
#     import cassis_lte_python
#     print(f"✓ cassis_lte_python found at: {cassis_lte_python.__file__}")
# except ImportError as e:
#     print(f"✗ cassis_lte_python import failed: {e}")
#
# try:
#     import cassis_lte_python.utils
#     print(f"✓ cassis_lte_python.utils found")
# except ImportError as e:
#     print(f"✗ cassis_lte_python.utils import failed: {e}")
#
# try:
#     import cassis_lte_python.utils.utils
#     print(f"✓ cassis_lte_python.utils.utils found at: {cassis_lte_python.utils.utils.__file__}")
#     print(f"  Functions: {[x for x in dir(cassis_lte_python.utils.utils) if not x.startswith('_')]}")
# except ImportError as e:
#     print(f"✗ cassis_lte_python.utils.utils import failed: {e}")
#
# print("\n" + "-"*50)
# print("FILE STRUCTURE CHECK")
# print("-"*50)
# base_path = os.path.abspath('..')
# print(f"Looking in: {base_path}")
# for root, dirs, files in os.walk(base_path):
#     level = root.replace(base_path, '').count(os.sep)
#     if level < 3:  # Only show 3 levels deep
#         indent = ' ' * 2 * level
#         print(f'{indent}{os.path.basename(root)}/')
#         subindent = ' ' * 2 * (level + 1)
#         for file in files[:10]:  # Limit files shown
#             if file.endswith(('.py', '.rst')):
#                 print(f'{subindent}{file}')
#
# print("="*50 + "\n")


# try:
#     import cassis_lte_python.utils.utils as test_module
#     print("✓ Successfully imported module")
#     print("Functions found:", [f for f in dir(test_module) if not f.startswith('_')])
# except ImportError as e:
#     print("✗ Import failed:", e)

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
copyright = '2022-2026, CASSIS Team'
author = 'CASSIS Team'
release = '0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_toolbox.confval',
    # 'sphinx.ext.duration',
    # 'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    'autoapi.extension',
    'sphinx_design',  # For cards and grids
    # 'sphinx_rtd_theme',
    # 'sphinx.ext.napoleon',  # if you use Google/NumPy style docstrings
]

# autoapi settings

# def skip_member(app, what, name, obj, skip, options):
#     # skip submodules
#     if what == "module":
#         skip = True
#     return skip
#
# def setup(sphinx):
#     sphinx.connect("autoapi-skip-member", skip_member)

autoapi_dirs = [os.path.join(package_dir, "cassis_lte_python")]
autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi"
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    # 'imported-members',  # Remove this line to avoid documenting imports
]
autodoc_typehints = "signature"
autoapi_ignore = [
    '*/database/constantsdb.py',
    '*/gui/basic_units.py',
    '*/utils/logger.py',
    '*/utils/observer.py',
    '*/utils/settings*'
]
suppress_warnings = ["autoapi.python_import_resolution"]

rst_prolog = """
.. role:: summarylabel
"""

# Keep AutoAPI generated files for debugging
# autoapi_keep_files = True


templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- sphinx_togglebutton options ---------------------------------------------
# togglebutton_hint = str(_("Click to expand"))
# togglebutton_hint_hide = str(_("Click to collapse"))


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_last_updated_fmt = ""

# html_theme = 'classic'
html_theme = 'pydata_sphinx_theme'
# html_theme = "sphinx_book_theme"  # https://sphinx-book-theme.readthedocs.io
html_title = "CASSIS LTE Python"

# options, book theme
# html_theme_options = {
    # "home_page_in_toc": True,  # book theme
    # "show_navbar_depth": 2,  # book theme
    # "toc_title": "On this page",  # book theme
    # "repository_url": "https://gitlab.in2p3.fr/sandrine.bottinelli/cassis-lte-python",  # book theme
    # "use_repository_button": True,  # book theme
    # "use_source_button": True,  # book theme ; source button
    # "repository_branch": "master",  # book theme ; source button
    # "path_to_docs": "docs/source",  # book theme ; source button
    # "header_links_before_dropdown": 4,
    # "show_toc_level": 3,  # Show subsections in TOC (RHS)
    # "navigation_depth": 4,
    # "collapse_navigation": False,  # Keep nav expanded by default
    # # [left, content, right] For testing that the navbar items align properly
    # "navbar_align": "left",
    # "show_nav_level": 2,  # Alternative to navigation_depth
    # "secondary_sidebar_items": {
    #     "**/*": ["page-toc", "edit-this-page", "sourcelink"],
    #     "examples/no-sidebar": [],
    # },
# }

# options, pydata theme
html_theme_options = {
    "show_nav_level": 2,
    "show_toc_level": 3,  # Show subsections in TOC (RHS)
    "navigation_depth": 4,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    # "navbar_persistent": ["search-button"],
    # "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],  # search-field
    # "search_bar_text": "Search",
    "navbar_align": "content",  # content or left or right
    # "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],  # , "last-updated"
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
    "icon_links": [
        {
            # Label for this link
            "name": "GitLab",
            # URL where the link will redirect
            "url": "https://gitlab.in2p3.fr/sandrine.bottinelli/cassis-lte-python",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-gitlab",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            "name": "CASSIS",
            "url": "https://cassis.irap.omp.eu",
            "icon": "_static/logoCassis.gif",
            "type": "local",
        },

    ],
    "back_to_top_button": True,
    "content_footer_items": ["last-updated"],
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
