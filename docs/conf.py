# Configuration file for the Sphinx documentation builder.

project = "KGLite"
copyright = "2024, Kristian dF Kollsgård"
author = "Kristian dF Kollsgård"

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

# -- MyST (Markdown) settings ------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 6

# -- Sphinx-AutoAPI settings --------------------------------------------------
# Parses .pyi stubs directly — no need to import the Rust extension module.

autoapi_dirs = ["../kglite"]
autoapi_type = "python"
autoapi_file_patterns = ["*.pyi"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_python_class_content = "both"  # show class docstring + __init__ docstring
autoapi_member_order = "groupwise"

# AutoAPI reads stubs without importing the compiled extension. The Cypher
# Pygments lexer does not recognize every supported KGLite expression, so keep
# that presentation-only warning narrow; broken MyST references remain fatal.
suppress_warnings = ["autoapi.python_import_resolution", "misc.highlighting_failure"]

# -- General settings ---------------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = "KGLite"
html_theme_options = {
    "source_repository": "https://github.com/kkollsga/kglite",
    "source_branch": "main",
    "source_directory": "docs/",
}
