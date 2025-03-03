'''
##############################################################
# Created Date: Monday, March 3rd 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from __future__ import absolute_import
import pyufunc
import inspect
import logging
import os
import sys
import warnings
from pathlib import Path
import datetime

root = Path(__file__).resolve().parents[2]
sys.path = [str(root)] + sys.path


logger = logging.getLogger(__name__)

# Python's default allowed recursion depth is 1000.
sys.setrecursionlimit(5000)

sys.path.insert(0, os.path.abspath("./sphinxext"))

# General information about the project.
project = "pyufunc"
copyright = f'2023 - {datetime.datetime.now().year} , Xiangyong Luo'
author = 'Xiangyong Luo'
version = str(pyufunc.pkg_version)
release = version
language = "en"

source_suffix = [".rst"]
source_encoding = "utf-8"
master_doc = "index"

# -- General configuration -----------------------------------------------
extensions = [
    'sphinx.ext.napoleon',
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "nbsphinx",
    "numpydoc",
]

source_path = os.path.dirname(os.path.abspath(__file__))
pattern = os.environ.get("SPHINX_PATTERN")
single_doc = pattern is not None and pattern not in ("-api", "whatsnew")
include_api = pattern is None or pattern == "whatsnew"

# prepare index.rst from index.rst.template
# with open(os.path.join(source_path, "index.rst.template"), encoding="utf-8") as f:
#     t = jinja2.Template(f.read())
# with open(os.path.join(source_path, "index.rst"), "w", encoding="utf-8") as f:
#     f.write(
#         t.render(
#             include_api=include_api,
#             single_doc=(pattern if single_doc else None),
#         )
#     )

autosummary_generate = True if include_api else ["index"]
autodoc_typehints = "none"

# numpydoc
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = False

# nbsphinx do not use requirejs (breaks bootstrap)
nbsphinx_requirejs_path = ""

# https://sphinx-toggleprompt.readthedocs.io/en/stable/#offset
toggleprompt_offset_right = 35

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = []

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output ---------------------------------------------

# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = "sphinx-rtd-theme"

# "navigation_with_keys": True,
# "navbar-start": ["version-dropdown"],
html_theme_options = {
    'prev_next_buttons_location': 'both',
    "logo_only": False,
    "style_external_links": True,
    "style_nav_header_background": "#2980b9",
    "version_selector": True,
    "language_selector": True,
    "navigation_depth": 4,

    # navbar
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],

    "external_links": [],

    "github_url": "https://github.com/ORNL-Real-Sim",

    "logo": {"image_dark": "./_static/realsim_logo.png"},

    "show_version_warning_banner": True,
    "icon_links": [
        # {
        #     "name": "LinkedIn",
        #     "url": "https://www.linkedin.com/in/xiangyong-roy-luo/",
        #     "icon": "fab fa-linkedin",
        # }
    ],

    # footer
    "footer_start": ["copyright", "sphinx-version"],
    "footer_end": [],
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = ["themes"]

# so a file named "default.css" will overwrite the builtin "default.css".
templates_path = ["./_templates/"]
html_static_path = ["_static"]
html_title = "realtwin"
html_short_title = "realtwin"
html_logo = "./_static/realsim_logo.png"
html_favicon = "./_static/realsim_logo.ico"

html_css_files = [
    "css/getting_started.css",
    "css/pandas.css",
]

# If false, no module index is generated.
html_use_modindex = True

# Output file base name for HTML help builder.
htmlhelp_basename = "pyufunc"

# -- Options for nbsphinx ------------------------------------------------
nbsphinx_allow_errors = True

# -- Options for LaTeX output --------------------------------------------
latex_elements = {}

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples (source start
# file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    (
        "index",
        "pandas.tex",
        "pyufunc: powerful Python utility function toolkit",
        "Xiangyong Luo and the pyufunc Development Team",
        "manual",
    )
]

if include_api:
    intersphinx_mapping = {
        "dateutil": ("https://dateutil.readthedocs.io/en/latest/", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "python": ("https://docs.python.org/3/", None),
        "pyarrow": ("https://arrow.apache.org/docs/", None),
    }

# extlinks alias
extlinks = {
    "issue": ("https://github.com/xyluo25/pyufunc/issues/%s", "GH %s"),
}

ipython_execlines = [
    "import numpy as np",
    "import pandas as pd",
    # This ensures correct rendering on system with console encoding != utf8
    # (windows). It forces pandas to encode its output reprs using utf8
    # wherever the docs are built. The docs' target is the browser, not
    # the console, so this is fine.
    # 'pd.options.display.encoding="utf8"',
]


# based on numpy doc/source/conf.py
def linkcode_resolve(domain, info) -> str | None:
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            with warnings.catch_warnings():
                # Accessing deprecated objects will generate noisy warnings
                warnings.simplefilter("ignore", FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""
