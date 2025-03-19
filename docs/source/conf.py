'''
##############################################################
# Created Date: Monday, March 3rd 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from __future__ import absolute_import
import logging
import os
import sys
from pathlib import Path
import datetime
import os
import sys
import sphinx_rtd_theme
import warnings
import inspect

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(1, os.path.abspath('../../realtwin'))

root = Path(__file__).resolve().parents[2]
sys.path = [str(root)] + sys.path

logger = logging.getLogger(__name__)

# Python's default allowed recursion depth is 1000.
sys.setrecursionlimit(5000)

# General information about the project.
project = "realtwin"
copyright = f'2025 - {datetime.datetime.now().year}, ORNL-RealTwin'
author = 'ORNL-RealTwin'
version = "0.1"
release = version
language = "en"

source_suffix = {'.rst': 'restructuredtext'}
source_encoding = "utf-8"
master_doc = "index"

# -- General configuration -----------------------------------------------
extensions = [
    'sphinx.ext.napoleon',
    # "sphinx_copybutton",
    # "sphinx_design",
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
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
templates_path = ["./_templates/"]

# If true, section author and module author directives will be shown in the
# output. They are ignored by default.
show_authors = True

# -- Options for HTML output ---------------------------------------------
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'prev_next_buttons_location': 'both',
    "logo_only": False,
    # "style_external_links": True,
    "style_nav_header_background": "#2980b9",
    "version_selector": True,
    "language_selector": True,
    "navigation_depth": 4,
}

# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# so a file named "default.css" will overwrite the builtin "default.css".
html_title = "realtwin"
html_short_title = "realtwin"
html_logo = "./_static/realsim_logo.ico"
html_favicon = "./_static/realsim_logo_00.ico"
html_show_sourcelink = True
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "realtwin"


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
