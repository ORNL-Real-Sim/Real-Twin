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
# import sphinx_rtd_theme
# import warnings
# import inspect

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
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.extlinks',
    'sphinx.ext.todo',
    'sphinx_copybutton',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
templates_path = ["./_templates/"]

external_links = {
    "GitHub": ("https: // github.com/ORNL-Real-Sim/Real-Twin")
}

# If true, section author and module author directives will be shown in the
# output. They are ignored by default.
show_authors = True

# -- Options for HTML output ---------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_css_files = [
    "css/realtwin_css.css",
]

# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# so a file named "default.css" will overwrite the builtin "default.css".
html_title = "realtwin"
html_short_title = "realtwin"
# html_logo = "./_static/realsim_logo.ico"
# html_favicon = "./_static/realsim_logo_01.ico"
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "realtwin"

#
# html_sidebars = {
#     '**': [
#         'index.html',
#         'genindex.html',   # adds a link to the generated index page
#         'searchbox.html',
#         'py-modindex.html',
#     ]
# }
rst_prolog = """
 .. include:: <s5defs.txt>

 """
