# Configuration file for the Sphinx documentation builder.
# -- Path setup --------------------------------------------------------------
import os
import sys

# sys.path.insert(0, os.path.abspath('../../'))
current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, target_dir)

# -- Project information -----------------------------------------------------
from datetime import date

project = "ExoSim2"

author = (
    "L. V. Mugnai, E. Pascale, A. Bocchieri, A. Lorenzani, A. Papageorgiou"
)

copyright = "2020-{:d}, {}".format(date.today().year, author)

# The full version, including alpha/beta/rc tags

from exosim import __version__

release = version = str(__version__)

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # 'sphinx.ext.imgmath',
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "matplotlib.sphinxext.plot_directive",
    "autoapi.extension",
    # TODO update to sphinx-design and sphinx>5
    "sphinx_panels",
    "sphinx_design",
]
# ------------------------------------------------------------------------------
# Matplotlib plot_directive options
# ------------------------------------------------------------------------------

# Determine if the matplotlib has a recent enough version of the
# plot_directive.
from matplotlib.sphinxext import plot_directive

if plot_directive.__version__ < 2:
    raise RuntimeError("You need a recent enough version of matplotlib")
# Do some matplotlib config in case users have a matplotlibrc that will break
# things
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

plt.ioff()

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    # 'member-order': 'bysource',
    "undoc-members": True,
    #    'exclude-members': '__weakref__'
}
napoleon_use_ivar = True
autodoc_typehints = (
    "description"  # show type hints in doc body instead of signature
)
autoclass_content = (
    "both"  # get docstring from class level and init simultaneously
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
# The master toctree document.
master_doc = "index"

language = "en"

# -----------------------------------------------------------------------------
# Auto-API
# -----------------------------------------------------------------------------
autoapi_dirs = ["../../exosim"]
autoapi_root = "api"
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/dev", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "matplotlib": ("https://matplotlib.org", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "astropy": ("http://docs.astropy.org/en/latest/", None),
    "h5py": ("https://docs.h5py.org/en/latest/", None),
    "photutils": ("https://photutils.readthedocs.io/en/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

# html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
#     'collapse_navigation': False,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False
# }

# TODO change this from the example reported in pydata to ExoSim.
# TODO change the switcher.json links to ExoSim's
# Define the json_url for our version switcher.
# TODO release this with correct link when move to readthedocs
# json_url = "https://pydata-sphinx-theme.readthedocs.io/en/latest/_static/switcher.json"

# Inizializziamo json_url con un valore di default
json_url = "_static/switcher.json"

# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
if not version_match or version_match.isdigit():
    # For local development, infer the version to match from the package.
    release = __version__
    version_match = "v" + release

html_theme = "pydata_sphinx_theme"
# panels_add_bootstrap_css = False
# panels_dev_mode = True

html_theme_options = {
    "github_url": "https://github.com/arielmission-space/ExoSim2-public",
    "collapse_navigation": True,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_additional_pages = {
    #    "index": "index.html",
    "user/index": "user_index.html",
    "contributing/index": "developer_index.html",
}
html_title = "{} Manual".format(project)
html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"

htmlhelp_basename = "exosim"

add_module_names = False
add_function_parentheses = False

nbsphinx_execute = "never"

if "sphinx.ext.pngmath" in extensions:
    pngmath_use_preview = True
    pngmath_dvipng_args = ["-gamma", "1.5", "-D", "96", "-bg", "Transparent"]

# mathjax_path = "scipy-mathjax/MathJax.js?config=scipy-mathjax"
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

plot_html_show_formats = False
plot_html_show_source_link = False

# -----------------------------------------------------------------------------
# Latex output
# -----------------------------------------------------------------------------

latex_logo = "_static/exosim.png"
# inside conf.py
latex_elements = {
    "geometry": "\\usepackage[vmargin=2.5cm, hmargin=3cm]{geometry}",
    "preamble": """\
\\usepackage{xcolor}
\\usepackage[titles]{tocloft}
\\cftsetpnumwidth {1.25cm}\\cftsetrmarg{1.5cm}
\\setlength{\\cftchapnumwidth}{0.75cm}
\\setlength{\\cftsecindent}{\\cftchapnumwidth}
\\setlength{\\cftsecnumwidth}{1.25cm}
\\definecolor{mintbg}{rgb}{.63,.79,.95}
\\colorlet{lightmintbg}{mintbg!40}
""",
    "printindex": "\\footnotesize\\raggedright\\printindex",
    "sphinxsetup": """
    verbatimwithframe=false, VerbatimColor={RGB}{245,245,245},
    tipBorderColor={RGB}{0,153,0}, warningBorderColor={RGB}{153, 0,0},
    cautionBgColor = {RGB}{255,203,149}, noteBorderColor={RGB}{0,0,153},,
    noteborder = 1pt, cautionborder = 0pt, tipborder = 1pt""",
}

latex_show_urls = "footnote"
