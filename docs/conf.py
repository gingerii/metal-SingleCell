"""Sphinx configuration for metalsinglecell documentation."""
from datetime import datetime
import importlib.metadata as _md

# -- Project information -----------------------------------------------------
project = "metalsinglecell"
author = "Ian Gingerich"
copyright = f"{datetime.now():%Y}, {author}"
try:
    release = _md.version("metalsinglecell")
except _md.PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",                      # Markdown + executable/ipynb notebooks
    "sphinx_design",                # grid cards / buttons on the landing page
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # NumPy/Google docstring parsing
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
master_doc = "index"

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# The GPU/heavy backends are lazy-imported at runtime; mock them so the API
# reference builds anywhere (incl. ReadTheDocs' Linux workers with no Metal).
autodoc_mock_imports = ["mlx", "scanpy", "squidpy", "sklearn", "umap", "igraph",
                        "leidenalg", "pynndescent", "numba", "harmonypy", "scrublet"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_param = True

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]
nb_execution_mode = "off"           # notebooks ship with their saved outputs

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "metalsinglecell"
html_theme_options = {
    "github_url": "https://github.com/gingerii/metal-SingleCell",
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/metalsinglecell/",
         "icon": "fa-brands fa-python"},
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": False,
}
html_static_path = ["_static"]
