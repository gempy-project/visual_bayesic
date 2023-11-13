import time
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

# Find the site-packages directory
site_packages_path = next(p for p in sys.path if 'site-packages' in p)

# Remove the site-packages path
sys.path.remove(site_packages_path)

# Add your project root to the sys.path
sys.path.insert(0, os.path.abspath('..'))

# Optionally, re-add site-packages at the end of sys.path
sys.path.append(site_packages_path)

import warnings
from visual_bayesic import __version__

from sphinx_gallery.sorting import FileNameSortKey
import numpy as np

# External examples:
import make_external_gallery
make_external_gallery.make_example_gallery()

# -- General configuration ------------------------------------------------

# Load extensions
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    "sphinx_design",
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]

linkcheck_retries = 3
linkcheck_timeout = 500

autosummary_generate = True
add_module_names = True
numpydoc_show_class_members = False

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pyvista': ('https://docs.pyvista.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None)
}

napoleon_google_docstring = True

description = 'Visual Scripting for Probabilistic Network data in Python.'

# The templates path.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Visual Bayesic'
author = 'Miguel de la Varga and Maximilian Hallenberger'
copyright = f'2023-{time.strftime("%Y")}, {author}'

# |version| and |today| tags (|release|-tag is not used).
version = __version__
release = __version__
today_fmt = '%d %B %Y'

# List of patterns to ignore, relative to source directory.
exclude_patterns = ['_build', '../tests']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Sphinx Gallery Options

root_examples_dir = "../../examples/examples/"

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        "../../examples/",
    ],
    # path where to save gallery generated examples
    "gallery_dirs": [
        "examples",
    ],
    # Patter to search for example files
    "filename_pattern": r"\.py$",
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": 'gen_modules/backreferences',
    # Modules for which function level galleries are created.  In
    "doc_module": ('subsurface', 'numpy', 'pandas'),
    "image_scrapers": ('matplotlib'),
    'first_notebook_cell': ("%matplotlib inline"),
    'reference_url': {
        # The module you locally document uses None
        'subsurface': None,
        'numpy': 'https://numpy.org/doc/stable/'
    },
}

linkcheck_ignore = [r'https://github.com/cgre-aachen/gempy_data/raw/master/',
                    r'https://raw.githubusercontent.com/softwareunderground/subsurface/main/tests/data/borehole/'
                    ]

linkcheck_request_header = {
    '*': {'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8'},
    'https://github.com': {}
}

# -- Options for HTML output ----------------------------------------------
html_theme = 'alabaster'
html_theme_options = {
    'github_user'     : 'terranigma-solutions',
    'github_repo'     : 'visual-bayesic',
    'github_type'     : 'star',
    'logo_name'       : False,
    'travis_button'   : False,
    'show_related'    : True,
    'sidebar_collapse': True,
    'prev_next_buttons_location': 'both',
}

html_static_path = ['_static']
html_logo = '_static/logos/logo-black.svg'
html_favicon = '_static/logos/favicon.ico'
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
    ]
}

html_context = {
    'menu_links_name': 'Links',
    'menu_links': [
        ('<i class="fa fa-link fa-fw"></i> SWUNG',
         'https://softwareunderground.org'),
        ('<i class="fa fa-slack fa-fw"></i> Slack',
         'https://swu.ng/slack'),
        ('<i class="fa fa-github fa-fw"></i> Source Code',
         'https://github.com/softwareunderground/subsurface'),
    ],
}

htmlhelp_basename = 'visual-bayesic-doc'

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')


# -- CSS fixes --
def setup(app):
    app.add_css_file("style.css")
