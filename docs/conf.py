project = 'pyS3M'
copyright = '2026, University of Cambridge'
author = 'Joseph S. Beckwith'
release = 'v0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autodocsumm',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

napoleon_custom_sections = [('Returns', 'params_style')]
# Render docstring "Attributes:" sections as :ivar: cross-references into the
# real attribute object instead of a second, separate `.. attribute::` block --
# without this, autodoc's :undoc-members: (which documents dataclass fields
# directly from introspection) and napoleon's own Attributes:-section expansion
# both register the same fully-qualified name, causing "duplicate object
# description" warnings for every documented dataclass.
napoleon_use_ivar = True
auto_doc_default_options = {'autosummary': True}

# Mock heavy optional dependencies that are unavailable on headless build servers
autodoc_mock_imports = [
    'PyQt6',
    'numba',
    'datashader',   # imports numba at module level; numba is mocked so version check fails
    'fast_hdbscan', # runs HDBSCAN().fit() at import time, crashes on empty random_data
    # matplotlib's Qt backend does a real Qt-version compatibility check at import
    # time (QLibraryInfo.version().toString(), parsed and compared against its
    # minimum-supported Qt version) -- against the mocked PyQt6 above, that check
    # sees a fake string instead of a real version and raises ImportError, which
    # breaks autodoc on every gui module that imports it (app.py, main_window.py,
    # results_panel.py). Mocking the backend module itself skips that check
    # entirely, same treatment as PyQt6 above.
    'matplotlib.backends.backend_qtagg',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
