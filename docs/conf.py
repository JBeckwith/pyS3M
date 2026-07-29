import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'pyS3M'
copyright = '2026, University of Cambridge'
author = 'Joseph S. Beckwith'
release = 'v0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autodocsumm',
    'sphinx.ext.coverage',
]

napoleon_custom_sections = [('Returns', 'params_style')]
auto_doc_default_options = {'autosummary': True}

# Mock heavy optional dependencies that are unavailable on headless build servers
autodoc_mock_imports = [
    'PyQt6',
    'napari',
    'napari_animation',
    'numba',
    'datashader',   # imports numba at module level; numba is mocked so version check fails
    'fast_hdbscan', # runs HDBSCAN().fit() at import time, crashes on empty random_data
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
