import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'pyS3M'
copyright = '2024, Joseph S. Beckwith, Steven F. Lee'
author = 'Joseph S. Beckwith, Steven F. Lee'
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
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
