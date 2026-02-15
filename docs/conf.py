import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

# print("sys.path:", sys.path)

project = 'PyLabFlow'
copyright = '2025, BBEK-Anand'
author = 'BBEK-Anand'
release = '1.0'


extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # for Google/Numpy-style docstrings
    'sphinx.ext.viewcode',
    'sphinx_copybutton'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_typehints = 'description'  # keeps type hints in descriptions, less clutter

html_theme = "sphinx_rtd_theme" # if installed
html_static_path = [] #['_static']
add_module_names = False
autodoc_class_signature = "mixed"#"separated"  # or "mixed"