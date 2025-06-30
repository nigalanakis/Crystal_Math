# docs/source/conf.py
# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.

# Add the source directory to Python path
docs_dir = Path(__file__).parent.parent.absolute()
project_root = docs_dir.parent
src_dir = project_root / "src"

sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------
project = 'CrystalMath Structure Analysis (CSA)'
copyright = '2024, Nikolaos Galanakis'
author = 'Crystal Math Team'
release = '2.0.0'
version = '2.0'

# -- General configuration ---------------------------------------------------
extensions = [
    # Core Sphinx extensions
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx.ext.viewcode',          # Add source code links
    'sphinx.ext.napoleon',          # Support Google/NumPy style docstrings
    'sphinx.ext.intersphinx',       # Link to other documentation
    'sphinx.ext.mathjax',           # Render math equations
    'sphinx.ext.todo',              # Support TODO items
    'sphinx.ext.coverage',          # Check documentation coverage
    
    # Third-party extensions
    'sphinx_rtd_theme',             # Read the Docs theme
    'sphinx_copybutton',            # Copy button for code blocks
    'sphinx_design',                # Design elements (cards, tabs, etc.)
    'myst_parser',                  # Markdown support
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS
html_css_files = [
    'custom.css',
]

# Logo and favicon
html_logo = "_static/images/logo.png"  # Add your logo
html_favicon = "_static/images/favicon.ico"  # Add your favicon

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Automatically extract typehints
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# -- Options for autosummary ------------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True

# -- Options for napoleon ----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for coverage extension ------------------------------------------
coverage_show_missing_items = True

# -- Options for copy button extension ---------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Custom configuration ----------------------------------------------------

# Mock imports for modules that might not be available during doc build
autodoc_mock_imports = [
    'ccdc',  # CCDC software - requires license
    'torch', # PyTorch - large dependency
    'h5py',  # HDF5 Python bindings
]

# Add any custom roles or directives
def setup(app):
    """Custom setup function for Sphinx."""
    app.add_css_file('custom.css')
    
    # Add custom directives for common patterns
    from docutils.parsers.rst import directives
    from docutils import nodes
    from sphinx.util.docutils import SphinxDirective
    
    class PerformanceNote(SphinxDirective):
        """Custom directive for performance notes."""
        has_content = True
        
        def run(self):
            node = nodes.admonition()
            node += nodes.title(text='Performance Note')
            self.state.nested_parse(self.content, self.content_offset, node)
            node['classes'].append('performance-note')
            return [node]
    
    app.add_directive('performance-note', PerformanceNote)

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'fncychap': '',
    'maketitle': '',
    'printindex': '',
}

latex_documents = [
    (master_doc, 'CrystalStructureAnalysis.tex', 
     'Crystal Structure Analysis Documentation',
     'Crystal Math Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'crystal-structure-analysis', 
     'Crystal Structure Analysis Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'CrystalStructureAnalysis', 
     'Crystal Structure Analysis Documentation',
     author, 'CrystalStructureAnalysis', 
     'Comprehensive analysis of molecular crystal structures.',
     'Miscellaneous'),
]