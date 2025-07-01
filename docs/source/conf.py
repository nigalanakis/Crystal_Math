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
project = 'Crystal Structure Analysis (CSA)'
copyright = '2025, Nikolaos Galanakis '
author = 'Nikolaos Galanakis'
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
    'sphinx.ext.githubpages',       # GitHub Pages support
    
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
exclude_patterns = [
    '_build',
    'Thumbs.db', 
    '.DS_Store',
    '**.ipynb_checkpoints'
]

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
    'analytics_anonymize_ip': False,
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

# Custom CSS and JS
html_css_files = [
    'custom.css',
]

html_js_files = [
    'custom.js',
]

# Logo and favicon
# html_logo = "_static/images/logo.png"  # Uncomment when you have a logo
# html_favicon = "_static/images/favicon.ico"  # Uncomment when you have a favicon

# HTML context for template variables
html_context = {
    'display_github': True,
    'github_user': 'your-org',  # Update with your GitHub username/org
    'github_repo': 'crystal-structure-analysis',  # Update with your repo name
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

# Custom sidebar
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'inherited-members': True,
}

# Automatically extract typehints
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_typehints_format = 'short'

# Mock imports for modules that might not be available during doc build
autodoc_mock_imports = [
    'ccdc',        # CCDC software - requires license
    'torch',       # PyTorch - may not be available on ReadTheDocs
    'torchvision',
    'torchaudio',
    'h5py',        # HDF5 Python bindings
    'networkx',    # NetworkX
    'pandas',      # Pandas
    'numpy',       # NumPy
    'scipy',       # SciPy
    'matplotlib',  # Matplotlib
    'plotly',      # Plotly
]

# -- Options for autosummary ------------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

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
napoleon_attr_annotations = True

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True
todo_emit_warnings = True

# -- Options for coverage extension ------------------------------------------
coverage_show_missing_items = True
coverage_ignore_modules = [
    'ccdc',  # External CCDC modules
]

# -- Options for copy button extension ---------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_exclude = '.linenos, .gp, .go'

# -- Options for MyST parser ------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath", 
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_substitutions = {
    "project": project,
    "version": version,
    "release": release,
}

# -- Options for sphinx-design -----------------------------------------------
# Enable the design extension features
sd_fontawesome_latex = True

# -- Custom configuration ----------------------------------------------------

# Add any custom roles or directives
def setup(app):
    """Custom setup function for Sphinx."""
    # Add custom CSS
    app.add_css_file('custom.css')
    
    # Add custom directives for common patterns
    from docutils.parsers.rst import directives
    from docutils import nodes
    from sphinx.util.docutils import SphinxDirective
    
    class PerformanceNote(SphinxDirective):
        """Custom directive for performance notes."""
        has_content = True
        required_arguments = 0
        optional_arguments = 1
        final_argument_whitespace = True
        
        def run(self):
            node = nodes.admonition()
            node += nodes.title(text='Performance Note')
            self.state.nested_parse(self.content, self.content_offset, node)
            node['classes'].extend(['admonition', 'performance-note'])
            return [node]
    
    class GPUNote(SphinxDirective):
        """Custom directive for GPU-specific notes."""
        has_content = True
        required_arguments = 0
        optional_arguments = 1
        final_argument_whitespace = True
        
        def run(self):
            node = nodes.admonition()
            node += nodes.title(text='GPU Acceleration')
            self.state.nested_parse(self.content, self.content_offset, node)
            node['classes'].extend(['admonition', 'gpu-note'])
            return [node]
    
    class ConfigNote(SphinxDirective):
        """Custom directive for configuration notes."""
        has_content = True
        required_arguments = 0
        optional_arguments = 1
        final_argument_whitespace = True
        
        def run(self):
            node = nodes.admonition()
            node += nodes.title(text='Configuration')
            self.state.nested_parse(self.content, self.content_offset, node)
            node['classes'].extend(['admonition', 'config-note'])
            return [node]
    
    # Register custom directives
    app.add_directive('performance-note', PerformanceNote)
    app.add_directive('gpu-note', GPUNote)
    app.add_directive('config-note', ConfigNote)

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
}

latex_documents = [
    (master_doc, 'CrystalStructureAnalysis.tex', 
     'Crystal Structure Analysis Documentation',
     'Crystal Math Team', 'manual'),
]

latex_show_urls = 'footnote'

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
     'Comprehensive analysis of molecular crystal structures using GPU acceleration.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']

# -- Additional configuration ------------------------------------------------

# Add version info to the documentation
html_last_updated_fmt = '%b %d, %Y'

# Don't show the "View page source" link
html_show_sourcelink = True

# Don't show the "Built with Sphinx" text
html_show_sphinx = True

# Show the "Edit on GitHub" link (if you have the GitHub integration)
html_show_copyright = True

# Add numbering to figures, tables and code-blocks
numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
    'section': 'Section %s',
}

# Enable cross-references to other documentation
intersphinx_disabled_domains = ['std']

# Suppress warnings for external links
suppress_warnings = [
    'toc.secnum',
    'ref.citation',
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
if not os.path.exists('_static'):
    os.makedirs('_static')

# Ensure all the required directories exist
for directory in ['_static', '_templates']:
    if not os.path.exists(directory):
        os.makedirs(directory)