from pathlib import Path
import re

# Parse version from include/ceres/version.h so it stays in sync automatically.
_version_header = (
    Path(__file__).resolve().parents[2] / 'include' / 'ceres' / 'version.h'
).read_text(encoding='utf-8')
_major = re.search(
    r'#define\s+CERES_VERSION_MAJOR\s+(\d+)', _version_header
).group(1)
_minor = re.search(
    r'#define\s+CERES_VERSION_MINOR\s+(\d+)', _version_header
).group(1)
_patch = re.search(
    r'#define\s+CERES_VERSION_REVISION\s+(\d+)', _version_header
).group(1)

project = 'Ceres Solver'
copyright = '2026 Google Inc'
author = 'Sameer Agarwal, Keir Mierle & Others'
version = f'{_major}.{_minor}'
release = f'{_major}.{_minor}.{_patch}'

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinxcontrib.jquery',
]

master_doc = 'index'
exclude_patterns = []
pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'
html_title = 'Ceres Solver'
html_domain_indices = True
html_use_index = True
html_split_index = False
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True
htmlhelp_basename = 'CeresSolverdoc'

mathjax3_config = {
    'HTML-CSS': {
        'availableFonts': ['TeX'],
        'scale': 90,
    },
}
