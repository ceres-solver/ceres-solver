from pathlib import Path
import re

# Parse version from include/ceres/version.h so it stays in sync automatically.
_repo_root = Path(__file__).resolve().parents[2]
_version_header = (_repo_root / 'include' / 'ceres' / 'version.h').read_text(
    encoding='utf-8'
)
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

# Verify that package.xml matches include/ceres/version.h.
_package_xml = (_repo_root / 'package.xml').read_text(encoding='utf-8')
if f'<version>{release}</version>' not in _package_xml:
    raise RuntimeError(
        f'package.xml version does not match version.h ({release})'
    )

# Parse the latest stable release metadata from CITATION.cff for index.rst.
_citation_cff = (_repo_root / 'CITATION.cff').read_text(encoding='utf-8')
_stable_version = re.search(
    r'^version:\s*(\S+)', _citation_cff, re.MULTILINE
).group(1)
_stable_date = re.search(
    r'^date-released:\s*(\d{4})-(\d{2})-\d{2}', _citation_cff, re.MULTILINE
)
_stable_year = _stable_date.group(1)
_stable_month = str(int(_stable_date.group(2)))

rst_epilog = f"""
.. |stable_version| replace:: {_stable_version}
.. |stable_year| replace:: {_stable_year}
.. |stable_month| replace:: {_stable_month}
"""

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
