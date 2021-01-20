# Configuration file for the Sphinx documentation builder.

import os
import sys
import time
import six
import twinpy
import sphinx_rtd_theme


# --------
# packages
# --------
needs_sphinx = '3.4.3'
html_theme = 'sphinx_rtd_theme'

extensions = [
    'sphinx.ext.intersphinx',   # link to other projects
    'sphinxcontrib.contentui',  # content html
    'sphinx.ext.autodoc',       # read doc automatically
    'sphinx.ext.imgmath',       # math support
    'sphinx.ext.viewcode',      # go to class and def to search the docstring
    'sphinx.ext.napoleon',      # to read various kinds of style of docstring
    'sphinx.ext.todo',
]


# -------------------
# project and version
# -------------------
project = 'twinpy'
release = twinpy.__version__
version = '.'.join(release.split('.'))


# ---------
# copyright
# ---------
copyright_first_year= '2020'
copyright_owners = 'kei0822kei'
current_year = str(time.localtime().tm_year)

if current_year == copyright_first_year:
    copyright_year_string = current_year
else:
    copyright_year_string = "{}-{}".format(
            copyright_first_year, current_year)
copyright = '{}, {}. All rights reserved'.format(
        copyright_year_string,
        copyright_owners)


# ---------------------
# ignore error patterns
# ---------------------
# ref. https://www.366service.com/jp/qa/eb1d0865228adc35d6bdcdbf18749793
nitpicky = True
nitpick_ignore = []

for line in open('nitpick-exceptions'):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, six.u(target)))



# nitpick_ignore = [
#     ('py:class', 'np.array'),
#     ('py:class', 'Phonopy'),
#     ('py:class', 'plt.figure'),
#     ('py:class', 'numpy.array'),  # If delete this, Warning occurs.
# ]
