# Configuration file for the Sphinx documentation builder.

import os
import sys
import time
import twinpy
import sphinx_rtd_theme


# --------
# packages
# --------
needs_sphinx = '2.4.4'
html_theme = 'sphinx_rtd_theme'

extensions = [
    'sphinx.ext.intersphinx',   # link to other projects
    'sphinxcontrib.contentui',  # content html
    'sphinx.ext.autodoc',       # read doc automatically
    'sphinx.ext.mathjax',       # math support
    'sphinx.ext.viewcode',      # go to class and def to search the docstring
    'sphinx.ext.napoleon',      # to read various kinds of style of docstring
    'sphinx.ext.todo',
]


# -------------------
# project and version
# -------------------
project = 'twinpy'
release = twinpy.__version__
version = '.'.join(release.split('.')[:2])


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
nitpick_ignore = [
    ('py:exc', 'ArithmeticError'),
    ('py:exc', 'AssertionError'),
    ('py:exc', 'AttributeError'),
    ('py:exc', 'BaseException'),
    ('py:exc', 'BufferError'),
    ('py:exc', 'DeprecationWarning'),
    ('py:exc', 'EOFError'),
    ('py:exc', 'EnvironmentError'),
    ('py:exc', 'Exception'),
    ('py:exc', 'FloatingPointError'),
    ('py:exc', 'FutureWarning'),
    ('py:exc', 'GeneratorExit'),
    ('py:exc', 'IOError'),
    ('py:exc', 'ImportError'),
    ('py:exc', 'ImportWarning'),
    ('py:exc', 'IndentationError'),
    ('py:exc', 'IndexError'),
    ('py:exc', 'KeyError'),
    ('py:exc', 'KeyboardInterrupt'),
    ('py:exc', 'LookupError'),
    ('py:exc', 'MemoryError'),
    ('py:exc', 'NameError'),
    ('py:exc', 'NotImplementedError'),
    ('py:exc', 'OSError'),
    ('py:exc', 'OverflowError'),
    ('py:exc', 'PendingDeprecationWarning'),
    ('py:exc', 'ReferenceError'),
    ('py:exc', 'RuntimeError'),
    ('py:exc', 'RuntimeWarning'),
    ('py:exc', 'StandardError'),
    ('py:exc', 'StopIteration'),
    ('py:exc', 'SyntaxError'),
    ('py:exc', 'SyntaxWarning'),
    ('py:exc', 'SystemError'),
    ('py:exc', 'SystemExit'),
    ('py:exc', 'TabError'),
    ('py:exc', 'TypeError'),
    ('py:exc', 'UnboundLocalError'),
    ('py:exc', 'UnicodeDecodeError'),
    ('py:exc', 'UnicodeEncodeError'),
    ('py:exc', 'UnicodeError'),
    ('py:exc', 'UnicodeTranslateError'),
    ('py:exc', 'UnicodeWarning'),
    ('py:exc', 'UserWarning'),
    ('py:exc', 'VMSError'),
    ('py:exc', 'ValueError'),
    ('py:exc', 'Warning'),
    ('py:exc', 'WindowsError'),
    ('py:exc', 'ZeroDivisionError'),
    ('py:obj', 'str'),
    ('py:obj', 'list'),
    ('py:obj', 'tuple'),
    ('py:obj', 'int'),
    ('py:obj', 'float'),
    ('py:obj', 'bool'),
    ('py:obj', 'Mapping'),
    ('py:class', 'list'),
    ('py:class', 'np.array'),
    ('py:class', 'function'),
]
