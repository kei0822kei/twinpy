#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is for pytest fixtures.
"""

import pytest
from twinpy.properties.hexagonal import (get_hexagonal_lattice_from_a_c,
                                         get_hcp_atom_positions)

@pytest.fixture(autouse=True, scope='session')
def ti_cell():
    """
    Ti hexagonal cell.

    Returns:
        tuple: Ti hexagonal cell.
    """
    a = 2.93
    c = 4.65
