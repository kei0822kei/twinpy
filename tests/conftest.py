#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is for pytest fixtures.
"""

import pytest
from twinpy.properties.hexagonal import get_hcp_cell

a = 2.93
c = 4.65
symbol = 'Ti'


@pytest.fixture(autouse=True, scope='session')
def ti_cell_wyckoff_c() -> tuple:
    """
    Ti hexagonal cell.

    Returns:
        tuple: Ti hexagonal cell.
    """
    wyckoff = 'c'
    cell = get_hcp_cell(a=a, c=c, symbol=symbol, wyckoff=wyckoff)
    return cell


@pytest.fixture(autouse=True, scope='session')
def ti_cell_wyckoff_d() -> tuple:
    """
    Ti hexagonal cell.

    Returns:
        tuple: Ti hexagonal cell.
    """
    wyckoff = 'd'
    cell = get_hcp_cell(a=a, c=c, symbol=symbol, wyckoff=wyckoff)
    return cell
