#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hexagonal property.
"""

import numpy as np


def get_atom_positions(wyckoff:str) -> np.array:
    """
    Get atom positions in Hexagonal Close-Packed.

    Args:
        wyckoff (str): wyckoff letter, choose 'c' or 'd'

    Returns:
        np.array: atom positions
    """
    assert wyckoff in ['c', 'd'], "wyckoff must be 'c' or 'd'"
    if wyckoff == 'c':
        atom_positions = \
            np.array([[ 1/3, -1/3,  1/4],
                      [-1/3,  1/3, -1/4]])  # No.194, wyckoff 'c'
    else:
        atom_positions = \
            np.array([[ 1/3, -1/3, -1/4],
                      [-1/3,  1/3,  1/4]])  # No.194, wyckoff 'd'
    return atom_positions
