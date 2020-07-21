#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make structures
"""

import numpy as np
from typing import Sequence
import re

epsilon = 1e-8

def write_poscar(
        lattice:np.array,
        scaled_positions:np.array,
        symbols:Sequence[str],
        filename:str='POSCAR'):
    """
    write out structure to file
    In this function, structure is not fixed
    even if its lattice basis is left handed.

    Args:
        lattice (np.array): lattice, 3x3 numpy array
        scaled_positions (np.array): fractional positioninates
        symbols: list of symbols
        filename (str): poscar filename
    """
    symbol_sets = list(set(symbols))
    nums = []
    idx = []
    for symbol in symbol_sets:
        index = [ i for i, s in enumerate(symbols) if s == symbol ]
        nums.append(str(len(index)))
        idx.extend(index)
    positions = np.round(np.array(scaled_positions)[idx, :], decimals=9).astype(str)

    strings = ''
    strings += 'generated by twinpy\n'
    strings += '1.0\n'
    for i in range(3):
        strings += ' '.join(list(np.round(
            lattice[i], decimals=9).astype(str))) + '\n'
    strings += ' '.join(symbol_sets) + '\n'
    strings += ' '.join(nums) + '\n'
    strings += 'Direct\n'
    for position in positions:
        strings += ' '.join(list(position)) + '\n'
    print("export filename:")
    print("    %s" % filename)

    with open(filename, 'w') as f:
        f.write(strings)
