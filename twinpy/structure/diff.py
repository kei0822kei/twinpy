#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module deals with structures difference.
"""

import numpy as np
from twinpy.structure.lattice import CrystalLattice


def get_structure_diff(cells:list,
                       base_index:int=0,
                       include_base:bool=True) -> dict:
    """
    Get structure diff with first cell in cells.

    Args:
        cells: List of cells.
        base_index: Base cell index.
        include_base: If True, include base cell in output dict.

    Returns:
        dict: Containing 'lattice_diffs', 'frac_posi_diffs',
              'cart_posi_diffs' and 'cart_norm_diffs'.

    Note:
        Return diff compared with base cell.
        ex. lattice_diffs[i] = ith_lattice - base_lattice.
        The value 'frac_posi_diffs' are the difference between two
        fractional coordinates which is not consider lattice change,
        which can be defined as 'shuffle'.
        The value 'cart_posi_diffs' are the difference between two
        cartesian coordinates which is automatically consider lattice
        periodicity.
    """
    cart_posis = [ np.dot(cells[i][0].T, cells[i][1].T).T
                       for i in range(len(cells)) ]
    base_lat, base_frac_posi, _ = cells[base_index]
    base_cart_posi = cart_posis[base_index]
    lattice_diffs = [ cell[0] - base_lat for cell in cells ]
    _frac_posi_diffs = [ np.round(cell[1] - base_frac_posi, decimals=8)
                               for cell in cells ]
    frac_posi_diffs = [ np.where(diff>0.5, diff-1, diff)
                              for diff in _frac_posi_diffs ]

    cart_posi_diffs = []
    for cell, cart_posi in zip(cells, cart_posis):
        lattice = Lattice(lattice=cell[0])
        diff = lattice.get_diff(first_coords=base_cart_posi,
                                second_coords=cart_posi,
                                is_cartesian=True,
                                with_periodic=True)
        cart_posi_diffs.append(np.dot(lattice.lattice.T, diff.T).T)

    cart_norm_diffs = [ np.linalg.norm(cart_posi_diff, axis=1)
                            for cart_posi_diff in cart_posi_diffs ]

    if not include_base:
        del lattice_diffs[base_index]
        del frac_posi_diffs[base_index]
        del cart_posi_diffs[base_index]
        del cart_norm_diffs[base_index]

    return {
              'lattice_diffs': np.array(lattice_diffs),
              'frac_posi_diffs': np.array(frac_posi_diffs),
              'cart_posi_diffs': np.array(cart_posi_diffs),
              'cart_norm_diffs': np.array(cart_norm_diffs),
           }
