#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare structures
"""

import numpy as np
from twinpy.lattice.lattice import Lattice

def get_structure_diff(cells:list,
                       base_index:int=0,
                       include_base:bool=True) -> dict:
    """
    Get structure diff with first cell in cells.

    Args:
        cells (list): list of cells
        base_index (int): base cell index
        include_base (bool): if True, include base cell in output dict

    Returns:
        dict: containing 'lattice_diffs', 'scaled_posi_diffs'
              and 'cart_posi_diffs'

    Note:
        Return diff compared with base cell.
        ex. lattice_diffs[i] = ith_lattice - base_lattice.
        The value 'scaled_posi_diffs' are the difference between two
        fractional coordinates which is not consider lattice change.
        The value 'cart_posi_diffs' are the difference between two
        cartesian coordinates which is automatically consider lattice
        change.
    """
    cart_posis = [ np.dot(cells[i][0].T, cells[i][1].T).T
                       for i in range(len(cells)) ]
    base_lat, base_scaled_posi, _ = cells[base_index]
    base_cart_posi = cart_posis[base_index]
    lattice_diffs = [ cell[0] - base_lat for cell in cells ]
    _scaled_posi_diffs = [ np.round(cell[1] - base_scaled_posi, decimals=8) for cell in cells ]
    scaled_posi_diffs = [ np.where(diff>0.5, diff-1, diff) for diff in _scaled_posi_diffs ]

    cart_posi_diffs = []
    for cell, cart_posi in zip(cells, cart_posis):
        lattice = Lattice(lattice=cell[0])
        diff = lattice.get_diff(first_positions=base_cart_posi,
                                second_positions=cart_posi,
                                is_cartesian=True,
                                with_periodic=True)
        cart_posi_diffs.append(np.dot(lattice.lattice.T, diff.T).T)

    cart_norm_diffs = [ np.linalg.norm(cart_posi_diff, axis=1)
                            for cart_posi_diff in cart_posi_diffs ]

    if not include_base:
        del lattice_diffs[base_index]
        del scaled_posi_diffs[base_index]
        del cart_posi_diffs[base_index]
        del cart_norm_diffs[base_index]
    return {
            'lattice_diffs': np.array(lattice_diffs),
            'scaled_posi_diffs': np.array(scaled_posi_diffs),
            'cart_posi_diffs': np.array(cart_posi_diffs),
            'cart_norm_diffs': np.array(cart_norm_diffs),
           }
