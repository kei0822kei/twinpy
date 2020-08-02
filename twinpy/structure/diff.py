#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare structures
"""

import numpy as np

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
    base_cart_posi = np.dot(base_lat.T, base_scaled_posi.T).T
    lattice_diffs = [ cell[0] - base_lat for cell in cells ]
    scaled_posi_diffs = [ cell[1] - base_scaled_posi for cell in cells ]
    cart_posi_diffs = [ cart_posis[i] - base_cart_posi
                            for i in range(len(cells)) ]
    if not include_base:
        del lattice_diffs[base_index]
        del scaled_posi_diffs[base_index]
        del cart_posi_diffs[base_index]
    return {
            'lattice_diffs': lattice_diffs,
            'scaled_posi_diffs': scaled_posi_diffs,
            'cart_posi_diffs': cart_posi_diffs,
           }
