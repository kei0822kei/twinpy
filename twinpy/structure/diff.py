#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare structures
"""

import numpy as np

def get_structure_diff(*cells):
    """
    get structure diff

    Args:
        cells: cell = (lattice, scaled_positions, symbols)

    Returns:
        dict: diff info

    Note:
        return diff compared with first cell
        ex. lattice_diffs[i] = ith_lattice - 1st_lattice
    """
    cart_posis = [ np.dot(cells[i][0].T, cells[i][1].T).T
                       for i in range(len(cells)) ]
    base_lat, base_scaled_posi, _ = cells[0]
    base_cart_posi = np.dot(base_lat.T, base_scaled_posi.T).T
    lattice_diffs = [ cell[0] - base_lat for cell in cells ]
    scaled_posi_diffs = [ cell[1] - base_scaled_posi for cell in cells ]
    cart_posi_diffs = [ cart_posis[i] - base_cart_posi
                            for i in range(len(cells)) ]
    return {
            'lattice_diffs': lattice_diffs,
            'scaled_posi_diffs': scaled_posi_diffs,
            'cart_posi_diffs': cart_posi_diffs,
           }
