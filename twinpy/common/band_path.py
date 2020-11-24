#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Band plot.
"""

import seekpath
from twinpy.structure.base import get_numbers_from_symbols


def get_seekpath(cell:tuple) -> dict:
    """
    Get seekpath results.

    Args:
        cell (tuple): Cell.

    Returns:
        dict: Labels and corresponding qpoints.
    """
    if type(cell[2][0]) is str:
        numbers = get_numbers_from_symbols(cell[2])
        _cell = (cell[0], cell[1], numbers)
    else:
        _cell = cell
    skp = seekpath.get_path(_cell)

    return skp


def get_labels_band_paths_from_seekpath(cell:tuple):
    """
    Get labels and band paths from seekpath result.

    Args:
        cell (tuple): Cell.

    Returns:
        list: labels
        np.array: band paths
    """
    skp = get_seekpath(cell)
    paths = skp['path']
    labels = []
    for path in paths:
        try:
            if labels[-1] != path[0]:
                labels.extend(['', path[0]])
        except IndexError:
            labels.append(path[0])
        labels.append(path[1])
    labels_qpoints = skp['point_coords']
    lb, band_paths = get_band_paths_from_labels(
            labels=labels,
            labels_qpoints=labels_qpoints)
    return lb, band_paths


def get_labels_for_twin() -> dict:
    """
    Get labels for hexagonal twin.

    Returns:
        dict: Contain keys and qpoints.

    Examples:
        Labels and qpoints for twin is as bellow.

        >>> label_qpoints = {
                'GAMMA': [0, 0, 0],
                'M_1'  : [1/2, 0, 0],
                'M_2'  : [-1/2, 1/2, 0],
                'K_1'  : [1/3, 1/3, 0],
                'K_2'  : [-1/3, 2/3, 0],
                'A'    : [0, 0, 1/2],
                'L_1'  : [1/2, 0, 1/2],
                'L_2'  : [-1/2, 1/2, 1/2],
                'H_1'  : [1/3, 1/3, 1/2],
                'H_2'  : [-1/3, 2/3, 1/2],
                }
    """
    label_qpoints = {
        'GAMMA': [0, 0, 0],
        'M_1'  : [1/2, 0, 0],
        'M_2'  : [-1/2, 1/2, 0],
        'K_1'  : [1/3, 1/3, 0],
        'K_2'  : [-1/3, 2/3, 0],
        'A'    : [0, 0, 1/2],
        'L_1'  : [1/2, 0, 1/2],
        'L_2'  : [-1/2, 1/2, 1/2],
        'H_1'  : [1/3, 1/3, 1/2],
        'H_2'  : [-1/3, 2/3, 1/2],
    }
    return label_qpoints


def get_band_paths_from_labels(labels:list,
                               labels_qpoints:dict):
    """
    Get segment qpoints which is input for phonopy.

    Args:
        labels (list): List of labels. If you want to separate band structure,
                       add '' between labels. See Examples.
        labels_qpoints (dict): Dictionary for labels and corresponding qpoints.

    Returns:
        list: labels
        np.array: band paths

    Examples:
        >>> labels = ['GAMMA', 'M_1', '', 'K_1', 'GAMMA']
        >>> get_band_paths_from_labels(labels)
            [[[  0,   0, 0],
              [1/2,   0, 0]],
             [[1/3, 1/3, 0],
              [  0,   0, 0]]]
    """
    seg_qpt = []
    qpt = []
    lb = []
    for label in labels:
        if label == '':
            seg_qpt.append(qpt)
            qpt = []
        else:
            qpt.append(labels_qpoints[label])
            lb.append(label)
    seg_qpt.append(qpt)

    return lb, seg_qpt
