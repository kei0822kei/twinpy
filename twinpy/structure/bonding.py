#!/usr/bin/env pythoo
# -*- coding: utf-8 -*-

"""
Bonding base.
"""

import numpy as np
import itertools
from twinpy.lattice.lattice import Lattice


def get_neighbors(cell:tuple,
                  idx:int,
                  distance_cutoff:float,
                  get_distances:bool=False) -> list:
    """
    Get neighboring atoms from idx th atom.

    Args:
        cell (tuple): cell
        idx (int): index of specific atom
        distance_cutoff (float): distance cutoff
        get_distances (bool): if True, return also distances

    Returns:
        list: List of neighboring atoms. Each data contains
              atom_index in the first element and periorics
              in the remaining elements.
        list: Optional. If get_distances=True, return distances
    """
    lattice = Lattice(cell[0])
    periodics = np.floor(distance_cutoff / lattice.abc).astype(int)  # round up

    neighbors = []
    distances = []

    for i, posi in enumerate(cell[1]):
        grids = [ [ i for i in range(-(x+1), (x+2)) ] for x in periodics ]
        for l, m, n in itertools.product(*grids):
            if i == idx and [l, m, n] == [0, 0, 0]:
                continue

            distance = lattice.get_distance(
                    cell[1][idx],
                    posi+np.array([l, m, n]))
            if distance < distance_cutoff:
                neighbors.append([i, l, m, n])
                distances.append(distance)

    # sort
    sort_idx = np.argsort(distances)
    distances = list(np.array(distances)[sort_idx])
    neighbors = list(np.array(neighbors)[sort_idx])
    neighbors = list(map(tuple, neighbors))

    if get_distances:
        return (neighbors, distances)
    else:
        return neighbors


def get_nth_neighbor_distance(cell:tuple,
                              idx:int,
                              n:int,
                              max_distance:float) -> float:
    """
    Check atoms within max_distance and get n th neighbor distance.

    Args:
        cell (tuple): cell
        idx (int): index of specific atom
        n (int): n th neighbor
        max_distance (float): distance cutoff

    Returns:
        float: n th neighbor distance

    Todo:
        results may become different depends on the values of
        distance_cutoff and decimals
        Calculation takes a lot of time.
    """
    assert n > 0, "n must be positive integer"
    _, distances = get_neighbors(cell=cell,
                                 idx=idx,
                                 distance_cutoff=max_distance,
                                 get_distances=True)
    try:
        nth_distance = np.sort(np.unique(np.round(distances, decimals=1)))[n-1]
    except IndexError:
        raise RuntimeError("Please use greater max_distance")
    return nth_distance


def common_neighbor_analysis(cell:tuple) -> list:
    """
    Common neighbor analysis.

    Args:
        cell (tuple): cell
        cutoff (float): bonding cutoff

    Returns:
        list: states

    Todo:
        results may become different depends on the values of
        distance_cutoff and decimals
    """
    def _get_label(bondings):
        state = 'unknown'
        if len(bondings) == 12:
            status_1 = len([ bond for bond in bondings if bond == [1,4,2,1] ])
            status_2 = len([ bond for bond in bondings if bond == [1,4,2,2] ])
            if status_1 == 6 and status_2 == 6:
                state = 'hcp'
            elif status_1 == 12 and status_2 == 0:
                state = 'fcc'
        if len(bondings) == 14:
            status_1 = len([ bond for bond in bondings if bond == [1,4,4,1] ])
            status_2 = len([ bond for bond in bondings if bond == [1,6,6,1] ])
            if status_1 == 6 and status_2 == 8:
                state = 'bcc'
        return state

    def _get_bonding_state(cell, idx, n):
        max_distance = 3.
        flag = True
        while flag:
            try:
                cutoff = get_nth_neighbor_distance(cell=cell,
                                                   idx=idx,
                                                   n=n,
                                                   max_distance=max_distance)
                flag = False
            except IndexError:
                max_distance += 0.5
        distance_cutoff = cutoff + 0.5
        j_atoms = get_neighbors(cell=cell,
                                idx=idx,
                                distance_cutoff=distance_cutoff,
                                get_distances=False)
        i_common_neighbors = []
        for j_atom in j_atoms:
            j, a, b, c = j_atom
            prim_j_atom_neighbors = \
                    get_neighbors(cell=cell,
                                  idx=j,
                                  distance_cutoff=distance_cutoff,
                                  get_distances=False)
            j_atom_neighbors = \
                list(map(tuple,
                         np.array(prim_j_atom_neighbors)
                             + np.array([0,a,b,c])))
            k_atoms = set(j_atoms) & set(j_atom_neighbors)
            k_total_bonds = 0
            bond_nums = [0]
            for k_atom in k_atoms:
                k, a, b, c = k_atom
                prim_k_atom_neighbors = \
                        get_neighbors(cell=cell,
                                      idx=k,
                                      distance_cutoff=distance_cutoff,
                                      get_distances=False)
                k_atom_neighbors = list(map(tuple,
                                            np.array(prim_k_atom_neighbors)
                                                + np.array([0,a,b,c])))
                num = len(set(k_atoms) & set(k_atom_neighbors))
                k_total_bonds += num
                bond_nums.append(num)
            k_total_bonds = int(k_total_bonds / 2)
            max_bond_nums = max(bond_nums)
            if len(k_atoms) == 4 and k_total_bonds == 2:
                fourth_label = max_bond_nums
            else:
                fourth_label = 1
            i_common_neighbors.append(
                    [1, len(k_atoms), k_total_bonds, fourth_label])
        state = _get_label(i_common_neighbors)
        return state

    atoms_num = len(cell[1])
    states = []
    for i in range(atoms_num):
        state = _get_bonding_state(cell, i, 1)
        if state == 'unknown':
            state = _get_bonding_state(cell, i, 2)
        states.append(state)

    return states
