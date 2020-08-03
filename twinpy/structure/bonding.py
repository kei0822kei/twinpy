#!/usr/bin/env pythoo
# -*- coding: utf-8 -*-

"""
Bonding base.
"""

import numpy as np
import itertools
from twinpy.lattice.lattice import Lattice


def get_neighbor(cell:tuple,
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

    neighbors = list(map(tuple, neighbors))

    if get_distances:
        return (neighbors, distances)
    else:
        return neighbors


def common_neighbor_analysis(cell:tuple,
                             cutoff:float):
    """
    Common neighbor analysis.

    Args:
        cell (tuple): cell
        cutoff (float): bonding cutoff

    Todo:
        cutoff is not unique, which means it is needed to use
        different cutoff distance when, for exaple, checking hcp status and bcc
        status.
    """
    def _get_bonding_state(bondings):
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

    atoms_num = len(cell[1])
    all_neighbors = []
    all_distances = []
    for i in range(atoms_num):
        neighbors, distances = get_neighbor(cell=cell,
                                            idx=i,
                                            distance_cutoff=cutoff,
                                            get_distances=True)
        all_neighbors.append(neighbors)
        all_distances.append(distances)

    states = []
    for i in range(atoms_num):
        j_atoms = all_neighbors[i]
        i_common_neighbors = []
        for j_atom in j_atoms:
            j, a, b, c = j_atom
            j_atom_neighbors = list(map(tuple,
                np.array(all_neighbors[j]) + np.array([0,a,b,c])))
            k_atoms = set(j_atoms) & set(j_atom_neighbors)
            k_total_bonds = 0
            bond_nums = [0]
            for k_atom in k_atoms:
                k, a, b, c = k_atom
                k_atom_neighbors = list(map(tuple,
                    np.array(all_neighbors[k]) + np.array([0,a,b,c])))
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
        print(len(i_common_neighbors))
        state = _get_bonding_state(i_common_neighbors)
        states.append(state)

    return states

    # common_neighbors = np.zeros([atoms_num, atoms_num, 4]).astype(int)
    # idx_list = [ i for i in range(atoms_num) ]
    # for i, j in itertools.product(idx_list, idx_list):
    #     print(i,j)
    #     if j not in all_neighbors[i][:,0]:
    #         common_neighbors[i,j] = [2,0,0,0]
    #     else:
    #         share_idx_set = set(map(tuple, all_neighbors[i])) \
    #                             and set(map(tuple, all_neighbors[j]))
    #         bond_num = 0
    #         bond_shares = []
    #         print(share_idx_set)
    #         for idx_set in share_idx_set:
    #             fixed_neighbors = all_neighbors[idx_set[0]]
    #             _, a, b, c = idx_set
    #             fixed_neighbors = np.array(fixed_neighbors) \
    #                     + np.array([0, a, b, c])
    #             fixed_neighbors_set = set(map(tuple, fixed_neighbors))
    #             num = len(share_idx_set & fixed_neighbors_set)
    #             bond_num += num
    #             bond_shares.append(num)
    #         bn_among_cn = int(bond_num / 2)

    #         if len(share_idx_set) == 4 and bn_among_cn == 2:
    #             max_bond = max(bond_shares)
    #             common_neighbors[i,j] = \
    #                     [1, len(share_idx_set), bn_among_cn, max_bond]
    #         else:
    #             common_neighbors[i,j] = \
    #                     [1, len(share_idx_set), bn_among_cn, 1]
    # return common_neighbors
