#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make structures
"""

from copy import deepcopy
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import numpy as np
from twinpy.structure.lattice import CrystalLattice


def write_poscar(
        cell:tuple,
        filename:str='POSCAR'):
    """
    Write out structure to file.
    In this function, structure is not fixed
    even if its lattice basis is left handed.

    Args:
        cell: (lattice, scaled_positions, symbols).
        filename: Poscar filename.
    """
    lattice, scaled_positions, symbols = cell
    symbol_sets = list(set(symbols))
    nums = []
    idx = []
    for symbol in symbol_sets:
        index = [ i for i, s in enumerate(symbols) if s == symbol ]
        nums.append(str(len(index)))
        idx.extend(index)
    positions = np.round(np.array(scaled_positions)[idx, :],
                         decimals=9).astype(str)

    # create strings
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


def write_yaml(dic:dict, filename:str):
    """
    Write yaml file from dic.

    Args:
        dict: Dictionary object.
        filename: Output file name.
    """
    with open(filename, 'w') as f:
        yaml.dump(dic, f, indent=4, default_flow_style=False, Dumper=Dumper)


def read_yaml(filename:str) -> dict:
    """
    Return dic from yaml.

    Args:
        filename: Output file name.

    Returns:
        dict: Dictionary object.
    """
    with open(filename, 'r') as f:
        dic = yaml.load(f, Loader=Loader)
    return dic


def write_thermal_ellipsoid(cell:tuple,
                            matrices:np.array,
                            temperatures:list,
                            filetype:str='CrystalMaker',
                            header:str=''):
    """
    Write thermal ellipsoid.

    Args:
        cell: (lattice, scaled_positions, symbols).
        matrices: Thermal ellipsoid.
        temperatures: Temperature list.
        filetype: Currently only 'CrystalMaker' is supported.
        header: Header of filename.
    """
    if filetype != 'CrystalMaker':
        raise ValueError("Only filetype='CrystalMaker' is supported.")
    lines = []
    lattice = CrystalLattice(cell[0])
    abc = lattice.abc
    abc_str = ' '.join(map(str, list(abc)))
    angles = lattice.angles
    angles_str = ' '.join(map(str, list(angles)))
    cell_str = "CELL {} {}".format(abc_str, angles_str)
    lines.append(cell_str)
    lines.append("")
    lines.append("ATOM")

    # crystal maker => xx yy zz xy xz yz
    tensor_idx = [[0,0], [1,1], [2,2], [0,1], [0,2], [1,2]]

    for i in range(len(cell[2])):
        frac_str = ' '.join(map(str, list(cell[1][i])))
        lines.append("{} {} {}".format(cell[2][i],
                                       cell[2][i]+str(i+1),
                                       frac_str))
    lines.append("")
    lines.append("UANI")
    for i, temp in enumerate(temperatures):
        temp_lines = deepcopy(lines)
        for j in range(len(cell[2])):
            mat = np.round(matrices[i,j], decimals=4)
            tensor = [ str(mat.item(*idx)) for idx in tensor_idx ]
            tensor_str = ' '.join(tensor)
            temp_lines.append("{} {}".format(cell[2][j]+str(1+j), tensor_str))

        strings = '\n'.join(temp_lines)
        filename = header + "temp{}K.cmtx".format(temp)
        with open(filename, 'w') as f:
            f.write(strings)
