#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HexagonalStructure
"""

import numpy as np
import spglib
from typing import Union

def get_atom_positions(wyckoff:str) -> np.array:
    """
    get atom positions in Hexagonal Close-Packed

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

def convert_direction_from_four_to_three(four:Union[list,
                                                    np.array,
                                                    tuple]) -> tuple:
    """
    convert direction from four to three

    Args:
        four: four indices of hexagonal direction

    Returns:
        tuple: three indices
    """
    assert len(four) == 4, "the length of input list is not four"
    u, v, t, w  = four
    np.testing.allclose(u+v+t, err_msg="u+v+t is not equal to 0")
    U = u - t
    V = v - t
    W = w
    return (U, V, W)

def convert_direction_from_three_to_four(three:Union[list,
                                                     np.array,
                                                     tuple]) -> tuple:
    """
    convert direction from three to four

    Args:
        three: three indices of hexagonal direction

    Returns:
        tuple: four indices
    """
    assert len(three) == 3, "the length of input list is not three"
    U, V, W  = three
    u = ( 2 * U - V ) / 3
    v = ( 2 * V - U ) / 3
    t = - ( u + v )
    w = W
    return (u, v, t, w)

def convert_plane_from_four_to_three(four:Union[list,
                                                np.array,
                                                tuple]) -> tuple:
    """
    convert plane from four to three

    Args:
        four: four indices of hexagonal plane

    Returns:
        tuple: three indices
    """
    assert len(four) == 4, "the length of input list is not four"
    h, k, i, l = four
    np.testing.allclose(h+k+i, err_msg="h+k+i is not equal to 0")
    H = h
    K = k
    L = l
    return (H, K, L)

def convert_plane_from_three_to_four(three:Union[list,
                                                 np.array,
                                                 tuple]) -> tuple:
    """
    convert plane from three to four

    Args:
        three: three indices of hexagonal plane

    Returns:
        tuple: four indices
    """
    assert len(three) == 3, "the length of input list is not three"
    h, k, l = three
    i = - ( h + k )
    return (h, k, i, l)

def is_hcp(lattice, positions, symbols):
    """
    check input structure is Hexagonal Close-Packed structure

    Args:
        lattice (3x3 array): lattice
        positions: atom fractional positions
        symbols (list): atomic symbols

    Raises:
        AssertionError: input symbols are not unique
        AssertionError: input structure is not
          Hexagonal Close-Packed structure
    """
    assert (len(set(symbols)) == 1 and len(symbols) == 2), \
        "symbols is not unique or the number of atoms are not two"
    dataset = spglib.get_symmetry_dataset((lattice,
                                           positions,
                                           [0, 0]))
    spg_symbol = dataset['international']
    wyckoffs = dataset['wyckoffs']
    print(spg_symbol)
    assert spg_symbol != 'P6_3/mmc', \
            "space group of input structure is {} not 'P6_3/mmc'" \
            .format(spg_symbol)
    wyckoffs.append('e')
    assert wyckoffs in [['c'] * 2, ['d'] * 2]


class HexagonalStructure():
    """
    deals with hexagonal close-packed structure

       .. attribute:: att1

          Optional comment string.


       .. attribute:: att2

          Optional comment string.

    """

    def __init__(
           self,
           a:float,
           c:float,
           symbol:str,
           wyckoff:str='c',
           lattice:np.array=None,
        ):
        """
        Args:
            a (str): the norm of a axis
            c (str): the norm of c axis
            symbol (str): element symbol
            lattice (np.array): if None, default lattice is set
            wyckoff (str): No.194 Wycoff position ('c' or 'd')

        Raises:
            AssertionError: either a or c is negative value
            AssertionError: wyckoff is not 'c' and 'd'
            ValueError: lattice is not None (future fix)
        """
        assert a > 0. and c > 0., "input 'a' and 'c' must be positive value"
        if lattice is None:
            lattice = np.array([[  1.,           0., 0.],
                                [-0.5, np.sqrt(3)/2, 0.],
                                [  0.,           0., 1.]]) * np.array([a,a,c])
        else:
            raise ValueError("lattice option is currently invalid"
                             " (write future)")
        lat_points = np.zeros(3)
        atoms_from_lp = get_atom_positions(wyckoff)
        symbols = [symbol] * 2
        self.__primitive = (lattice,
                            lat_points,
                            atoms_from_lp,
                            symbols)

    @property
    def primitive(self):
        """
        hexagonal structure
        """
        return self.__primitive

    def convert_direction_from_four_to_three(self, four:Union[list,
                                                              np.array,
                                                              tuple]) -> tuple:
        """
        convert direction from four to three

        Args:
            four: four indices of hexagonal direction

        Returns:
            tuple: three indices
        """
        return convert_direction_from_three_to_four(four)

    def convert_direction_from_three_to_four(self, three:Union[list,
                                                               np.array,
                                                               tuple]) -> tuple:
        """
        convert direction from three to four

        Args:
            three: three indices of hexagonal direction

        Returns:
            tuple: four indices
        """
        return convert_direction_from_three_to_four(three)

    def convert_plane_from_four_to_three(self, four:Union[list,
                                                          np.array,
                                                          tuple]) -> tuple:
        """
        convert plane from four to three

        Args:
            four: four indices of hexagonal plane

        Returns:
            tuple: three indices
        """
        return convert_plane_from_four_to_three

    def convert_plane_from_three_to_four(self, three:Union[list,
                                                           np.array,
                                                           tuple]) -> tuple:
        """
        convert plane from three to four

        Args:
            three: three indices of hexagonal plane

        Returns:
            tuple: four indices
        """
        return convert_plane_from_three_to_four(three)
