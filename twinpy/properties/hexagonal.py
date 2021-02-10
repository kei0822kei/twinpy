#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module deals with hexagonal property.
"""

import numpy as np
import spglib
from twinpy.structure.lattice import CrystalLattice


def check_hexagonal_lattice(lattice:np.array):
    """
    Check input lattice is hexagonal lattice.

    Args:
        lattice: Lattice matrix.

    Raises:
        AssertionError: The angles are not (90, 90, 120).

    Note:
        Check the angles of input lattice are (90, 90, 120).
    """
    hexagonal = CrystalLattice(lattice)
    expected = np.array([90., 90., 120.])
    actual = hexagonal.angles
    err_msg = "The angles of lattice was {}, \
               which does not match [90., 90., 120.]".format(actual)
    np.testing.assert_allclose(
            actual,
            expected,
            err_msg=err_msg,
            )


def check_cell_is_hcp(cell:tuple):
    """
    Check input cell is hexagonal close-packed.

    Args:
        cell: (lattice, scaled_positions, symbols).

    Raises:
        RuntimeError: Both positions and scaled_positions are specified.
        RuntimeError: Input symbol is not unique.
        RuntimeError: The number of symbols are not two,
                      which is the number of atoms
                      for hexagonal close-packed structure.
        RuntimeError: Space group of input cell is not 'P6_3/mmc'.
        AssertionError: Input cell is not
                        hexagonal close-packed.

    Note:
        The spglib software is used to get space group.
    """
    lattice, scaled_positions, symbols = cell

    if len(set(symbols)) != 1:
        raise RuntimeError("Symbol is not unique.")
    if len(symbols) != 2:
        raise RuntimeError("The number of symbols are not two.")

    dataset = spglib.get_symmetry_dataset((lattice,
                                           scaled_positions,
                                           [0, 0]))

    spg_symbol = dataset['international']
    if spg_symbol != 'P6_3/mmc':
        raise RuntimeError("Space group of input structure is {} "
                           "not 'P6_3/mmc'.".format(spg_symbol))

    wyckoffs = dataset['wyckoffs']
    if wyckoffs not in [['c'] * 2, ['d'] * 2]:
        raise RuntimeError("wyckoff letters are {}, which must be "
                           "['c', 'c'] or ['d', 'd']".format(wyckoffs))


def get_wyckoff_from_hcp(cell) -> str:
    """
    Get wyckoff letter from HCP crystal structure.

    Args:
        cell: (lattice, scaled_positions, symbols).

    Returns:
        str: Wyckoff letter.
    """
    check_cell_is_hcp(cell=cell)
    lattice, scaled_positions, _ = cell
    dataset = spglib.get_symmetry_dataset((lattice,
                                           scaled_positions,
                                           [0, 0]))
    wyckoff = dataset['wyckoffs'][0]

    return wyckoff


def get_hexagonal_lattice_from_a_c(a:float, c:float) -> np.array:
    """
    Get hexagonal lattice from the norms of a and c axes.

    Args:
        a: The norm of a axis.
        c: The norm of c axis.

    Returns:
        np.array: Hexagonal lattice with its length of basis vectors
                  are [a, a, c].

    Raises:
        ValueError: Either a or c is negative value.
    """
    if not a > 0. and c > 0.:
        err_msg = "input value was a={} and c={} " \
                  "but both a and c must be positive".format(a, c)
        raise ValueError(err_msg)

    lattice = np.array([[  1.,           0., 0.],
                        [-0.5, np.sqrt(3)/2, 0.],
                        [  0.,           0., 1.]]) * np.array([a,a,c])
    return lattice


def get_hcp_atom_positions(wyckoff:str) -> np.array:
    """
    Get atom positions in Hexagonal Close-Packed.

    Args:
        wyckoff: Wyckoff letter, choose 'c' or 'd'.

    Raises:
        AssertionError: Input wyckoff is neither 'c' nor 'd'.

    Returns:
        np.array: Atom positions with fractional coordinate.

    Note:
        origin point is  plane in the middle of hcp primitive atom pairs.
        The atom positions of wyckoff 'c', for example, are
        [[ 1/3 -1/3  1/4 ], [ -1/3 1/3 -1/4 ]] not
        [[ 1/3  2/3  1/4 ], [  2/3 1/3  3/4 ]].
    """
    assert wyckoff in ['c', 'd'], \
             "Input wyckoff must be 'c' or 'd' (input:{})".format(wyckoff)

    if wyckoff == 'c':
        atom_positions = \
            np.array([[ 1/3, -1/3,  1/4],
                      [-1/3,  1/3, -1/4]])  # No.194, wyckoff 'c'
    else:
        atom_positions = \
            np.array([[ 1/3, -1/3, -1/4],
                      [-1/3,  1/3,  1/4]])  # No.194, wyckoff 'd'

    return atom_positions


def get_hcp_cell(a:float,
                 c:float,
                 symbol:str,
                 wyckoff:str='c') -> tuple:
    """
    Get hexagonal close-packed cell.

    Args:
        a: The norm of a axis.
        c: The norm of c axis.
        symbol: Element symbol.
        wyckoff: Wyckoff letter.

    Note:
        Input wyckoff must be 'c' or 'd'.

    Returns:
        tuple: Hexagonal close-packed cell.
    """
    lattice = get_hexagonal_lattice_from_a_c(a=a, c=c)
    scaled_positions = get_hcp_atom_positions(wyckoff=wyckoff)
    symbols = [symbol] * len(scaled_positions)
    return (lattice, scaled_positions, symbols)


def convert_direction_from_four_to_three(four:Union[list,
                                                    np.array,
                                                    tuple]) -> np.array:
    """
    Convert direction from four to three.

    Args:
        four: Four indices of hexagonal direction [uvtw].

    Raises:
        AssertionError: The number of elements of input list
                       'four' is not four.
        AssertionError: Let be four=[u,v,t,w], u + v + t is not zero.

    Returns:
        np.array: Direction with three indices.
    """
    assert len(four) == 4, "The number of elements of input list is not four."
    u, v, t, w = four
    assert (u+v+t) == 0, "Input elements u+v+t is not equal to 0."
    U = u - t
    V = v - t
    W = w
    return np.array([U, V, W])


def convert_direction_from_three_to_four(three:Union[list,
                                                     np.array,
                                                     tuple]) -> np.array:
    """
    Convert direction from three to four.

    Args:
        three: Three indices of hexagonal direction [UVW].

    Raises:
        AssertionError: The number of element of input list 'three'
                        is not three.

    Returns:
        np.array: Direction with four indices.
    """
    assert len(three) == 3, "The number of elements of input list \
                             is not three."
    U, V, W = three
    u = ( 2 * U - V ) / 3
    v = ( 2 * V - U ) / 3
    t = - ( u + v )
    w = W
    return np.array([u, v, t, w])


class HexagonalDirection(CrystalLattice):
    """
    This class deals with hexagonal direction.
    """

    def __init__(
           self,
           lattice:np.array,
           three:Union[list,np.array,tuple]=None,
           four:Union[list,np.array,tuple]=None,
           ):
        """
        Setup.

        Args:
            lattice: Lattice matrix.
            three: Direction indice (three).
            four: Direction indice (four).
        """
        super().__init__(lattice=lattice)
        self._three = three
        self._four = four
        self.reset_indices(self._three, self._four)

    @property
    def three(self):
        """
        Hexagonal direction with three direction indices.
        """
        return self._three

    @property
    def four(self):
        """
        Hexagonal direction with four direction indices.
        """
        return self._four

    def reset_indices(self,
                      three:Union[list,np.array,tuple]=None,
                      four:Union[list,np.array,tuple]=None):
        """
        The indices for hexagonal direction are reset.

        Args:
            three: Direction with three indices.
            four: Direction with four indices.

        Raises:
            RuntimeError: Both 'three' and 'four' are None or
                          both 'three' and 'four' are not None.
        """
        if three is not None:
            if four is None:
                _three = three
                _four = convert_direction_from_three_to_four(three)
            else:
                raise RuntimeError("Both 'three' and 'four' are not None.")
        else:
            if four is not None:
                _three = convert_direction_from_four_to_three(four)
                _four = four
            else:
                raise RuntimeError("Both 'three' and 'four' are None.")

        self._three = np.array(_three)
        self._four = np.array(_four)

    def inverse(self):
        """
        Set inversed plane. In the case direction is [10-12],
        indices are reset from [10-12] to [-101-2].
        """
        self.reset_indices(three=self.three*(-1))

    def get_cartesian(self, normalize:bool=False) -> np.array:
        """
        Get direction with the cartesian coordinate.

        Args:
            normalize: If True, the norm of the vector is normalized to one.

        Returns:
            np.array: Direction with the cartesian coordinate.
        """
        cart_coords = np.dot(self.lattice.T,
                             self.three.reshape([3,1])).reshape(3)
        if normalize:
            cart_coords /= np.linalg.norm(cart_coords)

        return cart_coords


def convert_plane_from_four_to_three(four:Union[list,
                                                np.array,
                                                tuple]) -> np.array:
    """
    Convert plane from four to three.

    Args:
        four: Four indices of hexagonal plane.

    Raises:
        AssertionError: len(four) != 4
        AssertionError: (h+k+i) != 0

    Returns:
        tuple: Plane three indices.
    """
    assert len(four) == 4, "The number of elements of input list is not four."
    h, k, i, l = four
    assert (h+k+i) == 0, "h+k+i is not equal to 0."
    H = h
    K = k
    L = l
    return (H, K, L)


def convert_plane_from_three_to_four(three:Union[list,
                                                 np.array,
                                                 tuple]) -> np.array:
    """
    Convert plane from three to four.

    Args:
        three: Three indices of hexagonal plane.

    Raises:
        AssertionError: len(four) != 3

    Returns:
        np.array: Direction with three indices.
    """
    assert len(three) == 3, "The length of input list is not three."
    h, k, l = three
    i = - ( h + k )
    return np.array([h, k, i, l])


class HexagonalPlane(CrystalLattice):
    """
    Deals with hexagonal plane.
    """

    def __init__(
           self,
           lattice:np.array,
           three:Union[list,np.array,tuple]=None,
           four:Union[list,np.array,tuple]=None,
           ):
        """
        Setup.

        Args:
            lattice: Lattice.
            three: Plane indice (three).
            four: Plane indice (four).
        """
        super().__init__(lattice=lattice)
        if three is None:
            self._three = None
        else:
            self._three = np.array(three)
        if four is None:
            self._four = None
        else:
            self._four = np.array(four)
        self.reset_indices(three=self._three, four=self._four)

    @property
    def three(self):
        """
        Plane indice (HKL).
        """
        return self._three

    @property
    def four(self):
        """
        Plane indice (hkil).
        """
        return self._four

    def reset_indices(self,
                      three:Union[list,np.array,tuple]=None,
                      four:Union[list,np.array,tuple]=None):
        """
        The indices for hexagonal plane are reset.

        Args:
            three: Plane with three indices.
            four: Plane with four indices.

        Raises:
            RuntimeError: Both 'three' and 'four' are None or
                          both 'three' and 'four' are not None.
        """
        if three is not None:
            if four is None:
                _three = three
                _four = convert_plane_from_three_to_four(three)
            else:
                raise RuntimeError("Both 'three' and 'four' are not None.")
        else:
            if four is not None:
                _three = convert_plane_from_four_to_three(four)
                _four = four
            else:
                raise RuntimeError("Both 'three' and 'four' are None.")

        self._three = np.array(_three)
        self._four = np.array(_four)

    def inverse(self):
        """
        Set inversed plane ex. (10-12) => (-101-2).
        """
        self.reset_indices(three=self.three*(-1))

    def get_direction_normal_to_plane(self) -> HexagonalDirection:
        """
        Get direction normal to input plane.

        Returns:
            HexagonalDirection: Hexagonal direction object
                                which is normal to plane.
        """
        tf_matrix = self.lattice.T
        res_tf_matrix = self.reciprocal_lattice.T
        direction = np.dot(np.linalg.inv(tf_matrix),
                           np.dot(res_tf_matrix,
                                  self._three.reshape([3,1]))).reshape(3)
        hex_dr = HexagonalDirection(lattice=self.lattice,
                                    three=direction)
        return hex_dr

    def get_distance_from_plane(self, frac_coord:np.array) -> float:
        """
        Get dicstance from plane which contains origin point
        to input fractional coordinate.

        Args:
            frac_coord: Fractional coorinate.

        Returns:
            float: Distance from plane.
        """
        frac_coord = frac_coord.reshape(1,3)
        k = self.get_direction_normal_to_plane()
        k_cart = np.dot(self.lattice.T, k.three.reshape(1,3).T).reshape(3)
        e_k_cart = k_cart / np.linalg.norm(k_cart)
        x_cart = np.dot(self.lattice.T, frac_coord.T).reshape(3)
        d = np.dot(e_k_cart, x_cart)
        return d
