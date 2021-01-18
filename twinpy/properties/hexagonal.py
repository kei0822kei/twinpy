#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hexagonal property.
"""

import numpy as np
from typing import Union
from twinpy.structure.lattice import Lattice


def check_hexagonal_lattice(lattice:np.array):
    """
    Check input lattice is hexagonal lattice.

    Args:
        lattice (np.array): lattice

    Raises:
        AssertionError: The angles are not (90, 90, 120).

    Note:
        Check the angles of input lattice are (90, 90, 120).
    """
    hexagonal = Lattice(lattice)
    expected = np.array([90., 90., 120.])
    actual = hexagonal.angles
    err_msg = "The angles of lattice was {}, "
              "which does not match [90., 90., 120.]".format(actual)
    np.testing.assert_allclose(
            actual,
            expected,
            err_msg=err_msg,
                    format(actual),
            )


def get_hexagonal_lattice_from_a_c(a:float, c:float) -> np.array:
    """
    Get hexagonal lattice from the norms of a and c axes.

    Args:
        a (str): The norm of a axis.
        c (str): The norm of c axis.

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
        wyckoff (str): Wyckoff letter, choose 'c' or 'd'.

    Raises:
        AssertionError: Input wyckoff is neither 'c' nor 'd'.

    Returns:
        np.array: Atom positions with fractional coordinate.
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


def convert_direction_from_four_to_three(four:Union[list,
                                                    np.array,
                                                    tuple]) -> np.array:
    """
    Convert direction from four to three.

    Args:
        four: Four indices of hexagonal direction [uvtw].

    Raises:
        AssertionError: The number of elements of input list 'four' is not four.
        AssertionError: Let be four=[u,v,t,w], u + v + t is not zero.

    Returns:
        np.array: three indices
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
        AssertionError: The number of element of input list 'three' is not three.

    Returns:
        np.array: four indices
    """
    assert len(three) == 3, "The number of elements of input list is not three."
    U, V, W = three
    u = ( 2 * U - V ) / 3
    v = ( 2 * V - U ) / 3
    t = - ( u + v )
    w = W
    return np.array([u, v, t, w])


class HexagonalDirection(Lattice):
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
            lattice: lattice.
            three: direction indice (three).
            four: direction indice (four).
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
        The indices for hexagonal direction is reset.

        Args:
            three: Direction with three indices.
            four: Direction with four indices.

        Raises:
            RuntimeError: Both 'three' and 'four' are None or
                          both 'three' and 'four' are not None.
        """
        if three is None and four is None:
            raise RuntimeError("Both 'three' and 'four' are None.")
        elif three is not None:
            four = convert_direction_from_three_to_four(three)
        elif four is not None:
            three = convert_direction_from_four_to_three(four)
        else:
            raise RuntimeError("Both 'three' and 'four' are not None.")

        self._three = np.array(three)
        self._four = np.array(four)

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
            return cart_coords / np.linalg.norm(cart_coords)
        else:
            return cart_coords



def convert_plane_from_four_to_three(four:Union[list,
                                                np.array,
                                                tuple]) -> np.array:
    """
    Convert plane from four to three.

    Args:
        four: four indices of hexagonal plane

    Raises:
        AssertionError: len(four) != 4
        AssertionError: (h+k+i) != 0

    Returns:
        tuple: three indices
    """
    assert len(four) == 4, "the length of input list is not four"
    h, k, i, l = four
    assert (h+k+i) == 0, "h+k+i is not equal to 0"
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
        three: three indices of hexagonal plane

    Raises:
        AssertionError: len(four) != 3

    Returns:
        np.array: four indices
    """
    assert len(three) == 3, "the length of input list is not three"
    h, k, l = three
    i = - ( h + k )
    return (h, k, i, l)


class HexagonalPlane(Lattice):
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
        Args:
            lattice (np.array): lattice
            three: plane indice (three)
            four: plane indice (four)
        """
        super().__init__(lattice=lattice)
        self._three = three
        self._four = four
        self.reset_indices(self._three, self._four)

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
        Reset indices.

        Args:
            three: plane indice (three)
            four: plane indice (four)

        Raises:
            RuntimeError: both 'three' and 'four' are None or
                          both 'three' and 'four' are not None
        """
        if three is None and four is None:
            raise RuntimeError("both 'three' and 'four' are None")
        elif three is not None:
            four = convert_plane_from_three_to_four(three)
        elif four is not None:
            three = convert_plane_from_four_to_three(four)
        else:
            raise RuntimeError("both 'three' and 'four' are not None")

        self._three = np.array(three)
        self._four = np.array(four)

    def inverse(self):
        """
        Set inversed plane ex. (10-12) => (-101-2).
        """
        self.reset_indices(three=self.three*(-1))

    def get_direction_normal_to_plane(self) -> HexagonalDirection:
        """
        Get direction normal to input plane.

        Returns:
            HexagonalDirection: direction normal to plane
        """
        tf_matrix = self.lattice.T
        res_tf_matrix = self.reciprocal_lattice.T
        direction = np.dot(np.linalg.inv(tf_matrix),
                           np.dot(res_tf_matrix,
                                  self._three.reshape([3,1]))).reshape(3)
        return HexagonalDirection(lattice=self.lattice,
                                  three=direction)

    def get_distance_from_plane(self, frac_coord) -> np.array:
        """
        Get dicstance from plane.

        Args:
            frac_coord (np.array): fractional coorinate

        Returns:
            float: distance
        """
        frac_coord = frac_coord.reshape(1,3)
        k = self.get_direction_normal_to_plane()
        k_cart = np.dot(self.lattice.T, k.three.reshape(1,3).T).reshape(3)
        e_k_cart = k_cart / np.linalg.norm(k_cart)
        x_cart = np.dot(self.lattice.T, frac_coord.T).reshape(3)
        d = np.dot(e_k_cart, x_cart)
        return d

    def get_cartesian(self, frac_coord) -> np.array:
        """
        Get cartesian coordinate of the input frac_coord.

        Args:
            frac_coord (np.array): fractional coorinate

        Returns:
            np.array: cartesian coorinate
        """
        frac_coord = frac_coord.reshape(1,3)
        cart_coord = np.dot(self._lattice.T, frac_coord.T).reshape(3)
        return cart_coord
