#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
hexagonal direction
"""

import numpy as np
from .lattice import Lattice

def convert_direction_from_four_to_three(four:Union[list,
                                                    np.array,
                                                    tuple]) -> np.array:
    """
    convert direction from four to three

    Args:
        four: four indices of hexagonal direction [uvtw]

    Raises:
        AssertionError: len(four) != 4
        AssertionError: u + v + t != 0

    Returns:
        array: three indices
    """
    assert len(four) == 4, "the length of input list is not four"
    u, v, t, w  = four
    np.testing.allclose(u+v+t, err_msg="u+v+t is not equal to 0")
    U = u - t
    V = v - t
    W = w
    return np.array([U, V, W])

def convert_direction_from_three_to_four(three:Union[list,
                                                     np.array,
                                                     tuple]) -> array:
    """
    convert direction from three to four

    Args:
        three: three indices of hexagonal direction [UVW]

    Raises:
        AssertionError: len(four) != 3

    Returns:
        array: four indices
    """
    assert len(three) == 3, "the length of input list is not three"
    U, V, W  = three
    u = ( 2 * U - V ) / 3
    v = ( 2 * V - U ) / 3
    t = - ( u + v )
    w = W
    return np.array([u, v, t, w])


class HexagonalDirection():
    """
    deals with hexagonal direction

       .. attribute:: three

          direct indice (three)

       .. attribute:: four

          direct indice (four)
    """

    def __init__(
           self,
           lattice:Lattice,
           three:'list or np.array'=None,
           four:'list or np.array'=None,
       ):
        """
        Args:
            lattice (3x3 array): lattice
            three: direction indice (three)
            four: direction indice (four)
        """
        super().__init__(lattice=lattice)
        self._three = three
        self._four = four
        self.reset_indices(self._three, self._four)

    @property
    def three(self):
        return self._three

    @property
    def four(self):
        return self._four

    def reset_indices(self,
                      three:Union[list,np.array,tuple]=None,
                      four:Union[list,np.array,tuple]=None):
        """
        reset indices

        Args:
            three: direction indice (three)
            four: direction indice (four)

        Raises:
            ValueError: both 'three' and 'four' are None or
                        both 'three' and 'four' are not None
        """
        if three is None and four is None:
            raise ValueError("both 'three' and 'four' are None")
        elif three is not None:
            four = convert_direction_from_three_to_four(three)
        elif four is not None:
            three = convert_direction_from_four_to_three(four)
        else:
            raise ValueError("both 'three' and 'four' are not None")

        self._three = np.array(three)
        self._four = np.array(four)

    def get_cartesian(self, normalize:bool=False) -> np.array:
        """
        get direction with the cartesian coordinate

        Args:
            normalize: if True, normalized the norm of the vector to 1

        Returns:
            np.array: direction with the cartesian coordinate
        """
        cart_coords = np.dot(self.lattice.T,
                             self.three.reshape([3,1])).reshape(3)
        if normalize:
            return cart_coords / np.linalg.norm(cart_coords)
        else:
            return cart_coords
