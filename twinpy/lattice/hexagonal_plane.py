#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
hexagonal plane
"""

import numpy as np
from typing import Union
from .lattice import Lattice
from .hexagonal_direction import HexagonalDirection

def convert_plane_from_four_to_three(four:Union[list,
                                                np.array,
                                                tuple]) -> np.array:
    """
    convert plane from four to three

    Args:
        four: four indices of hexagonal plane

    Returns:
        np.array: three indices
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
    convert plane from three to four

    Args:
        three: three indices of hexagonal plane

    Returns:
        np.array: four indices
    """
    assert len(three) == 3, "the length of input list is not three"
    h, k, l = three
    i = - ( h + k )
    return (h, k, i, l)


class HexagonalPlane(Lattice):
    """
    deals with hexagonal plane
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
        plane indice (HKL)
        """
        return self._three

    @property
    def four(self):
        """
        plane indice (hkil)
        """
        return self._four

    def reset_indices(self,
                      three:Union[list,np.array,tuple]=None,
                      four:Union[list,np.array,tuple]=None):
        """
        reset indices

        Args:
            three: plane indice (three)
            four: plane indice (four)

        Raises:
            ValueError: both 'three' and 'four' are None or
                        both 'three' and 'four' are not None
        """
        if three is None and four is None:
            raise ValueError("both 'three' and 'four' are None")
        elif three is not None:
            four = convert_plane_from_three_to_four(three)
        elif four is not None:
            three = convert_plane_from_four_to_three(four)
        else:
            raise ValueError("both 'three' and 'four' are not None")

        self._three = np.array(three)
        self._four = np.array(four)

    def inverse(self):
        """
        set inversed plane ex. (10-12) => (-101-2)
        """
        self.reset_indices(three=self.three*(-1))

    def get_direction_normal_to_plane(self) -> HexagonalDirection:
        """
        get direction normal to input plane

        Returns:
            HexagonalDirection: direction normal to plane
        """
        tf_matrix = self.lattice.T
        res_tf_matrix = self.reciprocal_lattice.T
        direction = np.dot(np.linalg.inv(tf_matrix),
                np.dot(res_tf_matrix, self.three.reshape([3,1]))).reshape(3)
        return HexagonalDirection(lattice=self.lattice,
                                  three=direction)

    def get_distance_from_plane(self, frac_coord) -> np.array:
        """
        get dicstance from plane

        Args:
            frac_coord (np.array): fractional coorinate

        Returns:
            float: distance
        """
        frac_coord = frac_coord.reshape(1,3)
        k = self.get_direction_normal_to_plane()
        k_cart = np.dot(self.lattice.T, k.three.reshape(1,3).T).T
        d = abs(np.dot(k_cart / np.linalg.norm(k_cart),
                       np.dot(self.lattice.T, frac_coord.T))[0,0])
        return d

    def get_cartesian(self, frac_coord) -> np.array:
        """
        get cartesian coordinate of the input frac_coord

        Args:
            frac_coord (np.array): fractional coorinate

        Returns:
            np.array: cartesian coorinate
        """
        frac_coord = frac_coord.reshape(1,3)
        cart_coord = np.dot(self._lattice.T, frac_coord).reshape(3)
        return cart_coord




    # def get_equivalent_plane(self, unique=True, get_recp_operation=False):
    #     """
    #     get equivelent plane

    #     Args:
    #         unique (bool): if True,
    #           only get planes whose l > 0 where plane(hkil)
    #         get_operation (bool): whether get corresponding operations

    #     Returns:
    #         list: case get_operation=False, equivelent planes
    #         tuple: case get_operation=True, equivelent planes and
    #           corresponding operations

    #     Note:
    #         P6/mmm has 24 point group operations, but half of them
    #         becomes duplicated and they are automatically removed.

    #     Todo:
    #         If point group operation of reciprocal_lattice is used,
    #         error occurs because their rotations matrix is different
    #         from direct ones, but I do not understand deeply about this.

    #         And also error occurs if K1 indices is used,
    #         probably it is necessary to use indices of k1

    #         How to calculate get_recp_symmetry_operation ?
    #         How to transform recp symmetry operation to real space ?
    #     """
    #     operations = self.lattice.get_recp_symmetry_operation()
    #     threes = []
    #     planes_recp_operations = []
    #     for operation in operations:
    #         three = np.dot(
    #             operation.rotation_matrix, self.__three).T.astype(int).tolist()
    #         if unique:
    #             if three[2] < 0:
    #                 continue
    #         if three not in threes:
    #             threes.append(three)
    #             if get_recp_operation:
    #                 planes_recp_operations.append(operation)

    #     planes = []
    #     for equivelent in threes:
    #         plane = deepcopy(self)
    #         plane.reset_indices(three=equivelent)
    #         planes.append(plane)

    #     if get_recp_operation:
    #         return (planes, planes_recp_operations)
    #     else:
    #         return planes
