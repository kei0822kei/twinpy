#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lattice class
"""

import numpy as np


class Lattice():
    """
    Deals with lattice.
    """

    def __init__(self, lattice:np.array):
        """
        Args:
            lattice (np.array): lattice

        Raises:
            AssertionError: input lattice is not 3x3 numpy array
        """
        assert lattice.shape == (3, 3), "invalid lattice (not 3x3 numpy array)"

        self._lattice = lattice

        self._volume = None
        self._set_volume()

        self._reciprocal_lattice = None
        self._set_reciprocal_lattice()

        self._abc = None
        self._set_abc()

        self._angles = None
        self._cos_angles = None
        self._set_angles()

        self._metric = None
        self._set_metric()

    @property
    def lattice(self):
        """
        Lattice matrix.
        """
        return self._lattice

    def _set_volume(self):
        """
        Set volume.
        """
        self._volume = np.dot(np.cross(self._lattice[0],
                                       self._lattice[1]),
                              self._lattice[2])

    @property
    def volume(self):
        """
        Volume of lattice.
        """
        return self._volume

    def _set_reciprocal_lattice(self):
        """
        Set reciprocal lattice WITHOUT 2*pi.
        """
        recip_bases = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            recip_bases.append(
                np.cross(self._lattice[j], self._lattice[k]) / self._volume)
        self._reciprocal_lattice = np.array(recip_bases)

    @property
    def reciprocal_lattice(self):
        """
        Reciprocal lattice WITHOUT 2*pi.
        """
        return self._reciprocal_lattice

    def _set_abc(self):
        """
        Set norm of each axis.
        """
        self._abc = np.linalg.norm(self._lattice, axis=1)

    @property
    def abc(self):
        """
        Norm of each axis.
        """
        return self._abc

    def _set_angles(self):
        """
        Set angles and cosine angles.
        """
        lat = self.lattice
        abc = self.abc
        cos_angles = []
        angles = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            cos_angle = np.dot(lat[j],lat[k]) / (abc[j] * abc[k])
            cos_angles.append(cos_angle)
            angles.append(np.arccos((cos_angle)) * 180 / np.pi)
        self._cos_angles = tuple(cos_angles)
        self._angles = tuple(angles)

    @property
    def angles(self):
        """
        Angles of axes.
        """
        return self._angles

    @property
    def cos_angles(self):
        """
        Cosine angles of axes.
        """
        return self._cos_angles

    def _set_metric(self, is_old_style=False):
        """
        Set metric tensor.
        """
        if is_old_style:
            self._set_metric_legacy()
        else:
            metric = np.dot(self._lattice,
                            self._lattice.T)
            self._metric = metric

    def _set_metric_legacy(self):
        """
        FUTURE REMOVED.
        """
        metric = np.zeros([3,3])
        for i in range(3):
            for j in range(3):
                if i == j:
                    metric[i,j] = self._abc[i]**2
                else:
                    k = 3 - (i + j)
                    metric[i,j] = self._abc[i] \
                        * self._abc[j] \
                        * self._cos_angles[k]
        self._metric = metric

    @property
    def metric(self):
        """
        metric tensor
        """
        return self._metric

    def dot(self,
            frac_coord_first:np.array,
            frac_coord_second:np.array) -> float:
        """
        Return inner product of two vectors.
        Two vectors must be given as fractional coordinate.

        Args:
            frac_coord_first (np.array): frac coord
            frac_coord_second (np.array): frac coord

        Returns:
            float: inner product
        """
        first = frac_coord_first.reshape(3,1)
        second = frac_coord_second.reshape(3,1)
        return float(np.dot(np.dot(first.T,
                                   self._metric),
                            second))

    def is_hexagonal_lattice(self) -> bool:
        """
        Check that lattice is hexagonal.

        Returns:
            bool: If True, self.lattice is hexagonal lattice.

        Note:
            Run check_hexagonal_lattice in this definition.
        """
        try:
            check_hexagonal_lattice(self._lattice)
            flag = True
        except AssertionError:
            flag = False
        return flag


def check_hexagonal_lattice(lattice:np.array):
    """
    Check input lattice is hexagonal lattice.

    Args:
        lattice (np.array): lattice

    Raises:
        AssertionError: the angles are not (90, 90, 120)

    Note:
        Check the angles of input lattice are (90, 90, 120).

    TODO:
        Add tolerance of assert_allclose.
    """
    hexagonal = Lattice(lattice)
    expected = np.array([90., 90., 120.])
    actual = hexagonal.angles
    np.testing.assert_allclose(
            actual,
            expected,
            err_msg="angles of lattice was {}, which is not hexagonal".
                    format(actual),
            )


def create_hexagonal_lattice(a:float, c:float) -> np.array:
    """
    Create hexagonal lattice.

    Args:
        a (float): norm of a axis
        c (float): norm of c axis

    Returns:
        np.array: hexagonal lattice
    """
    lattice = np.array([[  1.,           0., 0.],
                        [-0.5, np.sqrt(3)/2, 0.],
                        [  0.,           0., 1.]]) \
              * np.array([[a,a,c]]).T
    return lattice


def get_hexagonal_lattice_from_a_c(a:float, c:float) -> np.array:
    """
    Get hexagonal lattice from the norms of a and c axes.

    Args:
        a (str): the norm of a axis
        c (str): the norm of c axis

    Returns:
        np.array: hexagonal lattice

    Raises:
        AssertionError: either a or c is negative value
    """
    assert a > 0. and c > 0., "input 'a' and 'c' must be positive value"
    lattice = np.array([[  1.,           0., 0.],
                        [-0.5, np.sqrt(3)/2, 0.],
                        [  0.,           0., 1.]]) * np.array([a,a,c])
    return lattice
