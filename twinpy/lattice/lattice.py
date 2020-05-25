#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lattice class
"""

import numpy as np

def check_hexagonal_lattice(lattice):
    """
    check input lattice is hexagonal lattice

    Args:
        lattice (np.array): lattice

    Raises:
        AssertionError: the angles are not (90, 90, 120)
    """
    hexagonal = Lattice(lattice)
    expected = np.array([90., 90., 120.])
    actual = hexagonal.angles
    np.testing.assert_allclose(
            actual,
            expected,
            err_msg="angles of lattice was {}, which is not hexagonal". \
                    format(actual),
            )

def create_hexagonal_lattice(a:float, c:float):
    """
    create hexagonal lattice

    Args:
        a (float): norm of a axis
        c (float): norm of c axis

    Returns:
        Lattice: lattice object
    """
    lattice = np.array([[  1.,           0., 0.],
                        [-0.5, np.sqrt(3)/2, 0.],
                        [  0.,           0., 1.]]) \
                       * np.array([[a,a,c]]).T
    return lattice


class Lattice():
    """
    deals with lattice

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

        self._norms = None
        self._set_norms()

        self._angles = None
        self._cos_angles = None
        self._set_angles()

        self._metric = None
        self._set_metric()


    @property
    def lattice(self):
        """
        lattice
        """
        return self._lattice

    @property
    def volume(self):
        """
        volume
        """
        return self._volume

    def _set_volume(self):
        self._volume = np.dot(np.cross(self._lattice[0],
                                       self._lattice[1]),
                              self._lattice[2])

    @property
    def reciprocal_lattice(self):
        """
        reciprocal lattice WITHOUT 2*pi
        """
        return self._reciprocal_lattice

    def _set_reciprocal_lattice(self):
        recip_bases = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            recip_bases.append(
                np.cross(self._lattice[j], self._lattice[k]) / self._volume)
        self._reciprocal_lattice = np.array(recip_bases)

    @property
    def norms(self):
        """
        norms of axes
        """
        return self._norms

    def _set_norms(self):
        self._norms = np.linalg.norm(self._lattice, axis=1)

    @property
    def angles(self):
        """
        angles of axes
        """
        return self._angles

    @property
    def cos_angles(self):
        """
        cosine angles of axes
        """
        return self._cos_angles

    def _set_angles(self):
        lat = self.lattice
        norms = self.norms
        cos_angles = []
        angles = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            cos_angle = np.dot(lat[j],lat[k]) / (norms[j] * norms[k])
            cos_angles.append(cos_angle)
            angles.append(np.arccos((cos_angle)) * 180 / np.pi)
        self._cos_angles = tuple(cos_angles)
        self._angles = tuple(angles)

    @property
    def metric(self):
        """
        metric
        """
        return self._metric

    def _set_metric(self):
        metric = np.zeros([3,3])
        for i in range(3):
            for j in range(3):
                if i == j:
                    metric[i,j] = self._norms[i]**2
                else:
                    k = 3 - (i + j)
                    metric[i,j] = self._norms[i] \
                                    * self._norms[j] \
                                        * self._cos_angles[k]
        self._metric = metric

    def dot(self,
            frac_coord_first:np.array,
            frac_coord_second:np.array):
        """
        inner product of frac_coord1 and frac_coord2

        Args:
            frac_coord_first (np.array): frac coord
            frac_coord_second (np.array): frac coord
        """
        first = frac_coord_first.reshape(3,1)
        second = frac_coord_second.reshape(3,1)
        return float(np.dot(np.dot(frac_coord_first.T,
                                   self.metric),
                            frac_coord_second))

    def is_hexagonal_lattice(self):
        """
        check lattice is hexagonal
        """
        try:
            check_hexagonal_lattice(self.lattice)
            flag = True
        except AssertionError:
            flag = False
        return flag

    def get_abc(self):
        """
        get norms of each axis
        """
        abc = tuple(np.linalg.norm(lattice, axis=1))
        return abc
