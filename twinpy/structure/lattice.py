#!/usr/bin/env python

"""
This module deals with crystal lattice.
"""

from copy import deepcopy
import warnings
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Supercell
from twinpy.common.utils import reshape_dimension


def get_lattice_points_from_supercell(lattice:np.array,
                                      dim:np.array) -> np.array:
    """
    Get lattice points from supercell.

    Args:
        lattice: Lattice matrix.
        dim: Dimension with its shape is (3,) or (3,3).

    Returns:
        np.array: Lattice points.
    """
    unitcell = PhonopyAtoms(symbols=['H'],
                            cell=lattice,
                            scaled_positions=np.array([[0.,0.,0]]),
                            )
    super_lattice = Supercell(unitcell=unitcell,
                              supercell_matrix=reshape_dimension(dim))
    lattice_points = super_lattice.scaled_positions

    return lattice_points


class CrystalLattice():
    """
    This class deals with crystal lattice.
    """

    def __init__(self, lattice:np.array):
        """
        Args:
            lattice: Lattice matrix.

        Raises:
            AssertionError: Input lattice is not 3x3 numpy array.
        """
        assert lattice.shape == (3, 3), \
                "Invalid lattice (not 3x3 numpy array)."

        self._lattice = lattice

        self._volume = None
        self._set_volume()

        self._reciprocal_lattice = None
        self._set_reciprocal_lattice()

        self._abc = None
        self._set_abc()

        self._angles = None
        self._sin_angles = None
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
        self._reciprocal_lattice = np.linalg.inv(np.transpose(self._lattice))

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
        sin_angles = []
        angles = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            cos_angle = np.dot(lat[j],lat[k]) / (abc[j] * abc[k])
            cos_angles.append(cos_angle)
            angles.append(np.arccos(cos_angle) * 180 / np.pi)
            sin_angles.append(np.sin(np.arccos(cos_angle)))
        self._cos_angles = tuple(cos_angles)
        self._sin_angles = tuple(sin_angles)
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
        Cosine angles.
        """
        return self._cos_angles

    @property
    def sin_angles(self):
        """
        Sine angles.
        """
        return self._sin_angles

    def _set_metric(self):
        """
        Set metric tensor.
        """
        metric = np.dot(self._lattice,
                        self._lattice.T)
        self._metric = metric

    @property
    def metric(self):
        """
        Metric tensor.
        """
        return self._metric

    def dot(self,
            first_coord:np.array,
            second_coord:np.array,
            is_cartesian:bool=False) -> float:
        """
        Return inner product of two vectors.
        Two vectors must be given as fractional coordinate.

        Args:
            frac_coord_first: Firest fractional coordinate.
            frac_coord_second: Second fractional coordinate.
            is_cartesian: If True input coord are recognized
                          as cartesian.

        Returns:
            float: Inner product.
        """
        if is_cartesian:
            frac_first = self.convert_cartesian_to_fractional(first_coord)
            frac_second = self.convert_cartesian_to_fractional(second_coord)
        else:
            frac_first = first_coord
            frac_second = second_coord
        frac_first = frac_first.reshape(3,1)
        frac_second = frac_second.reshape(3,1)

        return float(np.dot(np.dot(frac_first.T,
                                   self._metric),
                            frac_second))

    def get_norm(self,
                 coord:np.array,
                 is_cartesian:bool=False,
                 with_periodic:bool=True) -> float:
        """
        Get distance between input coord and origin.

        Args:
            coord: Coordinate.
            is_cartesian: If True input coord are recognized
                          as cartesian.
            with_periodic: If True, consider periodic condition.

        Returns:
            float: Distance in cartesian coordinate.
        """
        origin = np.array([0.,0.,0.])
        norm = self.get_distance(first_coord=coord,
                                 second_coord=origin,
                                 is_cartesian=is_cartesian,
                                 with_periodic=with_periodic)

        return norm

    def get_angle(self,
                  frac_coord_first:np.array,
                  frac_coord_second:np.array,
                  get_acute:bool=False) -> float:
        """
        Return angle between two fractional coordinate.

        Args:
            frac_coord_first: Firest fractional coordinate.
            frac_coord_second: Second fractional coordinate.
            get_acute: If True, get acute angle.

        Returns:
            float: Angle between first and second vector.
        """
        origin = np.array([0.,0.,0.])
        norm_first = self.get_distance(frac_coord_first,
                                       origin,
                                       with_periodic=False)
        norm_second = self.get_distance(frac_coord_second,
                                        origin,
                                        with_periodic=False)
        inner_product = self.dot(first_coord=frac_coord_first,
                                 second_coord=frac_coord_second,
                                 is_cartesian=False)
        cos_angle = np.round(inner_product / (norm_first * norm_second),
                             decimals=8)
        angle = np.arccos(cos_angle) * 180 / np.pi

        if get_acute:
            angle = min(angle, 180-angle)

        return angle

    def get_diff(self,
                 first_coord:np.array,
                 second_coord:np.array,
                 is_cartesian:bool=False,
                 with_periodic:bool=True) -> np.array:
        """
        Get diff between first coord and second coord.

        Args:
            first_coord: First coordinate.
            second_coord: Second coordinate.
            is_cartesian: If True input coord are recognized
                                 as cartesian.
            with_periodic: If True, consider periodic condition.

        Returns:
            np.array: Atom diff (second_coord - first_coord)
                      in fractional coordinate.

        Todo:
            Check it is best to use deepcopy.
            Currenly get_diff with with_periodic=True may return incorrect
            result.
        """
        if is_cartesian:
            _first_frac = \
                self.convert_cartesian_to_fractional(cart_coords=first_coord)
            _second_frac = \
                self.convert_cartesian_to_fractional(cart_coords=second_coord)
        else:
            _first_frac = deepcopy(first_coord)
            _second_frac = deepcopy(second_coord)

        if with_periodic:
            warnings.warn("with_periodic=True may returns incorrect result.")
            _diff = np.round((_second_frac - _first_frac) % 1., decimals=8)
            diff = np.where(_diff>0.5, _diff-1, _diff)
        else:
            diff = _second_frac - _first_frac

        return diff

    def convert_fractional_to_cartesian(self, frac_coords:np.array):
        """
        Convert list of fractional coordinates to cartesian coordinates.

        Args:
            frac_coords: List of fractional coordinates.

        Returns:
            np.array: List of cartesian coordinates.
        """
        return np.dot(np.transpose(self._lattice), frac_coords.T).T

    def convert_cartesian_to_fractional(self, cart_coords:np.array):
        """
        Convert list of cartesian coordinates to fractional coordinates.

        Args:
            cart_coords: Cartesian coordinates.

        Returns:
            np.array: Fractional coordinates.
        """
        return np.dot(np.linalg.inv(np.transpose(self._lattice)),
                      cart_coords.T).T

    def get_distance(self,
                     first_coord:np.array,
                     second_coord:np.array,
                     is_cartesian:bool=False,
                     with_periodic:bool=True) -> float:
        """
        Get cartesian distance between two coordinates.

        Args:
            first_coord: First coordinate.
            second_coord: Second coordinate.
            is_cartesian: If True input coord are recognized
                                 as cartesian.
            with_periodic: If True, consider periodic condition.

        Returns:
            float: Distance in cartesian coordinate.
        """
        diff = self.get_diff(first_coord=first_coord,
                             second_coord=second_coord,
                             is_cartesian=is_cartesian,
                             with_periodic=with_periodic)
        distance = np.linalg.norm(np.dot(self.lattice.T, diff.T))

        return distance

    def get_midpoint(self,
                     first_coord:np.array,
                     second_coord:np.array,
                     is_cartesian:bool=False,
                     with_periodic:bool=True) -> np.array:
        """
        Get midpoint.

        Args:
            first_coord: First fractional coordinate.
            second_coord: Second fractional coordinate.
            is_cartesian: If True input coord is recognized
                          as cartesian.
            with_periodic: If True, consider periodic condition.

        Returns:
            np.array: Atom diff (second_coord - first_coord)
                      in fractional coordinate.

        Todo:
            It is necessary to fix this function because
            in the case 'with_periodic=True', there are two different points
            which can be defined as midpoint.
        """
        if is_cartesian:
            _first_frac = \
                self.convert_cartesian_to_fractional(cart_coords=first_coord)
            _second_frac = \
                self.convert_cartesian_to_fractional(cart_coords=second_coord)
        else:
            _first_frac = deepcopy(first_coord)
            _second_frac = deepcopy(second_coord)

        if with_periodic:
            diff = self.get_diff(first_coord=first_coord,
                                 second_coord=second_coord,
                                 is_cartesian=is_cartesian,
                                 with_periodic=with_periodic)
            midpoint = np.round((_first_frac + diff / 2) % 1., decimals=8)
        else:
            midpoint = (_second_frac - _first_frac) / 2

        return midpoint

    def get_expanded_lattice(self,
                             expansion_ratios:np.array=np.eye(3)) -> np.array:
        """
        Get super lattice.

        Args:
            expansion_ratios: Expansion ratios.

        Returns:
            np.array: Expanded lattice.
        """
        _expansion_ratios = expansion_ratios
        if not isinstance(_expansion_ratios, np.ndarray):
            _expansion_ratios = np.array(_expansion_ratios)
        assert _expansion_ratios.shape == (3,), \
                   "Shape of expansion_ratios is {}, which must be (3,)".format(
                           np.array(_expansion_ratios).shape)
        expanded_lattice = np.transpose(np.transpose(self._lattice) * _expansion_ratios)

        return expanded_lattice
