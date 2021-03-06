#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deals with kpoints.
"""

from typing import Union
import numpy as np
from twinpy.properties.hexagonal import check_hexagonal_lattice
from twinpy.structure.lattice import CrystalLattice


class Kpoints():
    """
    This class deals with kpoints.
    """

    def __init__(
           self,
           lattice:np.array,
           ):
        """
        Args:
            lattice: Lattice matrix.
        """
        self._lattice = lattice
        self._reciprocal_lattice = None
        self._reciprocal_abc = None
        self._reciprocal_volume = None
        self._is_hexagonal = False
        self._set_properties()

    def _set_properties(self):
        """
        Set properties.
        """
        cry_lat = CrystalLattice(lattice=self._lattice)
        self._reciprocal_lattice = cry_lat.reciprocal_lattice
        recip_cry_lat = CrystalLattice(lattice=self._reciprocal_lattice)
        self._reciprocal_abc = recip_cry_lat.abc
        self._reciprocal_volume = recip_cry_lat.volume
        try:
            check_hexagonal_lattice(self._lattice)
            self._is_hexagonal = True
        except AssertionError:
            pass

    def get_mesh_from_interval(self,
                               interval:Union[float,np.array],
                               decimal_handling:str=None,
                               include_two_pi:bool=True) -> list:
        """
        Get mesh from interval.

        Args:
            interval: Grid interval.
            decimal_handling: Decimal handling. Available choise is 'floor',
                              'ceil' and 'round'. If 'decimal_handling' is not
                              'floor' and 'ceil', 'round' is set automatically.
            include_two_pi: If True, include 2 * pi.

        Returns:
            list: Sampling mesh.

        Note:
            The basis norms of reciprocal lattice is divided by interval and
            make float to int using the rule specified with 'decimal_handling'.
            If the final mesh includes 0, fix 0 to 1.
        """
        if isinstance(interval, np.ndarray):
            assert interval.shape == (3,), \
                       "Shape of interval is {}, which must be (3,)".format(
                               interval.shape)

        recip_abc = self._reciprocal_abc.copy()
        if include_two_pi:
            recip_abc *= 2 * np.pi

        mesh_float = np.round(recip_abc / interval, decimals=5)
        if decimal_handling == 'floor':
            mesh = np.int64(np.floor(mesh_float))
        elif decimal_handling == 'ceil':
            mesh = np.int64(np.ceil(mesh_float))
        else:
            mesh = np.int64(np.round(mesh_float))

        fixed_mesh = np.where(mesh==0, 1, mesh)

        return fixed_mesh.tolist()

    def get_intervals_from_mesh(self,
                                mesh:list,
                                include_two_pi:bool=True) -> np.array:
        """
        Get intervals from mesh.

        Args:
            mesh: Sampling mesh.
            include_two_pi: If True, include 2 * pi.

        Returns:
            np.array: Get intervals for each axis.
        """
        recip_abc = self._reciprocal_abc.copy()
        if include_two_pi:
            recip_abc *= 2 * np.pi

        intervals = recip_abc / np.array(mesh)

        return intervals

    def fix_mesh_based_on_symmetry(self, mesh:list) -> list:
        """
        Fix mesh based on lattice symmetry.

        Args:
            mesh: Sampling mesh.

        Returns:
            list: Fixed sampling mesh.

        Note:
            Currenly, check only hexagonal lattice.
            If crystal lattice is hexagonal,
            mesh is fixed as: (odd odd even).
            Else mesh: (even even even).
            But '1' is kept fixed during this operation.
        """
        if self._is_hexagonal:
            condition = lambda x: int(x%2==0)  # If True, get 1, else get 0.
            arr = [ condition(m) for m in mesh[:2] ]
            if (mesh[2]!=1 and mesh[2]%2==1):
                arr.append(1)
            else:
                arr.append(0)
            arr = np.array(arr)
        else:
            condition = lambda x: int(x%2==1)
            arr = np.array([ condition(m) for m in mesh ])
        fixed_mesh = np.array(mesh) + arr

        return fixed_mesh.tolist()

    def get_offset(self, use_gamma_center:bool=False, mesh:list=None) -> list:
        """
        Get offset.

        Args:
            use_gamma_center: If use_gamma_center is True, return [0., 0., 0.].
            mesh: Optional. If mesh is parsed,
                            set offset element 0 where mesh is 1.

        Returns:
            list: Offset from origin centered mesh grids.
        """
        if use_gamma_center:
            return [0., 0., 0.]

        if self._is_hexagonal:
            offset = [0., 0., 0.5]
        else:
            offset = [0.5, 0.5, 0.5]

        if mesh is not None:
            offset = np.array(offset)
            offset[np.where(np.array(mesh)==1)] = 0
            offset = offset.tolist()

        return offset

    def get_mesh_offset_auto(self,
                             interval:Union[float,np.array]=None,
                             mesh:list=None,
                             include_two_pi:bool=True,
                             decimal_handling:str='round',
                             use_symmetry:bool=True,
                             use_gamma_center:bool=False):
        """
        Get mesh and offset.

        Args:
            interval: Grid interval.
            mesh: Sampling mesh.
            include_two_pi: If True, include 2 * pi.
            decimal_handling: Decimal handling. Available choise is 'floor',
                              'ceil' and 'round'. If 'decimal_handling' is not
                              'floor' and 'ceil', 'round' is set automatically.
            use_symmetry: When 'mesh' is None, 'use_symmetry' is called.
                          If True, run 'fix_mesh_based_on_symmetry'.
            use_gamma_center: If use_gamma_center is True,
                              offset becomes [0., 0., 0.].

        Raises:
            ValueError: Both mesh and interval are not specified.
            ValueError: Both mesh and interval are specified.

        Returns:
            tuple: (mesh, offset).
        """
        if interval is None and mesh is None:
            raise ValueError("both mesh and interval are not specified")
        if interval is not None and mesh is not None:
            raise ValueError("both mesh and interval are specified")

        if mesh is None:
            _mesh = self.get_mesh_from_interval(
                    interval=interval,
                    decimal_handling=decimal_handling,
                    include_two_pi=include_two_pi)
            if use_symmetry:
                _mesh = self.fix_mesh_based_on_symmetry(mesh=_mesh)
        else:
            _mesh = mesh
        offset = self.get_offset(use_gamma_center=use_gamma_center,
                                 mesh=_mesh)
        return (_mesh, offset)

    def get_dict(self,
                 interval:Union[float,np.array]=None,
                 mesh:list=None,
                 include_two_pi:bool=True,
                 decimal_handling:str='round',
                 use_symmetry:bool=True):
        """
        Get dict including all properties and settings.

        Args:
            interval: Grid interval.
            mesh: Sampling mesh.
            include_two_pi: If True, include 2 * pi.
            decimal_handling: Decimal handling. Available choise is 'floor',
                              'ceil' and 'round'. If 'decimal_handling' is not
                              'floor' and 'ceil', 'round' is set automatically.
            use_symmetry: If True, run 'fix_mesh_based_on_symmetry'.

        Raises:
            ValueError: Both mesh and interval are not specified.
            ValueError: Both mesh and interval are specified.

        Returns:
            dict: All properties and settings.
        """
        mesh, offset = self.get_mesh_offset_auto(
                interval=interval,
                mesh=mesh,
                include_two_pi=include_two_pi,
                decimal_handling=decimal_handling,
                use_symmetry=use_symmetry)
        intervals = self.get_intervals_from_mesh(
                mesh=mesh,
                include_two_pi=include_two_pi,
                )
        recip_lattice = self._reciprocal_lattice.copy()
        recip_abc = self._reciprocal_abc.copy()
        recip_vol = self._reciprocal_volume
        total_mesh = mesh[0] * mesh[1] * mesh[2]
        if include_two_pi:
            recip_lattice *= 2 * np.pi
            recip_abc *= 2 * np.pi
            recip_vol *= (2 * np.pi)**3
        dic = {
                'reciprocal_lattice': recip_lattice,
                'reciprocal_abc': recip_abc,
                'reciprocal_volume': recip_vol,
                'total_mesh': total_mesh,
                'mesh': mesh,
                'offset': offset,
                'input_interval': interval,
                'intervals': intervals,
                'include_two_pi': include_two_pi,
                'decimal_handling': decimal_handling,
                'use_symmetry': use_symmetry,
              }

        return dic
