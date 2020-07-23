#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API for twinpy
"""

import numpy as np
from twinpy.structure.base import is_hcp
from twinpy.structure.shear import get_shear
from twinpy.structure.standardize import StandardizeCell
from twinpy.structure.twinboundary import get_twinboundary


def get_twinpy_from_cell(cell:tuple,
                         twinmode:str):
    """
    Get Twinpy object from cell.

    Args:
        cell: tuple (lattice, scaled_positions, symbols)
        twinmode (str): twinmode

    Note:
        return Twinpy class object
    """
    lattice, scaled_positions, symbols = cell
    wyckoff = is_hcp(lattice=lattice,
                     scaled_positions=scaled_positions,
                     symbols=symbols,
                     get_wyckoff=True)
    twinpy = Twinpy(lattice=lattice,
                    twinmode=twinmode,
                    symbol=symbols[0],
                    wyckoff=wyckoff)
    return twinpy


class Twinpy():
    """
    API for twinpy
    """
    def __init__(self,
                 lattice:np.array,
                 twinmode:str,
                 symbol:str,
                 wyckoff:str='c'):
        """
        Args:
            lattice (np.array): 3x3 lattice
            twinmode (str): twinmode
            symbol (str): element symbol
            wyckoff (str): No.194 Wycoff position ('c' or 'd')
        """
        self._hcp_matrix = lattice
        self._twinmode = twinmode
        self._symbol = symbol
        self._wyckoff = wyckoff
        self._shear = None
        self._twinboundary = None

    @property
    def shear(self):
        """
        Shear structure object.
        """
        return self._shear

    def set_shear(self,
                  xshift:float=0.,
                  yshift:float=0.,
                  dim:np.array=np.ones(3, dtype='intc'),
                  shear_strain_ratio:float=0.,
                  is_primitive:bool=False,
                  ):
        """
        Set shear structure object.

        Args:
            xshift (float): x shift
            yshift (float): y shift
            dim (np.array): dimension
            shear_strain_ratio (float): shear strain ratio
        """
        self._shear = get_shear(
                lattice=self._hcp_matrix,
                twinmode=self._twinmode,
                symbol=self._symbol,
                wyckoff=self._wyckoff,
                xshift=xshift,
                yshift=yshift,
                dim=dim,
                shear_strain_ratio=shear_strain_ratio,
                is_primitive=is_primitive)

    def set_twinboundary(self,
                         twintype:int=1,
                         xshift:float=0.,
                         yshift:float=0.,
                         dim:np.array=np.ones(3, dtype='intc'),
                         shear_strain_ratio:float=0.,
                         make_tb_flat=True,
                         ):
        """
        Set twinboundary structure object.

        Args:
            twintype (int): twintype, choose from 1 and 2
            xshift (float): x shift
            yshift (float): y shift
            dim (np.array): dimension
            shear_strain_ratio (float): shear twinboundary ratio
            make_tb_flat (bool): whether make twin boundary flat
        """
        self._twinboundary = get_twinboundary(
                lattice=self._hcp_matrix,
                twinmode=self._twinmode,
                wyckoff=self._wyckoff,
                symbol=self._symbol,
                twintype=twintype,
                xshift=xshift,
                yshift=yshift,
                dim=dim,
                shear_strain_ratio=shear_strain_ratio,
                make_tb_flat=make_tb_flat)

    @property
    def twinboundary(self):
        """
        Twinboundary structure object.
        """
        return self._twinboundary

    def _shear_structure_is_not_set(self):
        """
        Check shear structure is set.

        Raises:
            RuntimeError: when shear structure is not set
        """
        if self._shear is None:
            msg = "shear structure is not set, run please set shear"
            raise RuntimeError(msg)

    def _twinboundary_structure_is_not_set(self):
        """
        Check twinboudnary is set.

        Raises:
            RuntimeError: when twinboudary structure is not set
        """
        if self._twinboundary is None:
            msg = "twinboundary structure is not set, \
                   run please set twinboundary"
            raise RuntimeError(msg)

    def get_shear_standardize(self,
                              get_lattice:bool=False,
                              move_atoms_into_unitcell:bool=True,
                              ) -> StandardizeCell:
        """
        Get shear standardized structure object.
        """
        self._shear_structure_is_not_set()
        cell = self._shear.get_cell_for_export(
                get_lattice=get_lattice,
                move_atoms_into_unitcell=move_atoms_into_unitcell,
                )
        return StandardizeCell(cell)

    def get_twinboundary_standardize(self,
                                     get_lattice:bool=False,
                                     move_atoms_into_unitcell:bool=True,
                                     ) -> StandardizeCell:
        """
        Get twinboundary standardized structure object.
        """
        self._twinboundary_structure_is_not_set()
        cell = self._twinboundary.get_cell_for_export(
                get_lattice=get_lattice,
                move_atoms_into_unitcell=move_atoms_into_unitcell,
                )
        return StandardizeCell(cell)
