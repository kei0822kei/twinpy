#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API for twinpy
"""

import numpy as np
from twinpy.lattice.lattice import Lattice
from twinpy.structure.shear import get_shear
from twinpy.structure.twinboundary import get_twinboundary

class Twinpy():
    """
    API for twinpy

       .. attribute:: att1

          Optional comment string.


       .. attribute:: att2

          Optional comment string.

    """
    def __init__(
           self,
           lattice:np.array,
           twinmode:str,
           symbol:str,
           wyckoff:str='c',
        ):
        """
        initialize

        Args:
            lattice (np.array): lattice
            twinmode (str): twinmode
            symbol (str): element symbol
            wyckoff (str): No.194 Wycoff position ('c' or 'd')

        Raises:
            AssertionError: wyckoff is not 'c' and 'd'
            ValueError: lattice is not None (future fix)
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
        shear structure object
        """
        return self._shear

    def set_shear(self,
                  xshift=0.,
                  yshift=0.,
                  dim=np.ones(3, dtype='intc'),
                  shear_strain_ratio=0.):
        """
        set shear structure object

        Args:
            xshift (float): x shift
            yshift (float): y shift
            dim (3, numpy array): dimension
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
                shear_strain_ratio=shear_strain_ratio)

    def get_shear(self):
        """
        get shear structure
        """
        return self.shear

    @property
    def twinboundary(self):
        """
        twinboundary structure object
        """
        return self._twinboundary

    def set_twinboundary(self,
                         twintype:int=1,
                         xshift:float=0.,
                         yshift:float=0.,
                         dim:np.array=np.ones(3, dtype='intc'),
                         shear_strain_ratio:float=0.,
                         make_tb_flat=True,
                         ):
        """
        set twinboundary structure object

        Args:
            twintype (int): twintype, choose from 1 and 2
            xshift (float): x shift
            yshift (float): y shift
            dim (3, numpy array): dimension
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

    def get_twinboundary(self):
        """
        get twinboundary structure
        """
        return self.twinboundary

    def _shear_structure_is_not_set(self):
        """
        raise RuntimeError when shear structure is not set
        """
        if self._shear is None:
            msg = "shear structure is not set, run please set shear"
            raise RuntimeError(msg)

    def _twinboundary_structure_is_not_set(self):
        """
        raise RuntimeError when twinboundary structure is not set
        """
        if self._twinboundary is None:
            msg = "twinboundary structure is not set, run please set twinboundary"
            raise RuntimeError(msg)

    def write_shear_lattice(self, filename:str='shear_lattice.poscar'):
        """
        create shear lattice poscar file

        Args:
            filename (str): POSCAR filename
        """
        self._shear_structure_is_not_set()
        self._shear.write_poscar(filename=filename,
                                 get_lattice=True)

    def write_twinboundary_lattice(
            self, filename:str='twinboundary_lattice.poscar'):
        """
        create twinboundary lattice poscar file

        Args:
            filename (str): POSCAR filename
        """
        self._twinboundary_structure_is_not_set()
        self._twinboundary.write_poscar(filename=filename,
                                        get_lattice=True)

    def get_shear_phonopy_structure(self,
                                    structure_type:str='base',
                                    symprec:float=1e-5):
        """
        return shear phonopy strucutre

        Args:
            structure_type (str): choose from 'base', 'conventional'
                                  and 'primitive'
            symprec (float): use when shearching conventional unitcell
        """
        self._shear_structure_is_not_set()
        ph_structure = self._shear.get_phonopy_structure(
                structure_type=structure_type,
                symprec=symprec)
        return ph_structure

    def get_twinboundary_phonopy_structure(self,
                                           structure_type:str='base',
                                           symprec:float=1e-5):
        """
        return twinboundary phonopy strucutre

        Args:
            structure_type (str): choose from 'base', 'conventional'
                                  and 'primitive'
            symprec (float): use when shearching conventional unitcell
        """
        self._twinboundary_structure_is_not_set()
        ph_structure = self._twinboundary.get_phonopy_structure(
                structure_type=structure_type,
                symprec=symprec)
        return ph_structure
