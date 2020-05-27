#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API for twinpy
"""

from twinpy.lattice import Lattice
from twinpy.structure import get_shear, TwinBoundaryStructure

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

    def _shear_structure_is_not_set(self):
        """
        raise RuntimeError when shear structure is not set
        """
        if self._shear is None:
            msg = "shear structure is not set, run please set shear"
            raise RuntimeError(msg)

    def get_shear(self):
        """
        get shear structure object
        """
        return self.shear

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
        self._shear = get_shear(lattice=self._hcp_matrix,
                                twinmode=self._twinmode,
                                wyckoff=self._wyckoff,
                                xshift=xshift,
                                yshift=yshift,
                                dim=dim,
                                shear_strain_ratio=shear_strain_ratio)

    def write_shear_lattice(self, filename:str='shear_lattice.poscar'):
        """
        create shear lattice poscar file

        Args:
            filename (str): POSCAR filename
        """
        self._shear_structure_is_not_set()
        self._shear.write_poscar(filename=filename,
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
