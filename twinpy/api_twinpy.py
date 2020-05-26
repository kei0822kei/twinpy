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
           symbol:str,
           wyckoff:str='c',
        ):
        """
        initialize

        Args:
            lattice (np.array): lattice
            symbol (str): element symbol
            wyckoff (str): No.194 Wycoff position ('c' or 'd')

        Raises:
            AssertionError: wyckoff is not 'c' and 'd'
            ValueError: lattice is not None (future fix)
        """
        self._hcp_matrix = lattice
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

    def get_shear(self):
        """
        get shear structure object
        """
        return self.shear

    def set_shear(self,
                  twinmode,
                  xshift=0.,
                  yshift=0.,
                  dim=np.ones(3, dtype='intc'),
                  ratio=0.):
        """
        set shear structure object

        Args:
            dim (3, numpy array): dimension
            ratio (float): shear strain ratio
        """
        self._shear = get_shear(lattice=self._hcp_matrix,
                                twinmode=twinmode,
                                wyckoff=self._wyckoff,
                                xshift=xshift,
                                yshift=yshift,
                                dim=dim,
                                ratio=ratio)
