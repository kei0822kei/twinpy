#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analize shear
"""
import numpy as np
from twinpy.structure.base import get_phonopy_structure

class AnalizeShear():
    """
    analize shear result

       .. attribute:: att1

          Optional comment string.


       .. attribute:: att2

          Optional comment string.

    """

    def __init__(
           self,
           structure_type:str,
           orig_cells:list,
           input_cells:list,
           symprec:float=1e-5,
       ):
        """
        Args:
            structure_type (str): 'base', 'primitive' or 'conventional'
            comment (str): description
            selective_dynamics: description
        """
        structures = [ get_phonopy_structure(cell=cell,
                                             structure_type=structure_type,
                                             symprec=symprec)
                       for cell in orig_cells ]
        same_lattices = [ np.allclose(structure.get_cell(), input_cells[i])
                          for i, structure in enumerate(structures) ]
        if False in same_lattices:
            print(same_lattices)
            raise RuntimeError("some lattices could not be detected relation "
                               "between original cell and input cell")
        else:
            self.structures = structures
