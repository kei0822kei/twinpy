#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analize shear
"""
import warnings
import numpy as np
from twinpy.structure.base import (get_phonopy_structure,
                                   get_cell_from_phonopy_structure)
from twinpy.structure.diff import get_structure_diff

class ShearAnalizer():
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
            orig_cells (list): original cells, orig_cells=(cell1, cell2, ...)
            input_cells (list): input cells for vasp
        """
        if structure_type == 'conventional':
            warnings.warn("structure_type = 'conventional' \
                           may occur some errors")

        structures = [ get_phonopy_structure(cell=cell,
                                             structure_type=structure_type,
                                             symprec=symprec)
                       for cell in orig_cells ]
        same_lattices = [ np.allclose(structure.get_cell(), input_cells[i][0])
                          for i, structure in enumerate(structures) ]
        if False in same_lattices:
            raise RuntimeError("some lattices could not be detected relation "
                               "between original cell and input cell")
        else:
            self.structures = structures
        self._original_structures = structures
        self._relax_structures = None

    def set_relax_structures(self, relax_cells:list):
        """
        Args:
            relax_cells (list): relax cells
        """
        self._relax_structures = [ get_phonopy_structure(cell)
                                     for cell in relax_cells ]

    @property
    def original_structures(self):
        """
        original structures
        """
        return self._original_structures

    @property
    def relax_structures(self):
        """
        relax structures
        """
        return self._relax_structures

    def get_relax_diffs(self):
        """
        get structure diffs between original and relax structures
        """
        diffs = [ get_structure_diff(get_cell_from_phonopy_structure(
                                         self._original_structures[i]),
                                     get_cell_from_phonopy_structure(
                                         self._relax_structures[i]))
                  for i in range(len(self._original_structures)) ]
        return diffs
