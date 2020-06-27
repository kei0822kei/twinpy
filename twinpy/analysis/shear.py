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
from twinpy.common.plot import (bands_plot,
                                get_plot_properties_from_trajectory)

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
        are_same_lattices = [ np.allclose(structure.get_cell(), input_cells[i][0])
                              for i, structure in enumerate(structures) ]
        if False in are_same_lattices:
            raise RuntimeError("some lattices could not be detected relation "
                               "between original cell and input cell")
        self._original_cells = orig_cells
        self._relax_input_structures = structures
        self._relax_output_structures = None
        self._phonons = None

    def set_relax_output_structures(self, relax_cells:list):
        """
        Args:
            relax_cells (list): relax cells
        """
        self._relax_output_structures = [ get_phonopy_structure(cell)
                                     for cell in relax_cells ]

    def set_phonons(self, phonons):
        """
        set phonons
        """
        self._phonons = phonons

    def get_phonons(self):
        """
        get phonons
        """
        return self._phonons

    @property
    def original_cells(self):
        """
        original cells
        """
        return self._original_cells

    @property
    def relax_input_structures(self):
        """
        original structures
        """
        return self._relax_input_structures

    @property
    def relax_output_structures(self):
        """
        relax structures
        """
        return self._relax_output_structures

    def get_relax_diffs(self):
        """
        get structure diffs between original and relax structures
        """
        diffs = [ get_structure_diff(get_cell_from_phonopy_structure(
                                         self._relax_input_structures[i]),
                                     get_cell_from_phonopy_structure(
                                         self._relax_output_structures[i]))
                  for i in range(len(self._relax_input_structures)) ]
        return diffs

    def get_shear_diffs(self):
        """
        get structure diffs between original and sheared structures

        TODO:
            rotation matrix does not considered, which is not correct
        """
        cells = [ get_cell_from_phonopy_structure(ph_structure)
                    for ph_structure in self._relax_output_structures ]
        diffs = get_structure_diff(*cells)
        return diffs

    def plot_bands(self,
                   fig,
                   with_dos=False,
                   mesh=None,
                   band_labels=None,
                   segment_qpoints=None,
                   is_auto=False,
                   xscale=20,
                   npoints=51,
                   labels=None,):
        """
        plot phonon bands

        Args:
            arg1 (str): description
            arg2 (3x3 numpy array): description

        Returns:
            dict: description

        Raises:
            ValueError: description

        Examples:
            description

            >>> print_test ("test", "message")
              test message

        Note:
            description
        """
        cs, alphas, linewidths, linestyles = \
                get_plot_properties_from_trajectory(
                        plot_nums=len(self._phonons))
        bands_plot(fig=fig,
                   phonons=self._phonons,
                   with_dos=with_dos,
                   mesh=mesh,
                   band_labels=band_labels,
                   segment_qpoints=segment_qpoints,
                   is_auto=is_auto,
                   xscale=xscale,
                   npoints=npoints,
                   cs=cs,
                   alphas=alphas,
                   linewidths=linewidths,
                   linestyles=linestyles,
                   labels=labels,
                   )
