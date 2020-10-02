#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analize shear calculation.
"""
from twinpy.structure.shear import ShearStructure
from twinpy.structure.diff import get_structure_diff
from twinpy.analysis.phonon_analyzer import PhononAnalyzer


class _BaseShearAnalyzer():
    """
    Base for ShearAnalyzer and TwinboundaryShearAnalyzer.
    """

    def __init__(
           self,
           phonon_analyzers:list
           ):
        """
        Args:
            phonon_analyzers (list): List of PhononAnalyzer class object.
        """
        self._phonon_analyzers = phonon_analyzers

    @property
    def phonon_analyzers(self):
        """
        List of phonon analyzers.
        """
        return self._phonon_analyzers

    def get_shear_diffs(self):
        """
        Get structure diffs between original relax and sheared relax cells
        IN ORIGINAL FRAME.
        """
        relax_analyzers = [ phonon_analyzer.relax_analyzer
                                for relax_analyzer in relax_analyzers ]
        relax_cells_original_frame = \
                [ relax_analyzer.final_cell_in_original_frame
                      for relax_analyzer in relax_analyzers ]
        diffs = get_structure_diff(cells=relax_cells_original_frame,
                                   base_index=0,
                                   include_base=True)
        return diffs

    def get_band_paths(self, base_band_paths:list) -> list:
        """
        Get band paths for all shear cells from band paths for first cell.

        Args:
            base_band_paths (np.array): Path connections for first
                                             primitive standardized structure.

        Examples:
            >>> base_band_paths = [[[  0, 0, 0.5],
                                    [  0, 0, 0  ]],
                                   [[0.5, 0,   0],
                                    [0.5, 0, 0.5],
                                    [  0, 0, 0.5]]]

        Note:
            Get path_connections for each shear structure considering
            structure body rotation.
        """
        base_pha = self._phonon_analyzers[0]
        bps_orig_cart = base_pha.get_band_paths_from_primitive_to_original(
                band_paths=base_band_paths,
                input_is_cart=False,
                output_is_cart=True,
                )
        band_paths_for_all = []
        for pha in self._phonon_analyzers:
            bps = pha.get_band_paths_from_original_to_primitive(
                    band_paths=bps_orig_cart,
                    input_is_cart=True,
                    output_is_cart=False,
                    )
            band_paths_for_all.append(bps)

        return band_paths_for_all

    def get_band_structures(self,
                            base_band_paths:list,
                            labels:list=None,
                            npoints:int=51,
                            with_eigenvectors:bool=False) -> list:
        """
        Get BandStructure objects.

        Args:
            base_band_paths (np.array): Path connections for first
                                             primitive standardized structure.
            labels (list): Band labels for first band paths.
            npoints (int): The number of qpoints along the band path.
            with_eigenvectors (bool): If True, compute eigenvectors.

        Notes:
            Reciprocal lattices for each structure are set automatically.
            For more detail, see 'get_band_qpoints_and_path_connections'
            in phonopy.phonon.band_structure.
        """
        band_paths_for_all = self.get_band_paths(
                base_band_paths=base_band_paths)
        band_structures = []
        for i, phonon_analyzer in enumerate(self._phonon_analyzers):
            if i == 0:
                lbs = labels
            else:
                lbs = None
            band_structure = phonon_analyzer.get_band_structure(
                    band_paths=band_paths_for_all[i],
                    labels=lbs,
                    npoints=npoints,
                    with_eigenvectors=with_eigenvectors,
                    )
            band_structures.append(band_structure)

        return band_structures


class ShearAnalyzer(_BaseShearAnalyzer):
    """
    Analize shear result.
    """

    def __init__(
           self,
           shear_structure:ShearStructure,
           phonon_analyzers:list,
           ):
        """
        Init.

        Args:
            shear_structure: ShearStructure class object.
            phonon_analyzers (list): List of PhononAnalyzer class object.

        Todo:
            Currently not supported the case the number of original_cells
            and input_cells changes because it is difficult to construct
            the relax cells in the original frame. But future fix this
            problem. One solution is to make attribute
            'self._original_primitive' which contains two atoms
            in the unit cell and original basis.
            Twinboundary shaer structure also use this class.
            If this is inconvenient, I have to create
            _BaseShaerAnalyzer, ShearAnalyzer TwinboundaryShearAnalyzer
            classes separately.
        """
        super().__init__(phonon_analyzers=phonon_analyzers)
        self._shear_structure = shear_structure

    @property
    def shear_structure(self):
        """
        Shear structure.
        """
        return self._shear_structure


class TwinboundaryShearAnalyzer(_BaseShearAnalyzer):
    """
    Analize twinboundary shear result.
    """

    def __init__(
           self,
           phonon_analyzers:list,
           shear_strain_ratios:list,
           ):
        """
        Init.

        Args:
            phonon_analyzers (list): List of PhononAnalyzer class object.
        """
        super().__init__(phonon_analyzers=phonon_analyzers)
        self._shear_strain_ratios = shear_strain_ratios

    @property
    def shear_strain_ratios(self):
        """
        Shear structure.
        """
        return self._shear_strain_ratios
