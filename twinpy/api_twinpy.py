#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides API for twinpy.
"""

from pprint import pprint
import numpy as np
from copy import deepcopy
from phonolammps import Phonolammps
from aiida.orm import load_node
from matplotlib import pyplot as plt
from twinpy.file_io import write_poscar
from twinpy.properties.hexagonal import (get_wyckoff_from_hcp,
                                         get_hcp_atom_positions)
from twinpy.structure.shear import get_shear
from twinpy.structure.standardize import StandardizeCell
from twinpy.structure.twinboundary import get_twinboundary
from twinpy.interfaces.aiida.shear import AiidaShearWorkChain
from twinpy.interfaces.aiida.twinboundary \
        import AiidaTwinBoudnaryRelaxWorkChain
from twinpy.interfaces.aiida.twinboundary_shear \
        import AiidaTwinBoudnaryShearWorkChain
from twinpy.interfaces.aiida.base import load_aiida_profile
from twinpy.interfaces.lammps import (get_lammps_relax,
                                      get_phonon_from_phonolammps,
                                      get_relax_analyzer_from_lammps_static,
                                      get_phonon_analyzer_from_lammps_static,
                                      get_twinboundary_analyzer_from_lammps,
                                      get_twinboundary_shear_analyzer_from_lammps,
                                      )
from twinpy.analysis.relax_analyzer import RelaxAnalyzer
from twinpy.analysis.phonon_analyzer import PhononAnalyzer
from twinpy.analysis.twinboundary_analyzer import TwinBoundaryAnalyzer




load_aiida_profile()


class Twinpy():
    """
    API for twinpy.
    """
    def __init__(self,
                 lattice:np.array,
                 twinmode:str,
                 symbol:str,
                 wyckoff:str='c'):
        """
        Args:
            lattice: Lattice matrix.
            twinmode: Twinmode.
            symbol: Element symbol.
            wyckoff: No.194 Wycoff position ('c' or 'd').
        """
        self._hexagonal_lattice = lattice
        self._twinmode = twinmode
        self._symbol = symbol
        self._wyckoff = wyckoff
        self._shear = None
        self._twinboundary = None
        self._is_shear_primitive = None
        self._shear_analyzer = None
        self._twinboundary_analyzer = None
        self._twinboundary_shear_analyzer = None

    def _check_shear_is_set(self):
        """
        Check shear structure is set.

        Raises:
            RuntimeError: When shear structure is not set.
        """
        if self._shear is None:
            msg = "Shear structure is not set, please run set_shear."
            raise RuntimeError(msg)

    def _check_twinboundary_is_set(self):
        """
        Check twinboudnary is set.

        Raises:
            RuntimeError: When twinboundary structure is not set.
        """
        if self._twinboundary is None:
            msg = "Twinboundary structure is not set, \
                   please run set_twinboundary."
            raise RuntimeError(msg)

    def _check_shear_analyzer_is_set(self):
        """
        Check shear analyzer is set.

        Raises:
            RuntimeError: When shear analyzer is not set.
        """
        if self._shear_analyzer is None:
            msg = "Shear analyzer is not set, please run \
                   initialize_from_aiida_shear."
            raise RuntimeError(msg)

    def _check_twinboundary_analyzer_is_set(self):
        """
        Check twinboudnary analyzer is set.

        Raises:
            RuntimeError: When twinboundary analyzer is not set.
        """
        if self._twinboundary_analyzer is None:
            msg = "Twinboundary analyzer is not set, \
                   please run initialize_from_aiida_twinboundary."
            raise RuntimeError(msg)

    def _check_twinboundary_shear_analyzer_is_set(self):
        """
        Check twinboudnary shear analyzer is set.

        Raises:
            RuntimeError: When twinboundary analyzer is not set.
        """
        if self._twinboundary_shear_analyzer is None:
            msg = "Twinboundary shear analyzer is not set, " \
                  "please run set_twinboundary_shear_analyzer."
            raise RuntimeError(msg)

    @staticmethod
    def initialize_from_aiida_shear(shear_pk:int):
        """
        Set shear from AiidaShearWorkChain.
        """
        aiida_shear = AiidaShearWorkChain(load_node(shear_pk))
        shear_analyzer = aiida_shear.get_shear_analyzer()
        twinmode = aiida_shear.shear_conf['twinmode']
        cell = aiida_shear.cells['hexagonal']
        twinpy = get_twinpy_from_cell(
                     cell=cell,
                     twinmode=twinmode)
        twinpy.set_shear(xshift=0.,
                         yshift=0.,
                         dim=[1,1,1],
                         shear_strain_ratio=0.,
                         is_primitive=True)
        twinpy.set_shear_analyzer(shear_analyzer)

        return twinpy

    @staticmethod
    def initialize_from_aiida_twinboundary(twinboundary_relax_pk:int,
                                           twinboundary_phonon_pk:int=None,
                                           additional_relax_pks:list=None,
                                           hexagonal_relax_pk:int=None,
                                           hexagonal_phonon_pk:int=None):
        """
        Set twinboundary from AiidaTwinBoudnaryRelaxWorkChain.

        Args:
            twinboundary_relax_pk: Twinboundary relax pk.
            twinboundary_phonon_pk: Twinboundary phonon pk.
            additional_relax_pks: Additional relax pks.
            hexagonal_relax_pk: Hexagonal relax pk.
            hexagonal_phonon_pk: Hexagonal phonon pk.
        """
        aiida_tb_relax = AiidaTwinBoudnaryRelaxWorkChain(
                load_node(twinboundary_relax_pk))
        twinboundary_analyzer = \
                aiida_tb_relax.get_twinboundary_analyzer(
                    twinboundary_phonon_pk=twinboundary_phonon_pk,
                    additional_relax_pks=additional_relax_pks,
                    hexagonal_relax_pk=hexagonal_relax_pk,
                    hexagonal_phonon_pk=hexagonal_phonon_pk)
        twinpy = _get_twinpy_from_twinboundary_relax(twinboundary_relax_pk)
        twinpy.set_twinboundary_analyzer(twinboundary_analyzer)

        return twinpy

    @staticmethod
    def initialize_from_aiida_twinboundary_shear(
            twinboundary_shear_pk:int,
            twinboundary_shear_phonon_pks:list=None,
            twinboundary_phonon_pk:int=None,
            hexagonal_relax_pk:int=None,
            hexagonal_phonon_pk:int=None):
        """
        Set twinboundary from AiidaTwinBoudnaryRelaxWorkChain.

        Args:
            twinboundary_shear_pk: Twinboundary shear pk.
            twinboundary_shear_phonon_pks: Twinboundary shear phonon pks.
            twinboundary_phonon_pk: Twinboundary phonon pk.
            hexagonal_relax_pk: Hexagonal relax pk.
            hexagonal_phonon_pk: Hexagonal phonon pk.
        """
        aiida_tb_shr = AiidaTwinBoudnaryShearWorkChain(
                load_node(twinboundary_shear_pk))
        tb_rlx_pk = aiida_tb_shr.get_pks()['twinboundary_relax_pk']
        aiida_tb_shr.set_twinboundary_analyzer(
                twinboundary_phonon_pk,
                hexagonal_relax_pk,
                hexagonal_phonon_pk)
        tb_shr_analyzer = aiida_tb_shr.get_twinboundary_shear_analyzer(
                shear_phonon_pks=twinboundary_shear_phonon_pks)
        twinpy = _get_twinpy_from_twinboundary_relax(
                     twinboundary_relax_pk=tb_rlx_pk)
        twinpy.set_twinboundary_analyzer(aiida_tb_shr.twinboundary_analyzer)
        twinpy.set_twinboundary_shear_analyzer(tb_shr_analyzer)

        return twinpy

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
                  expansion_ratios:np.array=np.ones(3),
                  is_primitive:bool=False,
                  ):
        """
        Set shear structure object.

        Args:
            xshift: x shift.
            yshift: y shift.
            dim: Dimension.
            shear_strain_ratio: Shear strain ratio.
            expansion_ratios: Expansion ratios.
            is_primitive: If True, output shear structure is primitive.
        """
        self._shear = get_shear(
                lattice=self._hexagonal_lattice,
                twinmode=self._twinmode,
                symbol=self._symbol,
                wyckoff=self._wyckoff,
                xshift=xshift,
                yshift=yshift,
                dim=dim,
                shear_strain_ratio=shear_strain_ratio,
                expansion_ratios=expansion_ratios,
                is_primitive=is_primitive)
        self._is_shear_primitive = is_primitive

    @property
    def is_shear_primitive(self):
        """
        Shear structure object.
        """
        return self._is_shear_primitive

    def set_twinboundary(self,
                         layers:int,
                         delta:float=0.,
                         twintype:int=1,
                         xshift:float=0.,
                         yshift:float=0.,
                         shear_strain_ratio:float=0.,
                         expansion_ratios:np.array=np.ones(3),
                         make_tb_flat:bool=True,
                         ):
        """
        Set twinboundary structure object.

        Args:
            layers: The number of layers.
            delta: Additional interval both sites of twin boundary.
            twintype: Twintype, choose from 1 and 2.
            xshift: x shift.
            yshift: y shift.
            shear_strain_ratio: Shear twinboundary ratio.
            expansion_ratios: Expansion ratios.
            make_tb_flat: If True, atoms on the twin boundary plane are
                          projected to twin boundary.
        """
        self._twinboundary = get_twinboundary(
                lattice=self._hexagonal_lattice,
                twinmode=self._twinmode,
                wyckoff=self._wyckoff,
                symbol=self._symbol,
                twintype=twintype,
                xshift=xshift,
                yshift=yshift,
                shear_strain_ratio=shear_strain_ratio,
                expansion_ratios=expansion_ratios,
                layers=layers,
                delta=delta,
                make_tb_flat=make_tb_flat)

    @property
    def twinboundary(self):
        """
        Twinboundary structure object.
        """
        return self._twinboundary

    def set_shear_analyzer(self, shear_analyzer):
        """
        Set shear_analyzer.
        """
        self._shear_analyzer = shear_analyzer

    @property
    def shear_analyzer(self):
        """
        Shear anylzer.
        """
        return self._shear_analyzer

    def set_twinboundary_analyzer(self, twinboundary_analyzer):
        """
        Set twinboundary_analyzer.
        """
        self._twinboundary_analyzer = twinboundary_analyzer

    def set_twinboundary_analyzer_from_lammps(
            self,
            pair_style:str,
            pair_coeff:str=None,
            pot_file:str=None,
            is_relax_lattice:bool=False,
            move_atoms_into_unitcell:bool=True,
            no_standardize:bool=True,
            is_run_phonon:bool=True,
            supercell_matrix:np.array=np.eye(3),
            hexagonal_relax_analyzer:RelaxAnalyzer=None,
            hexagonal_phonon_analyzer:PhononAnalyzer=None,
            ):
        """
        Set twinboundary_analyzer from lammps.
        """
        tb_analyzer = get_twinboundary_analyzer_from_lammps(
                          twinboundary_structure=self._twinboundary,
                          pair_style=pair_style,
                          pair_coeff=pair_coeff,
                          pot_file=pot_file,
                          is_relax_lattice=is_relax_lattice,
                          move_atoms_into_unitcell=move_atoms_into_unitcell,
                          no_standardize=no_standardize,
                          is_run_phonon=is_run_phonon,
                          supercell_matrix=supercell_matrix,
                          hexagonal_relax_analyzer=hexagonal_relax_analyzer,
                          hexagonal_phonon_analyzer=hexagonal_phonon_analyzer,
                          )
        self.set_twinboundary_analyzer(tb_analyzer)

    def set_twinboundary_shear_analyzer(self, twinboundary_shear_analyzer):
        """
        Set twinboundary_shear_analyzer.
        """
        self._twinboundary_shear_analyzer = \
                twinboundary_shear_analyzer

    def set_twinboundary_shear_analyzer_from_lammps(
            self,
            pair_style:str,
            supercell_matrix,
            shear_strain_ratios:list,
            pair_coeff:str=None,
            pot_file:str=None,
            is_relax_lattice:bool=False,
            move_atoms_into_unitcell:bool=True,
            no_standardize:bool=True,
            hexagonal_relax_analyzer:RelaxAnalyzer=None,
            hexagonal_phonon_analyzer:PhononAnalyzer=None,
            ):
        """
        Set twinboundary_analyzer from lammps.
        """
        tb_analyzer, tb_shr_analyzer = \
                get_twinboundary_shear_analyzer_from_lammps(
                    twinboundary_structure=self._twinboundary,
                    pair_style=pair_style,
                    supercell_matrix=supercell_matrix,
                    shear_strain_ratios=shear_strain_ratios,
                    pair_coeff=pair_coeff,
                    pot_file=pot_file,
                    is_relax_lattice=is_relax_lattice,
                    move_atoms_into_unitcell=move_atoms_into_unitcell,
                    no_standardize=no_standardize,
                    hexagonal_relax_analyzer=hexagonal_relax_analyzer,
                    hexagonal_phonon_analyzer=hexagonal_phonon_analyzer,
                    is_return_twinboundary_analyzer=True,
                    )
        self.set_twinboundary_analyzer(tb_analyzer)
        self.set_twinboundary_shear_analyzer(tb_shr_analyzer)

    @property
    def twinboundary_analyzer(self):
        """
        Twinboundary anylzer.
        """
        return self._twinboundary_analyzer

    @property
    def twinboundary_shear_analyzer(self):
        """
        Twinboundary shear analyzer.
        """
        return self._twinboundary_shear_analyzer

    def get_shear_standardize(self,
                              get_lattice:bool=False,
                              move_atoms_into_unitcell:bool=True,
                              ) -> StandardizeCell:
        """
        Get shear standardized structure object.

        Args:
            get_lattice: Get lattice not crystal structure.
            move_atoms_into_unitcell: If True, move atoms to unitcell.
        """
        self._check_shear_is_set()
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

        Args:
            get_lattice: Get lattice not crystal structure.
            move_atoms_into_unitcell: If True, move atoms to unitcell.
        """
        self._check_twinboundary_is_set()
        cell = self._twinboundary.get_cell_for_export(
                get_lattice=get_lattice,
                move_atoms_into_unitcell=move_atoms_into_unitcell,
                )
        return StandardizeCell(cell)

    def get_shear_cells(self,
                        is_original_frame:bool=True,
                        is_relax:bool=True):
        """
        Get shear cells.

        Args:
            is_original_frame: If True, returns cells in original frame.
            is_relax: If True, return relax cells.
        """
        self._check_shear_analyzer_is_set()
        relax_analyzers = self._shear_analyzer.relax_analyzers
        cells = _get_cells(relax_analyzers=relax_analyzers,
                           is_original_frame=is_original_frame,
                           is_relax=is_relax)

        return cells

    def write_shear_cells(self,
                          is_original_frame:bool=True,
                          is_relax:bool=True,
                          header:str='shear'):
        """
        Write shear cells to POSCAR.

        Args:
            is_original_frame: If True, returns cells in original frame.
            is_relax: If True, return relax cells.
            header: File header.
        """
        cells = self.get_shear_cells(
                is_original_frame=is_original_frame,
                is_relax=is_relax)
        ratios = self._shear_analyzer.shear_strain_ratios
        for cell, ratio in zip(cells, ratios):
            filename = header + '_%1.2f.poscar' % ratio
            write_poscar(cell=cell,
                         filename=filename)

    def get_twinboundary_shear_cells(self,
                                     is_original_frame:bool=True,
                                     is_relax:bool=True):
        """
        Get twinboundary shear cells.

        Args:
            is_original_frame: If True, returns cells in original frame.
            is_relax: If True, return relax cells.
        """
        self._check_twinboundary_shear_analyzer_is_set()
        relax_analyzers = self._twinboundary_shear_analyzer.relax_analyzers
        cells = _get_cells(relax_analyzers=relax_analyzers,
                           is_original_frame=is_original_frame,
                           is_relax=is_relax)

        return cells

    def write_twinboundary_shear_cells(self,
                                       is_original_frame:bool=True,
                                       is_relax:bool=True,
                                       header:str='twinboundary_shear'):
        """
        write twinboundary shear cells to POSCAR.

        Args:
            is_original_frame: If True, returns cells in original frame.
            is_relax: If True, return relax cells.
            header: File header.
        """
        cells = self.get_twinboundary_shear_cells(
                is_original_frame=is_original_frame,
                is_relax=is_relax)
        for i, cell in enumerate(cells):
            ratio = self._twinboundary_shear_analyzer.shear_strain_ratios[i]
            filename = header + '_%1.2f.poscar' % ratio
            write_poscar(cell=cell,
                         filename=filename)

    def plot_twinboundary_shear_bandstructures(
            self,
            npoints:int=51,
            with_eigenvectors:bool=False,
            use_reciprocal_lattice:bool=True):
        """
        Plot twinboundary shear band structure.
        """
        from twinpy.common.band_path \
                import get_labels_band_paths_from_seekpath
        from twinpy.plot.band_structure import BandsPlot

        self._check_twinboundary_shear_analyzer_is_set()

        tb_shear = self._twinboundary_shear_analyzer
        base_phn = self._twinboundary_shear_analyzer.phonon_analyzers[0]
        base_cell = base_phn.primitive_cell
        labels, base_band_paths = \
                get_labels_band_paths_from_seekpath(cell=base_cell)
        band_structures = \
                tb_shear.get_band_structures(
                        base_band_paths=base_band_paths,
                        labels=labels,
                        npoints=npoints,
                        with_eigenvectors=with_eigenvectors,
                        use_reciprocal_lattice=use_reciprocal_lattice,
                        )
        bsp = BandsPlot(band_structures=band_structures)
        _, _ = bsp.plot_band_structures()
        plt.show()

    def show_twinboundary_reciprocal_high_symmetry_points(self):
        """
        Show twinboundary reciprocal high symmetry points.
        """
        phonon_analyzer = self._twinboundary_analyzer.phonon_analyzer
        recip_high_sym = phonon_analyzer.get_reciprocal_high_symmetry_points()
        pprint(recip_high_sym)


def get_twinpy_from_cell(cell:tuple,
                         twinmode:str) -> Twinpy:
    """
    Get Twinpy object from cell.

    Args:
        cell: (lattice, scaled_positions, symbols)
        twinmode: twinmode

    Note:
        Return Twinpy class object.
    """
    wyckoff = get_wyckoff_from_hcp(cell)
    lattice, _, symbols = cell
    twinpy = Twinpy(lattice=lattice,
                    twinmode=twinmode,
                    symbol=symbols[0],
                    wyckoff=wyckoff)
    return twinpy


def _get_cells(relax_analyzers:list,
               is_original_frame:bool=True,
               is_relax:bool=True):
    """
    Get cells.

    Args:
        relax_analyzers: List of relax analyzer.
        is_original_frame: If True, returns cells in original frame.
        is_relax: If True, return relax cells.
    """
    if is_original_frame:
        if is_relax:
            cells = [ relax.final_cell_in_original_frame
                          for relax in relax_analyzers ]
        else:
            cells = [ relax.original_cell for relax in relax_analyzers ]
    else:
        if is_relax:
            cells = [ relax.final_cell for relax in relax_analyzers ]
        else:
            cells = [ relax.initial_cell for relax in relax_analyzers ]

    return cells


def _get_twinpy_from_twinboundary_relax(
        twinboundary_relax_pk:int):
    """
    Get twinpy from twinboundary_relax_pk.

    Args:
        twinboundary_relax_pk: Twinboundary relax pk.
    """
    aiida_tb_relax = AiidaTwinBoudnaryRelaxWorkChain(
            load_node(twinboundary_relax_pk))
    tb_parameters = aiida_tb_relax.twinboundary_parameters
    twinpy = get_twinpy_from_cell(
                 cell=aiida_tb_relax.cells['hexagonal'],
                 twinmode=tb_parameters['twinmode'])
    twinpy.set_twinboundary(
            layers=tb_parameters['layers'],
            delta=tb_parameters['delta'],
            twintype=tb_parameters['twintype'],
            xshift=tb_parameters['xshift'],
            yshift=tb_parameters['yshift'],
            shear_strain_ratio=tb_parameters['shear_strain_ratio'],
            expansion_ratios=tb_parameters['expansion_ratios'],
            make_tb_flat=tb_parameters['make_tb_flat'],
            )

    return twinpy
