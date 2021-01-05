#!/usr/bin/env pythoo
# -*- coding: utf-8 -*-

"""
Analize twinboudnary relax calculation.
"""
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from aiida.orm import load_node
from twinpy.structure.bonding import _get_atomic_environment
from twinpy.structure.twinboundary import TwinBoundaryStructure
from twinpy.structure.standardize import StandardizeCell, get_standardized_cell

from twinpy.interfaces.aiida.vasp import AiidaRelaxWorkChain
from twinpy.interfaces.aiida.phonopy import AiidaPhonopyWorkChain
from twinpy.analysis.relax_analyzer import RelaxAnalyzer
from twinpy.analysis.phonon_analyzer import PhononAnalyzer
from twinpy.analysis.shear_analyzer import TwinBoundaryShearAnalyzer
from twinpy.plot.relax import plot_atom_diff
from twinpy.plot.twinboundary import plot_plane, plot_angle, plot_pair_distance


class TwinBoundaryAnalyzer():
    """
    Analyze shear result.
    """

    def __init__(
           self,
           twinboundary_structure:TwinBoundaryStructure,
           twinboundary_relax_analyzer:RelaxAnalyzer=None,
           twinboundary_phonon_analyzer:PhononAnalyzer=None,
           hexagonal_phonon_analyzer:PhononAnalyzer=None,
           ):
        """
        Args:
            twinboundary_structure:TwinBoundaryStructure object.
            hexagonal_phonon_analyzer: PhononAnalyzer class object.
            twinboundary_phonon_analyzer: PhononAnalyzer class object.
        """
        self._twinboundary_structure = twinboundary_structure
        self._relax_analyzer = None
        self._phonon_analyzer = None
        self._set_analyzers(twinboundary_relax_analyzer,
                            twinboundary_phonon_analyzer)
        self._hexagonal_relax_analyzer = None
        self._hexagonal_phonon_analyzer = None
        self._set_hexagonal_analyzers(hexagonal_phonon_analyzer)
        self._standardize = None
        self._set_standardize()

    def _check_phonon_analyzer_is_set(self):
        if self._phonon_analyzer is None:
            raise RuntimeError("phonon analyzer does not set.")

    def _check_hexagonal_analyzer_is_set(self):
        if self._hexagonal_relax_analyzer is None:
            raise RuntimeError("hexagonal analyzer does not set.")

    def _set_analyzers(self, relax_analyzer, phonon_analyzer):
        """
        Set analyzer.
        """
        if [relax_analyzer, phonon_analyzer] == [None, None]:
            raise RuntimeError("Both twinboundary_relax_analyzer and "
                               "twinboundary_phonon_analyzer do not set.")

        if phonon_analyzer is None:
            self._relax_analyzer = relax_analyzer
        else:
            self._relax_analyzer = phonon_analyzer.relax_analyzer
            self._phonon_analyzer = phonon_analyzer

    def _set_hexagonal_analyzers(self, hex_phonon_analyzer):
        """
        Set hexagonal analyzers.
        """
        if hex_phonon_analyzer is not None:
            self._hexagonal_relax_analyzer = hex_phonon_analyzer.relax_analyzer
            self._hexagonal_phonon_analyzer = hex_phonon_analyzer

    def _set_standardize(self):
        """
        Set standardize.
        """
        cell = self._twinboundary_structure.get_cell_for_export(
                    get_lattice=False,
                    move_atoms_into_unitcell=True)
        self._standardize = StandardizeCell(cell)

    @property
    def twinboundary_structure(self):
        """
        TwinBoundaryStructure class object
        """
        return self._twinboundary_structure

    @property
    def relax_analyzer(self):
        """
        Twinboundary relax.
        """
        return self._relax_analyzer

    @property
    def phonon_analyzer(self):
        """
        Twinboundary phonon.
        """
        return self._phonon_analyzer

    @property
    def hexagonal_relax_analyzer(self):
        """
        Bulk relax.
        """
        return self._hexagonal_relax_analyzer

    @property
    def hexagonal_phonon_analyzer(self):
        """
        Bulk phonon.
        """
        return self._hexagonal_phonon_analyzer

    @property
    def standardize(self):
        """
        Stadardize object of twinpy original cell.
        """
        return self._standardize

    def get_formation_energy(self, use_relax_lattice:bool=True):
        """
        Get formation energy. Unit [mJ/m^(-2)]

        Args:
            use_relax_lattice (bool): If True, relax lattice is used.
        """
        def __get_natom_energy(relax_analyzer):
            natom = len(relax_analyzer.final_cell[2])
            energy = relax_analyzer.energy
            return (natom, energy)

        def __get_excess_energy():
            tb_natoms, tb_energy = __get_natom_energy(self._relax_analyzer)
            hex_natoms, hex_energy = \
                    __get_natom_energy(self._hexagonal_relax_analyzer)
            excess_energy = tb_energy - hex_energy * (tb_natoms / hex_natoms)
            return excess_energy

        self._check_hexagonal_analyzer_is_set()
        eV = 1.6022 * 10 ** (-16)
        ang = 10 ** (-10)
        unit = eV / ang ** 2

        if use_relax_lattice:
            lattice = self._relax_analyzer.final_cell_in_original_frame[0]
        else:
            lattice = self._relax_analyzer.original_cell[0]
        A = lattice[0,0] * lattice[1,1]
        excess_energy = __get_excess_energy()
        formation_energy = np.array(excess_energy) / (2*A) * unit
        return formation_energy

    def _get_shear_twinboundary_lattice(self,
                                        tb_lattice:np.array,
                                        shear_strain_ratio:float) -> np.array:
        """
        Get shear twinboudnary lattice.
        """
        lat = deepcopy(tb_lattice)
        e_b = lat[1] / np.linalg.norm(lat[1])
        shear_func = \
            self._twinboundary_structure.indices.get_shear_strain_function()
        lat[2] += np.linalg.norm(lat[2]) \
                  * shear_func(self._twinboundary_structure.r) \
                  * shear_strain_ratio \
                  * e_b
        return lat

    def get_shear_cell(self,
                       shear_strain_ratio:float,
                       atom_positions:np.array=None,
                       is_standardize:bool=False) -> tuple:
        """
        Get shear introduced twinboundary cell.

        Args:
            shear_strain_ratio (float): Shear strain ratio.
            atom_positions (np.array): Atom positions in fractional coordinate.
                                       If atom_positions is None, the ones of
                                       original relax cell is used.
            is_standardize (bool): If True, get standardized cell.

        Returns:
            tuple: shear introduced cell

        Notes:
            If you want to shear structure step by step, you can get next step
            structure by inputting atom positions of previous step using
            'atom_positions' input parameter.
        """
        orig_relax_cell = self._relax_analyzer.final_cell_in_original_frame

        shear_lat = self._get_shear_twinboundary_lattice(
            tb_lattice=orig_relax_cell[0],
            shear_strain_ratio=shear_strain_ratio)
        if atom_positions is None:
            frac_coords = orig_relax_cell[1]
        else:
            assert orig_relax_cell[1].shape == atom_positions.shape, \
                    "numpy shape of atom_positions is {} " \
                    "which is not the same as original relax cell {}.".format(
                            orig_relax_cell[1].shape, atom_positions.shape)
            frac_coords = atom_positions

        shear_cell = (shear_lat, frac_coords, orig_relax_cell[2])
        if is_standardize:
            std_cell = get_standardized_cell(
                    cell=shear_cell,
                    to_primitive=True,
                    no_idealize=False,
                    no_sort=True)
            return std_cell
        else:
            return shear_cell

    def get_twinboundary_shear_analyzer(self,
                                        shear_strain_ratios:list,
                                        relax_analyzers:list=None,
                                        shear_phonon_analyzers:list=None,
                                        ):
        """
        Get TwinBoundaryShearAnalyzer class object.

        Args:
            shear_phonon_analyzers (list): List of additional shear
                                           phonon analyzers.
            shear_strain_ratios (list): Shear shear_strain_ratios.
        """
        if shear_phonon_analyzers is None:
            _relax_analyzers = [self.phonon_analyzer.relax_analyzer]
            _relax_analyzers.extend(relax_analyzers)
            phonon_analyzers = None
        else:
            _relax_analyzers = None
            phonon_analyzers = [self.phonon_analyzer]
            phonon_analyzers.extend(shear_phonon_analyzers)
        strain_ratios = [0.]
        strain_ratios.extend(shear_strain_ratios)
        twinboundary_shear_analyzer = TwinBoundaryShearAnalyzer(
                relax_analyzers=_relax_analyzers,
                phonon_analyzers=phonon_analyzers,
                shear_strain_ratios=strain_ratios,
                layer_indices=self.get_layer_indeces())
        return twinboundary_shear_analyzer

    def get_twinboundary_shear_analyzer_from_pks(self,
                                                 shear_relax_pks:list,
                                                 shear_strain_ratios:list,
                                                 shear_phonon_pks:list=None,
                                                 ):
        """
        Get TwinBoundaryShearAnalyzer class object from pks.

        Args:
            shear_relax_pks (list): Relaxes for shear structures.
            shear_phonon_pks (list): Phonon calculations for shear structures.
            shear_strain_ratios (list): Shear shear_strain_ratios.
        """
        original_cells = [ self.get_shear_cell(shear_strain_ratio=ratio,
                                               is_standardize=False)
                               for ratio in shear_strain_ratios ]
        aiida_relaxes = [ AiidaRelaxWorkChain(load_node(pk))
                              for pk in shear_relax_pks ]
        relax_analyzers = [ relax.get_relax_analyzer(
                                    original_cell=original_cells[i])
                                for i, relax in enumerate(aiida_relaxes) ]
        if shear_phonon_pks is None:
            phonon_analyzers = None
            _relax_analyzers = relax_analyzers
        else:
            aiida_phonons = [ AiidaPhonopyWorkChain(load_node(pk))
                                  for pk in shear_phonon_pks ]
            phonon_analyzers = [ phonon.get_phonon_analyzer(
                                         relax_analyzer=relax_analyzers[i])
                                   for i, phonon in enumerate(aiida_phonons) ]
            _relax_analyzers = None
        twinboundary_shear_analyzer = self.get_twinboundary_shear_analyzer(
                relax_analyzers=_relax_analyzers,
                shear_phonon_analyzers=phonon_analyzers,
                shear_strain_ratios=shear_strain_ratios)

        return twinboundary_shear_analyzer

    def get_atomic_environment(self) -> list:
        """
        Get plane coords from lower plane to upper plane.
        Return list of z coordinates of original cell frame.
        Plane coordinates (z coordinates) are fractional.

        Returns:
            tuple: First and second element are for initial and final cell
                   respectively. Each contains (planes, distances, angles)
        """
        initial_cell = self._relax_analyzer.original_cell
        final_cell = self._relax_analyzer.final_cell_in_original_frame
        cells = [ initial_cell, final_cell ]
        layer_indices = self.get_layer_indeces()
        atm_envs = [ _get_atomic_environment(cell, layer_indices)
                         for cell in cells ]
        return atm_envs

    def plot_plane_diff(self):
        """
        Plot plane diff.
        """
        initial_env, final_env = self.get_atomic_environment()

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)

        plot_plane(ax, distances=initial_env[1], z_coords=initial_env[0],
                   decorate=False, label='Initial')
        plot_plane(ax, distances=final_env[1], z_coords=final_env[0],
                   label='Final')

        return fig

    def plot_angle_diff(self) -> plt.figure:
        """
        Plot angle diff.
        """
        initial_env, final_env = self.get_atomic_environment()

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)

        plot_angle(ax, angles=initial_env[2], z_coords=initial_env[0],
                   decorate=False, label='Initial')
        plot_angle(ax, angles=final_env[2], z_coords=final_env[0],
                   label='Final')

        return fig

    def plot_pair_distance(self) -> plt.figure:
        """
        Plot angle diff.
        """
        initial_env, final_env = self.get_atomic_environment()

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)

        plot_pair_distance(ax,
                           pair_distances=initial_env[3],
                           z_coords=initial_env[0],
                           decorate=False, label='Initial')
        plot_pair_distance(ax,
                           pair_distances=final_env[3],
                           z_coords=final_env[0],
                           label='Final')

        return fig

    def plot_atom_diff(self,
                       shuffle:bool=True):
        """
        Plot atom diff.

        Args:
            shuffle (bool): If True, diffrence of scaled positions,
                            which ignore lattice shear, are ploted.
        """
        initial_cell = self._relax_analyzer.original_cell
        final_cell = self._relax_analyzer.final_cell_in_original_frame

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)

        for direction in ['x', 'y', 'z']:
            if direction == 'z':
                decorate = True
            else:
                decorate = False
            plot_atom_diff(ax,
                           initial_cell=initial_cell,
                           final_cell=final_cell,
                           decorate=decorate,
                           direction=direction,
                           shuffle=shuffle,
                           label=direction,
                           )

    def get_layer_indeces(self):
        """
        Get layzer indices.

        Returns:
            list: Layer indices.
        """
        orig_atoms = self._relax_analyzer.original_cell[1]
        sort_indices = np.argsort(orig_atoms[:,2])
        layer_indices = sort_indices.reshape(int(len(sort_indices)/2), 2)

        return layer_indices
