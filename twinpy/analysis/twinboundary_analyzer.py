#!/usr/bin/env python

"""
Analize twinboudnary relax calculation.
"""
from copy import deepcopy
import warnings
import numpy as np
from matplotlib import pyplot as plt
from twinpy.structure.bonding import _get_atomic_environment
from twinpy.structure.twinboundary import TwinBoundaryStructure
from twinpy.structure.standardize import StandardizeCell, get_standardized_cell

# future delete because error occurs when users have not aiida.
# from aiida.orm import load_node
# from twinpy.interfaces.aiida.vasp import AiidaRelaxWorkChain
# from twinpy.interfaces.aiida.phonopy import AiidaPhonopyWorkChain

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
           hexagonal_relax_analyzer:RelaxAnalyzer=None,
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
        self._set_hexagonal_analyzers(hexagonal_relax_analyzer,
                                      hexagonal_phonon_analyzer)
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

    def _set_hexagonal_analyzers(self, hex_relax_analyzer, hex_phonon_analyzer):
        """
        Set hexagonal analyzers.
        """
        if hex_phonon_analyzer is not None:
            self._hexagonal_relax_analyzer = hex_phonon_analyzer.relax_analyzer
            self._hexagonal_phonon_analyzer = hex_phonon_analyzer
        else:
            self._hexagonal_relax_analyzer = hex_relax_analyzer

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

        return shear_cell

    def get_twinboundary_shear_analyzer(self,
                                        shear_strain_ratios:list,
                                        shear_relax_analyzers:list=None,
                                        shear_phonon_analyzers:list=None,
                                        ):
        """
        Get TwinBoundaryShearAnalyzer class object.

        Args:
            shear_phonon_analyzers: List of additional shear
                                           phonon analyzers.
            shear_strain_ratios: Shear shear_strain_ratios.
        """
        _relax_analyzers = [self.phonon_analyzer.relax_analyzer]
        _relax_analyzers.extend(shear_relax_analyzers)

        if shear_phonon_analyzers is None:
            phonon_analyzers = None
        else:
            assert len(shear_strain_ratios) == len(shear_phonon_analyzers)
            phonon_analyzers = [self.phonon_analyzer]
            phonon_analyzers.extend(shear_phonon_analyzers)
        strain_ratios = [0.]
        strain_ratios.extend(shear_strain_ratios)
        twinboundary_shear_analyzer = TwinBoundaryShearAnalyzer(
                relax_analyzers=_relax_analyzers,
                phonon_analyzers=phonon_analyzers,
                shear_strain_ratios=strain_ratios,
                layer_indices=self.get_layer_indices())
        return twinboundary_shear_analyzer

    # def get_twinboundary_shear_analyzer_from_relax_pks(self,
    #                                                    shear_relax_pks:list,
    #                                                    shear_strain_ratios:list,
    #                                                    shear_phonon_pks:list=None,
    #                                                    ):
    #     """
    #     Get TwinBoundaryShearAnalyzer class object from pks.

    #     Args:
    #         shear_relax_pks: Relaxes for shear structures.
    #         shear_phonon_pks: Phonon calculations for shear structures.
    #         shear_strain_ratios: Shear shear_strain_ratios.
    #     """
    #     def _get_finished_idx(aiida_rlxes):
    #         idx = len(aiida_rlxes) - 1
    #         for i, aiida_rlx in enumerate(aiida_rlxes):
    #             if not aiida_rlx.process_state == 'finished':
    #                 warnings.warn(
    #                     "{}th RelaxWorkChain has not finished yet.".format(i))
    #                 idx = i - 1
    #         return idx

    #     aiida_relaxes = [ AiidaRelaxWorkChain(load_node(pk))
    #                           for pk in shear_relax_pks ]
    #     ix = _get_finished_idx(aiida_relaxes) + 1
    #     original_cells = [ self.get_shear_cell(shear_strain_ratio=ratio,
    #                                            is_standardize=False)
    #                           for ratio in shear_strain_ratios][:ix]
    #     relax_analyzers = [ relax.get_relax_analyzer(
    #                                 original_cell=original_cells[i])
    #                                 for i, relax in enumerate(aiida_relaxes[:ix]) ]
    #     _relax_analyzers = relax_analyzers

    #     if shear_phonon_pks is None:
    #         phonon_analyzers = None
    #     else:
    #         phonon_analyzers = []
    #         for rlx_analyzer, ph_pk in zip(relax_analyzers, shear_phonon_pks):
    #             if ph_pk is None:
    #                 phonon_analyzers.append(None)
    #             else:
    #                 aiida_phonon = AiidaPhonopyWorkChain(load_node(ph_pk))
    #                 ph_analyzer = aiida_phonon.get_phonon_analyzer(
    #                         relax_analyzer=rlx_analyzer)
    #                 phonon_analyzers.append(ph_analyzer)

    #     twinboundary_shear_analyzer = self.get_twinboundary_shear_analyzer(
    #             shear_relax_analyzers=_relax_analyzers,
    #             shear_phonon_analyzers=phonon_analyzers[:ix],
    #             shear_strain_ratios=shear_strain_ratios[:ix])

    #     return twinboundary_shear_analyzer

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
        layer_indices = self.get_layer_indices()
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
            decorate = (direction == 'z')
            plot_atom_diff(ax,
                           initial_cell=initial_cell,
                           final_cell=final_cell,
                           decorate=decorate,
                           direction=direction,
                           shuffle=shuffle,
                           label=direction,
                           )

    def get_layer_indices(self):
        """
        Get layzer indices.

        Returns:
            list: Layer indices.
        """
        orig_atoms_num = len(self._relax_analyzer.original_cell[1])
        atoms_num_per_layer = self._twinboundary_structure.indices.atom_num_per_layer
        layer_indices = np.array([ i for i in range(orig_atoms_num) ])
        layer_indices = layer_indices.reshape(
                int(orig_atoms_num/atoms_num_per_layer),
                    atoms_num_per_layer)

        return layer_indices
