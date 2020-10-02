#!/usr/bin/env pythoo
# -*- coding: utf-8 -*-

"""
Analize twinboudnary relax calculation.
"""
import numpy as np
from copy import deepcopy
from twinpy.structure.diff import get_structure_diff
from twinpy.structure.twinboundary import TwinBoundaryStructure
from twinpy.structure.standardize import StandardizeCell
from twinpy.plot.base import get_plot_properties_for_trajectory
from twinpy.common.kpoints import get_mesh_offset_from_direct_lattice
# from twinpy.interfaces.aiida import (AiidaRelaxWorkChain,
#                                      AiidaPhononWorkChain)
from twinpy.analysis import (RelaxAnalyzer,
                             PhononAnalyzer,
                             ShearAnalyzer,
                             TwinboundaryShearAnalyzer)
from phonopy import Phonopy


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

        # self._twinboundary_phonon_analyzer = twinboundary_phonon_analyzer
        # self._hexagonal_phonon_analyzer = hexagonal_phonon_analyzer
        # self._standardize = None
        # self._set_standardize()

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
            natom_energy = [ __get_natom_energy(relax_analyzer)
                                 for relax_analyzer in relax_analyzers ]
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
                       is_standardize:bool=False) -> tuple:
        """
        Get shear introduced twinboundary cell.

        Args:
            shear_strain_ratio (float): shear strain ratio
            is_standardize (bool): if True, get standardized cell

        Returns:
            tuple: shear introduced cell

        Notes:
            original relax cell is use to create shear cell which is a little
            bit different shear value with respect to bulk shear value
            but I expect this is neglectable.
        """
        orig_relax_cell = self._relax_analyzer.final_cell_in_original_frame

        shear_lat = self._get_shear_twinboundary_lattice(
            tb_lattice=orig_relax_cell[0],
            shear_strain_ratio=shear_strain_ratio)
        shear_cell = (shear_lat, orig_relax_cell[1], orig_relax_cell[2])
        if is_standardize:
            std_cell = get_standardized_cell(
                    cell=shear_cell,
                    to_primitive=True,
                    no_idealize=False,
                    no_sort=True)
            return std_cell
        else:
            return shear_cell

    # def get_twinboundary_shear_analyzer(self,
    #                                     shear_phonon_analyzers:list,
    #                                     shear_strain_ratios:list):
    #     """
    #     Get TwinBoundaryShearAnalyzer class object.

    #     Args:
    #         shear_phonon_analyzers (list): List of additional shear
    #                                        phonon analyzers.
    #     """
    #     phonon_analyzers = deepcopy(shear_phonon_analyzers)
    #     ratios = deepcopy(shear_strain_ratios)
    #     phonon_analyzers.insert(0, self._twinboundary_phonon_analyzer)
    #     ratios.insert(0, 0.)
    #     twinboundary_shear_analyzer = TwinBoundaryShearAnalyzer(
    #             phonon_analyzers=phonon_analyzers,
    #             shear_strain_ratios=ratios)
    #     return twinboundary_shear_analyzer






    # def _set_rotation_matrices(self):
    #     """
    #     Set rotation matrix.
    #     """
    #     self._hexagonal_to_original_rotation_matrix = \
    #             self._twinboundary_structure.rotation_matrix
    #     self._twinboundary_to_original_rotation_matrix = \
    #             np.linalg.inv(self._standardize.rotation_matrix)

    # @property
    # def hexagonal_to_original_rotation_matrix(self):
    #     """
    #     Hexagonal to original rotation matrix.
    #     """
    #     return self._hexagonal_to_original_rotation_matrix

    # @property
    # def twinboundary_to_original_rotation_matrix(self):
    #     """
    #     Twinboundary to original rotation matrix.
    #     """
    #     return self._twinboundary_to_original_rotation_matrix

    # def set_shears(self, shear_relaxes:list, shear_phonons:list=None):
    #     """
    #     Set shear relaxes and corresponding shear phonons

    #     Args:
    #         shear_relaxes: list of AiidaRelaxWorkChain
    #         shear_phonons: list of AiidaPhonopyWorkChain
    #     """
    #     self._shear_phonons = shear_phonons
    #     self._shear_phonons = shear_phonons

    # @property
    # def shear_relaxes(self):
    #     """
    #     Shear relaxes, list of AiidaRelaxWorkChain objects.
    #     """
    #     return self._shear_phonons

    # @property
    # def shear_phonons(self):
    #     """
    #     Shear phonons, list of AiidaPhonopyWorkChain objects.
    #     """
    #     return self._shear_phonons

    # def get_shear_analyzer(self):
    #     """
    #     Get ShearAnalyzer class object.

    #     Returns:
    #         ShearAnalyzer: ShearAnalyzer class object.
    #     """

    # def run_mesh(self, interval:float=0.1):
    #     """
    #     Run mesh for both hexagonal and twinboundary phonon.

    #     Args:
    #         interval (float): mesh interval
    #     """
    #     phonons = (self._hexagonal_phonon, self._twinboundary_phonon)
    #     structure_types = ['hexagonal', 'twinboundary']
    #     for structure_type, phonon in zip(structure_types, phonons):
    #         lattice = phonon.primitive.get_cell()
    #         kpt = get_mesh_offset_from_direct_lattice(
    #                 lattice=lattice,
    #                 interval=interval,
    #                 )
    #         print("run mesh with {} ({})".format(
    #             kpt['mesh'], structure_type))
    #         phonon.run_mesh
    #         phonon.set_mesh(
    #             mesh=kpt['mesh'],
    #             shift=None,
    #             is_time_reversal=True,
    #             is_mesh_symmetry=False,  # necessary for calc ellipsoid
    #             is_eigenvectors=True,
    #             is_gamma_center=False,
    #             run_immediately=True)

    # def get_thermal_displacement_matrices(
    #         self,
    #         t_step:int=100,
    #         t_max:int=1000,
    #         t_min:int=0,
    #         with_original_cart:bool=True,
    #         def_cif:bool=False,
    #         ):
    #     """
    #     Get ThermalDisplacementMatrices object for
    #     both hexagonal and twinboundary.

    #     Args:
    #         t_step (int): temperature interval
    #         t_max (int): max temperature
    #         t_min (int): minimum temperature
    #         with_original_cart (bool): if True, use twinboundary
    #                                    original frame
    #         def_cif (bool): if True, use cif definition

    #     Todo:
    #         I do not know how to rotate 4d array (temp, atoms, 3, 3).
    #     """
    #     phonons = (self._hexagonal_phonon, self._twinboundary_phonon)
    #     tdm_matrices = []
    #     rotation_matrices = (self._hexagonal_to_original_rotation_matrix,
    #                          self._twinboundary_to_original_rotation_matrix)
    #     for phonon, rotation_matrix in zip(phonons, rotation_matrices):
    #         phonon.set_thermal_displacement_matrices(
    #             t_step=t_step,
    #             t_max=t_max,
    #             t_min=t_min,
    #             freq_min=None,
    #             freq_max=None,
    #             t_cif=None)
    #         tdm = phonon.thermal_displacement_matrices
    #         if def_cif:
    #             matrix = tdm.thermal_displacement_matrices_cif
    #         else:
    #             matrix = tdm.thermal_displacement_matrices
    #         if with_original_cart:
    #             rot_matrices = []
    #             shape = matrix.shape
    #             lst = []
    #             for i in range(shape[0]):
    #                 atom_lst = []
    #                 for j in range(shape[1]):
    #                     mat = np.dot(rotation_matrix,
    #                                  np.dot(matrix[i,j],
    #                                         rotation_matrix.T))
    #                     atom_lst.append(mat)
    #                 lst.append(atom_lst)
    #             tdm_matrices.append(np.array(lst))
    #         else:
    #             tdm_matrices.append(tdm.thermal_displacement_matrices)
    #     return tuple(tdm_matrices)

    # def get_diff(self, use_additional_relax:bool=False) -> dict:
    #     """
    #     Get diff between vasp input and output twinboundary structure.

    #     Args:
    #         use_additional_relax (bool): if True, output twinboundary structure
    #                                      becomes the final structure of
    #                                      additional_relax

    #     Returns:
    #         dict: diff between vasp input and output twinboundary structure

    #     Raises:
    #         AssertionError: lattice matrix is not identical
    #     """
    #     if use_additional_relax:
    #         final_cell = self._cells['twinboundary_additional_relax']
    #     else:
    #         final_cell = self._cells['twinboundary_relax']
    #     cells = (self._cells['twinboundary'],
    #              final_cell)
    #     diff = get_structure_diff(cells=cells,
    #                               base_index=0,
    #                               include_base=False)
    #     if not use_additional_relax:
    #         np.testing.assert_allclose(
    #                 diff['lattice_diffs'][0],
    #                 np.zeros((3,3)),
    #                 atol=1e-8,
    #                 err_msg="lattice matrix is not identical")
    #     return diff

    # def get_planes_angles(self,
    #                       is_fractional:bool=False,
    #                       use_additional_relax:bool=False) -> dict:
    #     """
    #     Get plane coords from lower plane to upper plane.
    #     Return list of z coordinates of original cell frame.

    #     Args:
    #         is_fractional (bool): if True, return with fractional coordinate
    #         use_additional_relax (bool): if True, output twinboundary structure
    #                                      becomes the final structure of
    #                                      additional_relax

    #     Returns:
    #         dict: plane coords of input and output twinboundary structures.
    #     """
    #     if not is_fractional and use_additional_relax:
    #         raise RuntimeError(
    #                 "is_fractional=False and use_additional_relax=True "
    #                 "is not allowed because in the case "
    #                 "use_additional_relax=True, c norm changes.")

    #     if use_additional_relax:
    #         orig_final_cell = \
    #                 self._cells['twinboundary_additional_relax_original']
    #     else:
    #         orig_final_cell = self._cells['twinboundary_relax_original']
    #     cells = [self._cells['twinboundary_original'],
    #              orig_final_cell]

    #     coords_list = []
    #     angles_list = []
    #     for cell in cells:
    #         lattice = Lattice(cell[0])
    #         atoms = cell[1]
    #         c_norm = lattice.abc[2]
    #         k_1 = np.array([0,0,1])
    #         sort_atoms = atoms[np.argsort(atoms[:,2])]

    #         # planes
    #         coords = [ lattice.dot(np.array(atom), k_1) / c_norm
    #                        for atom in sort_atoms ]
    #         num = len(coords)
    #         ave_coords = np.sum(
    #                 np.array(coords).reshape(int(num/2), 2), axis=1) / 2

    #         epsilon = 1e-9    # to remove '-0.0'
    #         if is_fractional:
    #             d = np.round(ave_coords+epsilon, decimals=8) / c_norm
    #         else:
    #             d = np.round(ave_coords+epsilon, decimals=8)
    #         coords_list.append(list(d))

    #         # angles
    #         sub_coords_orig = \
    #                 sort_atoms[[i for i in range(1,num,2)]] \
    #                     - sort_atoms[[i for i in range(0,num,2)]]
    #         sub_coords_plus = \
    #                 sort_atoms[[i for i in range(1,num,2)]]+np.array([0,1,0]) \
    #                     - sort_atoms[[i for i in range(0,num,2)]]
    #         sub_coords_minus = \
    #                 sort_atoms[[i for i in range(1,num,2)]]-np.array([0,1,0]) \
    #                     - sort_atoms[[i for i in range(0,num,2)]]
    #         coords = []
    #         for i in range(len(sub_coords_orig)):
    #             norm_orig = lattice.get_norm(sub_coords_orig[i])
    #             norm_plus = lattice.get_norm(sub_coords_plus[i])
    #             norm_minus = lattice.get_norm(sub_coords_minus[i])
    #             norms = [norm_orig, norm_plus, norm_minus]
    #             if min(norms) == norm_orig:
    #                 coords.append(sub_coords_orig[i])
    #             elif min(norms) == norm_plus:
    #                 coords.append(sub_coords_plus[i])
    #             else:
    #                 coords.append(sub_coords_minus[i])

    #         angles = [ lattice.get_angle(frac_coord_first=coord,
    #                                      frac_coord_second=np.array([0,1,0]),
    #                                      get_acute=True)
    #                    for coord in coords ]
    #         angles_list.append(np.array(angles))

    #     return {
    #             'planes': {'before': coords_list[0],
    #                        'relax': coords_list[1]},
    #             'angles': {'before': angles_list[0],
    #                        'relax': angles_list[1]},
    #             }

    # def get_distances(self,
    #                   is_fractional:bool=False,
    #                   use_additional_relax:bool=False) -> dict:
    #     """
    #     Get distances from the result of 'get_planes'.

    #     Args:
    #         is_fractional (bool): if True, return with fractional coordinate
    #         use_additional_relax (bool): if True, output twinboundary structure
    #                                      becomes the final structure of
    #                                      additional_relax

    #     Returns:
    #         dict: distances between planes of input and output
    #               twinboundary structures.
    #     """
    #     planes = self.get_planes_angles(
    #             is_fractional=is_fractional,
    #             use_additional_relax=use_additional_relax)['planes']
    #     lattice = Lattice(self._cells['twinboundary_original'][0])
    #     c_norm = lattice.abc[2]
    #     keys = ['before', 'relax']
    #     dic = {}
    #     for key in keys:
    #         coords = planes[key]
    #         if is_fractional:
    #             coords.append(1.)
    #         else:
    #             coords.append(c_norm)
    #         distances = [ coords[i+1] - coords[i]
    #                           for i in range(len(coords)-1) ]
    #         dic[key] = distances
    #     return dic

    # def plot_convergence(self):
    #     """
    #     Plot convergence.
    #     """
    #     relax_pk = self.get_pks()['relax_pk']
    #     aiida_relax = AiidaRelaxWorkChain(load_node(relax_pk))
    #     aiida_relax.set_additional_relax(self._additional_relax_pks)
    #     aiida_relax.plot_convergence()

    # def plot_plane_diff(self,
    #                     is_fractional:bool=False,
    #                     is_decorate:bool=True,
    #                     use_additional_relax=False):
    #     """
    #     Plot plane diff.

    #     Args:
    #         is_fractional (bool): if True, z coords with fractional coordinate
    #         is_decorate (bool): if True, decorate figure
    #         use_additional_relax (bool): if True, output twinboundary structure
    #                                      becomes the final structure of
    #                                      additional_relax
    #     """
    #     def _get_data(bl):
    #         c_norm = self.cells['twinboundary_original'][0][2,2]
    #         distances = self.get_distances(
    #                 is_fractional=is_fractional,
    #                 use_additional_relax=bl)
    #         planes = self.get_planes_angles(
    #                 is_fractional=is_fractional,
    #                 use_additional_relax=bl)['planes']
    #         before_distances = distances['before'].copy()
    #         bulk_interval = before_distances[1]
    #         rlx_distances = distances['relax'].copy()
    #         z_coords = planes['before'].copy()
    #         z_coords.insert(0, z_coords[0] - before_distances[-1])
    #         z_coords.append(z_coords[-1] + before_distances[0])
    #         rlx_distances.insert(0, rlx_distances[-1])
    #         rlx_distances.append(rlx_distances[0])
    #         ydata = np.array(z_coords) + bulk_interval / 2
    #         if is_fractional:
    #             rlx_distances = np.array(rlx_distances) * c_norm
    #         return (rlx_distances, ydata, z_coords, bulk_interval)

    #     if use_additional_relax:
    #         datas = [ _get_data(bl) for bl in [False, True] ]
    #         labels = ['isif7', 'isif3']
    #     else:
    #         datas = [ _get_data(False) ]
    #         labels = ['isif7']

    #     fig = plt.figure(figsize=(8,13))
    #     ax = fig.add_subplot(111)
    #     for i in range(len(datas)):
    #         xdata, ydata, z_coords, bulk_interval = datas[i]
    #         line_chart(ax=ax,
    #                    xdata=xdata,
    #                    ydata=ydata,
    #                    xlabel='distance',
    #                    ylabel='z coords',
    #                    sort_by='y',
    #                    label=labels[i])

    #     if is_decorate:
    #         num = len(z_coords)
    #         tb_idx = [1, int(num/2), num-1]
    #         xmax = max([ max(data[0]) for data in datas ])
    #         xmin = min([ min(data[0]) for data in datas ])
    #         ymax = max([ max(data[1]) for data in datas ])
    #         ymin = min([ min(data[1]) for data in datas ])
    #         for idx in tb_idx:
    #             ax.hlines(z_coords[idx],
    #                       xmin=xmin-0.005,
    #                       xmax=xmax+0.005,
    #                       linestyle='--',
    #                       linewidth=1.5)
    #         yrange = ymax - ymin
    #         if is_fractional:
    #             c_norm = self.cells['twinboundary_original'][0][2,2]
    #             vline_x = bulk_interval * c_norm
    #         else:
    #             vline_x = bulk_interval
    #         ax.vlines(vline_x,
    #                   ymin=ymin-yrange*0.01,
    #                   ymax=ymax+yrange*0.01,
    #                   linestyle='--',
    #                   linewidth=0.5)
    #         ax.legend()

    # def plot_angle_diff(self,
    #                     is_fractional:bool=False,
    #                     is_decorate:bool=True,
    #                     use_additional_relax:bool=False):
    #     """
    #     Plot angle diff.

    #     Args:
    #         is_fractional (bool): if True, z coords with fractional coordinate
    #         is_decorate (bool): if True, decorate figure
    #         use_additional_relax (bool): if True, output twinboundary structure
    #                                      becomes the final structure of
    #                                      additional_relax
    #     """
    #     def _get_data(bl):
    #         distances = self.get_distances(
    #                 is_fractional=is_fractional,
    #                 use_additional_relax=bl)
    #         planes_angles = self.get_planes_angles(
    #                 is_fractional=is_fractional,
    #                 use_additional_relax=bl)
    #         planes = planes_angles['planes']
    #         before_distances = distances['before'].copy()
    #         z_coords = planes['before'].copy()
    #         z_coords.insert(0, z_coords[0] - before_distances[-1])
    #         z_coords.append(z_coords[-1] + before_distances[0])

    #         angles = planes_angles['angles']
    #         rlx_angles = list(angles['relax'])
    #         bulk_angle = angles['before'][1]
    #         rlx_angles.insert(0, rlx_angles[-1])
    #         rlx_angles.append(rlx_angles[1])
    #         ydata = np.array(z_coords)
    #         rlx_angles = np.array(rlx_angles)
    #         return (rlx_angles, ydata, z_coords, bulk_angle)

    #     if use_additional_relax:
    #         datas = [ _get_data(bl) for bl in [False, True] ]
    #         labels = ['isif7', 'isif3']
    #     else:
    #         datas = [ _get_data(False) ]
    #         labels = ['isif7']

    #     fig = plt.figure(figsize=(8,13))
    #     ax = fig.add_subplot(111)
    #     for i in range(len(datas)):
    #         xdata, ydata, z_coords, bulk_angle = datas[i]
    #         line_chart(ax=ax,
    #                    xdata=xdata,
    #                    ydata=ydata,
    #                    xlabel='angle',
    #                    ylabel='z coords',
    #                    sort_by='y',
    #                    label=labels[i])

    #     if is_decorate:
    #         num = len(z_coords)
    #         tb_idx = [1, int(num/2), num-1]
    #         xmax = max([ max(data[0]) for data in datas ])
    #         xmin = min([ min(data[0]) for data in datas ])
    #         ymax = max([ max(data[1]) for data in datas ])
    #         ymin = min([ min(data[1]) for data in datas ])
    #         for idx in tb_idx:
    #             ax.hlines(z_coords[idx],
    #                       xmin=xmin-0.005,
    #                       xmax=xmax+0.005,
    #                       linestyle='--',
    #                       linewidth=1.5)
    #         yrange = ymax - ymin
    #         vline_x = bulk_angle
    #         ax.vlines(vline_x,
    #                   ymin=ymin-yrange*0.01,
    #                   ymax=ymax+yrange*0.01,
    #                   linestyle='--',
    #                   linewidth=0.5)
    #         ax.legend()

    # def get_description(self):
    #     """
    #     Get description.
    #     """
    #     self._print_common_information()
    #     print_header('PKs')
    #     pprint(self.get_pks())
    #     print("\n\n")
    #     print_header('twinboudnary settings')
    #     pprint(self.twinboundary_settings)

    # def _get_shear_twinboundary_lattice(self,
    #                                     tb_lattice:np.array,
    #                                     shear_strain_ratio:float) -> np.array:
    #     """
    #     Get shear twinboudnary lattice.
    #     """
    #     lat = deepcopy(tb_lattice)
    #     e_b = lat[1] / np.linalg.norm(lat[1])
    #     shear_func = \
    #             self._twinpy.twinboundary.indices.get_shear_strain_function()
    #     lat[2] += np.linalg.norm(lat[2]) \
    #               * shear_func(self._twinpy.twinboundary.r) \
    #               * shear_strain_ratio \
    #               * e_b
    #     return lat

    # def get_shear_cell(self,
    #                    shear_strain_ratio:float,
    #                    is_standardize:bool=False) -> tuple:
    #     """
    #     Get shear introduced twinboundary cell.

    #     Args:
    #         shear_strain_ratio (float): shear strain ratio
    #         is_standardize (bool): if True, get standardized cell

    #     Returns:
    #         tuple: shear introduced cell
    #     """
    #     try:
    #         rlx_cell = self._cells['twinboundary_additional_relax_original']
    #         print("get cell (twinboundary_additional_relax_original cell)")
    #     except KeyError:
    #         print("could not find twinboundary_additional_relax cell")
    #         print("so get cell (twinboundary_relax_original cell)")
    #         rlx_cell = self._cells['twinboundary_relax_original']
    #     shear_lat = self._get_shear_twinboundary_lattice(
    #         tb_lattice=rlx_cell[0],
    #         shear_strain_ratio=shear_strain_ratio)
    #     shear_cell = (shear_lat, rlx_cell[1], rlx_cell[2])
    #     if is_standardize:
    #         std_cell = get_standardized_cell(
    #                 cell=shear_cell,
    #                 to_primitive=True,
    #                 no_idealize=False,
    #                 no_sort=True)
    #         return std_cell
    #     else:
    #         return shear_cell

    # def get_shear_relax_builder(self,
    #                             shear_strain_ratio:float):
    #     """
    #     Get relax builder for shear introduced relax twinboundary structure.

    #     Args:
    #         shear_strain_ratio (float): shear strain ratio
    #     """
    #     cell = self.get_shear_cell(
    #             shear_strain_ratio=shear_strain_ratio,
    #             is_standardize=False)  # in order to get rotation matrix
    #     std = StandardizeCell(cell=cell)
    #     std_cell = std.get_standardized_cell(to_primitive=True,
    #                                          no_idealize=False,
    #                                          no_sort=True)
    #     try:
    #         rlx_pk = self._additional_relax_pks[-1]
    #     except KeyError:
    #         rlx_pk = self.get_pks()['relax_pk']
    #     rlx_node = load_node(rlx_pk)
    #     builder = rlx_node.get_builder_restart()

    #     # fix kpoints
    #     mesh, offset = map(np.array, builder.kpoints.get_kpoints_mesh())
    #     orig_mesh = np.abs(np.dot(np.linalg.inv(
    #         self._standardize.transformation_matrix), mesh).astype(int))
    #     orig_offset = np.round(np.abs(np.dot(np.linalg.inv(
    #         std.transformation_matrix), offset)), decimals=2)
    #     std_mesh = np.abs(np.dot(std.transformation_matrix,
    #                              orig_mesh).astype(int))
    #     std_offset = np.round(np.abs(np.dot(std.transformation_matrix,
    #                                         orig_offset)), decimals=2)
    #     kpt = KpointsData()
    #     kpt.set_kpoints_mesh(std_mesh, offset=std_offset)
    #     builder.kpoints = kpt

    #     # fix structure
    #     builder.structure = get_aiida_structure(cell=std_cell)

    #     # fix relax conf
    #     builder.relax.convergence_max_iterations = Int(100)
    #     builder.relax.positions = Bool(True)
    #     builder.relax.shape = Bool(False)
    #     builder.relax.volume = Bool(False)
    #     builder.relax.convergence_positions = Float(1e-4)
    #     builder.relax.force_cutoff = \
    #             Float(AiidaRelaxWorkChain(node=rlx_node).get_max_force())
    #     builder.metadata.label = "tbr:{} rlx:{} shr:{} std:{}".format(
    #             self._pk, rlx_node.pk, shear_strain_ratio, True)
    #     builder.metadata.description = \
    #             "twinboundary_relax_pk:{} relax_pk:{} " \
    #             "shear_strain_ratio:{} standardize:{}".format(
    #                 self._pk, rlx_node.pk, shear_strain_ratio, True)
    #     return builder
