#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
import numpy as np
from pprint import pprint
import warnings
from matplotlib import pyplot as plt
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.orm import (load_node,
                       Node)
from twinpy.interfaces.aiida import (check_process_class,
                                     get_cell_from_aiida,
                                     _WorkChain,
                                     AiidaRelaxWorkChain)
from twinpy.structure.base import check_same_cells
from twinpy.structure.diff import get_structure_diff
from twinpy.api_twinpy import get_twinpy_from_cell
from twinpy.common.utils import print_header
from twinpy.lattice.lattice import Lattice
from twinpy.plot.base import line_chart


@with_dbenv()
class AiidaTwinBoudnaryRelaxWorkChain(_WorkChain):
    """
    TwinBoundaryRelax work chain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            relax_pk (int): relax pk
        """
        process_class = 'TwinBoundaryRelaxWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        self._twinboundary_settings = None
        self._structures = None
        self._structure_pks = None
        self._set_twinboundary()
        self._twinpy = None
        self._standardize = None
        self._set_twinpy()
        self._check_structures()
        self._additional_relax_pks = []

    def _set_twinboundary(self):
        """
        Set twinboundary from vasp.
        """
        parameters = self._node.called[-1].outputs.parameters.get_dict()
        aiida_hexagonal = self._node.inputs.structure
        aiida_tb = self._node.called[-1].outputs.twinboundary
        aiida_tb_original = self._node.called[-1].outputs.twinboundary_orig
        aiida_tb_relax = self._node.called[-2].outputs.relax__structure
        tb = get_cell_from_aiida(aiida_tb)
        tb_original = get_cell_from_aiida(aiida_tb_original)
        tb_relax = get_cell_from_aiida(aiida_tb_relax)

        round_cells = []
        for cell in [ tb, tb_original, tb_relax ]:
            round_lattice = np.round(cell[0], decimals=8)
            round_lattice = cell[0]
            round_atoms = np.round(cell[1], decimals=8) % 1
            round_cell = (round_lattice, round_atoms, cell[2])
            round_cells.append(round_cell)

        self._twinboundary_settings = parameters
        self._structures = {
                'hexagonal': get_cell_from_aiida(aiida_hexagonal),
                'twinboundary': round_cells[0],
                'twinboundary_original': round_cells[1],
                'twinboundary_relax': round_cells[2],
                }
        self._structure_pks = {
                'hexagonal_pk': aiida_hexagonal.pk,
                'twinboundary_pk': aiida_tb.pk,
                'twinboundary_original_pk': aiida_tb_original.pk,
                'twinboundary_relax_pk': aiida_tb_relax.pk,
                }

    @property
    def twinboundary_settings(self):
        """
        Twinboundary settings.
        """
        return self._twinboundary_settings

    @property
    def structures(self):
        """
        Twinboundary structures
        """
        return self._structures

    @property
    def structure_pks(self):
        """
        Twinboundary structure pks
        """
        return self._structure_pks

    def __get_relax_twinboudnary_original_frame(self, rlx_cell, std):
        """
        Todo:
            Future replace convert_to_original_frame in standardize.py
        """
        M_bar_p = rlx_cell[0].T
        x_p = rlx_cell[1].T
        P_c = std.conventional_to_primitive_matrix
        R = std.rotation_matrix
        P = std.transformation_matrix
        p = std.origin_shift.reshape(3,1)

        M_p = np.dot(np.linalg.inv(R), M_bar_p)
        M_s = np.dot(M_p, np.linalg.inv(P_c))
        M = np.dot(M_s, P)
        x_s = np.dot(P_c, x_p)
        x = np.round(np.dot(np.linalg.inv(P), x_s)
                       - np.dot(np.linalg.inv(P), p), decimals=8) % 1
        cell = (M.T, x.T, rlx_cell[2])
        return cell

    def _set_twinpy(self):
        """
        Set twinpy structure object and standardize object.
        """
        params = self._twinboundary_settings
        cell = get_cell_from_aiida(
                load_node(self._structure_pks['hexagonal_pk']))
        twinpy = get_twinpy_from_cell(
                cell=cell,
                twinmode=params['twinmode'])
        twinpy.set_twinboundary(
                layers=params['layers'],
                delta=params['delta'],
                twintype=params['twintype'],
                xshift=params['xshift'],
                yshift=params['yshift'],
                shear_strain_ratio=params['shear_strain_ratio'],
                )
        std = twinpy.get_twinboundary_standardize(
                get_lattice=params['get_lattice'],
                move_atoms_into_unitcell=params['move_atoms_into_unitcell'],
                )
        tb_relax_orig = self.__get_relax_twinboudnary_original_frame(
                rlx_cell=self._structures['twinboundary_relax'],
                std=std,
                )
        self._structures['twinboundary_relax_original'] = tb_relax_orig
        self._twinpy = twinpy
        self._standardize = std

    @property
    def twinpy(self):
        """
        Twinpy structure object.
        """
        return self._twinpy

    @property
    def standardize(self):
        """
        Stadardize object of twinpy original cell.
        """
        return self._standardize

    def _check_structures(self):
        """
        Check structures by reconstucting twinboundary.
        """
        params = self._twinboundary_settings
        tb_orig_cell = self._standardize.cell
        tb_std_cell = self._standardize.get_standardized_cell(
                to_primitive=params['to_primitive'],
                no_idealize=params['no_idealize'],
                symprec=params['symprec'],
                no_sort=params['no_sort'],
                get_sort_list=params['get_sort_list'],
                )
        check_same_cells(first_cell=self._structures['twinboundary_original'],
                         second_cell=tb_orig_cell)
        check_same_cells(first_cell=self._structures['twinboundary'],
                         second_cell=tb_std_cell)
        np.testing.assert_allclose(
                self._structures['twinboundary_original'][0],
                self._structures['twinboundary_relax_original'][0],
                atol=1e-6)

    def set_additional_relax(self, aiida_relax_pks:list):
        """
        Set additional_relax_pks in the case final structure of
        TwinBoundaryRelax WorkChain is further relaxed.

        Args:
            aiida_relaxes (list): list of relax pks

        Raises:
            RuntimeError: Input node is not RelaxWorkChain
            RuntimeError: Output structure and next input structure
                          does not match.
        """
        previous_rlx = self.get_pks()['relax_pk']
        structure_pk = self._structure_pks['twinboundary_relax_pk']
        aiida_relaxes = [ load_node(relax_pk) for relax_pk in aiida_relax_pks ]
        for aiida_relax in aiida_relaxes:
            if aiida_relax.process_class.get_name() != 'RelaxWorkChain':
                raise RuntimeError(
                        "Input node (pk={}) is not RelaxWorkChain".format(
                            aiida_relax.pk))
            if structure_pk == aiida_relax.inputs.structure.pk:
                if aiida_relax.process_state.value == 'finished':
                    structure_pk = aiida_relax.outputs.relax__structure.pk
                else:
                    warnings.warn("RelaxWorkChain (pk={}) state is {}".format(
                        aiida_relax.pk, aiida_relax.process_state.value))
                previous_rlx = aiida_relax.pk
            else:
                print("previous relax: pk={}".format(previous_rlx))
                print("next relax: pk={}".format(aiida_relax.pk))
                raise RuntimeError("Output structure and next input structure "
                                   "does not match.")
        self._additional_relax_pks = aiida_relax_pks

    @property
    def additional_relax_pks(self):
        """
        Additional relax pks.
        """
        return self._additional_relax_pks

    def get_pks(self):
        """
        Get pks.
        """
        relax_pk = self._node.called[-2].pk
        pks = self._structure_pks.copy()
        pks['relax_pk'] = relax_pk
        return pks

    def _get_additional_relax_final_cell(self):
        """
        Get additional relax final structure.
        """
        if self._additional_relax_pks == []:
            raise RuntimeError("additional_relax_pks is not set.")
        else:
            aiida_relax = AiidaRelaxWorkChain(
                    load_node(self._additional_relax_pks[-1]))
            final_cell = get_cell_from_aiida(load_node(
                aiida_relax.get_pks()['current_final_structure_pk']))
        return final_cell

    def get_diff(self, get_additional_relax:bool=False) -> dict:
        """
        Get diff between vasp input and output twinboundary structure.

        Args:
            get_additional_relax (bool): if True, output twinboundary structure
                                         becomes the final structure of
                                         additional_relax

        Returns:
            dict: diff between vasp input and output twinboundary structure

        Raises:
            AssertionError: lattice matrix is not identical
        """
        if get_additional_relax:
            final_cell = self._get_additional_relax_final_cell()
        else:
            final_cell = self._structures['twinboundary_relax']
        cells = (self._structures['twinboundary'],
                 final_cell)
        diff = get_structure_diff(cells=cells,
                                  base_index=0,
                                  include_base=False)
        if not get_additional_relax:
            np.testing.assert_allclose(
                    diff['lattice_diffs'][0],
                    np.zeros((3,3)),
                    atol=1e-8,
                    err_msg="lattice matrix is not identical")
        return diff

    def get_planes_angles(self,
                          is_fractional:bool=False,
                          get_additional_relax:bool=False) -> dict:
        """
        Get plane coords from lower plane to upper plane.
        Return list of z coordinates of original cell frame.

        Args:
            is_fractional (bool): if True, return with fractional coordinate
            get_additional_relax (bool): if True, output twinboundary structure
                                         becomes the final structure of
                                         additional_relax

        Returns:
            dict: plane coords of input and output twinboundary structures.
        """
        if not is_fractional and get_additional_relax:
            raise RuntimeError(
                    "is_fractional=False and get_additional_relax=True "
                    "is not allowed because in the case "
                    "get_additional_relax=True, c norm changes.")

        if get_additional_relax:
            final_cell = self._get_additional_relax_final_cell()
            orig_final_cell = \
                    self.__get_relax_twinboudnary_original_frame(
                            rlx_cell=final_cell,
                            std=self._standardize)
        else:
            orig_final_cell = self._structures['twinboundary_relax_original']
        cells = [self._structures['twinboundary_original'],
                 orig_final_cell]

        coords_list = []
        angles_list = []
        for cell in cells:
            lattice = Lattice(cell[0])
            atoms = cell[1]
            c_norm = lattice.abc[2]
            k_1 = np.array([0,0,1])
            sort_atoms = atoms[np.argsort(atoms[:,2])]

            # planes
            coords = [ lattice.dot(np.array(atom), k_1) / c_norm
                           for atom in sort_atoms ]
            num = len(coords)
            ave_coords = np.sum(
                    np.array(coords).reshape(int(num/2), 2), axis=1) / 2

            epsilon = 1e-9    # to remove '-0.0'
            if is_fractional:
                d = np.round(ave_coords+epsilon, decimals=8) / c_norm
            else:
                d = np.round(ave_coords+epsilon, decimals=8)
            coords_list.append(list(d))

            # angles
            sub_coords_orig = \
                    sort_atoms[[i for i in range(1,num,2)]] \
                        - sort_atoms[[i for i in range(0,num,2)]]
            sub_coords_plus = \
                    sort_atoms[[i for i in range(1,num,2)]]+np.array([0,1,0]) \
                        - sort_atoms[[i for i in range(0,num,2)]]
            sub_coords_minus = \
                    sort_atoms[[i for i in range(1,num,2)]]-np.array([0,1,0]) \
                        - sort_atoms[[i for i in range(0,num,2)]]
            coords = []
            for i in range(len(sub_coords_orig)):
                norm_orig = lattice.get_norm(sub_coords_orig[i])
                norm_plus = lattice.get_norm(sub_coords_plus[i])
                norm_minus = lattice.get_norm(sub_coords_minus[i])
                norms = [norm_orig, norm_plus, norm_minus]
                if min(norms) == norm_orig:
                    coords.append(sub_coords_orig[i])
                elif min(norms) == norm_plus:
                    coords.append(sub_coords_plus[i])
                else:
                    coords.append(sub_coords_minus[i])

            angles = [ lattice.get_angle(frac_coord_first=coord,
                                         frac_coord_second=np.array([0,1,0]),
                                         get_acute=True)
                       for coord in coords ]
            angles_list.append(np.array(angles))

        return {
                'planes': {'before': coords_list[0],
                           'relax': coords_list[1]},
                'angles': {'before': angles_list[0],
                           'relax': angles_list[1]},
                }

    def get_distances(self,
                      is_fractional:bool=False,
                      get_additional_relax:bool=False) -> dict:
        """
        Get distances from the result of 'get_planes'.

        Args:
            is_fractional (bool): if True, return with fractional coordinate
            get_additional_relax (bool): if True, output twinboundary structure
                                         becomes the final structure of
                                         additional_relax

        Returns:
            dict: distances between planes of input and output
                  twinboundary structures.
        """
        planes = self.get_planes_angles(
                is_fractional=is_fractional,
                get_additional_relax=get_additional_relax)['planes']
        lattice = Lattice(self._structures['twinboundary_original'][0])
        c_norm = lattice.abc[2]
        keys = ['before', 'relax']
        dic = {}
        for key in keys:
            coords = planes[key]
            if is_fractional:
                coords.append(1.)
            else:
                coords.append(c_norm)
            distances = [ coords[i+1] - coords[i]
                              for i in range(len(coords)-1) ]
            dic[key] = distances
        return dic

    def get_formation_energy(self, bulk_relax_pk:int):
        """
        Get formation energy. Unit [mJ/m^(-2)]

        Args:
            bulk_relax_pk: relax pk of bulk relax calculation
        """
        def __get_excess_energy():
            rlx_pks = [ self.get_pks()['relax_pk'], bulk_relax_pk ]
            energies = []
            natoms = []
            for i, rlx_pk in enumerate(rlx_pks):
                rlx = load_node(rlx_pk)
                cell = get_cell_from_aiida(rlx.inputs.structure)
                if i == 1:
                    check_same_cells(cell, self._structures['hexagonal'])
                energy = rlx.outputs.energies.get_array('energy_no_entropy')[0]
                energies.append(energy)
                natoms.append(len(cell[2]))
            excess_energy = energies[0] - energies[1] * (natoms[0] / natoms[1])
            return excess_energy

        eV = 1.6022 * 10 ** (-16)
        ang = 10 ** (-10)
        unit = eV / ang ** 2
        orig_lattice = self._structures['twinboundary_original'][0]
        A = orig_lattice[0,0] * orig_lattice[1,1]
        excess_energy = __get_excess_energy()
        formation_energy = np.array(excess_energy) / (2*A) * unit
        return formation_energy

    def plot_convergence(self):
        """
        Plot convergence.
        """
        relax_pk = self.get_pks()['relax_pk']
        aiida_relax = AiidaRelaxWorkChain(load_node(relax_pk))
        aiida_relax.set_additional_relax(self._additional_relax_pks)
        aiida_relax.plot_convergence()

    def plot_plane_diff(self,
                        is_fractional:bool=False,
                        is_decorate:bool=True):
        """
        Plot plane diff.

        Args:
            is_fractional (bool): if True, z coords with fractional coordinate
            is_decorate (bool): if True, decorate figure
        """
        def _get_data(get_additional_relax):
            c_norm = self.structures['twinboundary_original'][0][2,2]
            distances = self.get_distances(
                    is_fractional=is_fractional,
                    get_additional_relax=get_additional_relax)
            planes = self.get_planes_angles(
                    is_fractional=is_fractional,
                    get_additional_relax=get_additional_relax)['planes']
            before_distances = distances['before'].copy()
            bulk_interval = before_distances[1]
            rlx_distances = distances['relax'].copy()
            z_coords = planes['before'].copy()
            z_coords.insert(0, z_coords[0] - before_distances[-1])
            z_coords.append(z_coords[-1] + before_distances[0])
            rlx_distances.insert(0, rlx_distances[-1])
            rlx_distances.append(rlx_distances[0])
            ydata = np.array(z_coords) + bulk_interval / 2
            if is_fractional:
                rlx_distances = np.array(rlx_distances) * c_norm
            return (rlx_distances, ydata, z_coords, bulk_interval)

        if self.additional_relax_pks == []:
            datas = [ _get_data(False) ]
            labels = ['isif7']
        else:
            datas = [ _get_data(bl) for bl in [False, True] ]
            labels = ['isif7', 'isif3']

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)
        for i in range(len(datas)):
            xdata, ydata, z_coords, bulk_interval = datas[i]
            line_chart(ax=ax,
                       xdata=xdata,
                       ydata=ydata,
                       xlabel='distance',
                       ylabel='z coords',
                       sort_by='y',
                       label=labels[i])

        if is_decorate:
            num = len(z_coords)
            tb_idx = [1, int(num/2), num-1]
            xmax = max([ max(data[0]) for data in datas ])
            xmin = min([ min(data[0]) for data in datas ])
            ymax = max([ max(data[1]) for data in datas ])
            ymin = min([ min(data[1]) for data in datas ])
            for idx in tb_idx:
                ax.hlines(z_coords[idx],
                          xmin=xmin-0.005,
                          xmax=xmax+0.005,
                          linestyle='--',
                          linewidth=1.5)
            yrange = ymax - ymin
            if is_fractional:
                c_norm = self.structures['twinboundary_original'][0][2,2]
                vline_x = bulk_interval * c_norm
            else:
                vline_x = bulk_interval
            ax.vlines(vline_x,
                      ymin=ymin-yrange*0.01,
                      ymax=ymax+yrange*0.01,
                      linestyle='--',
                      linewidth=0.5)
            ax.legend()

    def plot_angle_diff(self,
                        is_fractional:bool=False,
                        is_decorate:bool=True):
        """
        Plot angle diff.

        Args:
            is_fractional (bool): if True, z coords with fractional coordinate
            is_decorate (bool): if True, decorate figure
        """
        def _get_data(get_additional_relax):
            distances = self.get_distances(
                    is_fractional=is_fractional,
                    get_additional_relax=get_additional_relax)
            planes_angles = self.get_planes_angles(
                    is_fractional=is_fractional,
                    get_additional_relax=get_additional_relax)
            planes = planes_angles['planes']
            before_distances = distances['before'].copy()
            z_coords = planes['before'].copy()
            z_coords.insert(0, z_coords[0] - before_distances[-1])
            z_coords.append(z_coords[-1] + before_distances[0])

            angles = planes_angles['angles']
            rlx_angles = list(angles['relax'])
            bulk_angle = angles['before'][1]
            rlx_angles.insert(0, rlx_angles[-1])
            rlx_angles.append(rlx_angles[1])
            ydata = np.array(z_coords)
            rlx_angles = np.array(rlx_angles)
            return (rlx_angles, ydata, z_coords, bulk_angle)

        if self.additional_relax_pks == []:
            datas = [ _get_data(False) ]
            labels = ['isif7']
        else:
            datas = [ _get_data(bl) for bl in [False, True] ]
            labels = ['isif7', 'isif3']

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)
        for i in range(len(datas)):
            xdata, ydata, z_coords, bulk_angle = datas[i]
            line_chart(ax=ax,
                       xdata=xdata,
                       ydata=ydata,
                       xlabel='angle',
                       ylabel='z coords',
                       sort_by='y',
                       label=labels[i])

        if is_decorate:
            num = len(z_coords)
            tb_idx = [1, int(num/2), num-1]
            xmax = max([ max(data[0]) for data in datas ])
            xmin = min([ min(data[0]) for data in datas ])
            ymax = max([ max(data[1]) for data in datas ])
            ymin = min([ min(data[1]) for data in datas ])
            for idx in tb_idx:
                ax.hlines(z_coords[idx],
                          xmin=xmin-0.005,
                          xmax=xmax+0.005,
                          linestyle='--',
                          linewidth=1.5)
            yrange = ymax - ymin
            vline_x = bulk_angle
            ax.vlines(vline_x,
                      ymin=ymin-yrange*0.01,
                      ymax=ymax+yrange*0.01,
                      linestyle='--',
                      linewidth=0.5)
            ax.legend()

    def get_description(self):
        """
        Get description.
        """
        self._print_common_information()
        print_header('PKs')
        pprint(self.get_pks())
        print("\n\n")
        print_header('twinboudnary settings')
        pprint(self.twinboundary_settings)
