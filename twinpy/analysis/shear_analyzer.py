#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analize shear calculation.
"""
import numpy as np
from phonopy.phonon.band_structure import (get_band_qpoints_and_path_connections,
                                           BandStructure)
from twinpy.structure.base import check_same_cells
from twinpy.structure.diff import get_structure_diff
from twinpy.structure.standardize import StandardizeCell
from twinpy.plot.base import get_plot_properties_for_trajectory


class ShearAnalyzer():
    """
    Analize shear result.
    """

    def __init__(
           self,
           original_cells:list,
           input_cells:list,
           relax_cells:list,
           ):
        """
        Args:
            original_cells (list): primitivie original cells, which is output
                               cells of ShearStructure class
            input_cells (list): input cells for vasp
            relax_cells (list): relax cells of vasp

        Raises:
            RuntimeError: The number of atoms changes between original cells
                          and input cells, which is not supported.

        Todo:
            Currently not supported the case the number of original_cells
            and input_cells changes because it is difficult to construct
            the relax cells in the original frame. But future fix this
            problem. One solution is to make attribute
            'self._original_primitive' which contains two atoms
            in the unit cell and original basis.
        """
        def __check_number_of_atoms_not_changed(original_cells,
                                                input_cells):
            for original_cell, input_cell in zip(original_cells, input_cells):
                if not len(original_cell) == len(input_cell):
                    raise RuntimeError(
                            "The number of atoms changes between "
                            "original cells and input cells, "
                            "which is not supported.")

        __check_number_of_atoms_not_changed(original_cells, input_cells)
        self._original_cells = original_cells
        self._input_cells = input_cells
        self._relax_cells = None
        self._set_relax_cells(input_cells, relax_cells)
        self._standardizes = None
        self._set_standardizes()
        self._relax_cells_in_original_frame = None
        self._set_relax_cells_in_original_frame()
        self._phonons = None

    @property
    def original_cells(self):
        """
        Original cells, which are output of ShearStructure.
        """
        return self._original_cells

    @property
    def input_cells(self):
        """
        Relax input cells.
        """
        return self._input_cells

    def _set_relax_cells(self, input_cells, relax_cells):
        """
        Check lattice does not change in relaxation and
        set relax_cells.
        """
        for input_cell, relax_cell in zip(input_cells, relax_cells):
            np.testing.assert_allclose(input_cell[0], relax_cell[0],
                                       atol=1e-5)
        self._relax_cells = relax_cells

    @property
    def relax_cells(self):
        """
        Relax output cells.
        """
        return self._relax_cells

    def _set_standardizes(self):
        """
        Set standardizes.
        """
        to_primitive = True
        no_idealize = False
        symprec = 1e-5
        no_sort = False
        get_sort_list = False

        standardizes = [ StandardizeCell(cell)
                             for cell in self._original_cells ]

        for i, standardize in enumerate(standardizes):
            std_cell = standardize.get_standardized_cell(
                    to_primitive=to_primitive,
                    no_idealize=no_idealize,
                    symprec=symprec,
                    no_sort=no_sort,
                    get_sort_list=get_sort_list,
                    )
            cells_are_same = check_same_cells(self._input_cells[i],
                                              std_cell)
            if not cells_are_same:
                raise RuntimeError("standardized cell do not match "
                                   "with input cell")

        self._standardizes = standardizes

    @property
    def standardizes(self):
        """
        Standardizes.
        """
        return self._standardizes

    def _set_relax_cells_in_original_frame(self) -> list:
        """
        Set relax cells in original frame which is not
        conventional and its angles are close to (90., 90., 120.).

        Returns:
            list: relax cells in original frame

        Note:
            For variable definitions in this definition,
            see Eq.(1.8) and (1.17) in Crystal Structure documention.
            Note that by crystal body rotation, fractional
            coordinate of atom positions are not changed.
        """
        def __get_relax_atoms_in_original_frame(prim_atoms,
                                                conv_to_prim_matrix,
                                                transformation_matrix,
                                                origin_shift):
            X_p = prim_atoms.T
            P_c = conv_to_prim_matrix
            P = transformation_matrix
            p = origin_shift.reshape(3,1)

            X_s = np.dot(P_c, X_p)
            X = np.dot(np.linalg.inv(P), X_s) \
                    - np.dot(np.linalg.inv(P), p)
            orig_atoms = np.round(X.T, decimals=8) % 1.
            return orig_atoms

        relax_orig_cells = []
        for i in range(len(self._relax_cells)):
            lattice = self._original_cells[i][0]
            conv_to_prim_matrix = \
                self._standardizes[i].conventional_to_primitive_matrix
            transformation_matrix = \
                self._standardizes[i].transformation_matrix
            origin_shift = self._standardizes[i].origin_shift
            scaled_positions = __get_relax_atoms_in_original_frame(
                    prim_atoms=self._relax_cells[i][1],
                    conv_to_prim_matrix=conv_to_prim_matrix,
                    transformation_matrix=transformation_matrix,
                    origin_shift=origin_shift,
                    )
            symbols = self._original_cells[i][2]
            relax_orig_cell = (lattice, scaled_positions, symbols)
            relax_orig_cells.append(relax_orig_cell)

        self._relax_cells_in_original_frame = relax_orig_cells

    @property
    def relax_cells_in_original_frame(self):
        """
        Relax cells in original frame
        """
        return self._relax_cells_in_original_frame

    def set_phonons(self, phonons):
        """
        Set phonons.
        """
        self._phonons = phonons

    @property
    def phonons(self):
        """
        Phonons.
        """
        return self._phonons

    def get_relax_diffs(self):
        """
        Get structure diffs between input and relax cells
        IN ORIGINAL FRAME.
        """
        diffs = []
        for input_cell, relax_cell in \
                zip(self._original_cells, self._relax_cells_in_original_frame):
            cells = (input_cell, relax_cell)
            diff = get_structure_diff(cells=cells,
                                      base_index=0,
                                      include_base=False)
            diffs.append(diff)

        return diffs

    def get_shear_diffs(self):
        """
        Get structure diffs between original relax and sheared relax cells
        IN ORIGINAL FRAME.
        """
        cells = self._relax_cells_in_original_frame
        diffs = get_structure_diff(cells=cells,
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
        def _get_band_paths_cart(base_bps, base_lat, rotation):
            bps_orig_cart = []
            for bp in base_bps:
                qmat = np.array(bp).T
                rec_mat = np.linalg.inv(base_lat)
                qmat_cart = np.dot(rec_mat, qmat)
                qmat_orig_cart = np.dot(np.linalg.inv(rotation), qmat_cart)
                bps_orig_cart.append(qmat_orig_cart.T)
            return bps_orig_cart

        def _get_rotated_band_paths(bps_cart, lat, rotation):
            rot_bps = []
            for bp in bps_cart:
                qmat_cart = np.array(bp).T
                rec_mat = np.linalg.inv(lat)
                qmat_rot_cart = np.dot(rotation, qmat_cart)
                qmat_rot = np.dot(np.linalg.inv(rec_mat), qmat_rot_cart)
                rot_bps.append(qmat_rot.T)
            return rot_bps

        bps_orig_cart = _get_band_paths_cart(
                base_bps=base_band_paths,
                base_lat=self._phonons[0].get_primitive().get_cell(),
                rotation=self._standardizes[0].rotation_matrix)
        band_paths_for_all = []
        for phonon, standardize in zip(self._phonons, self._standardizes):
            lattice = phonon.get_primitive().get_cell()
            rotation = standardize.rotation_matrix
            rot_bps = _get_rotated_band_paths(
                    bps_cart=bps_orig_cart,
                    lat=lattice,
                    rotation=rotation)
            band_paths_for_all.append(rot_bps)

        return band_paths_for_all

    def get_band_paths_by_seekpath(self):
        """
        Get band paths for all shear cells from band paths for first cell
        by using seekpath toward first cell.
        """
        import seekpath

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
        for i in range(len(self._phonons)):
            phonon = self._phonons[i]
            band_paths = band_paths_for_all[i]
            lattice = phonon.get_primitive().get_cell()
            rec_lattice = np.linalg.inv(lattice)
            qpoints, connections = get_band_qpoints_and_path_connections(
                    band_paths=band_paths,
                    npoints=npoints,
                    rec_lattice=rec_lattice)
            if i == 0:
                lbs = labels
            else:
                lbs = None
            band_structure = BandStructure(
                    paths=qpoints,
                    dynamical_matrix=phonon.get_dynamical_matrix(),
                    with_eigenvectors=with_eigenvectors,
                    is_band_connection=False,
                    group_velocity=None,
                    path_connections=connections,
                    labels=lbs,
                    is_legacy_plot=False)
            band_structures.append(band_structure)
        return band_structures
