#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analize phonopy calculation.
"""
from copy import deepcopy
import warnings
import numpy as np
from phonopy import Phonopy
from phonopy.phonon.band_structure import (
        get_band_qpoints_and_path_connections,
        BandStructure)
from twinpy.interfaces.phonopy import get_cell_from_phonopy_structure
from twinpy.common.kpoints import Kpoints
from twinpy.common.band_path import (get_seekpath,
                                     get_labels_band_paths_from_seekpath)
from twinpy.structure.base import check_same_cells
from twinpy.analysis.relax_analyzer import RelaxAnalyzer
from twinpy.plot.band_structure import BandPlot
from twinpy.plot.dos import TotalDosPlot


class PhononAnalyzer():
    """
    Analize phonopy result.
    """

    def __init__(
           self,
           phonon:Phonopy,
           relax_analyzer:RelaxAnalyzer=None,
           ):
        """
        Args:
            phonon: Phonopy class object.
        """
        self._phonon = phonon
        self._symmetrize_fc()
        self._relax_analyzer = None
        if relax_analyzer is not None:
            self.set_relax_analyzer(relax_analyzer)
            if relax_analyzer.no_standardize:
                self._rotation_matrix = np.identity(3)
            else:
                if relax_analyzer.standardize:
                    self._rotation_matrix = \
                            self.relax_analyzer.standardize.rotation_matrix
                else:
                    warnings.warn("standardize is not set in relax_analyzer."
                                  +" set np.identity(3) automatically.")
                    self._rotation_matrix = np.identity(3)
        self._primitive_cell = None
        self._set_primitive_cell()
        self._reciprocal_lattice = None
        self._original_reciprocal_lattice = None
        self._set_reciprocal_lattice()

    def _symmetrize_fc(self):
        """
        Symmetrize force constants.
        """
        self._phonon.symmetrize_force_constants()

    def _set_primitive_cell(self):
        """
        Set primitive cell.
        """
        self._primitive_cell = get_cell_from_phonopy_structure(
                self._phonon.get_primitive())

    @property
    def primitive_cell(self):
        """
        Primitive cell.

        Note:
        Remind this is user defined primitive cell.
        Therefore, this is not necessarily true primitive cell.
        """
        return self._primitive_cell

    def _set_reciprocal_lattice(self):
        """
        Set reciprocal lattice.
        """
        self._reciprocal_lattice = np.linalg.inv(self._primitive_cell[0]).T
        if self._relax_analyzer is not None:
            self._original_reciprocal_lattice = np.linalg.inv(
                    self._relax_analyzer.original_cell[0])

    @property
    def reciprocal_lattice(self):
        """
        Reciprocal lattice.
        """
        return self._reciprocal_lattice

    @property
    def original_reciprocal_lattice(self):
        """
        Reciprocal lattice.
        """
        return self._reciprocal_lattice

    @property
    def phonon(self):
        """
        Phonopy class object.
        """
        return self._phonon

    @property
    def relax_analyzer(self):
        """
        RelaxAnalyzer class object.
        """
        return self._relax_analyzer

    def set_relax_analyzer(self, relax_analyzer):
        """
        Set relax analyzer.

        Args:
            relax_analyzer: RelaxAnalyzer class object.
        """
        relax_cell = relax_analyzer.final_cell
        unitcell = get_cell_from_phonopy_structure(
                self._phonon.get_primitive())
        check_same_cells(first_cell=relax_cell,
                         second_cell=unitcell,
                         raise_error=True)
        self._relax_analyzer = relax_analyzer

    @property
    def rotation_matrix(self):
        """
        Rotation matrix.
        """
        return self._rotation_matrix

    def export_phonon(self, filename:str=None):
        """
        Export phonopy object to yaml file.

        Args:
            filename (str): Output filename. If None, filename becomes
                            pk<number>_phonopy.yaml.
        """
        if filename is None:
            filename = "phonopy_params.yaml"
        self._phonon.save(filename)

    def get_reciprocal_high_symmetry_points(self):
        """
        Get reciprocal high symmetry points.

        Returns:
            dict: Contains point coordinates for primitive and original.
        """
        point_coords = get_seekpath(cell=self._primitive_cell)['point_coords']
        point_coords_orig = {}
        for key in point_coords.keys():
            orig_qpt = self.get_qpoints_from_primitive_to_original(
                               qpoints=point_coords[key])
            point_coords_orig[key] = list(orig_qpt)

        return {'primitive': point_coords, 'original': point_coords_orig}

    def get_qpoints_from_original_to_primitive(self,
                                               qpoints:np.array,
                                               input_is_cart:bool=False,
                                               output_is_cart:bool=False):
        """
        Get qpoints in original frame to primitive frame.

        Args:
            qpoints (np.array): Qpoints.
            input_is_cart (bool): Whether input qpoints are cartesian.
            output_is_cart (bool): Whether output qpoints are cartesian.
        """
        recip_mat = self._reciprocal_lattice.T
        orig_recip_mat = self._original_reciprocal_lattice.T
        rotation_matrix = self._rotation_matrix
        if input_is_cart:
            orig_qmat_cart = np.array(qpoints).T
        else:
            orig_qmat_cart = np.dot(orig_recip_mat, np.array(qpoints).T)
        qmat_cart = np.dot(rotation_matrix, orig_qmat_cart)
        if output_is_cart:
            output_qpoints = qmat_cart.T
        else:
            output_qpoints = np.dot(np.linalg.inv(recip_mat),
                                    qmat_cart).T
        return output_qpoints

    def get_qpoints_from_primitive_to_original(self,
                                               qpoints:np.array,
                                               input_is_cart:bool=False,
                                               output_is_cart:bool=False):
        """
        Get qpoints in primitive frame to original frame.

        Args:
            qpoints (np.array): Qpoints.
            input_is_cart (bool): Whether input qpoints are cartesian.
            output_is_cart (bool): Whether output qpoints are cartesian.
        """
        recip_mat = self._reciprocal_lattice.T
        orig_recip_mat = self._original_reciprocal_lattice.T
        rotation_matrix = self._rotation_matrix
        if input_is_cart:
            qmat_cart = np.array(qpoints).T
        else:
            qmat_cart = np.dot(recip_mat, np.array(qpoints).T)
        qmat_orig_cart = np.dot(np.linalg.inv(rotation_matrix), qmat_cart)

        if output_is_cart:
            output_qpoints = qmat_orig_cart.T
        else:
            output_qpoints = np.dot(np.linalg.inv(orig_recip_mat),
                                    qmat_orig_cart).T
        return output_qpoints

    def get_band_paths_from_original_to_primitive(self,
                                                  band_paths,
                                                  input_is_cart:bool=False,
                                                  output_is_cart:bool=False):
        """
        Get band paths in original frame to primitive frame.

        Args:
            band_paths (list): Band paths.
            input_is_cart (bool): Whether input qpoints are cartesian.
            output_is_cart (bool): Whether output qpoints are cartesian.
        """
        primitive_band_paths = []
        for band_path in band_paths:
            primitive_band_path = \
                    self.get_qpoints_from_original_to_primitive(
                        qpoints=band_path,
                        input_is_cart=input_is_cart,
                        output_is_cart=output_is_cart)
            primitive_band_paths.append(primitive_band_path)

        return primitive_band_paths

    def get_band_paths_from_primitive_to_original(self,
                                                  band_paths,
                                                  input_is_cart:bool=False,
                                                  output_is_cart:bool=False):
        """
        Get band paths in primitive frame to original frame.

        Args:
            band_paths (list): Band paths.
            input_is_cart (bool): Whether input qpoints are cartesian.
            output_is_cart (bool): Whether output qpoints are cartesian.
        """
        original_band_paths = []
        for band_path in band_paths:
            original_band_path = \
                    self.get_qpoints_from_primitive_to_original(
                        qpoints=band_path,
                        input_is_cart=input_is_cart,
                        output_is_cart=output_is_cart)
            original_band_paths.append(original_band_path)

        return original_band_paths

    def get_band_structure(self,
                           band_paths:list,
                           labels:list=None,
                           npoints:int=51,
                           with_eigenvectors:bool=False,
                           use_reciprocal_lattice:bool=True):
        """
        Get BandStructure class object.

        Args:
            band_paths (list): Band paths.
            labels (list): Band labels.
            npoints (int): The number of sampling points.
            with_eigenvectors (bool): If True, calculte eigenvectors.
        """
        if use_reciprocal_lattice:
            rec_lattice = self._reciprocal_lattice
        else:
            rec_lattice = None

        qpoints, connections = get_band_qpoints_and_path_connections(
                band_paths=band_paths,
                npoints=npoints,
                rec_lattice=rec_lattice)
        band_structure = BandStructure(
                paths=qpoints,
                dynamical_matrix=self._phonon.get_dynamical_matrix(),
                with_eigenvectors=with_eigenvectors,
                is_band_connection=False,
                group_velocity=None,
                path_connections=connections,
                labels=labels,
                is_legacy_plot=False)

        return band_structure

    def run_mesh(self,
                 interval:float=0.1,
                 is_store:bool=True,
                 get_phonon:bool=False,
                 dry_run:bool=False,
                 is_eigenvectors:bool=False,
                 is_gamma_center:bool=True,
                 verbose:bool=True):
        """
        Run mesh for both hexagonal and twinboundary phonon.

        Args:
            interval (float): mesh interval
            is_store (bool): If True, result is stored in self._phonon.
            get_phonon (bool): If True, return Phonopy class obejct,
                               which is independent from self._phonon.
            dry_run (bool): If True, show sampling mesh information
                            and not run.

        Todo:
            Fix get_phonon because currenly cannot get thermal_displacement_matrices for gottern phonon object.
        """
        kpt = Kpoints(lattice=self._primitive_cell[0])
        mesh = kpt.get_mesh_from_interval(interval=interval)
        if dry_run:
            print("# interval: {}".format(interval))
            print("# sampling mesh: {}".format(mesh))
            print("# dry_run is evoked.")

        else:
            if verbose:
                print("run mesh with {}".format(mesh))
            if is_store:
                _phonon = self._phonon
            else:
                _phonon = deepcopy(self._phonon)
            _phonon.set_mesh(
                    mesh=mesh,
                    shift=None,
                    is_time_reversal=True,
                    is_mesh_symmetry=False,  # necessary for calc ellipsoid
                    is_eigenvectors=is_eigenvectors,
                    is_gamma_center=is_gamma_center,
                    run_immediately=True)

        if get_phonon:
            return _phonon

    def get_total_dos(self,
                      is_store:bool=True,
                      sigma=None,
                      freq_min=None,
                      freq_max=None,
                      freq_pitch=None,
                      use_tetrahedron_method=True):
        """
        Get total dos.

        Args:
            is_store (bool): If True, result is stored in self._phonon.total_dos
        """
        if is_store:
            _phonon = self._phonon
        else:
            _phonon = deepcopy(self._phonon)

        if use_tetrahedron_method and sigma is not None:
            raise RuntimeError("It is prohibited to set use_tetrahedron_method=True and sigma=<float_val>")

        _phonon.run_total_dos(sigma=sigma,
                              freq_min=freq_min,
                              freq_max=freq_max,
                              freq_pitch=freq_pitch,
                              use_tetrahedron_method=use_tetrahedron_method)

        return _phonon.total_dos

    def get_projected_dos(self,
                        is_store:bool=True,
                        sigma=None,
                        freq_min=None,
                        freq_max=None,
                        freq_pitch=None,
                        use_tetrahedron_method=True,
                        direction=None,
                        xyz_projection=None):
        """
        Get projected dos.

        Args:
            is_store (bool): If True, result is stored in self._phonon.projected_dos
        """
        if use_tetrahedron_method and sigma is not None:
            raise RuntimeError("It is prohibited to set use_tetrahedron_method=True and sigma=<float_val>")

        if is_store:
            _phonon = self._phonon
        else:
            _phonon = deepcopy(self._phonon)

        _phonon.run_projected_dos(sigma=sigma,
                                  freq_min=freq_min,
                                  freq_max=freq_max,
                                  freq_pitch=freq_pitch,
                                  use_tetrahedron_method=use_tetrahedron_method,
                                  direction=direction,
                                  xyz_projection=xyz_projection)

        return _phonon.projected_dos

    def get_thermal_displacement_matrices(self,
                                          t_step:int=100,
                                          t_max:int=1000,
                                          t_min:int=0,
                                          with_original_cart:bool=True,
                                          def_cif:bool=False,
                                          ):
        """
        Get ThermalDisplacementMatrices object for
        both hexagonal and twinboundary.

        Args:
            t_step (int): temperature interval
            t_max (int): max temperature
            t_min (int): minimum temperature
            with_original_cart (bool): if True, use twinboundary
                                       original frame
            def_cif (bool): if True, use cif definition

        Todo:
            I do not know how to rotate 4d array (temp, atoms, 3, 3).
        """
        phonon = self._phonon
        inv_rotation_matrix = self._rotation_matrix.T

        phonon.set_thermal_displacement_matrices(
            t_step=t_step,
            t_max=t_max,
            t_min=t_min,
            freq_min=None,
            freq_max=None,
            t_cif=None)
        tdm = phonon.thermal_displacement_matrices
        if def_cif:
            _matrix = tdm.thermal_displacement_matrices_cif
        else:
            _matrix = tdm.thermal_displacement_matrices

        if with_original_cart:
            shape = _matrix.shape
            lst = []
            for i in range(shape[0]):
                atom_lst = []
                for j in range(shape[1]):
                    mat = np.dot(inv_rotation_matrix,
                                 np.dot(_matrix[i,j],
                                        inv_rotation_matrix.T))
                    atom_lst.append(mat)
                lst.append(atom_lst)
            matrix = np.array(lst)
        else:
            matrix = _matrix

        return matrix

    def plot_band_dos_auto(self, figsize=(8,6)):
        """
        Plot band structure and dos using seekpath.
        """
        from matplotlib import pyplot as plt
        self.run_mesh()
        ds = self.get_total_dos()
        dp = TotalDosPlot(ds, flip_xy=True)
        labels, band_paths = \
            get_labels_band_paths_from_seekpath(self.primitive_cell)
        bs = self.get_band_structure(band_paths=band_paths, labels=labels)
        bp = BandPlot(band_structure=bs)
        bp.plot_band_structure(dosplot=dp, figsize=figsize)
        plt.show()
