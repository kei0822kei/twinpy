#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analize shear calculation.
"""
from matplotlib import pyplot as plt
from twinpy.structure.shear import ShearStructure
from twinpy.structure.diff import get_structure_diff
from twinpy.structure.bonding import _get_atomic_environment
from twinpy.plot.twinboundary import plot_plane, plot_angle, plot_pair_distance
from twinpy.plot.relax import plot_atom_diff
from twinpy.file_io import write_poscar


class _BaseShearAnalyzer():
    """
    Base for ShearAnalyzer and TwinBoundaryShearAnalyzer.
    """

    def __init__(self,
                 relax_analyzers:list=None,
                 phonon_analyzers:list=None,
                 ):
        self._relax_analyzers = None
        self._phonon_analyzers = None
        self._set_analyzers(relax_analyzers,
                            phonon_analyzers)

    def _set_analyzers(self, relax_analyzers, phonon_analyzers):
        """
        Set analyzer.
        """
        if [relax_analyzers, phonon_analyzers] == [None, None]:
            raise RuntimeError("Both relax_analyzers and "
                               "phonon_analyzers do not set.")

        if phonon_analyzers is None:
            self._relax_analyzers = relax_analyzers
        else:
            self._relax_analyzers = \
                    [ phonon_analyzer.relax_analyzer
                          for phonon_analyzer in phonon_analyzers]
            self._phonon_analyzers = phonon_analyzers

    @property
    def relax_analyzers(self):
        """
        List of relax analyzers.
        """
        return self._relax_analyzers

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
        relax_cells_original_frame = \
                [ relax_analyzer.final_cell_in_original_frame
                      for relax_analyzer in self._relax_analyzers ]
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
                            with_eigenvectors:bool=False,
                            use_reciprocal_lattice:bool=True) -> list:
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
            # if i == 0:
            #     lbs = labels
            # else:
            #     lbs = None
            band_structure = phonon_analyzer.get_band_structure(
                    band_paths=band_paths_for_all[i],
                    # labels=lbs,
                    labels=labels,
                    npoints=npoints,
                    with_eigenvectors=with_eigenvectors,
                    use_reciprocal_lattice=use_reciprocal_lattice,
                    )
            band_structures.append(band_structure)

        return band_structures

    def run_mesh(self,
                 interval:float=0.1,
                 is_store:bool=True,
                 is_gamma_center:bool=True,
                 dry_run:bool=False,
                 is_eigenvectors:bool=False,
                 verbose:bool=True):
        """
        Run mesh.

        Args:
            interval (float): mesh interval
            is_store (bool): If True, result is stored in self._phonon.
            dry_run (bool): If True, show sampling mesh information
                            and not run.
        """
        for phonon in self._phonon_analyzers:
            phonon.run_mesh(interval=interval,
                            is_store=is_store,
                            dry_run=dry_run,
                            is_eigenvectors=is_eigenvectors,
                            is_gamma_center=is_gamma_center,
                            verbose=True)

    def get_total_doses(self,
                        is_store:bool=True,
                        sigma=None,
                        freq_min=None,
                        freq_max=None,
                        freq_pitch=None,
                        use_tetrahedron_method=True):
        """
        Get total doses.
        """
        tdoses = []
        for phonon in self._phonon_analyzers:
            tdos = phonon.get_total_dos(
                    is_store=is_store,
                    sigma=sigma,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    freq_pitch=freq_pitch,
                    use_tetrahedron_method=use_tetrahedron_method)
            tdoses.append(tdos)

        return tdoses

    def get_projected_doses(self,
                            is_store:bool=True,
                            sigma=None,
                            freq_min=None,
                            freq_max=None,
                            freq_pitch=None,
                            use_tetrahedron_method=True,
                            direction=None,
                            xyz_projection=None):
        """
        Get projected doses.
        """
        pdoses = []
        for phonon in self._phonon_analyzers:
            pdos = phonon.get_projected_dos(
                    is_store=is_store,
                    sigma=sigma,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    freq_pitch=freq_pitch,
                    use_tetrahedron_method=use_tetrahedron_method,
                    direction=direction,
                    xyz_projection=xyz_projection)
            pdoses.append(pdos)

        return pdoses


class ShearAnalyzer(_BaseShearAnalyzer):
    """
    Analize shear result.
    """

    def __init__(
           self,
           shear_structure:ShearStructure,
           shear_strain_ratios:list,
           phonon_analyzers:list,
           ):
        """
        Init.

        Args:
            shear_structure: ShearStructure class object.
            shear_strain_ratios: List of shear strain ratios.
            phonon_analyzer List of PhononAnalyzer class object.

        Todo:
            Currently not supported the case the number of original_cells
            and input_cells changes because it is difficult to construct
            the relax cells in the original frame. But future fix this
            problem. One solution is to make attribute
            'self._original_primitive' which contains two atoms
            in the unit cell and original basis.
            Twinboundary shaer structure also use this class.
            If this is inconvenient, I have to create
            _BaseShaerAnalyzer, ShearAnalyzer TwinBoundaryShearAnalyzer
            classes separately.
        """
        super().__init__(phonon_analyzers=phonon_analyzers)
        self._shear_strain_ratios = shear_strain_ratios
        self._shear_structure = shear_structure

    @property
    def shear_structure(self):
        """
        Shear structure.
        """
        return self._shear_structure

    @property
    def shear_strain_ratios(self):
        """
        Shear strain ratios.
        """
        return self._shear_strain_ratios


class TwinBoundaryShearAnalyzer(_BaseShearAnalyzer):
    """
    Analize twinboundary shear result.
    """

    def __init__(
           self,
           shear_strain_ratios:list,
           layer_indices:list,
           relax_analyzers:list=None,
           phonon_analyzers:list=None,
           ):
        """
        Init.

        Args:
            phonon_analyzers (list): List of PhononAnalyzer class object.
        """
        super().__init__(
                relax_analyzers=relax_analyzers,
                phonon_analyzers=phonon_analyzers)
        self._shear_strain_ratios = shear_strain_ratios
        self._layer_indices = layer_indices

    @property
    def layer_indices(self):
        """
        Layer indices.
        """
        return self._layer_indices

    @property
    def shear_strain_ratios(self):
        """
        Shear structure.
        """
        return self._shear_strain_ratios

    def get_atomic_environment(self) -> list:
        """
        Get plane coords from lower plane to upper plane.
        Return list of z coordinates of original cell frame.
        Plane coordinates (z coordinates) are fractional.
        """
        orig_cells = self.get_final_cells_in_original_frame()
        envs = [ _get_atomic_environment(cell, self._layer_indices)
                     for cell in orig_cells ]
        return envs

    def get_final_cells_in_original_frame(self) -> list:
        """
        Get final cells in original frame.
        """
        orig_cells = [ relax_analyzer.final_cell_in_original_frame
                           for relax_analyzer in self._relax_analyzers ]
        return orig_cells

    def plot_plane_diff(self):
        """
        Plot plane diff.
        """
        envs = self.get_atomic_environment()

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)

        for i, ratio in enumerate(self._shear_strain_ratios):
            if i == len(envs)-1:
                decorate = True
            else:
                decorate = False
            label = "shear ratio: %1.2f" % ratio
            plot_plane(ax,
                       distances=envs[i][1],
                       z_coords=envs[i][0],
                       label=label,
                       decorate=decorate,
                       )

        return fig

    def plot_angle_diff(self):
        """
        Plot angle diff.
        """
        envs = self.get_atomic_environment()

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)

        for i, ratio in enumerate(self._shear_strain_ratios):
            if i == len(envs)-1:
                decorate = True
            else:
                decorate = False
            label = "shear ratio: %1.2f" % ratio
            plot_angle(ax,
                       z_coords=envs[i][0],
                       angles=envs[i][2],
                       label=label,
                       decorate=decorate,
                       )

        return fig

    def plot_pair_distance(self):
        """
        Plot pair distance.
        """
        envs = self.get_atomic_environment()

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)

        for i, ratio in enumerate(self._shear_strain_ratios):
            if i == len(envs)-1:
                decorate = True
            else:
                decorate = False
            label = "shear ratio: %1.2f" % ratio
            plot_pair_distance(ax,
                               z_coords=envs[i][0],
                               pair_distances=envs[i][3],
                               label=label,
                               decorate=decorate,
                               )
            ax.legend()

        return fig

    def plot_atom_diff(self, direction='x', shuffle:bool=True):
        """
        Plot atom diff.
        """
        i_f_cells = [ [ relax_analyzer.original_cell,
                        relax_analyzer.final_cell_in_original_frame]
                            for relax_analyzer in self._relax_analyzers ]
        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)

        for i, cells in enumerate(i_f_cells):
            if i == len(i_f_cells)-1:
                decorate = True
            else:
                decorate = False
            label = "shear ratio: %1.2f" % self._shear_strain_ratios[i]
            plot_atom_diff(ax,
                           initial_cell=cells[0],
                           final_cell=cells[1],
                           decorate=decorate,
                           direction=direction,
                           label=label,
                           shuffle=shuffle,
                           )

    def write_poscars(self, header:str='', is_original_frame:bool=True):
        """
        Write poscars

        Args:
            header (str): File header.
            is_original_frame (bool): Poscar is in original frame.
        """
        orig_cells = [ relax_analyzer.final_cell_in_original_frame
                           for relax_analyzer in self._relax_analyzers ]
        for i, cell in enumerate(orig_cells):
            filename = header + "{}_s{}.poscar".format(
                    i, self._shear_strain_ratios[i])
            write_poscar(cell=cell, filename=filename)
