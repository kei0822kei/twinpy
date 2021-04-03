#!/usr/bin/env python

"""
Deals with twinpy twinboundary.
"""

import argparse
from matplotlib import pyplot as plt
from twinpy.api_twinpy import Twinpy
from twinpy.file_io import write_poscar


# argparse
def get_argparse():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # create twinpy
    parser.add_argument('--twinboundary_relax_pk', type=int,
                        help="AiidaTwinBoudnaryRelaxWorkChain pk.")
    parser.add_argument('--additional_relax_pks', type=str, default=None,
                        help="Additional relax pks.")
    parser.add_argument('--twinboundary_phonon_pk', type=int, default=None,
                        help="AiidaPhonopy pk "
                             "for relax twinboundary structure.")
    parser.add_argument('--hexagonal_relax_pk', type=int,
                        help="Hexagonal relax pk.")
    parser.add_argument('--hexagonal_phonon_pk', type=int, default=None,
                        help="AiidaPhonopy pk for relax hexagonal structure.")

    # plot configuration
    parser.add_argument('--plot_plane_diff', action='store_true',
                        help="Plot plane interval.")

    # plot band structure
    parser.add_argument('--plot_band', action='store_true',
                        help="Plot band structure")
    parser.add_argument('--show_band_points', action='store_true',
                        help="Show reciprocal high symmetry points.")

    # export
    parser.add_argument('--write_cell_in_original_frame', action='store_true',
                        help="Write out relax cell in original frame.")

    args = parser.parse_args()

    return args


def main(twinboundary_relax_pk:int,
         additional_relax_pks:list=None,
         twinboundary_phonon_pk:int=None,
         hexagonal_relax_pk:int=None,
         hexagonal_phonon_pk:int=None,
         plot_plane_diff:bool=False,
         plot_band:bool=False,
         show_band_points:bool=False,
         write_cell_in_original_frame:bool=False,
         ):

    twinpy = Twinpy.initialize_from_aiida_twinboundary(
                 twinboundary_relax_pk=twinboundary_relax_pk,
                 additional_relax_pks=additional_relax_pks,
                 twinboundary_phonon_pk=twinboundary_phonon_pk,
                 hexagonal_relax_pk=hexagonal_relax_pk,
                 hexagonal_phonon_pk=hexagonal_phonon_pk,
                 )

    # plot plane interval
    if plot_plane_diff:
        fig = twinpy.twinboundary_analyzer.plot_plane_diff()
        plt.show()


    # show band points
    if show_band_points:
        twinpy.show_twinboundary_reciprocal_high_symmetry_points()

    # plot band structure
    if plot_band:
        twinpy.plot_twinboundary_shear_bandstructure()

    # write out relax cell in original frame
    if write_cell_in_original_frame:
        rlx_analyzer = twinpy.twinboundary_analyzer.relax_analyzer
        rlx_cell_orig = rlx_analyzer.final_cell_in_original_frame
        write_poscar(cell=rlx_cell_orig,
                     filename='rlx_cell_orig.poscar')


if __name__ == '__main__':
    args = get_argparse()
    if args.additional_relax_pks is not None:
        additional_relax_pks = list(map(int, args.additional_relax_pks.split()))
    else:
        additional_relax_pks = None

    main(twinboundary_relax_pk=args.twinboundary_relax_pk,
         additional_relax_pks=additional_relax_pks,
         twinboundary_phonon_pk=args.twinboundary_phonon_pk,
         hexagonal_relax_pk=args.hexagonal_relax_pk,
         hexagonal_phonon_pk=args.hexagonal_phonon_pk,
         plot_plane_diff=args.plot_plane_diff,
         plot_band=args.plot_band,
         show_band_points=args.show_band_points,
         write_cell_in_original_frame=args.write_cell_in_original_frame,
         )
