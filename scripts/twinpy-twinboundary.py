#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get twinpy strucuture
"""

import argparse
from twinpy.api_twinpy import Twinpy


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
    parser.add_argument('--shear_relax_pks', type=str, default=None,
                        help="Twinboundary shear relax pks.")
    parser.add_argument('--shear_phonon_pks', type=str, default=None,
                        help="Twinboundary shear phonon pks.")
    parser.add_argument('--shear_strain_ratios', type=str, default=None,
                        help="Twinboundary shear strain ratios.")

    # write shear cells
    parser.add_argument('--write_shear_cells', action='store_true',
                        help="Get shear cells")
    parser.add_argument('--is_original_frame', action='store_true',
                        help="Original frame.")
    parser.add_argument('--is_relax', action='store_true',
                        help="Relax.")
    parser.add_argument('--file_header', type=str, default='',
                        help="File header.")

    # plot band structure
    parser.add_argument('--plot_band', action='store_true',
                        help="Plot band structure")
    parser.add_argument('--show_band_points', action='store_true',
                        help="Show reciprocal high symmetry points.")
    args = parser.parse_args()

    return args


def main(twinboundary_relax_pk:int,
         additional_relax_pks:list=None,
         twinboundary_phonon_pk:int=None,
         hexagonal_relax_pk:int=None,
         hexagonal_phonon_pk:int=None,
         shear_relax_pks:list=None,
         shear_phonon_pks:list=None,
         shear_strain_ratios:list=None,
         write_shear_cells:bool=False,
         is_original_frame:bool=False,
         is_relax:bool=False,
         file_header:str='',
         plot_band:bool=False,
         show_band_points:bool=False,
         ):

    twinpy = Twinpy.initialize_from_aiida_twinboundary(
                 twinboundary_relax_pk=twinboundary_relax_pk,
                 additional_relax_pks=additional_relax_pks,
                 twinboundary_phonon_pk=twinboundary_phonon_pk,
                 hexagonal_relax_pk=hexagonal_relax_pk,
                 hexagonal_phonon_pk=hexagonal_phonon_pk,
                 )

    if shear_relax_pks is not None:
        twinpy.set_twinboundary_shear_analyzer(
                shear_relax_pks=shear_relax_pks,
                shear_phonon_pks=shear_phonon_pks,
                shear_strain_ratios=shear_strain_ratios,
                )

    # write shear cells
    if write_shear_cells:
        print("write shear cells:")
        print("    is_original_frame:{}".format(is_original_frame))
        print("    is_relax:{}".format(is_relax))
        print("    file_header:{}".format(file_header))
        twinpy.write_twinboundary_shear_cells(
                is_original_frame=is_original_frame,
                is_relax=is_relax,
                header=file_header,
                )

    # show band points
    if show_band_points:
        twinpy.show_twinboundary_reciprocal_high_symmetry_points()

    # plot band structure
    if plot_band:
        twinpy.plot_twinboundary_shear_bandstructure()


if __name__ == '__main__':
    args = get_argparse()
    if args.additional_relax_pks is not None:
        additional_relax_pks = list(map(int, args.additional_relax_pks.split()))
    else:
        additional_relax_pks = None

    if args.shear_relax_pks is not None:
        shear_relax_pks = list(map(int, args.shear_relax_pks.split()))
    else:
        shear_relax_pks = None

    if args.shear_phonon_pks is not None:
        shear_phonon_pks = list(map(int, args.shear_phonon_pks.split()))
    else:
        shear_phonon_pks = None

    if args.shear_strain_ratios:
        shear_strain_ratios = list(map(float, args.shear_strain_ratios.split()))
    else:
        shear_strain_ratios = None

    main(twinboundary_relax_pk=args.twinboundary_relax_pk,
         additional_relax_pks=additional_relax_pks,
         twinboundary_phonon_pk=args.twinboundary_phonon_pk,
         hexagonal_relax_pk=args.hexagonal_relax_pk,
         hexagonal_phonon_pk=args.hexagonal_phonon_pk,
         shear_relax_pks=shear_relax_pks,
         shear_phonon_pks=shear_phonon_pks,
         shear_strain_ratios=shear_strain_ratios,
         write_shear_cells=args.write_shear_cells,
         is_original_frame=args.is_original_frame,
         is_relax=args.is_relax,
         file_header=args.file_header,
         plot_band=args.plot_band,
         show_band_points=args.show_band_points,
         )
