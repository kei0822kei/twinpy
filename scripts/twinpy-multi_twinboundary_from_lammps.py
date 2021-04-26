#!/usr/bin/env python

"""
Search twinboundary structure using lammps.
"""

import os
import argparse
import itertools
import joblib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from twinpy.api_twinpy import Twinpy
from twinpy.interfaces.pymatgen import get_pymatgen_structure
from twinpy.interfaces.lammps import (get_lammps_relax,
                                      get_relax_analyzer_from_lammps_static)
from twinpy.properties.hexagonal import (get_hexagonal_lattice_from_a_c,
                                         get_hcp_atom_positions)
from twinpy.file_io import write_poscar


# argparse
def get_argparse():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # create twinpy
    parser.add_argument('--a', type=float,
                        help="Axis a norm.")
    parser.add_argument('--c', type=float,
                        help="Axis a norm.")
    parser.add_argument('--symbol', type=str,
                        help="Symbol.")
    parser.add_argument('--twinmode', type=str,
                        help="Twinmode.")
    parser.add_argument('--twintype', type=int, default=1,
                        help="Twintype.")
    parser.add_argument('--layers', type=int,
                        help="Layers.")
    parser.add_argument('--expansion_ratios', type=str, default='1. 1. 1.',
                        help="Ex. --expansion_ratios '1. 1. 1.2'")
    parser.add_argument('--xshifts', type=str,
                        help="Ex. --xshifts '0. 0.25 0.5 0.75'")
    parser.add_argument('--yshifts', type=str,
                        help="Ex. --yshifts '0. 0.25 0.5 0.75'")

    # Boolians
    parser.add_argument('--no_make_tb_flat', action='store_true',
                        help="No make twinboundary flat.")
    parser.add_argument('--is_run_hexagonal_relax', action='store_true',
                        help="Is run hexagonal relax.")
    parser.add_argument('--is_relax_lattice', action='store_true',
                        help="Is relax twinboundary lattice.")

    # Lammps configuration
    parser.add_argument('--pair_style', type=str,
                        help="Lammps pair_style")
    parser.add_argument('--pot_file', type=str,
                        help="Lammps potential file path.")

    args = parser.parse_args()

    return args


def main(a:float,
         c:float,
         symbol:str,
         twinmode:str,
         layers:float,
         expansion_ratios:np.array,
         make_tb_flat:bool,
         xshifts:list,
         yshifts:list,
         pair_style:str,
         pot_file:str,
         twintype:int=1,
         is_run_hexagonal_relax:bool=True,
         is_relax_lattice:bool=False,
         ):
    lattice = get_hexagonal_lattice_from_a_c(a=a, c=c)
    scaled_positions = get_hcp_atom_positions(wyckoff='c')
    symbols = [ symbol ] * 2
    hex_cell = (lattice, scaled_positions, symbols)
    hex_lmp_stc = get_lammps_relax(cell=hex_cell,
                                   pair_style=pair_style,
                                   pot_file=pot_file,
                                   is_relax_lattice=is_run_hexagonal_relax,
                                   )
    hex_rlx_analyzer = get_relax_analyzer_from_lammps_static(
            lammps_static=hex_lmp_stc,
            no_standardize=True,
            )

    itr = itertools.product(xshifts, yshifts)
    data = []
    os.makedirs("./poscar")
    os.makedirs("./joblib")
    for i, xy in enumerate(itr):
        xshift, yshift = xy

        twinpy = Twinpy(lattice=lattice,
                        twinmode=twinmode,
                        symbol=symbol,
                        wyckoff='c')
        twinpy.set_twinboundary(
                layers=layers,
                twintype=twintype,
                xshift=xshift,
                yshift=yshift,
                shear_strain_ratio=0.,
                expansion_ratios=expansion_ratios,
                make_tb_flat=make_tb_flat,
                )
        twinpy.set_twinboundary_analyzer_from_lammps(
                pair_style=pair_style,
                pot_file=pot_file,
                is_relax_lattice=is_relax_lattice,
                is_run_phonon=False,
                hexagonal_relax_analyzer=hex_rlx_analyzer,
                )
        num = "%03d" % i
        filename = "./joblib/{}_xshift-{}_yshift-{}.joblib".format(num, xshift, yshift)
        joblib.dump(twinpy, filename)
        rlx_cell = twinpy.twinboundary_analyzer.relax_analyzer.final_cell
        posname = "./poscar/{}_xshift-{}_yshift-{}.poscar".format(num, xshift, yshift)
        write_poscar(rlx_cell, posname)
        formation_energy = twinpy.twinboundary_analyzer.get_formation_energy()
        data.append([xshift, yshift, formation_energy])

    data = np.array(data)
    print(data)

    # plot plane interval
    im = plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=cm.jet)
    plt.colorbar(im)
    plt.show()


if __name__ == '__main__':
    args = get_argparse()
    make_tb_flat = not bool(args.no_make_tb_flat)
    xshifts = list(map(float, args.xshifts.split()))
    yshifts = list(map(float, args.yshifts.split()))
    expansion_ratios = np.array(list(map(float, args.expansion_ratios.split())))
    main(a=args.a,
         c=args.c,
         symbol=args.symbol,
         twinmode=args.twinmode,
         layers=args.layers,
         expansion_ratios=expansion_ratios,
         make_tb_flat=make_tb_flat,
         xshifts=xshifts,
         yshifts=yshifts,
         pair_style=args.pair_style,
         pot_file=args.pot_file,
         twintype=args.twintype,
         is_run_hexagonal_relax=args.is_run_hexagonal_relax,
         is_relax_lattice=args.is_relax_lattice,
         )
