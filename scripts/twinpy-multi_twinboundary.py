#!/usr/bin/env python

"""
Deals with twinpy twinboundary.
"""

import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from aiida.orm import Group
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from twinpy.api_twinpy import Twinpy
from twinpy.interfaces.pymatgen import get_pymatgen_structure


# argparse
def get_argparse():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # create twinpy
    parser.add_argument('--hexagonal_relax_pk', type=int,
                        help="Hexagonal relax pk.")
    parser.add_argument('--group_pk', type=int,
                        help="Twinboundary multi relax group pk.")
    # plot configuration
    parser.add_argument('--plot_energies', action='store_true',
                        help="Plot plane interval.")
    parser.add_argument('--vmax', type=float, default=None,
                        help="Colorbar max")

    args = parser.parse_args()

    return args


def main(hexagonal_relax_pk:int,
         group_pk:int,
         plot_energies:bool=False,
         vmax:float=None,
         ):

    grp = Group.get(id=group_pk)

    data = []
    print("load data: total {} data".format(len(grp.nodes)))
    for i, node in enumerate(grp.nodes):
        print("loading data: number {}".format(i+1))
        twinpy = Twinpy.initialize_from_aiida_twinboundary(
                     twinboundary_relax_pk=node.pk,
                     hexagonal_relax_pk=hexagonal_relax_pk,
                     )
        pmgstruct = get_pymatgen_structure(
                twinpy.twinboundary_analyzer.relax_analyzer.final_cell)
        spg = SpacegroupAnalyzer(pmgstruct, symprec=1e-1)
        sg = spg.get_symmetry_dataset()['international']
        print( "pk: %d" % node.pk + " " + node.label + " space group: %s" % sg)
        tb_analyzer = twinpy.twinboundary_analyzer
        xshift = tb_analyzer.twinboundary_structure.xshift
        yshift = tb_analyzer.twinboundary_structure.yshift
        energy = tb_analyzer.get_formation_energy()
        data.append([xshift, yshift, energy])
    data = np.array(data)

    # plot plane interval
    if plot_energies:
        plt.figure()
        im = plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=cm.jet,
                         vmax=vmax)
        plt.colorbar(im)
        plt.show()


if __name__ == '__main__':
    args = get_argparse()
    main(hexagonal_relax_pk=args.hexagonal_relax_pk,
         group_pk=args.group_pk,
         plot_energies=args.plot_energies,
         vmax=args.vmax,
         )
