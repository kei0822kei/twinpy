#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
get structure from twinpy
"""

import argparse
import numpy as np
from phonopy.interface.calculator import write_crystal_structure
from twinpy.lattice.lattice import get_hexagonal_lattice_from_a_c
from twinpy.lattice.lattice import Lattice
from twinpy.structure.base import get_hexagonal_structure_from_pymatgen
from twinpy.lattice.brillouin import show_brillouin_zone
from twinpy.api_twinpy import Twinpy
from pymatgen.io.vasp import Poscar
from pymatgen.core.structure import Structure

# argparse
def get_argparse():
    structure_type_msg = "output structure type: choose from \
                          'base'(default) 'conventional' and 'primitive'"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--structure', type=str,
        help="'shear' or 'twinboundary'")
    parser.add_argument('-r', '--ratio', type=float, default=0.,
        help="shear strain ratio")
    parser.add_argument('--twinmode', type=str,
        help="twinmode")
    parser.add_argument('--twintype', type=int, default=None,
        help="twintype, when you specify this, twin boundary mode is evoked")
    parser.add_argument('--xshift', type=float, default=0.,
        help="x shift")
    parser.add_argument('--yshift', type=float, default=0.,
        help="y shift")
    parser.add_argument('--dim', type=str, default='1 1 1',
        help="dimension")
    parser.add_argument('-c', '--posfile', default=None,
        help="POSCAR file")
    parser.add_argument('--get_poscar', action='store_true',
        help="get poscar")
    parser.add_argument('--get_lattice', action='store_true',
        help="get lattice not structure")
    parser.add_argument('--structure_type', type=str, default='base',
        help=structure_type_msg)
    parser.add_argument('-o', '--output', default=None,
        help="POSCAR filename")
    # parser.add_argument('--show_BZ', action='store_true',
    #     help="show brillouin zone")
    args = parser.parse_args()
    return args

def _get_output_name(structure, get_lattice, ratio, twinmode):
    name = ''
    if structure == 'shear':
        if np.allclose(ratio, 0.):
            name += 'parent'
        else:
            name += 'shear'
    else:
        name += 'tb'
    name += '_%s' % twinmode
    if get_lattice:
        name += '_lat.poscar'
    else:
        name += '.poscar'
    return name

def main(
         structure,
         ratio,
         twinmode,
         twintype,
         xshift,
         yshift,
         dim,
         posfile,
         get_poscar,
         get_lattice,
         structure_type,
         output,
         # show_BZ,
        ):
    if posfile is None:
        print("POSCAR file did not specify.")
        print("set automatically, a=2.93, c=4.65, symbol='Ti', wyckoff='c'")
        lattice = get_hexagonal_lattice_from_a_c(a=2.93, c=4.65)
        symbol = 'Ti'
        wyckoff = 'c'
    else:
        poscar = Poscar.from_file(posfile)
        pmgstructure = poscar.structure
        lattice, _, symbol, wyckoff = \
                get_hexagonal_structure_from_pymatgen(pmgstructure)
    twinpy = Twinpy(lattice=lattice,
                    twinmode=twinmode,
                    symbol=symbol,
                    wyckoff=wyckoff)

    if get_poscar:
        if output is None:
            output = _get_output_name(structure=structure,
                                      get_lattice=get_lattice,
                                      ratio=ratio,
                                      twinmode=twinmode)

    if structure == 'shear':
        twinpy.set_shear(xshift=xshift,
                         yshift=yshift,
                         dim=dim,
                         shear_strain_ratio=ratio)
        if get_lattice:
            twinpy.write_shear_lattice(filename=output)
        else:
            ph_structure = twinpy.get_shear().get_phonopy_structure(
                    structure_type=structure_type)
            write_crystal_structure(
                    filename=output,
                    cell=ph_structure,
                    interface_mode='vasp')

    else:
        twinpy.set_twinboundary(twintype=twintype,
                                xshift=xshift,
                                yshift=yshift,
                                dim=dim,
                                shear_strain_ratio=ratio,
                                make_tb_flat=True)
        if get_lattice:
            twinpy.write_twinboundary_lattice(filename=output)
        else:
            ph_structure = twinpy.get_twinboundary().get_phonopy_structure(
                    structure_type=structure_type)
            write_crystal_structure(
                    filename=output,
                    cell=ph_structure,
                    interface_mode='vasp')

    # if show_BZ:
    #     output_lattice = Lattice(structure.output_structure['lattice'])
    #     reciprocal_lattice = output_lattice.reciprocal_lattice
    #     show_brillouin_zone(reciprocal_lattice)

if __name__ == '__main__':
    args = get_argparse()
    dimension = list(map(int, args.dim.split()))
    assert args.structure in ['shear', 'twinboundary'], \
        "structure must be 'shear' or 'twinboundary'"
    main(
         structure=args.structure,
         ratio=args.ratio,
         twinmode=args.twinmode,
         twintype=args.twintype,
         xshift=args.xshift,
         yshift=args.yshift,
         dim=dimension,
         posfile=args.posfile,
         get_poscar=args.get_poscar,
         get_lattice=args.get_lattice,
         structure_type=args.structure_type,
         output=args.output,
         # show_BZ=args.show_BZ,
        )
