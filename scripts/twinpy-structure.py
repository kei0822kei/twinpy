#!/usr/bin/env python

"""
Get twinpy strucuture
"""

import argparse
import numpy as np
from pymatgen.io.vasp import Poscar
from twinpy.properties.hexagonal import (get_hexagonal_lattice_from_a_c,
                                         get_wyckoff_from_hcp)
from twinpy.interfaces.pymatgen import get_cell_from_pymatgen_structure
from twinpy.api_twinpy import Twinpy
from twinpy.file_io import write_poscar


# argparse
def get_argparse():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--structure', type=str,
                        help="'shear' or 'twinboundary'")
    parser.add_argument('-r', '--shear_strain_ratio', type=float, default=0.,
                        help="shear strain ratio")
    parser.add_argument('--twinmode', type=str,
                        help="twinmode")
    parser.add_argument('--twintype', type=int, default=None,
                        help="twintype, when you specify this, \
                              twin boundary mode is evoked")
    parser.add_argument('--xshift', type=float, default=0.,
                        help="x shift")
    parser.add_argument('--yshift', type=float, default=0.,
                        help="y shift")
    parser.add_argument('--dim', type=str, default='1 1 1',
                        help="dimension")
    parser.add_argument('--layers', type=int,
                        help="layers for twin boundary structure")
    parser.add_argument('--delta', type=float, default=0.,
                        help="delta")
    parser.add_argument('--expansion_ratios', type=str, default='1 1 1',
                        help="expansion_ratios")
    parser.add_argument('--no_make_tb_flat', action='store_true',
                        help="do not project atoms on the twin boundary")
    parser.add_argument('-c', '--posfile', default=None,
                        help="POSCAR file")
    parser.add_argument('--get_poscar', action='store_true',
                        help="get poscar")
    parser.add_argument('--get_lattice', action='store_true',
                        help="get lattice not structure")
    parser.add_argument('-o', '--output', default=None,
                        help="POSCAR filename")
    parser.add_argument('--is_primitive', action='store_true',
                        help="get primitive shear structure")
    parser.add_argument('--get_primitive_standardized', action='store_true',
                        help="get primitive standardized")
    parser.add_argument('--get_conventional_standardized', action='store_true',
                        help="get conventional standardized")
    parser.add_argument('--dump', action='store_true',
                        help="dump twinpy structure object to yaml")
    parser.add_argument('--show_nearest_distance', action='store_true',
                        help="Show nearest atomic distance.")

    arguments = parser.parse_args()

    return arguments


def _get_output_name(structure, get_lattice, shear_strain_ratio, twinmode):
    name = ''
    if structure == 'shear':
        if np.allclose(shear_strain_ratio, 0.):
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


def main(structure,
         shear_strain_ratio,
         twinmode,
         twintype,
         xshift,
         yshift,
         dim,
         layers,
         delta,
         expansion_ratios,
         no_make_tb_flat,
         posfile,
         get_poscar,
         get_lattice,
         output,
         is_primitive,
         get_primitive_standardized,
         get_conventional_standardized,
         dump,
         show_nearest_distance,
         ):

    move_atoms_into_unitcell = True
    symprec = 1e-5
    no_idealize = False
    no_sort = True
    get_sort_list = False

    if posfile is None:
        print("Warning:")
        print("    POSCAR file did not specify")
        print("    Set automatically, a=2.93, c=4.65, symbol='Ti', "
              "wyckoff='c'")
        lattice = get_hexagonal_lattice_from_a_c(a=2.93, c=4.65)
        symbol = 'Ti'
        wyckoff = 'c'
    else:
        poscar = Poscar.from_file(posfile)
        pmgstructure = poscar.structure
        cell = get_cell_from_pymatgen_structure(pmgstructure)
        lattice = cell[0]
        symbol = cell[2][0]
        wyckoff = get_wyckoff_from_hcp(cell)

    twinpy = Twinpy(lattice=lattice,
                    twinmode=twinmode,
                    symbol=symbol,
                    wyckoff=wyckoff)

    if get_poscar:
        if output is None:
            output = _get_output_name(structure=structure,
                                      get_lattice=get_lattice,
                                      shear_strain_ratio=shear_strain_ratio,
                                      twinmode=twinmode)

    if structure == 'shear':
        twinpy.set_shear(xshift=xshift,
                         yshift=yshift,
                         dim=dim,
                         shear_strain_ratio=shear_strain_ratio,
                         expansion_ratios=expansion_ratios,
                         is_primitive=is_primitive)
        std = twinpy.get_shear_standardize(
                get_lattice=get_lattice,
                move_atoms_into_unitcell=move_atoms_into_unitcell,
                )

    else:
        make_tb_flat = not no_make_tb_flat
        twinpy.set_twinboundary(twintype=twintype,
                                xshift=xshift,
                                yshift=yshift,
                                layers=layers,
                                delta=delta,
                                shear_strain_ratio=shear_strain_ratio,
                                expansion_ratios=expansion_ratios,
                                make_tb_flat=make_tb_flat)
        std = twinpy.get_twinboundary_standardize(
                get_lattice=get_lattice,
                move_atoms_into_unitcell=move_atoms_into_unitcell,
                )

        if show_nearest_distance:
            from twinpy.structure.twinboundary \
                    import plot_nearest_atomic_distance_of_twinboundary
            plot_nearest_atomic_distance_of_twinboundary(
                    lattice=lattice,
                    symbol=symbol,
                    twinmode=twinmode,
                    layers=layers,
                    wyckoff=wyckoff,
                    delta=delta,
                    twintype=twintype,
                    shear_strain_ratio=shear_strain_ratio,
                    expansion_ratios=expansion_ratios,
                    make_tb_flat=make_tb_flat,
                    )

    if get_primitive_standardized:
        to_primitive = True
    elif get_conventional_standardized:
        to_primitive = False
    else:
        to_primitive = None

    if to_primitive is None:
        out_cell = std.cell
    else:
        out_cell = std.get_standardized_cell(
                       to_primitive=to_primitive,
                       no_idealize=no_idealize,
                       symprec=symprec,
                       no_sort=no_sort,
                       get_sort_list=get_sort_list,
                       )

    if output is not None:
        write_poscar(cell=out_cell,
                     filename=output)

    if dump:
        twinpy.dump_yaml()


if __name__ == '__main__':
    args = get_argparse()
    dimension = list(map(int, args.dim.split()))
    expand = list(map(float, args.expansion_ratios.split()))
    assert args.structure in ['shear', 'twinboundary'], \
        "structure must be 'shear' or 'twinboundary'"

    main(structure=args.structure,
         shear_strain_ratio=args.shear_strain_ratio,
         twinmode=args.twinmode,
         twintype=args.twintype,
         xshift=args.xshift,
         yshift=args.yshift,
         dim=dimension,
         layers=args.layers,
         delta=args.delta,
         expansion_ratios=expand,
         no_make_tb_flat=args.no_make_tb_flat,
         posfile=args.posfile,
         get_poscar=args.get_poscar,
         get_lattice=args.get_lattice,
         output=args.output,
         is_primitive=args.is_primitive,
         get_primitive_standardized=args.get_primitive_standardized,
         get_conventional_standardized=args.get_conventional_standardized,
         dump=args.dump,
         show_nearest_distance=args.show_nearest_distance,
         )
