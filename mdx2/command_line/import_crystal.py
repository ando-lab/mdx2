"""
Import crystal geometry using the dxtbx machinery
"""

import argparse
import json

import numpy as np
from dxtbx.model.experiment_list import ExperimentList

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Required arguments
    parser.add_argument("expt", help=".expt file containing scan metadata (e.g. from dials.refine)")
    parser.add_argument("--outfile", default="crystal.json", help="name of the output json file")

    return parser

def extract_crystal_info(dxtbx_crystal):
    """retrieve useful info from the dxtbx crystal object"""
    symm = dxtbx_crystal.get_crystal_symmetry()

    # space group info
    sg = symm.space_group()
    sgi = sg.info()
    sgt = sgi.type()

    # laue group info
    lg = sg.build_derived_laue_group()
    lgi = lg.info()
    lgt = lgi.type()

    # unit cell info
    uc = symm.unit_cell()

    # build a dictionary to export
    crystal = {
        'unit_cell':{
            'a':uc.parameters()[0],
            'b':uc.parameters()[1],
            'c':uc.parameters()[2],
            'alpha':uc.parameters()[3],
            'beta':uc.parameters()[4],
            'gamma':uc.parameters()[5],
            'orthogonalization_matrix':uc.orthogonalization_matrix(),
            'U':dxtbx_crystal.get_U(),
            'B':dxtbx_crystal.get_B(),
        },
        'symmetry':{
            'space_group_number':sgt.number(),
            'space_group_symbol':sgt.lookup_symbol(),
            'crystal_system':sg.crystal_system(),
            'space_group_operators':[op.as_double_array() for op in sg.all_ops()],
            'laue_group_number':lgt.number(),
            'laue_group_symbol':sg.laue_group_type(),
            'laue_group_operators':[op.r().as_double() for op in lg.all_ops()],
            'reciprocal_space_asu':sgi.reciprocal_space_asu().reference_as_string(),
        },
    }
    return crystal

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    print(f"importing geometry from {args.expt}")
    elist = ExperimentList.from_file(args.expt)

    crystal = extract_crystal_info(elist[0].crystal)

    print(f"saving geometry to {args.outfile}")
    with open(args.outfile,'w') as outfile:
        json.dump(crystal, outfile)


if __name__ == "__main__":
    run()
