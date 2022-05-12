"""
Import crystal geometry using the dxtbx machinery
"""

import argparse
import json

import numpy as np
from dxtbx.model.experiment_list import ExperimentList
from nexusformat.nexus import NXentry

from mdx2.geometry import Crystal, Symmetry

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Required arguments
    parser.add_argument("expt", help="experiments file, such as from dials.refine")
    parser.add_argument("--outfile", default="crystal.nxs", help="name of the output NeXus file")

    return parser


def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    print(f"importing geometry from {args.expt}")
    elist = ExperimentList.from_file(args.expt)
    dxtbx_crystal = elist[0].crystal

    crystal = Crystal.from_dxtbx_crystal(dxtbx_crystal)
    symmetry = Symmetry.from_dxtbx_crystal(dxtbx_crystal)

    print(f"saving geometry to {args.outfile}")
    nxs = NXentry(crystal.to_nexus(),symmetry.to_nexus())
    nxs.save(args.outfile)

    print("done!")
    print(f"\n{args.outfile}:\n",nxs.tree)

if __name__ == "__main__":
    run()
