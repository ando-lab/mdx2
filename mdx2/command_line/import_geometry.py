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
    parser.add_argument("expt", help=".expt file containing scan metadata (e.g. from dials.refine)")
    parser.add_argument("--outfile", default="geometry.nxs", help="name of the output nexus file")

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


if __name__ == "__main__":
    run()
