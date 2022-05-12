"""
Import miller index map using the dxtbx machinery
"""

import argparse

import numpy as np
from dxtbx.model.experiment_list import ExperimentList

from mdx2.geometry import MillerIndex

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Required arguments
    parser.add_argument("expt", help="experiments file, such as from dials.refine")
    parser.add_argument("--outfile", default="miller_index.nxs", help="name of the output NeXus file")
    parser.add_argument("--sampling", nargs=3, metavar='N', type=int, default=[1,10,10], help="inverval between samples in degrees or pixels (phi, iy, ix)")

    return parser


def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    print(f"importing geometry from {args.expt}")
    elist = ExperimentList.from_file(args.expt)
    iset = elist.imagesets()[0]

    index = MillerIndex.from_dxtbx_imageset(iset,sampling=tuple(args.sampling))

    print(f"saving corrections to {args.outfile}")
    nxs = index.to_nexus()
    nxs.save(args.outfile)

    print("done!")
    print(f"\n{args.outfile}:\n",nxs.tree)

if __name__ == "__main__":
    run()
