"""
Find pixels with counts above a threshold
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
    parser.add_argument("imgs", help="NeXus file containing the image stack")
    parser.add_argument("--outfile", default="peaks.nxs", help="name of the output NeXus file")
    parser.add_argument("--threshold", type=int, default=None], help="count > threshold are recorded as peaks")

    return parser


def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    print(f"looping through image data in {args.imgs}")
    elist = ExperimentList.from_file(args.expt)
    expt = elist[0]

    index = MillerIndex.from_dxtbx_experiment(expt,sampling=tuple(args.sampling))

    print(f"saving peak table to {args.outfile}")
    nxs = index.to_nexus()
    nxs.save(args.outfile)

    print("done!")
    print(f"\n{args.outfile}:\n",nxs.tree)

if __name__ == "__main__":
    run()
