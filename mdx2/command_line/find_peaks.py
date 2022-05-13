"""
Find pixels with counts above a threshold
"""

import argparse

import numpy as np
from nexusformat.nexus import nxload

from mdx2.data import Peaks

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Required arguments
    parser.add_argument("imgs", help="NeXus file containing the image stack")
    parser.add_argument("--outfile", default="peaks.nxs", help="name of the output NeXus file")
    parser.add_argument("--threshold", required=True, type=int, help="counts greater than threshold are recorded as peaks")

    return parser


def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    print(f"looping through image data in {args.imgs}")
    nxs = nxload(args.imgs)
    peaks = Peaks.from_image_counts_above_threshold(
        nxs.entry.data,
        threshold=args.threshold,
        )

    print(f"saving peak table to {args.outfile}")
    nxs = peaks.to_nexus()
    nxs.save(args.outfile)

    print("done!")
    print(f"\n{args.outfile}:\n",nxs.tree)

if __name__ == "__main__":
    run()
