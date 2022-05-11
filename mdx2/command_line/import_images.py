"""
Import images using the dxtbx machinery
"""

import argparse

import numpy as np
from dxtbx.model.experiment_list import ExperimentList
from mdx2.data import ImageSet


def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Required arguments
    parser.add_argument("expt", help="experiments file, such as from dials.import")
    parser.add_argument("--outfile", default="images.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)
    print(f"importing images from {args.expt}")

    elist = ExperimentList.from_file(args.expt)
    iset = elist.imagesets()[0]

    print(f"writing images to {args.outfile}")
    IS = ImageSet(iset)
    data = IS.to_nexus()
    f = data.save(args.outfile)
    IS.read_all(data.images,buffer=data.images.chunks[0])

    print("done!")
    print(f"{args.outfile}:")
    print(f.tree)

if __name__ == "__main__":
    run()
