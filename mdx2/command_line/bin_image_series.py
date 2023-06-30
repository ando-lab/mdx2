"""
Bin down an image stack
"""

import argparse

import numpy as np

from mdx2.utils import saveobj, loadobj

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("data", help="NeXus data file containing the image_series")
    parser.add_argument("bins", nargs=3, type=int, metavar='N', help="number per bin in each direction (frames, y, x)")
    parser.add_argument("--outfile", default="binned.nxs", help="name of the output NeXus file")
    parser.add_argument("--valid_range", nargs=2, type=int, metavar='N', help="minimum and maximum valid data values")
    parser.add_argument("--nproc", type=int, default=1, metavar='N', help="number of parallel processes")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    image_series = loadobj(args.data,'image_series')

    binned = image_series.bin_down(args.bins,valid_range=args.valid_range,nproc=args.nproc)

    print(f'saving to file: {args.outfile}')
    nxs = saveobj(binned,args.outfile,name='binned_image_series')


if __name__ == "__main__":
    run()
