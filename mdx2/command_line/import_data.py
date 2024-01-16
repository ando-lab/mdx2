"""
Import x-ray image data using the dxtbx machinery
"""

#<--- LEFT OFF HERE --->

import argparse

import numpy as np

from mdx2.data import ImageSeries
from mdx2.dxtbx_machinery import ImageSet
from mdx2.utils import saveobj

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("expt", help="experiments file, such as from dials.import")
    parser.add_argument("--outfile", default="data.nxs", help="name of the output NeXus file")
    parser.add_argument("--chunks", nargs=3, type=int, metavar='N', help="chunking for compression (frames, y, x)")
    parser.add_argument("--nproc", type=int, default=1, metavar='N', help="number of parallel processes")
    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    exptfile = args.expt

    image_series = ImageSeries.from_expt(exptfile)
    iset = ImageSet.from_file(exptfile)

    if args.chunks is not None:
        image_series.data.chunks=tuple(args.chunks)

    nxs = saveobj(image_series,args.outfile,name='image_series')

    nbatches = image_series.data.chunks[0]
    
    if args.nproc == 1:
        iset.read_all(image_series.data,nbatches)
    else:
        iset.read_all_parallel(image_series.data,nbatches,args.nproc)

    print("done!")

if __name__ == "__main__":
    run()
