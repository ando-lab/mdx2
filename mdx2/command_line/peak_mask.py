"""

"""

import argparse

import numpy as np
from nexusformat.nexus import nxload, NXdata, NXentry
import json

from mdx2.geometry import MillerIndex
from mdx2.data import Peaks

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Required arguments
    parser.add_argument("peak_model", help="NeXus file containing the peak analysis")
    parser.add_argument("miller_index", help="NeXus file containing the Miller indices")
    parser.add_argument('images', help="NeXus file containing the image stack")
    parser.add_argument("--sigma_cutoff", default=3, help="sigma value for masking")
    parser.add_argument("--outfile", default="peak_mask.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    print(f"loading outliers from {args.peak_model}")
    nxs = nxload(args.peak_model)
    P = Peaks.from_nexus(nxs.entry.outliers)

    print(f"computing outlier mask {args.images}")
    imnx = nxload(args.images)
    outlier_mask = P.to_mask(imnx.entry.data)

    print(f"loading miller indices from {args.miller_index}")
    nxs = nxload(args.miller_index)
    MI = MillerIndex.from_nexus(nxs.entry)

    print(f"indexing peaks")
    hklinterp = MI.interpolator
    h,k,l = hklinterp(P.phi,P.iy,P.ix)
    dh = h - np.round(h)
    dk = k - np.round(k)
    dl = l - np.round(l)

    print(f"saving outlier mask to {args.outfile}")
    nxs = NXentry(outlier_mask=NXdata(outlier_mask))
    nxs.save(args.outfile)

    print("done!")
    print(f"\n{args.outfile}:",nxs.tree)

if __name__ == "__main__":
    run()
