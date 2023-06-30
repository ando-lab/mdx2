"""
Find and analyze peaks in an image stack
"""

import numpy as np
import argparse

from mdx2.utils import saveobj, loadobj
from mdx2.geometry import GaussianPeak
from mdx2.data import Peaks
#from . import MDX2Parser

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required arguments
    parser.add_argument("geom", help="NeXus file w/ miller_index")
    parser.add_argument("data", help="NeXus file w/ image_series")
    parser.add_argument("--count_threshold", metavar='THRESH', required=True, type=float, help="pixels with counts above threshold are flagged as peaks")
    parser.add_argument("--sigma_cutoff", metavar='SIGMA', default=3, type=float, help="\for outlier rejection in Gaussian peak fitting")
    parser.add_argument("--outfile", default="peaks.nxs", help="name of the output NeXus file")
    parser.add_argument("--nproc", type=int, default=1, metavar='N', help="number of parallel processes")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    MI = loadobj(args.geom,'miller_index')
    IS = loadobj(args.data,'image_series')

    print(f'finding pixels with counts above threshold: {args.count_threshold}')
    P = IS.find_peaks_above_threshold(args.count_threshold,nproc=args.nproc)

    print('indexing peaks')
    h,k,l = MI.interpolate(P.phi,P.iy,P.ix)
    dh = h - np.round(h)
    dk = k - np.round(k)
    dl = l - np.round(l)

    print('fitting Gaussian peak model')
    GP, is_outlier = GaussianPeak.fit_to_points(dh,dk,dl,sigma_cutoff=args.sigma_cutoff)

    O = Peaks(P.phi[is_outlier],P.iy[is_outlier],P.ix[is_outlier])

    print(f'{np.sum(is_outlier)} peaks were rejected as outliers')
    print('GaussianPeak model:',GP)

    print(f"Saving results to {args.outfile}")

    saveobj(GP,args.outfile,name='peak_model',append=False)
    saveobj(P,args.outfile,name='peaks',append=True)
    saveobj(O,args.outfile,name='outliers',append=True)

    print('done!')

if __name__ == "__main__":
    run()
