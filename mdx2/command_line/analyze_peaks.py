"""

"""

import argparse

import numpy as np
from nexusformat.nexus import nxload, NXcollection, NXentry
import json

from mdx2.geometry import MillerIndex, fit_quadform_to_points
from mdx2.data import Peaks

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Required arguments
    parser.add_argument("peaks", help="NeXus file containing the peaks")
    parser.add_argument("miller_index", help="NeXus file containing the Miller indices")
    parser.add_argument("--sigma_cutoff", default=3, help="sigma value for outlier rejection")
    parser.add_argument("--outfile", default="peak_analysis.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    print(f"loading peaks from {args.peaks}")
    nxs = nxload(args.peaks)
    P = Peaks.from_nexus(nxs.entry.peaks)

    print(f"loading miller indices from {args.miller_index}")
    nxs = nxload(args.miller_index)
    MI = MillerIndex.from_nexus(nxs.entry)

    print(f"indexing peaks")
    h,k,l = MI.interpolate(P.phi,P.iy,P.ix)
    dh = h - np.round(h)
    dk = k - np.round(k)
    dl = l - np.round(l)

    print(f"fitting sigma matrix")
    center, sigma, is_outlier = fit_quadform_to_points(dh,dk,dl,sigma_cutoff=args.sigma_cutoff)
    print(f"  (rejected {np.sum(is_outlier)} outliers)")
    print('\nSigma =')
    print(sigma)
    print('\ncenter = ')
    print(center)

    print(f"\nsaving peak analysis to {args.outfile}")
    # should use an NXtransformation?
    fit = NXcollection(Sigma=sigma,center=center)
    P.phi = P.phi[is_outlier]
    P.ix = P.ix[is_outlier]
    P.iy = P.iy[is_outlier]
    outliers = P.to_nexus()
    nxs = NXentry(fit=fit,outliers=outliers)
    nxs.save(args.outfile)

    print("done!")
    print(f"\n{args.outfile}:\n",nxs.tree)

if __name__ == "__main__":
    run()
