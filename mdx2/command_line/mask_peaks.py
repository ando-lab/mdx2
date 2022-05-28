"""
Create a peak mask for in an image stack
"""

import argparse

import numpy as np

from mdx2.utils import saveobj, loadobj
from mdx2.geometry import GridData

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("geom", help="NeXus file w/ miller_index")
    parser.add_argument("data", help="NeXus file w/ image_series")
    parser.add_argument("peaks", help="NeXus file w/ peak_model and peaks")
    parser.add_argument("--sigma_cutoff", metavar='SIGMA', default=3, type=float, help="\sigma value to draw the peak mask")
    parser.add_argument("--outfile", default="mask.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    MI = loadobj(args.geom,'miller_index')
    IS = loadobj(args.data,'image_series')
    GP = loadobj(args.peaks,'peak_model')
    P = loadobj(args.peaks,'peaks')

    # initialize the mask using Peaks
    mask = np.zeros(IS.shape,dtype='bool')

    # loop over phi values
    print(f'masking peaks with sigma above threshold: {args.sigma_cutoff}')
    for ind,sl in enumerate(IS.chunk_slice_iterator()):
        print(f'indexing chunk {ind}')
        MIdense = MI.regrid(IS.phi[sl[0]],IS.iy[sl[1]],IS.ix[sl[2]])
        dh = MIdense.h - np.round(MIdense.h)
        dk = MIdense.k - np.round(MIdense.k)
        dl = MIdense.l - np.round(MIdense.l)
        mask[sl] = GP.mask(dh,dk,dl,sigma_cutoff=args.sigma_cutoff)

    # add peaks
    print(f'masking count threshold peaks')
    P.to_mask(IS.phi,IS.iy,IS.ix,mask_in=mask)

    print(f"Saving mask to {args.outfile}")

    maskobj = GridData((IS.phi,IS.iy,IS.ix),mask)

    saveobj(maskobj,args.outfile,name='mask',append=False)

    print('done!')

if __name__ == "__main__":
    run()
