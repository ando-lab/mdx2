"""
Integrate counts in an image stack on a Miller index grid
"""

import argparse

import numpy as np
import pandas as pd

from mdx2.utils import saveobj, loadobj
from mdx2.data import ImageSeries
from mdx2.data import HKLTable
from nexusformat.nexus import nxload # mask is too big to read all at once?
#from . import MDX2Parser

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("geom", help="NeXus file with miller_index")
    parser.add_argument("data", help="NeXus file with image_series")
    parser.add_argument("--mask", help="NeXus file with mask")
    parser.add_argument("--subdivide", nargs=3, metavar='N', type=int, default=[1,1,1], help="subdivisions of the Miller index grid")
    #parser.add_argument("--limits", nargs=6, metavar=('hmin,hmax,kmin,kmax,lmin,lmax'), type=float, help="included region")
    parser.add_argument("--max_spread", type=float, default=1.0, metavar='DEGREES', help="maximum angular spread for binning partial observations")
    parser.add_argument("--outfile", default="integrated.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    MI = loadobj(args.geom,'miller_index')
    IS = loadobj(args.data,'image_series')

    if args.mask is not None:
        #MA = loadobj(args.mask,'mask')
        #mask = MA.data
        nxs = nxload(args.mask) # <-- loadobj fails if the array is too large. weird
        mask = nxs.entry.mask.signal # nxfield

    else:
        mask = None

    #if opts.limits is not None:
    #    lim = opts.limits
    #else:
    #    lim = None

    ndiv = args.subdivide

    max_degrees = args.max_spread

    T = [] # list of tables
    print(f'Looping through chunks')
    for ind,sl in enumerate(IS.chunk_slice_iterator()):
        ims = IS[sl]
        if mask is not None:
            tab = ims.index(MI,mask=mask[sl].nxdata) # added nxdata to deal with NXfield wrapper
        else:
            tab = ims.index(MI)
        tab.ndiv = ndiv
        tab_binned = tab.bin(count_name='pixels')

        print(f'  binned chunk {ind} from {len(tab)} to {len(tab_binned)} voxels')
        T.append(tab_binned)
        #if ind > 5: break # for debugging

    print(f'Summing partial observations over {len(T)} chunks')
    df = HKLTable.concatenate(T).to_frame()#.set_index(['h','k','l'])

    df['tmp'] = df['phi']/df['pixels']
    delta_phi = df.tmp - df.groupby(['h','k','l'])['tmp'].transform('min')
    df['n'] = np.floor(delta_phi/max_degrees)
    df = df.drop(columns=['tmp'])

    df = df.groupby(['h','k','l','n']).sum()

    # compute mean positions in the scan
    df['phi'] = df['phi']/df['pixels']
    df['iy'] = df['iy']/df['pixels']
    df['ix'] = df['ix']/df['pixels']

    print(f'  binned from {np.sum([len(t) for t in T])} to {len(df)} voxels')

    print(f"Saving table of integrated data to {args.outfile}")

    hkl_table = HKLTable.from_frame(df)
    hkl_table.ndiv = ndiv # lost in conversion to/from dataframe

    hkl_table.h = hkl_table.h.astype(np.float32)
    hkl_table.k = hkl_table.k.astype(np.float32)
    hkl_table.l = hkl_table.l.astype(np.float32)
    hkl_table.phi = hkl_table.phi.astype(np.float32)
    hkl_table.ix = hkl_table.ix.astype(np.float32)
    hkl_table.iy = hkl_table.iy.astype(np.float32)
    hkl_table.seconds = hkl_table.seconds.astype(np.float32)
    hkl_table.counts = hkl_table.counts.astype(np.int32)
    hkl_table.pixels = hkl_table.pixels.astype(np.int32)

    saveobj(hkl_table,args.outfile,name='hkl_table',append=False)

    print('done!')

if __name__ == "__main__":
    run()
