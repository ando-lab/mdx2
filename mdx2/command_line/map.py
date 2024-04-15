"""
Create a map from data in an hkl table
"""

import argparse

import numpy as np
import pandas as pd

from mdx2.utils import saveobj, loadobj
# from mdx2.data import ImageSeries
from mdx2.data import HKLTable
from mdx2.geometry import GridData

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("geom", help="NeXus file with symmetry and crystal")
    parser.add_argument("hkl", help="NeXus file with hkl_table")
    parser.add_argument("--symmetry", default=True, metavar='TF', help = "apply symmetry operators?")
    parser.add_argument('--limits', default=[0,10,0,10,0,10], type=float, nargs=6, metavar=('H1','H2','K1','K2','L1','L2'), help="region to map")
    parser.add_argument('--signal',default='intensity', help="column in hkl_table to map")
    parser.add_argument("--outfile", default="map.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)
    
    # fix argparse ~bug where booleans are given as strings
    for arg in ['symmetry']:
        if getattr(args,arg) in ['True','true','T','t']:
            setattr(args,arg,True)
        else:
            setattr(args,arg,False)

    T = loadobj(args.hkl,'hkl_table')
    Crystal = loadobj(args.geom,'crystal') # unused currently
    Symmetry = loadobj(args.geom,'symmetry') # used only if symmetry flag is set
    signal = args.signal
    hmin,hmax,kmin,kmax,lmin,lmax = args.limits
    ndiv = T.ndiv

    Hmin = np.round(hmin*ndiv[0]).astype(int)
    Hmax = np.round(hmax*ndiv[0]).astype(int)
    Kmin = np.round(kmin*ndiv[1]).astype(int)
    Kmax = np.round(kmax*ndiv[1]).astype(int)
    Lmin = np.round(lmin*ndiv[2]).astype(int)
    Lmax = np.round(lmax*ndiv[2]).astype(int)

    h_axis = np.arange(Hmin,Hmax+1)/ndiv[0]
    k_axis = np.arange(Kmin,Kmax+1)/ndiv[1]
    l_axis = np.arange(Lmin,Lmax+1)/ndiv[2]

    print('map region:')
    print(f'  h from {h_axis[0]} to {h_axis[-1]} ({h_axis.size} grid points)')
    print(f'  k from {k_axis[0]} to {k_axis[-1]} ({k_axis.size} grid points)')
    print(f'  l from {l_axis[0]} to {l_axis[-1]} ({l_axis.size} grid points)')

    print('generating Miller index array')
    h,k,l = np.meshgrid(h_axis,k_axis,l_axis,indexing='ij')

    Tgrid = HKLTable(h.ravel(),k.ravel(),l.ravel(),ndiv=ndiv)

    if args.symmetry:
        print('mapping Miller indices to asymmetric unit')
        Tgrid = Tgrid.to_asu(Symmetry)

    print(f'looking up {args.signal} in data table')
    # lookup in the table
    df = T.to_frame().set_index(['h','k','l'])
    dfgrid = Tgrid.to_frame().set_index(['h','k','l'])
    dfgrid = pd.merge(dfgrid,df[signal],sort=False,on=['h','k','l'],how='left')

    print('preparing output array')
    data = dfgrid[signal].to_numpy().reshape(h.shape)
    G = GridData((h_axis,k_axis,l_axis),data,axes_names=['h','k','l'])
    saveobj(G,args.outfile,name=signal,append=False)

    print('done!')

if __name__ == "__main__":
    run()
