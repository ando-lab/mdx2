"""
Fit scaling model to unmerged corrected intensities
"""

import argparse

import numpy as np
import pandas as pd

from mdx2.utils import saveobj, loadobj
from mdx2.data import HKLTable
from mdx2.scaling import ScaledData, ScalingModel

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("hkl", help="NeXus file with hkl_table")
    parser.add_argument('--smoothness',type=float,default=1,help="amount to multiply the regularization parameter")
    parser.add_argument('--phi_increment',type=float,default=1,metavar="DEGREES",help="spacing of phi control points in degrees")
    parser.add_argument('--iter',type=int,default=5,help="number of iterations")
    parser.add_argument("--outfile", default="scales.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    hkl = loadobj(args.hkl,'hkl_table')

    print('Grouping redundant observations')
    (h,k,l), index_map, counts = hkl.unique()

    S = ScaledData(hkl.intensity,hkl.intensity_error,index_map,hkl.phi)

    # for phi axis, just estimate the range to the nearest degree, and put the samples at one degree increments.
    # seems reasonable but might fail in some situations...
    dphi = args.phi_increment # one degree increments
    phi_min = np.floor(hkl.phi.min()*dphi)/dphi
    phi_max = np.ceil(hkl.phi.max()*dphi)/dphi
    nsamp = np.round((phi_max-phi_min)/dphi).astype(int) + 1
    phi_points = np.linspace(phi_min,phi_max,nsamp)
    phi_vals = np.ones_like(phi_points)

    print(f'initializing scaling model with {phi_points.size} samples')
    Model = ScalingModel(phi_points,phi_vals)

    for j in range(args.iter):
        print(f'iteration {j+1} of {args.iter}')
        print('  re-calculating scales')
        S.apply(Model)
        print('  merging')
        Im,sigmam,counts = S.merge()
        print('  fitting the model')
        Model,x2 = S.fit(Model,Im,args.smoothness)
        print(f'  current x2: {x2}')

    print('finished refining')

    #hkl.intensity=S.I.filled(fill_value=np.nan).astype(np.float32)
    #hkl.intensity_error=S.sigma.filled(fill_value=np.nan).astype(np.float32)
    #hkl.scale = S.scale.astype(np.float32)

    #saveobj(hkl,args.outfile,name='hkl_table',append=False)
    saveobj(Model,args.outfile,name='scaling_model',append=False)

    print('done!')

if __name__ == "__main__":
    run()
