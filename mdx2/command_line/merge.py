"""
Merge corrected intensities using a scaling model
"""

import argparse

import numpy as np
#import pandas as pd
from nexusformat.nexus import nxload

from mdx2.utils import saveobj, loadobj
from mdx2.data import HKLTable
from mdx2.scaling import ScaledData, BatchModelRefiner

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("hkl", 
                        nargs='+', 
                        help="NeXus file(s) with hkl_table")
    parser.add_argument("--scale", 
                        nargs='+', 
                        help="NeXus file(s) with scaling models")
    parser.add_argument("--outfile", 
                        default="merged.nxs", 
                        help="name of the output NeXus file")
    parser.add_argument('--outlier',
                        type=float,
                        metavar="NSIGMA",
                        help="optional standard error cutoff for outlier rejection")
    parser.add_argument('--split',
                        choices=['randomHalf','weightedRandomHalf','Friedel'], # TODO: batch, Laue, Friedel, etc
                        help="also merge data into separate columns based on splitting criteria",
                       )
    parser.add_argument('--geometry',
                        help="NeXus file containing the Laue group symmetry operators, required if --split is 'Friedel'")
    parser.add_argument('--no-scaling',action='store_true',help='do not apply scaling model')
    parser.add_argument('--no-offset',action='store_true',help='do not apply offset model')
    parser.add_argument('--no-absorption',action='store_true',help='do not apply absorption model')
    parser.add_argument('--no-detector',action='store_true',help='do not apply detector model')

    return parser

def wrs(index_map,w):
    """weigted random split -- vectorized version"""
    index_jitter = np.random.random_sample(index_map.shape)*.2
    sort_order = np.argsort(index_map + index_jitter)

    sorted_ind_map = index_map[sort_order]
    sorted_w = w[sort_order]
    breakpoints = [0] + list(np.nonzero(np.diff(sorted_ind_map))[0] + 1) + [len(sorted_ind_map)]
    starts = np.array(breakpoints[:-1])
    stops = np.array(breakpoints[1:])

    isgrp1 = np.full_like(index_map,False,dtype=bool)
    flips = np.random.randint(2,size=len(starts))
    cs = np.cumsum(sorted_w)
    dcs = cs - cs[starts][sorted_ind_map]
    isincl = dcs > 0.5*dcs[stops-1][sorted_ind_map]
    isincl = np.logical_xor(flips[sorted_ind_map],isincl)
    isgrp1[sort_order[isincl]] = True
    return isgrp1

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    # load data into a giant table
    tabs = []

    for n, fn in enumerate(args.hkl):
        tmp = loadobj(fn,'hkl_table')
        tmp.batch = n*np.ones_like(tmp.op)
        tabs.append(tmp)

    hkl = HKLTable.concatenate(tabs)

    print('Grouping redundant observations')
    (h,k,l), index_map, counts = hkl.unique()
    
    S = ScaledData(
        hkl.intensity,
        hkl.intensity_error,
        index_map,
        phi=hkl.phi,
        s=hkl.s,
        ix=hkl.ix,
        iy=hkl.iy,
        batch=hkl.batch,
    )

    MR = BatchModelRefiner(S)
    
    if args.scale is not None:
        for fn, refiner in zip(args.scale, MR._batch_refiners):
            a = nxload(fn)
            if not args.no_absorption and ('absorption_model' in a.entry.keys()):
                refiner.absorption.model = loadobj(fn,'absorption_model')
            if not args.no_offset and ('offset_model' in a.entry.keys()):
                refiner.offset.model = loadobj(fn,'offset_model')
            if not args.no_detector and ('detector_model' in a.entry.keys()):
                refiner.detector.model = loadobj(fn,'detector_model')
            if not args.no_scaling and ('scaling_model' in a.entry.keys()):
                refiner.scaling.model = loadobj(fn,'scaling_model')
    
    print(f"applying scale factors")
    MR.apply()
    print(f"merging")
    Im,sigmam,counts = MR.data.merge()

    if args.outlier is not None:
        nout = MR.data.mask_outliers(Im,args.outlier) 
        print(f"removed {nout} outliers > {args.outlier} sigma")  
        print(f"merging again")
        Im,sigmam,counts = MR.data.merge()

    cols = dict(
        intensity=Im.filled(fill_value=np.nan).astype(np.float32),
        intensity_error=sigmam.filled(fill_value=np.nan).astype(np.float32),
        count=counts.astype(np.int32),
    )
    
    if args.split is not None:
        if args.split == 'weightedRandomHalf':
            print('Splitting according to the weighted random half algorithm')
            w = 1/MR.data.sigma**2
            isgrp1 = wrs(index_map,w.filled(fill_value=0))
            groups = [isgrp1,~isgrp1]           
        elif args.split == 'randomHalf':
            print('Splitting into random half-datasets (unweighted)')
            isgrp1 = wrs(index_map,np.ones_like(index_map))
            groups = [isgrp1,~isgrp1]
        elif args.split == 'Friedel':
            print('Splitting into Friedel pairs')
            if args.geometry is None:
                raise('--geometry argument is required for symmetry-based splitting')
            symm = loadobj(args.geometry,'symmetry')
            has_inversion = np.array([np.linalg.det(op) for op in symm.laue_group_operators]) < 0
            isminus = has_inversion[hkl.op]
            groups = [~isminus,isminus]
        else:
            raise('something bad happened')
        for j,g in enumerate(groups):
            G = MR.data.copy()
            G.mask = G.mask | ~g
            print(f'Merging group {j}')
            Imj,sigmamj,countsj = G.merge()
            cols[f"group_{j}_intensity"] = Imj.filled(fill_value=np.nan).astype(np.float32)
            cols[f"group_{j}_intensity_error"] = sigmamj.filled(fill_value=np.nan).astype(np.float32)
            cols[f"group_{j}_count"] = countsj.astype(np.int32)
        
    # create the output table object
    hkl_table = HKLTable(h,k,l,ndiv=hkl.ndiv,**cols)

    #ndiv = T.ndiv # save for later

    # # use pandas for merging
    # df = T.to_frame()
    #
    # # do weighted least squares
    # df['w'] = 1/df.intensity_error**2
    # df['Iw'] = df.w*df.intensity
    #
    # df = df.dropna()
    #
    # df_merged = df.groupby(['h','k','l']).aggregate({
    #     'Iw':'sum',
    #     'w':'sum',
    #     'rs_volume':'sum',
    #     's':'mean',
    #     })
    #
    # df_merged['intensity'] = df_merged.Iw/df_merged.w
    # df_merged['intensity_error'] = np.sqrt(1/df_merged.w)
    # df_merged = df_merged.drop(columns=['w','Iw'])
    #
    # hkl_table = HKLTable.from_frame(df_merged)
    # hkl_table.ndiv = ndiv # lost in conversion to/from dataframe

    saveobj(hkl_table,args.outfile,name='hkl_table',append=False)

    print('done!')

if __name__ == "__main__":
    run()
