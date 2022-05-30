"""
Merge corrected intensities using a scaling model
"""

import argparse

import numpy as np
#import pandas as pd

from mdx2.utils import saveobj, loadobj
from mdx2.data import HKLTable
from mdx2.scaling import ScaledData

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("hkl", help="NeXus file with hkl_table")
    parser.add_argument("--scale", help="NeXus file with scaling_model")
    parser.add_argument("--outfile", default="merged.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    T = loadobj(args.hkl,'hkl_table')


    print('Grouping redundant observations')
    (h,k,l), index_map, counts = T.unique()

    S = ScaledData(T.intensity,T.intensity_error,index_map,T.phi)

    if args.scale is not None:
        M = loadobj(args.scale,'scaling_model')
        print('Calculating scales')
        S.apply(M)

    print('Merging')
    intensity,intensity_error,count = S.merge()

    hkl_table = HKLTable(h,k,l,ndiv=T.ndiv)
    hkl_table.intensity=intensity.filled(fill_value=np.nan).astype(np.float32)
    hkl_table.intensity_error=intensity_error.filled(fill_value=np.nan).astype(np.float32)
    hkl_table.count=count.astype(np.int32)

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
