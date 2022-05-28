"""
Merge corrected (scaled?) intensities
"""

import argparse

import numpy as np
import pandas as pd

from mdx2.utils import saveobj, loadobj
from mdx2.data import HKLTable

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("hkl", help="NeXus file with hkl_table")
    parser.add_argument("--outfile", default="merged.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    T = loadobj(args.hkl,'hkl_table')
    ndiv = T._ndiv # save for later

    # use pandas for merging
    df = T.to_frame()

    # do weighted least squares
    df['w'] = 1/df.intensity_error**2
    df['Iw'] = df.w*df.intensity

    df = df.dropna()

    df_merged = df.groupby(['h','k','l']).aggregate({
        'Iw':'sum',
        'w':'sum',
        'rs_volume':'sum',
        's':'mean',
        })

    df_merged['intensity'] = df_merged.Iw/df_merged.w
    df_merged['intensity_error'] = np.sqrt(1/df_merged.w)
    df_merged = df_merged.drop(columns=['w','Iw'])

    hkl_table = HKLTable.from_frame(df_merged)
    hkl_table._ndiv = ndiv # lost in conversion to/from dataframe

    saveobj(hkl_table,args.outfile,name='hkl_table',append=False)

    print('done!')

if __name__ == "__main__":
    run()
