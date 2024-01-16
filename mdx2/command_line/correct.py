"""
Apply corrections to integrated data
"""

import argparse

import numpy as np
# import pandas as pd

from mdx2.utils import saveobj, loadobj
# from mdx2.data import ImageSeries
# from mdx2.data import HKLTable

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("geom", help="NeXus file with miller_index")
    parser.add_argument("hkl", help="NeXus file with hkl_table")
    parser.add_argument("--background", help="NeXus file with background map")
    parser.add_argument("--attenuation", metavar='TF', default=True, help="apply attenuation correction?")
    parser.add_argument("--efficiency", metavar='TF', default=True, help="apply efficiency correction?")
    parser.add_argument("--polarization", metavar='TF', default=True, help="apply polarization correction?")
    parser.add_argument("--p1",action='store_true',help="map Miller indices to asymmetric unit for P1 (Friedel symmetry only)")
    parser.add_argument("--outfile", default="corrected.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    T = loadobj(args.hkl,'hkl_table')

    # hack to work with older versions
    if '_ndiv' in T.__dict__:
        T.ndiv = T._ndiv
        del(T._ndiv)

    Corrections = loadobj(args.geom,'corrections')
    Crystal = loadobj(args.geom,'crystal')

    if args.p1:
        print('ignoring space group information and using P1 symmetry only')
        Symmetry = None
    else:
        Symmetry = loadobj(args.geom,'symmetry')

    UB = Crystal.ub_matrix

    # computing scattering vector magnitude
    print('calculating scattering vector magnitude (s)')
    s = UB @ np.stack((T.h,T.k,T.l))
    T.s = np.sqrt(np.sum(s*s,axis=0))

    # map h,k,l to asymmetric unit
    print('mapping Miller indices to the asymmetric unit')
    T = T.to_asu(Symmetry)

    # apply corrections to intensities

    Cinterp = Corrections.interpolate(T.iy,T.ix)

    count_rate = T.counts/T.seconds
    count_rate_error = np.sqrt(T.counts)/T.seconds

    if args.background is not None:
        Bkg = loadobj(args.background,'binned_image_series')
        bkg_count_rate = Bkg.interpolate(T.phi,T.iy,T.ix)
        print('subtracting background from count rate')
        count_rate = count_rate - bkg_count_rate

    solid_angle = Cinterp['solid_angle']

    for corr in ['attenuation','efficiency','polarization']:
        if getattr(args,corr):
            print('correcting solid angle for',corr)
            solid_angle *= Cinterp[corr]

    print(f'computing the swept reciprocal space volume fraction (rs_volume)')
    T.rs_volume = T.pixels*Cinterp['d3s']/np.linalg.det(UB)

    print(f'computing intensity and intensity_error')
    T.intensity = count_rate/solid_angle
    T.intensity_error = count_rate_error/solid_angle

    # remove some unnecessary columns
    del(T.counts)
    del(T.seconds)
    del(T.pixels)

    # save some disk space
    T.h = T.h.astype(np.float32)
    T.k = T.k.astype(np.float32)
    T.l = T.l.astype(np.float32)
    T.s = T.s.astype(np.float32)
    T.intensity = T.intensity.astype(np.float32)
    T.intensity_error = T.intensity_error.astype(np.float32)
    T.ix = T.ix.astype(np.float32)
    T.iy = T.iy.astype(np.float32)
    T.phi = T.phi.astype(np.float32)
    T.rs_volume = T.rs_volume.astype(np.float32)
    T.n = T.n.astype(np.int32)
    T.op = T.op.astype(np.int32)

    saveobj(T,args.outfile,name='hkl_table',append=False)

    print('done!')

if __name__ == "__main__":
    run()
