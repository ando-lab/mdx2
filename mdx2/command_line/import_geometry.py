"""
Import experimental geometry using the dxtbx machinery
"""

import argparse
import json

import numpy as np

import mdx2.geometry as geom
from mdx2.utils import saveobj

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("expt", help="dials experiments file, such as refined.expt")
    parser.add_argument("--sample_spacing", nargs=3, metavar=('PHI','IY','IX'), type=int, default=[1,10,10], help="inverval between samples in degrees or pixels")
    parser.add_argument("--outfile", default="geometry.nxs", help="name of the output NeXus file")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)

    exptfile = args.expt
    spacing_phi_px = tuple(args.sample_spacing)
    spacing_px = spacing_phi_px[1:]

    print("Computing miller index lookup grid")
    miller_index = geom.MillerIndex.from_expt( exptfile,
        sample_spacing=spacing_phi_px,
        )

    print("Computing geometric correction factors")
    corrections = geom.Corrections.from_expt( exptfile,
        sample_spacing=spacing_px,
        )

    print("Gathering space group info")
    symmetry = geom.Symmetry.from_expt(exptfile)

    print("Gathering unit cell info")
    crystal = geom.Crystal.from_expt(exptfile)

    print(f"Saving geometry to {args.outfile}")

    saveobj(crystal,args.outfile,name='crystal',append=False)
    saveobj(symmetry,args.outfile,name='symmetry',append=True)
    saveobj(corrections,args.outfile,name='corrections',append=True)
    nxs = saveobj(miller_index,args.outfile,name='miller_index',append=True)

    print("done!")

if __name__ == "__main__":
    run()
