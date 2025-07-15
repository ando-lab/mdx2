"""
Fit scaling model to unmerged corrected intensities
"""

import argparse
import os

import numpy as np
import pandas as pd

from mdx2.utils import saveobj, loadobj
from mdx2.data import HKLTable
from mdx2.scaling import ScaledData, BatchModelRefiner

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("hkl", 
                        nargs='+', 
                        help="NeXus file(s) with hkl_table")
    parser.add_argument("--mca2020", 
                        action='store_true',
                        help="shortcut for --scaling.enable True --offset.enable True --detector.enable True --absorption.enable True")
    parser.add_argument("--outfile", nargs='+', help="name of the output NeXus file(s). If not specified, will attempt a sensible name such as scales.nxs for a single input file")

    scaling_group = parser.add_argument_group(title='Scaling parameters')
    scaling_group.add_argument('--scaling.enable',dest='scaling_enable',default=True,metavar='TF',
                              help="include smooth scale factor vs. phi")
    scaling_group.add_argument('--scaling.alpha',dest='scaling_alpha',default=1.0,type=float,metavar="ALPHA",
                             help="amount to rescale the default smoothness (regularization parameter)")
    scaling_group.add_argument('--scaling.dphi',dest='scaling_dphi',default=1.0,type=float,metavar="DEGREES",
                              help="spacing of phi control points in degrees")
    scaling_group.add_argument('--scaling.niter',dest='scaling_niter',default=10,type=int,metavar="N",
                              help="maximum iterations in refinement")
    scaling_group.add_argument('--scaling.x2tol',dest='scaling_x2tol',default=1E-4,type=float,metavar="TOL",
                              help="maximum change in x2 to stop refinement early")
    scaling_group.add_argument('--scaling.outlier',dest='scaling_outlier',default=10,type=float,metavar="NSIGMA",
                              help="standard error cutoff for outlier rejection after refinement")

    offset_group = parser.add_argument_group(title='Offset parameters')
    offset_group.add_argument('--offset.enable',dest='offset_enable',default=False,metavar='TF',
                             help="include smooth offset vs. resolution and phi")
    offset_group.add_argument('--offset.alpha_x',dest='offset_alpha_x',default=1.0,type=float,metavar="ALPHA",
                             help="smoothness vs. s (resolution): multiplies the regularization parameter")
    offset_group.add_argument('--offset.alpha_y',dest='offset_alpha_y',default=1.0,type=float,metavar="ALPHA",
                             help="smoothness vs. phi -- multiplies the regularization parameter")
    offset_group.add_argument('--offset.alpha_min',dest='offset_alpha_min',default=0.001,type=float,metavar="ALPHA",
                             help="deviation from offset.min: multiplies the regularization parameter")
    offset_group.add_argument('--offset.min',dest='offset_min',default=0.0,type=float,metavar="VAL",
                             help="minimum value of offset")
    offset_group.add_argument('--offset.dphi',dest='offset_dphi',default=2.5,type=float,metavar="DEGREES",
                              help="spacing of phi control points in degrees")
    offset_group.add_argument('--offset.ns',dest='offset_ns',default=31,type=int,metavar="N",
                              help="number of s (resolution) control points")
    offset_group.add_argument('--offset.niter',dest='offset_niter',default=5,type=int,metavar="N",
                              help="maximum iterations in refinement")
    offset_group.add_argument('--offset.x2tol',dest='offset_x2tol',default=1E-3,type=float,metavar="TOL",
                              help="maximum change in x2 to stop refinement early")
    offset_group.add_argument('--offset.outlier',dest='offset_outlier',default=5,type=float,metavar="NSIGMA",
                              help="standard error cutoff for outlier rejection after refinement")


    detector_group = parser.add_argument_group(title='Detector parameters')
    detector_group.add_argument('--detector.enable',dest='detector_enable',default=False,metavar='TF',
                             help="include smooth scale vs. detector xy position")
    detector_group.add_argument('--detector.alpha',dest='detector_alpha',default=1.0,type=float,metavar="ALPHA",
                             help="smoothness vs. xy position: multiplies the regularization parameter")
    detector_group.add_argument('--detector.nx',dest='detector_nx',default=200,type=float,metavar="N",
                              help="number of grid control points in the x direction")
    detector_group.add_argument('--detector.ny',dest='detector_ny',default=200,type=float,metavar="N",
                              help="number of grid control points in the y direction")
    detector_group.add_argument('--detector.niter',dest='detector_niter',default=5,type=int,metavar="N",
                              help="maximum iterations in refinement")
    detector_group.add_argument('--detector.x2tol',dest='detector_x2tol',default=1E-3,type=float,metavar="TOL",
                              help="maximum change in x2 to stop refinement early")
    detector_group.add_argument('--detector.outlier',dest='detector_outlier',default=5,type=float,metavar="NSIGMA",
                              help="standard error cutoff for outlier rejection after refinement")


    absorption_group = parser.add_argument_group(title='Absorption parameters')
    absorption_group.add_argument('--absorption.enable',dest='absorption_enable',default=False,metavar='TF',
                             help="include smooth scale vs. detector xy position and phi")
    absorption_group.add_argument('--absorption.alpha_xy',dest='absorption_alpha_xy',default=10.0,type=float,metavar="ALPHA",
                             help="smoothness vs. xy position: multiplies the regularization parameter")
    absorption_group.add_argument('--absorption.alpha_z',dest='absorption_alpha_z',default=1.0,type=float,metavar="ALPHA",
                             help="smoothness vs. phi: multiplies the regularization parameter")
    absorption_group.add_argument('--absorption.nx',dest='absorption_nx',default=20,type=float,metavar="N",
                              help="number of grid control points in the x direction")
    absorption_group.add_argument('--absorption.ny',dest='absorption_ny',default=20,type=float,metavar="N",
                              help="number of grid control points in the y direction")
    absorption_group.add_argument('--absorption.dphi',dest='absorption_dphi',default=5.0,type=float,metavar="DEGREES",
                              help="spacing of phi control points in degrees")
    absorption_group.add_argument('--absorption.niter',dest='absorption_niter',default=5,type=int,metavar="N",
                              help="maximum iterations in refinement")
    absorption_group.add_argument('--absorption.x2tol',dest='absorption_x2tol',default=1E-4,type=float,metavar="TOL",
                              help="maximum change in x2 to stop refinement early")
    absorption_group.add_argument('--absorption.outlier',dest='absorption_outlier',default=5,type=float,metavar="NSIGMA",
                              help="standard error cutoff for outlier rejection after refinement")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)
    
    if args.mca2020:
        args.scaling_enable=True
        args.detector_enable=True
        args.absorption_enable=True
        args.offset_enable=True

    def generate_default_outfiles(infiles):
        dirs = [os.path.dirname(fn) for fn in infiles]
        if len(set(dirs))==len(dirs): # dirs are unique
            return [os.path.join(d,'scales.nxs') for d in dirs]
        if len(set(dirs))==1: # dirs are identical
            roots = [os.path.splitext(os.path.split(fn)[-1])[0] for fn in infiles]
            if all(['_' in root for root in roots]):
                postfix = [root.split('_')[-1] for root in roots]
                if len(set(postfix))==len(postfix): # postfixes are unique
                    return [os.path.join(dirs[0],f'scales_{pf}.nxs') for pf in postfix]

    if args.outfile is None:
        args.outfile = generate_default_outfiles(args.hkl)
        if args.outfile is None:
            raise ValueError('unable to auto-generate output file names from input name pattern')
    
    # load data into a giant table
    tabs = []

    for n, fn in enumerate(args.hkl):
        tmp = loadobj(fn,'hkl_table')
        tmp.batch = n*np.ones_like(tmp.op)
        tabs.append(tmp)

    hkl = HKLTable.concatenate(tabs)

    print('Grouping redundant observations')
    (h,k,l), index_map, count
    s = hkl.unique()
    
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
    
    if args.scaling_enable:
        MR.add_scaling_models(
            dphi=args.scaling_dphi,
        )
    if args.offset_enable:
        MR.add_offset_models(
            dphi=args.offset_dphi,
            ns=args.offset_ns,
        )
    if args.absorption_enable:
        MR.add_absorption_models(
            dphi=args.absorption_dphi,
            nix=args.absorption_nx,
            niy=args.absorption_ny,
        )
    if args.detector_enable:
        MR.add_detector_model(
            nix=args.detector_nx,
            niy=args.detector_ny,
        )

    print(f"applying scale factors")
    MR.apply()
    print(f"merging")
    Im,sigmam,counts = MR.data.merge()
    
    old_x2 = 1E6 # initialize to some large number
        
    if args.scaling_enable:
        print("optimizing scale vs. phi (b)")
        for j in range(args.scaling_niter):
            print(f"  iteration {j+1} of {args.scaling_niter}")
            print(f"    fitting the model")
            x2 = MR.bfit(Im,args.scaling_alpha)
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im,sigmam,counts = MR.data.merge()
            if old_x2-x2 < args.scaling_x2tol:
                print(f"    change in x2 less than tolerance of {args.scaling_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im,args.scaling_outlier) 
        print(f"removed {nout} outliers > {args.scaling_outlier} sigma")
        
    if args.scaling_enable and args.offset_enable:
        print("optimizing scale and offset vs. phi and resolution (b/c)")
        for j in range(args.offset_niter):
            print(f"  iteration {j+1} of {args.offset_niter}")
            print(f"    fitting the model")
            MR.cfit(Im,
                     args.offset_alpha_x,
                     args.offset_alpha_y,
                     args.offset_alpha_min,
                     args.offset_min,
                    ) # 1,1,.1,min_c=0
            x2 = MR.bfit(Im,args.scaling_alpha)
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im,sigmam,counts = MR.data.merge()
            if old_x2-x2 < args.offset_x2tol:
                print(f"    change in x2 less than tolerance of {args.offset_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im,args.offset_outlier) 
        print(f"removed {nout} outliers > {args.offset_outlier} sigma")

    if args.offset_enable:
        print("optimizing offset vs. phi and resolution (c)")
        for j in range(args.offset_niter):
            print(f"  iteration {j+1} of {args.offset_niter}")
            print(f"    fitting the model")
            x2 = MR.cfit(Im,
                     args.offset_alpha_x,
                     args.offset_alpha_y,
                     args.offset_alpha_min,
                     args.offset_min,
                    ) # 1,1,.1,min_c=0
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im,sigmam,counts = MR.data.merge()
            if old_x2-x2 < args.offset_x2tol:
                print(f"    change in x2 less than tolerance of {args.offset_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im,args.offset_outlier) 
        print(f"removed {nout} outliers > {args.offset_outlier} sigma")

    if args.detector_enable:
        print("optimizing scale vs. detector position (d)")
        for j in range(args.detector_niter):
            print(f"  iteration {j+1} of {args.detector_niter}")
            print(f"    fitting the model")
            x2 = MR.dfit(Im,args.detector_alpha) 
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im,sigmam,counts = MR.data.merge()
            if old_x2-x2 < args.detector_x2tol:
                print(f"    change in x2 less than tolerance of {args.detector_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im,args.detector_outlier) 
        print(f"removed {nout} outliers > {args.detector_outlier} sigma")

    if args.absorption_enable:
        print("optimizing scale vs. detector position and phi (a)")
        for j in range(args.absorption_niter):
            print(f"  iteration {j+1} of {args.absorption_niter}")
            print(f"    fitting the model")
            x2 = MR.afit(Im,
                         args.absorption_alpha_xy,
                         args.absorption_alpha_z) 
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im,sigmam,counts = MR.data.merge()
            if old_x2-x2 < args.absorption_x2tol:
                print(f"    change in x2 less than tolerance of {args.absorption_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im,args.absorption_outlier) 
        print(f"removed {nout} outliers > {args.absorption_outlier} sigma")

    print('finished refining')

    for model_refiner, fn in zip(MR._batch_refiners, args.outfile):
        
        models = dict(
            scaling_model=model_refiner.scaling.model,
            detector_model=model_refiner.detector.model,
            absorption_model=model_refiner.absorption.model,
            offset_model=model_refiner.offset.model,
        )
        
        file_created = False
        for name, model in models.items():
            if model is not None:
                saveobj(model,fn,name=name,append=file_created)
                file_created = True

    print('done!')

if __name__ == "__main__":
    run()
