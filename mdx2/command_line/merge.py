"""
Merge corrected intensities using a scaling model
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger
from nexusformat.nexus import nxload
from simple_parsing import field  # pip install simple-parsing

from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.data import HKLTable
from mdx2.io import loadobj, saveobj
from mdx2.scaling import BatchModelRefiner, ScaledData


@dataclass
class Parameters:
    """Options for merging scaled intensities"""

    hkl: List[str] = field(positional=True, nargs="+")  # NeXus file(s) containing hkl_table
    scale: List[str] = field(nargs="+")  # NeXus file(s) with scaling models
    outlier: Optional[float] = None  # optional standard error cutoff for outlier rejection
    split: Optional[str] = field(default=None, choices=["randomHalf", "weightedRandomHalf", "Friedel"])
    """also merge data into separate columns based on splitting criteria"""
    geometry: Optional[str] = (
        None  # NeXus file containing the Laue group symmetry operators, required for --split Friedel
    )
    outfile: str = "merged.nxs"  # name of the output NeXus file
    scaling: bool = field(default=True, negative_prefix="--no-")  # apply scaling model if present
    offset: bool = field(default=True, negative_prefix="--no-")  # apply offset model if present
    absorption: bool = field(default=True, negative_prefix="--no-")  # apply absorption model if present
    detector: bool = field(default=True, negative_prefix="--no-")  # apply detector model if present
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)


    def __post_init__(self):
        """Validate inter-parameter dependencies and outlier parameter"""
        if self.split == "Friedel" and self.geometry is None:
            raise ValueError("--geometry argument is required for symmetry-based splitting")
        if self.outlier is not None and self.outlier <= 0:
            raise ValueError(f"outlier must be > 0, got {self.outlier}")
        if len(self.scale) != len(self.hkl):
            raise ValueError(
                f"Number of scale files ({len(self.scale)}) must match number of HKL inputs ({len(self.hkl)})"
            )


# note: negative_prefix is used to allow --no-scaling, --no-offset, etc. for consistency with old argparse api
# future: could switch split to an Enum type, to simplify selection of choices by alternate APIs


def wrs(index_map, w):
    """weigted random split -- vectorized version"""
    index_jitter = np.random.random_sample(index_map.shape) * 0.2
    sort_order = np.argsort(index_map + index_jitter)

    sorted_ind_map = index_map[sort_order]
    sorted_w = w[sort_order]
    breakpoints = [0] + list(np.nonzero(np.diff(sorted_ind_map))[0] + 1) + [len(sorted_ind_map)]
    starts = np.array(breakpoints[:-1])
    stops = np.array(breakpoints[1:])

    isgrp1 = np.full_like(index_map, False, dtype=bool)
    flips = np.random.randint(2, size=len(starts))
    cs = np.cumsum(sorted_w)
    dcs = cs - cs[starts][sorted_ind_map]
    isincl = dcs > 0.5 * dcs[stops - 1][sorted_ind_map]
    isincl = np.logical_xor(flips[sorted_ind_map], isincl)
    isgrp1[sort_order[isincl]] = True
    return isgrp1


def run_merge(params):
    """Run the merge script"""
    hkl = params.hkl
    scale = params.scale
    outfile = params.outfile
    outlier = params.outlier
    split = params.split
    apply_scaling = params.scaling
    apply_offset = params.offset
    apply_absorption = params.absorption
    apply_detector = params.detector
    geometry = params.geometry

    if params.nproc != 1:
        logger.warning("Serial execution only, ignoring nproc value")
 
    # load data into a giant table
    logger.info("Loading {} HKL table(s)...", len(hkl))
    tabs = []

    for n, fn in enumerate(hkl):
        tmp = loadobj(fn, "hkl_table")
        tmp.batch = n * np.ones_like(tmp.op)
        tabs.append(tmp)

    hkl = HKLTable.concatenate(tabs)
    logger.info("Total reflections loaded: {}", len(hkl))

    logger.info("Grouping redundant observations...")
    (h, k, l), index_map, counts = hkl.unique()

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

    if scale is not None:
        logger.info("Loading scaling models...")
        for fn, refiner in zip(scale, MR.batch_refiners):
            a = nxload(fn)
            if apply_absorption and ("absorption_model" in a.entry.keys()):
                refiner.absorption.model = loadobj(fn, "absorption_model")
            if apply_offset and ("offset_model" in a.entry.keys()):
                refiner.offset.model = loadobj(fn, "offset_model")
            if apply_detector and ("detector_model" in a.entry.keys()):
                refiner.detector.model = loadobj(fn, "detector_model")
            if apply_scaling and ("scaling_model" in a.entry.keys()):
                refiner.scaling.model = loadobj(fn, "scaling_model")

    logger.info("Applying scale factors...")
    MR.apply()
    logger.info("Merging observations...")
    Im, sigmam, counts = MR.data.merge()

    if outlier is not None:
        nout = MR.data.mask_outliers(Im, outlier)
        logger.info("Removed {} outliers (>{} sigma)", nout, outlier)
        logger.info("Merging again...")
        Im, sigmam, counts = MR.data.merge()

    cols = dict(
        intensity=Im.filled(fill_value=np.nan).astype(np.float32),
        intensity_error=sigmam.filled(fill_value=np.nan).astype(np.float32),
        count=counts.astype(np.int32),
    )

    if split is not None:
        if split == "weightedRandomHalf":
            logger.info("Splitting by weighted random half algorithm...")
            w = 1 / MR.data.sigma**2
            isgrp1 = wrs(index_map, w.filled(fill_value=0))
            groups = [isgrp1, ~isgrp1]
        elif split == "randomHalf":
            logger.info("Splitting into random half-datasets (unweighted)...")
            isgrp1 = wrs(index_map, np.ones_like(index_map))
            groups = [isgrp1, ~isgrp1]
        elif split == "Friedel":
            logger.info("Splitting into Friedel pairs...")
            symm = loadobj(geometry, "symmetry")
            has_inversion = np.array([np.linalg.det(op) for op in symm.laue_group_operators]) < 0
            isminus = has_inversion[hkl.op]
            groups = [~isminus, isminus]
        else:
            raise ValueError(f"Unhandled merge mode: {split!r}")
        for j, g in enumerate(groups):
            G = MR.data.copy()
            G.mask = G.mask | ~g
            logger.info("Merging group {}...", j)
            Imj, sigmamj, countsj = G.merge()
            cols[f"group_{j}_intensity"] = Imj.filled(fill_value=np.nan).astype(np.float32)
            cols[f"group_{j}_intensity_error"] = sigmamj.filled(fill_value=np.nan).astype(np.float32)
            cols[f"group_{j}_count"] = countsj.astype(np.int32)

    # create the output table object
    hkl_table = HKLTable(h, k, l, ndiv=hkl.ndiv, **cols)
    logger.info("Unique reflections in merged output: {}", len(hkl_table))

    # ndiv = T.ndiv # save for later

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

    logger.info("Saving merged data to {}...", outfile)
    saveobj(hkl_table, outfile, name="hkl_table", append=False)
    logger.info("Merge completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_merge))


if __name__ == "__main__":
    run()
