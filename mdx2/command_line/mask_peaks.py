"""
Create a peak mask for an image stack
"""

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field  # pip install simple-parsing

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.geometry import GridData
from mdx2.io import loadobj, saveobj


@dataclass
class Parameters:
    """Options for creating a peak mask"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    peaks: str = field(positional=True)  # NeXus data file containing peak_model and peaks
    sigma_cutoff: float = 3.0  # contour level for drawing the peak mask
    outfile: str = "mask.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.mask_peaks.log"  # name of the log file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    bragg: bool = False  # create a Bragg peak mask instead

    def __post_init__(self):
        """Validate sigma_cutoff parameter"""
        if self.sigma_cutoff <= 0:
            raise ValueError(f"sigma_cutoff must be > 0, got {self.sigma_cutoff}")


def run_mask_peaks(params):
    """Run the mask peaks script"""
    geom = params.geom
    data = params.data
    peaks = params.peaks
    outfile = params.outfile
    sigma_cutoff = params.sigma_cutoff
    nproc = params.nproc
    bragg = params.bragg

    logger.info("Loading geometry, image data, and peak model...")
    MI = loadobj(geom, "miller_index")
    Symm = loadobj(geom, "symmetry")
    IS = loadobj(data, "image_series")
    GP = loadobj(peaks, "peak_model")
    P = loadobj(peaks, "peaks")

    # initialize the mask using Peaks
    mask = np.zeros(IS.shape, dtype="bool")

    def maskchunk(sl):
        MIdense = MI.regrid(IS.phi[sl[0]], IS.iy[sl[1]], IS.ix[sl[2]])
        H = np.round(MIdense.h)
        K = np.round(MIdense.k)
        L = np.round(MIdense.l)
        dh = MIdense.h - H
        dk = MIdense.k - K
        dl = MIdense.l - L
        isrefl = Symm.is_reflection(H, K, L)
        return isrefl & GP.mask(dh, dk, dl, sigma_cutoff=sigma_cutoff)

    slices = list(IS.chunk_slice_iterator())
    logger.info("Computing peak mask for {} image chunks (requested n_jobs: {})...", len(slices), nproc)
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        masklist = parallel(delayed(maskchunk)(sl) for sl in slices)
    logger.info("Mask computation completed")
    for msk, sl in zip(masklist, slices):
        mask[sl] = msk  # <-- note, this copy step could be avoided with shared mem

    if bragg:
        logger.info("Inverting mask to retain Bragg peaks")
        mask = np.logical_not(mask)
    else:
        logger.info("Adding count threshold peaks to mask")
        P.to_mask(IS.phi, IS.iy, IS.ix, mask_in=mask)

    masked_fraction = np.sum(mask) / mask.size
    logger.info("Masked pixels: {:.2%}", masked_fraction)

    logger.info("Saving mask to {}...", outfile)
    maskobj = GridData((IS.phi, IS.iy, IS.ix), mask)
    saveobj(maskobj, outfile, name="mask", append=False)
    logger.info("Mask creation completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_mask_peaks))


if __name__ == "__main__":
    run()
