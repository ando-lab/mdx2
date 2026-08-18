"""
Bin down an image stack
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.geometry import GridData
from mdx2.io import loadobj, nxload, saveobj
from mdx2.utils import slice_sections


@dataclass
class Parameters:
    """Options for binning a series of images in the phi and xy directions"""

    data: str = field(positional=True)  # NeXus data file containing the image_series
    bins: Tuple[int, int, int] = field(positional=True)  # number per bin in each direction (frames, y, x)
    mask: Optional[str] = None  # name of NeXus file containing the mask
    valid_range: Optional[Tuple[int, int]] = None  # minimum and maximum valid data values
    outfile: str = "binned.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.bin_image_series.log"  # name of the log file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)

    def __post_init__(self):
        """Validate bins and valid_range parameters"""
        for i, bin_size in enumerate(self.bins):
            if bin_size <= 0:
                raise ValueError(f"bins[{i}] must be positive, got {bin_size}")
        if self.valid_range is not None and self.valid_range[0] >= self.valid_range[1]:
            raise ValueError(f"valid_range[0] must be < valid_range[1], got {self.valid_range}")


def run_bin_image_series(params):
    """Run the binning algorithm"""
    data = params.data
    bins = params.bins
    outfile = params.outfile
    valid_range = params.valid_range
    nproc = params.nproc
    maskfile = params.mask

    logger.info("Loading image series...")
    image_series = loadobj(data, "image_series")
    logger.info("Original image shape: {}", image_series.shape)

    if maskfile is not None:
        logger.info("Loading mask...")
        # Use nxload directly instead of loadobj because loadobj fails for very large arrays
        nxs = nxload(maskfile)
        mask = nxs.entry.mask.signal  # nxfield
    else:
        mask = None

    logger.info("Binning images with bin size: {}", bins)

    # Compute binning parameters
    bins_array = np.array(bins)
    nbins = np.ceil(image_series.shape / bins_array).astype(int)
    sl_0 = slice_sections(image_series.shape[0], nbins[0])
    sl_1 = slice_sections(image_series.shape[1], nbins[1])
    sl_2 = slice_sections(image_series.shape[2], nbins[2])

    new_phi = np.array([image_series.phi[sl].mean() for sl in sl_0])
    new_iy = np.array([image_series.iy[sl].mean() for sl in sl_1])
    new_ix = np.array([image_series.ix[sl].mean() for sl in sl_2])

    def binslab(sl):
        outslab = np.empty([len(sl_1), len(sl_2)], dtype=float)
        tmp = image_series._as_np(image_series.data[sl, :, :])
        tmp = np.ma.masked_equal(tmp, image_series._maskval, copy=False)
        if valid_range is not None:
            tmp = np.ma.masked_outside(tmp, valid_range[0], valid_range[1], copy=False)
        if mask is not None:
            msk = image_series._as_np(mask[sl, :, :])
            tmp = np.ma.masked_where(msk, tmp, copy=False)
        for ind_y, sl_y in enumerate(sl_1):  # not vectorized - could be slow?
            for ind_x, sl_x in enumerate(sl_2):
                val = tmp[:, sl_y, sl_x].mean()
                if np.ma.is_masked(val):
                    val = np.nan
                outslab[ind_y, ind_x] = val
        return outslab

    logger.info("Binning {} image slabs (requested n_jobs: {})...", len(sl_0), nproc)
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        new_data = np.stack(parallel(delayed(binslab)(sl) for sl in sl_0))
    logger.info("Binning computation completed")

    # Convert to count rate
    new_times = np.array([image_series.exposure_times[sl].mean() for sl in sl_0])
    new_data = new_data / new_times[:, np.newaxis, np.newaxis]

    binned = GridData((new_phi, new_iy, new_ix), new_data, axes_names=["phi", "iy", "ix"])
    logger.info("Binned image shape: {}", binned.data.shape)

    logger.info("Saving binned data to {}...", outfile)
    saveobj(binned, outfile, name="binned_image_series")
    logger.info("Binning completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_bin_image_series))


if __name__ == "__main__":
    run()
