"""
Integrate counts in an image stack on a Miller index grid
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import HKLTable
from mdx2.io import (
    loadobj,
    nxload,  # mask is too big to read all at once?
    saveobj,
)


@dataclass
class Parameters:
    """Options for integrating counts on a Miller index grid"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    mask: Optional[str] = None  # NeXus data file containing mask
    subdivide: Tuple[int, int, int] = (1, 1, 1)  # subdivisions of the Miller index grid
    max_spread: float = 1.0  # maximum angular spread (degrees) for binning partial observations
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    outfile: str = "integrated.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.integrate.log"  # name of the log file

    def __post_init__(self):
        """Validate subdivide and max_spread parameters"""
        for i, div in enumerate(self.subdivide):
            if div <= 0:
                raise ValueError(f"subdivide[{i}] must be > 0, got {div}")
        if self.max_spread <= 0:
            raise ValueError(f"max_spread must be > 0, got {self.max_spread}")


def run_integrate(params):
    """Run the integrate script"""
    geom = params.geom
    data = params.data
    outfile = params.outfile
    nproc = params.nproc
    ndiv = params.subdivide
    max_degrees = params.max_spread
    maskfile = params.mask

    logger.info("Loading geometry and image data...")
    MI = loadobj(geom, "miller_index")
    IS = loadobj(data, "image_series")

    if maskfile is not None:
        logger.info("Loading mask...")
        # Use nxload directly instead of loadobj because loadobj fails for very large arrays
        nxs = nxload(maskfile)
        mask = nxs.entry.mask.signal  # nxfield
    else:
        mask = None

    def intchunk(sl):
        ims = IS[sl]
        if mask is not None:
            tab = ims.index(MI, mask=mask[sl].nxdata)  # added nxdata to deal with NXfield wrapper
        else:
            tab = ims.index(MI)
        tab.ndiv = ndiv
        return tab.bin(count_name="pixels")

    slices = list(IS.chunk_slice_iterator())
    logger.info("Integrating {} image chunks (requested n_jobs: {})...", len(slices), nproc)
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        T = parallel(delayed(intchunk)(sl) for sl in slices)
    logger.info("Integration completed")

    logger.info("Summing partial observations over {} chunks...", len(T))

    # Handle empty results
    if not T:
        raise ValueError(
            "No integration results produced. The image series may be empty, or no valid data was found to integrate."
        )

    df = HKLTable.concatenate(T).to_frame()  # .set_index(['h','k','l'])

    df["tmp"] = df["phi"] / df["pixels"]
    delta_phi = df.tmp - df.groupby(["h", "k", "l"])["tmp"].transform("min")
    df["n"] = np.floor(delta_phi / max_degrees)
    df = df.drop(columns=["tmp"])

    df = df.groupby(["h", "k", "l", "n"]).sum()

    # compute mean positions in the scan
    df["phi"] = df["phi"] / df["pixels"]
    df["iy"] = df["iy"] / df["pixels"]
    df["ix"] = df["ix"] / df["pixels"]

    voxels_before = np.sum([len(t) for t in T])
    voxels_after = len(df)
    logger.info("Binned from {} to {} voxels", voxels_before, voxels_after)

    hkl_table = HKLTable.from_frame(df)
    hkl_table.ndiv = ndiv  # lost in conversion to/from dataframe

    hkl_table.h = hkl_table.h.astype(np.float32)
    hkl_table.k = hkl_table.k.astype(np.float32)
    hkl_table.l = hkl_table.l.astype(np.float32)
    hkl_table.phi = hkl_table.phi.astype(np.float32)
    hkl_table.ix = hkl_table.ix.astype(np.float32)
    hkl_table.iy = hkl_table.iy.astype(np.float32)
    hkl_table.seconds = hkl_table.seconds.astype(np.float32)
    hkl_table.counts = hkl_table.counts.astype(np.int32)
    hkl_table.pixels = hkl_table.pixels.astype(np.int32)

    logger.info("Saving integrated data to {}...", outfile)
    saveobj(hkl_table, outfile, name="hkl_table", append=False)
    logger.info("Integration completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_integrate))


if __name__ == "__main__":
    run()
