"""
Reintegrate on a different grid, applying corrections, scaling, and merging symmetry-equivalent observations.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import HKLGrid, HKLTable
from mdx2.io import (
    loadobj,
    nxload,  # mask is too big to read all at once?
    saveobj,
)

# TODO: implement splitting
# TODO: unify with mdx2.integrate, mdx2.correct
#   - allow reintegrate to run without scale and/or background
#   - add option to output unmerged data
#   - add option to use a dynamic mask
# TODO: properly deal with outlier detection and rejection, similar to mdx2.merge
#   - may require refactoring


@dataclass
class Parameters:
    """Options for integrating counts on a Miller index grid"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    mask: Optional[str] = None  # NeXus data file containing mask
    subdivide: Tuple[int, int, int] = (1, 1, 1)  # subdivisions of the Miller index grid
    scale: Optional[str] = None  # NeXus file with scaling model
    background: Optional[str] = None  # NeXus file with background binned_image_series
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    output: Literal["counts", "intensity"] = "counts"  # counts or intensity
    outfile: str = "reintegrated.nxs"  # name of the output data
    logfile: str = "mdx2.reintegrate.log"  # name of the log file

    def __post_init__(self):
        """Validate subdivide parameter"""
        for i, div in enumerate(self.subdivide):
            if div <= 0:
                raise ValueError(f"subdivide[{i}] must be > 0, got {div}")


def calc_corrections(
    tab,
    crystal,
    corrections,
    symmetry,
    scaling_model=None,
    absorption_model=None,
    detector_model=None,
    offset_model=None,
    background=None,
):
    """
    Apply correction factors, scaling models, and background subtraction to an HKLTable.

    This function computes various correction factors and applies optional scaling models
    to prepare integrated reflection data for merging. It modifies the input table in-place
    and returns the modified table.

    Parameters
    ----------
    tab : HKLTable
        Table with columns: h, k, l, phi, iy, ix, seconds, pixels, ndiv.
        Must contain partial observations binned by Miller index.
    crystal : Crystal
        Crystal object providing ub_matrix for computing scattering vectors.
    corrections : Corrections
        Corrections object providing interpolate(iy, ix) method that returns
        dict with keys: solid_angle, attenuation, polarization, efficiency, d3s.
    symmetry : Symmetry
        Symmetry object providing to_asu() method for converting Miller indices
        to asymmetric unit.
    scaling_model : ScalingModel, optional
        Time-dependent scaling model providing interp(phi) method.
    absorption_model : AbsorptionModel, optional
        Spatial and time-dependent absorption correction model providing
        interp(ix, iy, phi) method.
    detector_model : DetectorModel, optional
        Detector spatial efficiency model providing interp(ix, iy) method.
    offset_model : OffsetModel, optional
        Background offset model as function of resolution and time,
        providing interp(s, phi) method.
    background : BinnedImageSeries, optional
        Background image series providing interpolate(phi, iy, ix) method
        for background subtraction.

    Returns
    -------
    HKLTable
        Modified table with:
        - Added columns: scale, background_counts, multiplicity
        - Removed columns: phi, iy, ix, seconds, s, op, pixels
        - Miller indices (h, k, l) converted to asymmetric unit

    Notes
    -----
    The scale factor combines:
    - Exposure time (seconds)
    - Solid angle corrections (solid_angle * attenuation * polarization * efficiency)
    - Optional scaling models (absorption * scaling * detector)

    The background_counts includes both:
    - Direct background from binned_image_series
    - Offset model contribution (if provided)
    """
    UB = crystal.ub_matrix
    s = UB @ np.stack((tab.h, tab.k, tab.l))
    tab.s = np.sqrt(np.sum(s * s, axis=0))
    correction_factors = corrections.interpolate(tab.iy, tab.ix)
    solid_angle = correction_factors["solid_angle"]
    solid_angle *= correction_factors["attenuation"]
    solid_angle *= correction_factors["polarization"]
    solid_angle *= correction_factors["efficiency"]
    tab.multiplicity = tab.pixels * correction_factors["d3s"] * np.prod(tab.ndiv) / np.linalg.det(UB)
    b = scaling_model.interp(tab.phi) if scaling_model else 1.0
    a = absorption_model.interp(tab.ix, tab.iy, tab.phi) if absorption_model else 1.0
    d = detector_model.interp(tab.ix, tab.iy) if detector_model else 1.0
    c = offset_model.interp(tab.s, tab.phi) if offset_model else 0.0
    bg_rate = background.interpolate(tab.phi, tab.iy, tab.ix) if background else 0.0
    tab.scale = tab.seconds * solid_angle * a * b * d
    tab.background_counts = tab.seconds * (bg_rate + c * a * d * solid_angle)
    tab = tab.to_asu(symmetry)
    del tab.phi, tab.iy, tab.ix, tab.seconds, tab.s, tab.op, tab.pixels
    return tab


def run_reintegrate(params):
    logger.info("Loading geometry and image data...")
    miller_index = loadobj(params.geom, "miller_index")
    image_series = loadobj(params.data, "image_series")
    corrections = loadobj(params.geom, "corrections")
    crystal = loadobj(params.geom, "crystal")
    symmetry = loadobj(params.geom, "symmetry")

    logger.info("Loading optional models...")
    background = loadobj(params.background, "binned_image_series") if params.background else None
    if params.scale:
        scale_file = nxload(params.scale)
        scaling_model = loadobj(params.scale, "scaling_model") if "scaling_model" in scale_file.entry.keys() else None
        absorption_model = (
            loadobj(params.scale, "absorption_model") if "absorption_model" in scale_file.entry.keys() else None
        )
        offset_model = loadobj(params.scale, "offset_model") if "offset_model" in scale_file.entry.keys() else None
        detector_model = (
            loadobj(params.scale, "detector_model") if "detector_model" in scale_file.entry.keys() else None
        )
    else:
        scaling_model = None
        absorption_model = None
        offset_model = None
        detector_model = None
    # NXfield objects can be pickled safely (they serialize metadata and reopen the file in each worker)
    # so this works correctly with nproc > 1 despite being passed to joblib.Parallel
    mask = nxload(params.mask).entry.mask.signal if params.mask else None

    def intchunk(sl):
        ims = image_series[sl]
        if mask is not None:
            tab = ims.index(miller_index, mask=mask[sl].nxdata)  # added nxdata to deal with NXfield wrapper
        else:
            tab = ims.index(miller_index)
        if len(tab) == 0:
            return None  # signal that no pixels were integrated
        tab.ndiv = params.subdivide
        tab = tab.bin(count_name="pixels")
        tab.phi /= tab.pixels
        tab.iy /= tab.pixels
        tab.ix /= tab.pixels
        tab = calc_corrections(
            tab,
            crystal,
            corrections,
            symmetry,
            scaling_model,
            absorption_model,
            detector_model,
            offset_model,
            background,
        )
        tab.h = tab.h.astype(np.float32)
        tab.k = tab.k.astype(np.float32)
        tab.l = tab.l.astype(np.float32)
        return tab

    # TODO: check for memory limitations, consider using memory mapping for large datasets

    logger.info("Calculating Miller index range for output grid...")
    hkl = HKLTable(miller_index.h.ravel(), miller_index.k.ravel(), miller_index.l.ravel(), ndiv=params.subdivide)
    hkl = hkl.to_asu(symmetry)
    array_range = (min(hkl.H.min(), 0), hkl.H.max(), min(hkl.K.min(), 0), hkl.K.max(), min(hkl.L.min(), 0), hkl.L.max())
    array_size = tuple(hi - lo + 1 for lo, hi in zip(array_range[::2], array_range[1::2]))
    array_ori = tuple(lo / s for lo, s in zip(array_range[::2], params.subdivide))
    logger.info("Output grid size: {}", array_size)
    logger.info("Allocating empty output arrays...")
    data_arrays = {
        "counts": np.zeros(array_size, dtype=np.uint32),
        "background_counts": np.zeros(array_size, dtype=np.float32),
        "scale": np.zeros(array_size, dtype=np.float32),
        "multiplicity": np.zeros(array_size, dtype=np.float32),
    }
    grid = HKLGrid(data_arrays, ndiv=params.subdivide, ori=array_ori)

    slices = list(image_series.chunk_slice_iterator())
    logger.info("Reintegrating {} image chunks (requested n_jobs: {})...", len(slices), params.nproc)
    with Parallel(n_jobs=params.nproc, verbose=10, return_as="generator_unordered") as parallel:
        log_parallel_backend(parallel)
        tab_chunk = parallel(delayed(intchunk)(sl) for sl in slices)
        for tab in tab_chunk:
            if tab is None:
                continue
            grid.accumulate_from_table(tab, resize=False)
    logger.info("Reintegration completed")

    logger.info("Converting grid to sparse table...")
    hkl_table = grid.to_table(sparse=True)
    hkl_table.h = hkl_table.h.astype(np.float32)
    hkl_table.k = hkl_table.k.astype(np.float32)
    hkl_table.l = hkl_table.l.astype(np.float32)
    logger.info("Reflections in output: {}", len(hkl_table))

    if params.output == "intensity":
        logger.info("Computing intensities from counts...")
        zero_scale_mask = hkl_table.scale == 0
        if np.any(zero_scale_mask):
            logger.warning("Found {} voxels with zero scale; intensity will be inf/nan", np.sum(zero_scale_mask))
        hkl_table.intensity = (hkl_table.counts - hkl_table.background_counts) / hkl_table.scale
        hkl_table.intensity_error = np.sqrt(hkl_table.counts) / hkl_table.scale
        del hkl_table.counts, hkl_table.background_counts, hkl_table.scale
        hkl_table.intensity = hkl_table.intensity.astype(np.float32)
        hkl_table.intensity_error = hkl_table.intensity_error.astype(np.float32)

    logger.info("Saving reintegrated data to {}...", params.outfile)
    saveobj(hkl_table, params.outfile, name="hkl_table", append=False)
    logger.info("Reintegration completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_reintegrate))

if __name__ == "__main__":
    run()
