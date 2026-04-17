"""
Apply corrections to integrated data
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from simple_parsing import field  # pip install simple-parsing

from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.io import loadobj, saveobj


@dataclass
class Parameters:
    """Options for applying corrections to integrated data"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    hkl: str = field(positional=True)  # NeXus data file containing hkl_table
    background: Optional[str] = None  # NeXus file with background map
    attenuation: bool = True  # apply attenuation correction
    efficiency: bool = True  # apply efficiency correction
    polarization: bool = True  # apply polarization correction
    lorentz: bool = False  # apply Lorentz correction
    p1: bool = False  # map Miller indices to asymmetric unit for P1 (Friedel symmetry only)
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    outfile: str = "corrected.nxs"  # name of the output NeXus file


def run_correct(params):
    """Run the correct script"""
    hkl = params.hkl
    geom = params.geom
    background = params.background
    outfile = params.outfile
    p1 = params.p1
    attenuation = params.attenuation
    efficiency = params.efficiency
    polarization = params.polarization
    lorentz = params.lorentz

    if params.nproc != 1:
        logger.warning("Serial execution only, ignoring nproc value")

    logger.info("Loading integrated data and geometry...")
    T = loadobj(hkl, "hkl_table")

    # hack to work with older versions
    if "_ndiv" in T.__dict__:
        T.ndiv = T._ndiv
        del T._ndiv

    Corrections = loadobj(geom, "corrections")
    Crystal = loadobj(geom, "crystal")

    if p1:
        logger.info("Using P1 symmetry only (ignoring space group)")
        Symmetry = None
    else:
        Symmetry = loadobj(geom, "symmetry")

    UB = Crystal.ub_matrix

    logger.info("Calculating scattering vector magnitudes...")
    s = UB @ np.stack((T.h, T.k, T.l))
    T.s = np.sqrt(np.sum(s * s, axis=0))

    logger.info("Mapping Miller indices to asymmetric unit...")
    T = T.to_asu(Symmetry)

    # apply corrections to intensities
    logger.info("Interpolating correction factors...")
    Cinterp = Corrections.interpolate(T.iy, T.ix)

    count_rate = T.counts / T.seconds
    count_rate_error = np.sqrt(T.counts) / T.seconds

    if background is not None:
        Bkg = loadobj(background, "binned_image_series")
        bkg_count_rate = Bkg.interpolate(T.phi, T.iy, T.ix)
        logger.info("Subtracting background from count rate")
        count_rate = count_rate - bkg_count_rate

    solid_angle = Cinterp["solid_angle"].copy()

    corrections_applied = []
    if attenuation:
        solid_angle *= Cinterp["attenuation"]
        corrections_applied.append("attenuation")
    if efficiency:
        solid_angle *= Cinterp["efficiency"]
        corrections_applied.append("efficiency")
    if polarization:
        solid_angle *= Cinterp["polarization"]
        corrections_applied.append("polarization")

    if corrections_applied:
        logger.info("Applied corrections: {}", ", ".join(corrections_applied))

    logger.info("Computing reciprocal space volume fractions...")
    T.rs_volume = T.pixels * Cinterp["d3s"] / np.linalg.det(UB)

    logger.info("Computing intensities and errors...")
    T.intensity = count_rate / solid_angle
    T.intensity_error = count_rate_error / solid_angle

    if lorentz:
        logger.info("Applying Lorentz correction")
        T.intensity *= T.rs_volume
        T.intensity_error *= T.rs_volume

    # remove some unnecessary columns
    del T.counts
    del T.seconds
    del T.pixels

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

    logger.info("Reflections processed: {}", len(T))
    logger.info("Saving corrected data to {}...", outfile)
    saveobj(T, outfile, name="hkl_table", append=False)
    logger.info("Corrections completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_correct))


if __name__ == "__main__":
    run()
