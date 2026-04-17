"""
Create a map from data in an hkl table
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from loguru import logger
from simple_parsing import field

from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.data import HKLTable
from mdx2.geometry import GridData
from mdx2.io import loadobj, saveobj


@dataclass
class Parameters:
    """Options for creating an array from an hkl table"""

    geom: str = field(positional=True)  # NeXus file containing symmetry and crystal
    hkl: str = field(positional=True)  # NeXus file containing hkl_table
    symmetry: bool = True  # apply symmetry operators
    limits: Tuple[float, float, float, float, float, float] = (0, 10, 0, 10, 0, 10)
    """limits for the hkl grid (hmin, hmax, kmin, kmax, lmin, lmax)"""
    signal: str = "intensity"  # column in hkl_table to map
    outfile: str = "map.nxs"  # name of the output NeXus file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)

    def __post_init__(self):
        """Validate limits parameter"""
        hmin, hmax, kmin, kmax, lmin, lmax = self.limits
        if hmin > hmax:
            raise ValueError(f"limits: hmin must be <= hmax, got hmin={hmin}, hmax={hmax}")
        if kmin > kmax:
            raise ValueError(f"limits: kmin must be <= kmax, got kmin={kmin}, kmax={kmax}")
        if lmin > lmax:
            raise ValueError(f"limits: lmin must be <= lmax, got lmin={lmin}, lmax={lmax}")


# NOTE: should perhaps change so that limits is a required argument


def run_map(params):
    """Run the map script"""
    hkl = params.hkl
    geom = params.geom
    outfile = params.outfile
    apply_symmetry = params.symmetry
    signal = params.signal
    hmin, hmax, kmin, kmax, lmin, lmax = params.limits

    if params.nproc != 1:
        logger.warning("Serial execution only, ignoring nproc value")

    logger.info("Loading HKL table and geometry...")
    T = loadobj(hkl, "hkl_table")
    Symmetry = loadobj(geom, "symmetry")  # used only if symmetry flag is set
    ndiv = T.ndiv

    Hmin = np.round(hmin * ndiv[0]).astype(int)
    Hmax = np.round(hmax * ndiv[0]).astype(int)
    Kmin = np.round(kmin * ndiv[1]).astype(int)
    Kmax = np.round(kmax * ndiv[1]).astype(int)
    Lmin = np.round(lmin * ndiv[2]).astype(int)
    Lmax = np.round(lmax * ndiv[2]).astype(int)

    h_axis = np.arange(Hmin, Hmax + 1) / ndiv[0]
    k_axis = np.arange(Kmin, Kmax + 1) / ndiv[1]
    l_axis = np.arange(Lmin, Lmax + 1) / ndiv[2]

    logger.info("Map region:")
    logger.info("  h: {} to {} ({} points)", h_axis[0], h_axis[-1], h_axis.size)
    logger.info("  k: {} to {} ({} points)", k_axis[0], k_axis[-1], k_axis.size)
    logger.info("  l: {} to {} ({} points)", l_axis[0], l_axis[-1], l_axis.size)

    logger.info("Generating Miller index grid...")
    h, k, l = np.meshgrid(h_axis, k_axis, l_axis, indexing="ij")

    Tgrid = HKLTable(h.ravel(), k.ravel(), l.ravel(), ndiv=ndiv)

    if apply_symmetry:
        logger.info("Mapping Miller indices to asymmetric unit...")
        Tgrid = Tgrid.to_asu(Symmetry)

    logger.info("Looking up '{}' values in data table...", signal)
    data = T.lookup(Tgrid.h, Tgrid.k, Tgrid.l, signal).reshape(h.shape)

    logger.info("Saving map to {}...", outfile)
    G = GridData((h_axis, k_axis, l_axis), data, axes_names=["h", "k", "l"])
    saveobj(G, outfile, name=signal, append=False)
    logger.info("Map creation completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_map))

if __name__ == "__main__":
    run()
