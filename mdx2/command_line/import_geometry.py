"""
Import experimental geometry using the dxtbx machinery
"""

from dataclasses import dataclass
from typing import Tuple

from loguru import logger
from simple_parsing import field

import mdx2.geometry as geom
from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.io import saveobj


@dataclass
class Parameters:
    """Options for importing experimental geometry"""

    expt: str = field(positional=True)  # dials experiments file, such as refined.expt
    sample_spacing: Tuple[int, int, int] = (1, 10, 10)  # interval in degrees or pixels (phi, iy, ix)
    outfile: str = "geometry.nxs"  # name of the output NeXus file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)


    def __post_init__(self):
        """Validate sample_spacing parameter"""
        for i, spacing in enumerate(self.sample_spacing):
            if spacing <= 0:
                raise ValueError(f"sample_spacing[{i}] must be positive, got {spacing}")


def run_import_geometry(params):
    """Run the import geometry script with the given parameters"""
    exptfile = params.expt
    spacing_phi_px = tuple(params.sample_spacing)
    spacing_px = spacing_phi_px[1:]
    outfile = params.outfile

    if params.nproc != 1:
        logger.warning("Serial execution only, ignoring nproc value")

    logger.info("Computing miller index lookup grid...")
    miller_index = geom.MillerIndex.from_expt(
        exptfile,
        sample_spacing=spacing_phi_px,
    )
    logger.info(
        "Miller index grid shape (phi, iy, ix, hkl): {}",
        miller_index.data.shape,
    )

    logger.info("Computing geometric correction factors...")
    corrections = geom.Corrections.from_expt(
        exptfile,
        sample_spacing=spacing_px,
    )
    logger.info(
        "Correction factors grid shape (iy, ix, factors): {}",
        corrections.data.shape,
    )

    logger.info("Gathering crystal symmetry and unit cell...")
    symmetry = geom.Symmetry.from_expt(exptfile)
    crystal = geom.Crystal.from_expt(exptfile)
    logger.info("Space group: {}", symmetry.space_group_symbol)
    unit_cell_rounded = [f"{round(x, 4):g}" for x in crystal.unit_cell]
    logger.info("Unit cell: {}", ", ".join(unit_cell_rounded))

    logger.info("Saving geometry to {}...", outfile)
    saveobj(crystal, outfile, name="crystal", append=False)
    saveobj(symmetry, outfile, name="symmetry", append=True)
    saveobj(corrections, outfile, name="corrections", append=True)
    saveobj(miller_index, outfile, name="miller_index", append=True)
    logger.info("Geometry saved successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_import_geometry))

if __name__ == "__main__":
    run()
