"""
Convert merged HKL data from NeXus format to MTZ format.

Miller indices are scaled by ndiv so that half-integer positions (ndiv=2)
become integers in the output MTZ. The original unit cell is preserved.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from simple_parsing import field

from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.io import loadobj


@dataclass
class Parameters:
    """Options for converting merged NeXus data to MTZ format"""

    hkl: str = field(positional=True)  # NeXus file containing merged hkl_table
    geometry: str = field(positional=True)  # NeXus file containing crystal geometry
    output: str = field(alias=["-o"], default="merged.mtz")  # output MTZ file
    wavelength: float = 1.0  # wavelength in Angstroms
    title: str = "mdx2 merged data"  # MTZ title


def run_to_mtz(params):
    import warnings

    import gemmi

    warnings.filterwarnings("ignore", category=UserWarning, module="mdx2")

    from loguru import logger

    logger.info("Loading HKL table from {}...", params.hkl)
    hkl_table = loadobj(params.hkl, "hkl_table")
    df = hkl_table.to_frame()
    ndiv = hkl_table.ndiv

    logger.info("Loading crystal geometry from {}...", params.geometry)
    crystal = loadobj(params.geometry, "crystal")

    # Scale fractional indices to integers
    H = np.round(df["h"].values * ndiv[0]).astype(np.int32)
    K = np.round(df["k"].values * ndiv[1]).astype(np.int32)
    L = np.round(df["l"].values * ndiv[2]).astype(np.int32)

    I = df["intensity"].values.astype(np.float32)
    SIGI = df["intensity_error"].values.astype(np.float32)

    # Remove reflections with non-finite intensity or sigma
    mask = np.isfinite(I) & np.isfinite(SIGI) & (SIGI > 0)
    n_total = len(H)
    H, K, L, I, SIGI = H[mask], K[mask], L[mask], I[mask], SIGI[mask]
    logger.info("Reflections: {} total, {} with valid I/SIGI", n_total, mask.sum())

    uc = crystal.unit_cell  # [a, b, c, alpha, beta, gamma]
    sg_symbol = crystal.space_group

    # Scale a, b, c by ndiv to form the super-cell; angles are unchanged
    super_uc = [uc[i] * ndiv[i] if i < 3 else uc[i] for i in range(6)]
    if any(n != 1 for n in ndiv):
        logger.info(
            "ndiv={}: H,K,L scaled by ndiv; super-cell unit cell = {}",
            list(ndiv),
            [round(v, 4) for v in super_uc],
        )

    mtz = gemmi.Mtz(with_base=False)
    mtz.title = params.title
    mtz.cell = gemmi.UnitCell(*super_uc)
    mtz.spacegroup = gemmi.find_spacegroup_by_name(sg_symbol)

    ds = mtz.add_dataset("crystal")
    ds.crystal_name = "crystal"
    ds.project_name = "mdx2"
    ds.wavelength = params.wavelength

    mtz.add_column("H", "H").dataset_id = 0
    mtz.add_column("K", "H").dataset_id = 0
    mtz.add_column("L", "H").dataset_id = 0
    mtz.add_column("IMEAN", "J")
    mtz.add_column("SIGIMEAN", "Q")

    # Add half-dataset columns if present
    extra_cols = []
    for j in range(2):
        icol = f"group_{j}_intensity"
        scol = f"group_{j}_intensity_error"
        if icol in df.columns and scol in df.columns:
            extra_cols.append((f"IMEAN_half{j + 1}", "J", df[icol].values.astype(np.float32)[mask]))
            extra_cols.append((f"SIGIMEAN_half{j + 1}", "Q", df[scol].values.astype(np.float32)[mask]))

    for label, col_type, _ in extra_cols:
        mtz.add_column(label, col_type)

    cols = [H.astype(float), K.astype(float), L.astype(float), I, SIGI]
    cols += [arr for _, _, arr in extra_cols]
    mtz.set_data(np.column_stack(cols))

    mtz.write_to_file(params.output)
    logger.info("Wrote {} reflections to {}", len(H), params.output)


parse_arguments = make_argument_parser(Parameters, __doc__)
run = with_parsing(parse_arguments)(with_logging()(run_to_mtz))

if __name__ == "__main__":
    run()
