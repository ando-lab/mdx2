"""
Import x-ray image data using the dxtbx machinery
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import ImageSeries
from mdx2.dxtbx_machinery import ImageSet
from mdx2.io import nxload


@dataclass
class Parameters:
    """Options for importing x-ray image data"""

    expt: str = field(positional=True)  # experiments file, such as from dials.import
    chunks: Optional[Tuple[int, int, int]] = field(
        default=None, help="chunking for compression (frames, y, x). Use -1 for default"
    )
    outfile: str = "data.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.import_data.log"  # name of the log file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    datastore: str = "datastore"  # folder for storing source datasets


def run_import_data(params):
    exptfile = params.expt
    chunks = params.chunks
    nproc = params.nproc
    outfile = params.outfile
    datastore = params.datastore

    logger.info("Loading experiment metadata...")
    image_series = ImageSeries.from_expt(exptfile)
    iset = ImageSet.from_file(exptfile)
    logger.info("Image data shape (phi, iy, ix): {}", image_series.shape)

    if chunks is not None:
        # override the default chunking
        # if any of the chunks dimensions is <=0, use the default for that dimension
        default_chunks = image_series.data.chunks
        chunks = tuple(c if c > 0 else d for c, d in zip(chunks, default_chunks))
        image_series.data.chunks = chunks
        logger.info("Using chunking: {}", chunks)

    logger.info("Creating virtual dataset structure...")
    image_series.save(
        outfile,
        virtual=True,
        source_directory=datastore,
    )

    def write_stack(istart, istop, filename, datapath):
        source = nxload(filename, "r+")[datapath]
        data = iset.read_stack(istart, istop)
        source[:, :, :] = data

    slices = [sl for sl in image_series.chunk_slice_along_axis(0)]

    # Access virtual dataset information through public API
    # Reload the ImageSeries to ensure we have the updated virtual dataset
    image_series_reloaded = ImageSeries.load(outfile)
    files = image_series_reloaded.virtual_source_files
    vpath = image_series_reloaded.virtual_dataset_path

    # These edge cases should not happen if virtual datasets are implemented correctly, but check anyway
    if len(files) != len(slices):
        raise RuntimeError(f"Virtual dataset mismatch: {len(slices)=} vs {len(files)=}")
    if len(set(files)) != len(files):
        raise RuntimeError("Virtual_source_files contains duplicates, cannot proceed with import.")

    # Properties will raise RuntimeError if internal API has changed
    # If data is not a virtual dataset, they return None
    if files is None or vpath is None:
        raise RuntimeError(f"Expected virtual dataset in {outfile}, but virtual dataset information is not available.")

    logger.info("Writing {} image batches (requested n_jobs: {})...", len(slices), nproc)
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        parallel(delayed(write_stack)(sl.start, sl.stop, fn, vpath) for sl, fn in zip(slices, files))
    logger.info("Image data writing completed")

    logger.info("Image data import completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_import_data))


if __name__ == "__main__":
    run()
