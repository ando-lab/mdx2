[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10519719.svg)](https://doi.org/10.5281/zenodo.10519719)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/mdx2/badges/version.svg)](https://anaconda.org/conda-forge/mdx2)

# *mdx2*: macromolecular diffuse scattering data reduction in python

New here? Have a look at our recent [Blog post](https://prismscience.org/research/mdx2). See also [Examples](#examples) and [References](#references), below.

## Examples

- Introductory walkthrough using a small insulin dataset: [examples/insulin-tutorial](examples/insulin-tutorial/README.md)
- Scripts and notebooks to regenerate the figures from Meisburger & Ando, Acta Cryst. D (2024): [examples/insulin-multi-crystal](examples/insulin-multi-crystal).
- External repo that batch reprocesses Mac1 datasets for an upcoming article: [github.com/prism-science/diffuse-processing-mac1-east-west](https://github.com/prism-science/diffuse-processing-mac1-east-west)

## Versions

### Version 1.0.5

A minor update to prepare the CLI for pipeline integration. Previous versions had to be invoked from the working directory. This version allows arbitrary directories to be set for all input and output files.

- Feature: CLI tools have a `--logfile` option to explicitly set the path for logging (retaining the current default: `mdx2.<command>.log`)
- Bugfix: `mdx2.import_data` creates valid virtual dataset paths when `data.nxs` was written outside the current working directory

### Version 1.0.4

- Added `mdx2.report` to automatically generate and execute ipython notebook reports from the command-line. Available templates: `visualization`, `scaling_model`, and `map_statistics`. For example outputs, see: [examples/insulin-multi-crystal/reports](examples/insulin-multi-crystal/reports) and [examples/insulin-tutorial/reports](examples/insulin-tutorial/reports).

### Version 1.0.3

- Performance boost for `mdx2.import_data` using parallel read and write. The `data.nxs` file contains a virtual dataset linking to neXus files in a subdirectory (`datastore/` by default).
- `mdx2.reintegrate` -- New command-line tool to create fine maps after scaling (single-sweep only: multi-crystal datasets not yet implemented)
- Optional pre-scaling in `mdx2.scale` to correct anisotropic background
- Improved handling of command-line arguments via `dataclass` attributes and `simple-parsing` package
- Updated examples

### Version 1.0.2

- Rudimentary Bragg peak integration, in development
- Support for non-reference space group settings
- Bug fixes, including:
  - Symmetry operators now rotate in the correct direction
  - Gracefully skip missing or masked data chunks

## Installation

### Prerequisites

You'll need package manager for conda environments. If you don't have one already, we recommend installing [miniforge](https://github.com/conda-forge/miniforge), which includes conda and mamba commands, and has the conda-forge channel as its default. If you use a different manager, you'll need to add the flag `-c conda-forge`.

### User install (miniforge)

Minimal install:

```bash
mamba create -n mdx2
mamba activate mdx2
mamba install mdx2
```

Recommended: add packages for Bragg processing, jupyter notebooks, and nexus visualization

```bash
mamba install jupyterlab dials xia2 nexpy
```

## Contributing

Mdx2 is being developed in collaboration with [The Diffuse Project](https://diffuse.science), and the `dev` branch in the org's fork contains the latest changes: [diff-use/mdx2/tree/dev](https://github.com/diff-use/mdx2/tree/dev).

Contributions can be made by branching from `dev`, and submitting PRs to that branch. When it is time for a new release, the `dev` branch is merged upstream to the `ando-lab/mdx2` repo.

### Developer install (conda + pip from dev branch of diff-use fork)

```bash
git clone -b dev https://github.com/diff-use/mdx2.git
cd mdx2
mamba create -f env.yaml -n mdx2-dev
mamba activate mdx2-dev
pip install -e ".[dev]"
```

The last line installs mdx2 in editable mode, with optional development tools including pytest and ruff.

## References

Publications describing [ando-lab/mdx2](https://github.com/ando-lab/mdx2):

- Meisburger SP & Ando N. Scaling and merging macromolecular diffuse scattering with *mdx2*. Acta Cryst. D**80**, 299-313. [DOI](https://doi.org/10.1107/S2059798324002705)
- Meisburger SP & Ando N. Chapter Two - Processing macromolecular diffuse scattering data. In *Methods in Enzymology* Volume **688**, 43-86. [DOI](https://doi.org/10.1016/bs.mie.2023.06.010), [BioRxiv](https://www.biorxiv.org/content/10.1101/2023.06.04.543637v1)

*Mdx2* is based on algorithms and general philosophy of [ando-lab/mdx-lib](https://github.com/ando-lab/mdx-lib), described here:

- Meisburger SP, Case DA & Ando N. Diffuse X-ray scattering from correlated motions in a protein crystal. *Nature Communications* **11**, 1271 (2020). [DOI](https://doi.org/10.1038/s41467-020-14933-6)
- Meisburger SP, Case DA, & Ando N. Robust total X-ray scattering workflow to study correlated motion of proteins in crystals. *Nature Communications* **14**, 1228 (2023). [DOI](https://doi.org/10.1038/s41467-023-36734-3)