[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10519719.svg)](https://doi.org/10.5281/zenodo.10519719)

# *mdx2*: macromolecular diffuse scattering data reduction in python

## References

Publications describing [ando-lab/mdx2](https://github.com/ando-lab/mdx2):

- Meisburger SP & Ando N. Scaling and merging macromolecular diffuse scattering with *mdx2*. Acta Cryst. D**80**, 299-313. [DOI](https://doi.org/10.1107/S2059798324002705)
- Meisburger SP & Ando N. Chapter Two - Processing macromolecular diffuse scattering data. In *Methods in Enzymology* Volume **688**, 43-86. [DOI](https://doi.org/10.1016/bs.mie.2023.06.010), [BioRxiv](https://www.biorxiv.org/content/10.1101/2023.06.04.543637v1)

*Mdx2* is based on algorithms and general philosophy of [ando-lab/mdx-lib](https://github.com/ando-lab/mdx-lib), described here:

- Meisburger SP, Case DA & Ando N. Diffuse X-ray scattering from correlated motions in a protein crystal. *Nature Communications* **11**, 1271 (2020). [DOI](https://doi.org/10.1038/s41467-020-14933-6)
- Meisburger SP, Case DA, & Ando N. Robust total X-ray scattering workflow to study correlated motion of proteins in crystals. *Nature Communications* **14**, 1228 (2023). [DOI](https://doi.org/10.1038/s41467-023-36734-3)

## Examples

- Introductory walkthrough using a small insulin dataset: [examples/insulin-tutorial](examples/insulin-tutorial/README.md)
- Scripts and notebooks to regenerate the figures from Meisburger & Ando, Acta Cryst. D (2024): [examples/insulin-multi-crystal](examples/insulin-multi-crystal).

## Versions

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

For a conda-based installation, you'll need [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) or equivalent.

### User install (conda environment)

```bash
micromamba create -f https://raw.githubusercontent.com/ando-lab/mdx2/refs/tags/v1.0.3/env.yaml
micromamba activate mdx2
pip install mdx2==1.0.3
```

You'll probably want these packages too:

```bash
micromamba install -c conda-forge dials nexpy jupyterlab xia2
```

### Developer install

See The Diffuse Project's [fork](https://github.com/diff-use/mdx2) for instructions.
