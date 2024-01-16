[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10519719.svg)](https://doi.org/10.5281/zenodo.10519719)

# *mdx2*: macromolecular diffuse scattering data reduction in python

## Manuscripts

*Mdx2* is based on algorithms and general philosophy of the MATLAB library [mdx-lib](https://github.com/ando-lab/mdx-lib). The methods are described in the following publications:

> Meisburger, S.P., and Ando, N. Scaling and merging macromolecular diffuse scattering with *mdx2*. [In preparation] (2024).

> Meisburger, S.P., and Ando, N. Chapter Two - Processing macromolecular diffuse scattering data. In *Methods in Enzymology* Volume 688, 43-86 (2023). [BioRxiv](https://www.biorxiv.org/content/10.1101/2023.06.04.543637v1)

> Meisburger, S.P., Case, D.A. & Ando, N. Diffuse X-ray scattering from correlated motions in a protein crystal. *Nature Communications* **11**, 1271 (2020). [doi:10.1038/s41467-020-14933-6](https://doi.org/10.1038/s41467-020-14933-6)


## Examples

### Insulin tutorial

A introductory walkthrough is included. See [examples/insulin-tutorial](examples/insulin-tutorial/README.md) for instructions.

### Multi-crystal scaling

Scripts to process and analyze the multi-crystal insulin dataset from Meisburger & Ando 2024 are provided in [examples/insulin-multi-crystal](examples/insulin-multi-crystal).

## Versions

### Version 1.0.0

New:
- Implementation of the full scaling model from mdx-lib
- Scale and merge multi-sweep datasets
- Parallel processing
- Improved handling of systematic absences
- Example scripts and jupyter notebooks for multi-crystal data

### Version 0.3.0

Features:
- pip-installable via setup.py
- fully-featured command-line interface
- import geometry from *dials*
- read and write objects to nexus-formatted h5 files
- support for basic masking, integration, background subtraction, scaling, and merging
- construct 2D slices and 3D maps with symmetry expansion
- convert h,k,l tables to/from Pandas DataFrame

Limitations:
- single sweep datasets only (one experiment per expt file)
- not parallelized
- scaling model includes phi-dependent term only
- file format details will likely change in future releases

## Installation

### Using conda (for introductory tutorial)

Install version 0.3.0 using conda: see [examples/insulin-tutorial](examples/insulin-tutorial/README.md) for detailed instructions.

### Using micromamba (for latest version)

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

```bash
curl https://raw.githubusercontent.com/ando-lab/mdx2/main/env.yaml env.yaml
micromamba create -f env.yaml
micromamba activate mdx2
pip install git+https://github.com/ando-lab/mdx2.git
```