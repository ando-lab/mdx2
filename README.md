# *mdx2*: macromolecular diffuse scattering data reduction in python

## Manuscripts

*Mdx2* is based on algorithms and general philosophy of the MATLAB library [mdx-lib](https://github.com/ando-lab/mdx-lib). The methods are described in the following publications:

> Meisburger, S.P., Case, D.A. & Ando, N. Diffuse X-ray scattering from correlated motions in a protein crystal. *Nature Communications* **11**, 1271 (2020). [doi:10.1038/s41467-020-14933-6](https://doi.org/10.1038/s41467-020-14933-6)
> Meisburger, S.P., and Ando, N. Processing macromolecular diffuse scattering data. *Methods in Enzymology* [submitted] (2023).

## Tutorial

A detailed walkthrough is included. See ![tutorials/insulin/README.md](tutorials/insulin/README.md) for instructions.

## Versions

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

## Installation (conda)

```
conda create --name mdx2 python=3.10
conda activate mdx2
conda install -c conda-forge dxtbx nexusformat pandas numexpr
pip install git+https://github.com/ando-lab/mdx2.git
```
