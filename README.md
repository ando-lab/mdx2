# *mdx2*: macromolecular diffuse scattering data reduction in python

## Manuscripts

*mdx2* is based on the MATLAB library *mdx-lib*. The methods are described in the following publication:

> Meisburger, S.P., Case, D.A. & Ando, N. Diffuse X-ray scattering from correlated motions in a protein crystal. *Nature Communications* **11**, 1271 (2020). [doi:10.1038/s41467-020-14933-6](https://doi.org/10.1038/s41467-020-14933-6)

## Tutorial

For a detailed walkthrough check out the 2022 Erice workshop on data reduction: https://github.com/ando-lab/erice-2022-data-reduction

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

## Developer installation (conda)

Create a fresh conda environment
```
conda create --name mdx2-dev python=3.9
conda activate mdx2-dev
```

Install dependencies

```
conda install dxtbx
conda install pandas
conda install -c conda-forge nexusformat
```

Install `mdx2`

```
git clone https://github.com/ando-Lab/mdx2
cd mdx2
pip install -e .
```
