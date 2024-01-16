#!/bin/bash
set -e

mkdir -p mdx2
cd mdx2
for SUB in split_{00..16}; do
    mkdir -p $SUB
    cd $SUB
    EXPTFILE="../../dials/${SUB}.expt" # relative to subdir
    mdx2.import_geometry $EXPTFILE
    mdx2.import_data $EXPTFILE --chunks 20 211 493 --nproc 5 # single writer limits speed
    mdx2.find_peaks geometry.nxs data.nxs --count_threshold 20 --nproc 64
    mdx2.mask_peaks geometry.nxs data.nxs peaks.nxs --sigma_cutoff 3 --nproc 64
    mdx2.integrate geometry.nxs data.nxs --mask mask.nxs --subdivide 3 3 3 --nproc 64
    cd ..
done