#!/bin/bash
set -e

# process background files
cd mdx2
for sub in {1..2}_bkg; do
    mkdir -p $sub
    cd $sub
    EXPTFILE="../../dials/${sub}/imported.expt"
    mdx2.import_data $EXPTFILE --chunks 10 211 493 --nproc 5
    mdx2.bin_image_series data.nxs 10 20 20 --valid_range 0 200 --outfile binned.nxs
    cd ..
done

# apply the corrections
for sub in split_{00..08}; do
    cd $sub
    mdx2.correct geometry.nxs integrated.nxs --background ../1_bkg/binned.nxs
    cd ..
done

for sub in split_{09..16}; do
    cd $sub
    mdx2.correct geometry.nxs integrated.nxs --background ../2_bkg/binned.nxs
    cd ..
done