#!/bin/bash
set -e

DATADIR="/nfs/chess/scratch/user/spm82/data/insulin"

cd dials
for sub in {1..2}_bkg; do
    mkdir -p $sub
    cd $sub
    dials.import "${DATADIR}/insulin_${sub}_1_*.cbf"
    dials.generate_mask imported.expt untrusted.circle=1264,1242,50
    dials.apply_mask imported.expt input.mask=pixels.mask output.experiments=imported.expt
    cd ..
done