#!/bin/bash
set -e

DATADIR="/nfs/chess/scratch/user/spm82/data/insulin"
SUBS="1_1 1_2 1_3 1_4 1_5 1_6 1_7 1_8 1_9 2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8"
mkdir -p dials
cd dials
for sub in $SUBS; do
    mkdir -p $sub
    cd $sub
    dials.import "${DATADIR}/insulin_${sub}_*.cbf"
    dials.generate_mask imported.expt untrusted.circle=1264,1242,50
    dials.apply_mask imported.expt input.mask=pixels.mask output.experiments=imported.expt
    dials.find_spots imported.expt
    dials.index imported.expt strong.refl space_group=199
    dials.refine indexed.expt indexed.refl
    dials.integrate refined.expt refined.refl
    cd ..
done

dials.cosym **/integrated.{expt,refl} space_group=199
dials.scale symmetrized.{expt,refl} d_min=1.2
dials.split_experiments scaled.{expt,refl}
dials.report scaled.{expt,refl} output.json=report.json