#!/bin/bash
set -e

cd mdx2

# scale all / merge all
mdx2.scale split_{00..16}/corrected.nxs --mca2020 --outfile split_{00..16}/scales_all.nxs
mdx2.merge split_{00..16}/corrected.nxs --scale split_{00..16}/scales_all.nxs \
  --outlier 5 --split randomHalf --outfile merged_all.nxs
mdx2.merge split_{00..16}/corrected.nxs --scale split_{00..16}/scales_all.nxs \
  --outlier 5 --split Friedel --geometry split_00/geometry.nxs --outfile merged_all_Friedel.nxs

# scale crystal 1 / merge crystal 1
mdx2.scale split_{00..08}/corrected.nxs --mca2020 --outfile split_{00..08}/scales_crystal1.nxs
mdx2.merge split_{00..08}/corrected.nxs --scale split_{00..08}/scales_crystal1.nxs \
  --outlier 5 --split randomHalf --outfile merged_crystal1.nxs

# scale crystal 2 / merge crystal 2
mdx2.scale split_{09..16}/corrected.nxs --mca2020 --outfile split_{09..16}/scales_crystal2.nxs
mdx2.merge split_{09..16}/corrected.nxs --scale split_{09..16}/scales_crystal2.nxs \
  --outlier 5 --split randomHalf --outfile merged_crystal2.nxs