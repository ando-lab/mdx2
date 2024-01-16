#!/bin/bash
set -e

cd mdx2
mkdir -p partial_merge

mdx2.merge split_00/corrected.nxs --scale split_00/scales_all.nxs --outlier 5 --split randomHalf --outfile partial_merge/merged_1.nxs
mdx2.merge split_{00..01}/corrected.nxs --scale split_{00..01}/scales_all.nxs --outlier 5 --split randomHalf --outfile partial_merge/merged_2.nxs
mdx2.merge split_{00..03}/corrected.nxs --scale split_{00..03}/scales_all.nxs --outlier 5 --split randomHalf --outfile partial_merge/merged_4.nxs
mdx2.merge split_{00..07}/corrected.nxs --scale split_{00..07}/scales_all.nxs --outlier 5 --split randomHalf --outfile partial_merge/merged_8.nxs
mdx2.merge split_{00..11}/corrected.nxs --scale split_{00..11}/scales_all.nxs --outlier 5 --split randomHalf --outfile partial_merge/merged_12.nxs
mdx2.merge split_{00..16}/corrected.nxs --scale split_{00..16}/scales_all.nxs --outlier 5 --split randomHalf --outfile partial_merge/merged_17.nxs
cd ..