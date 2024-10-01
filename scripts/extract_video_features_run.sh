#!/bin/bash


for i in $(seq 0 47)
do
  sbatch alignment_scripts/1_extract_video_features.sh $i
done
