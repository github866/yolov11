#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path /data/zhaoheng_zhu/origin/output_by_parts/loc02/part_02.mp4 \
    --output_path ./results/clips_02_01.mp4 \
    --metadata_path /data/zhaoheng_zhu/human_data/filter_data/loc02-frame0001.data