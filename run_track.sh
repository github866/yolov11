#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path /data/leohsu/human_dataset/humans/loc_02/clip_01.mp4 \
    --output_path ./results/loc_02_01.mp4 \
    --metadata_path /data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame0001.data \

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path /data/leohsu/human_dataset/humans/loc_02/clip_02.mp4 \
    --output_path ./results/loc_02_02.mp4 \
    --metadata_path /data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame0901.data \

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path /data/leohsu/human_dataset/humans/loc_02/clip_03.mp4 \
    --output_path ./results/loc_02_03.mp4 \
    --metadata_path /data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame1801.data \

CUDA_VISIBLE_DEVICES=1 python leo_run_track.py \
    --video_path /data/leohsu/human_dataset/humans/loc_02/clip_04.mp4 \
    --output_path ./results/loc_02_04.mp4 \
    --metadata_path /data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame2701.data \

CUDA_VISIBLE_DEVICES=1 python leo_run_track.py \
    --video_path /data/leohsu/human_dataset/humans/loc_02/clip_05.mp4 \
    --output_path ./results/loc_02_05.mp4 \
    --metadata_path /data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame3601.data \

CUDA_VISIBLE_DEVICES=1 python leo_run_track.py \
    --video_path /data/leohsu/human_dataset/humans/loc_02/clip_06.mp4 \
    --output_path ./results/loc_02_06.mp4 \
    --metadata_path /data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame4501.data \