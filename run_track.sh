#!/bin/bash
video_dir="/data/leohsu/human_dataset/humans/loc_02"
metadata_dir="/data/leohsu/human_dataset/humans/metadata/init_frames"
# output_dir="./results/tracking/yolov11x_ours_tuned/dino_vits8"
output_dir="./results/tracking/yolov8m_coco_ours_tuned/dino_vits8"
mkdir -p ${output_dir}

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path ${video_dir}/clip_01.mp4 \
    --output_path ./${output_dir}/loc_02_01.mp4 \
    --metadata_path ${metadata_dir}/loc02-frame0001.data &

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path ${video_dir}/clip_02.mp4 \
    --output_path ./${output_dir}/loc_02_02.mp4 \
    --metadata_path ${metadata_dir}/loc02-frame0901.data &

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path ${video_dir}/clip_03.mp4 \
    --output_path ./${output_dir}/loc_02_03.mp4 \
    --metadata_path ${metadata_dir}/loc02-frame1801.data &

CUDA_VISIBLE_DEVICES=1 python leo_run_track.py \
    --video_path ${video_dir}/clip_04.mp4 \
    --output_path ./${output_dir}/loc_02_04.mp4 \
    --metadata_path ${metadata_dir}/loc02-frame2701.data &

CUDA_VISIBLE_DEVICES=1 python leo_run_track.py \
    --video_path ${video_dir}/clip_05.mp4 \
    --output_path ./${output_dir}/loc_02_05.mp4 \
    --metadata_path ${metadata_dir}/loc02-frame3601.data &

CUDA_VISIBLE_DEVICES=1 python leo_run_track.py \
    --video_path ${video_dir}/clip_06.mp4 \
    --output_path ./${output_dir}/loc_02_06.mp4 \
    --metadata_path ${metadata_dir}/loc02-frame4501.data &

wait

python leo_utils/postprocess/combine.py \
    --root_dir ${output_dir}

python leo_utils/postprocess/transform.py \
    --root_dir ${output_dir}

python eval/eval.py \
    --pred_dir ${output_dir}

# CUDA_VISIBLE_DEVICES=1 python leo_run_track.py     --video_path /data/leohsu/human_dataset/humans/loc_02/clip_04.mp4     --output_path ./results/tracking/yolov11x_ours_tuned/loc_02_0.mp4     --metadata_path /data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame2701.data