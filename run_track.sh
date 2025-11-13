#!/bin/bash
location_id="04"
video_dir="/data/leohsu/human_dataset/humans/loc_${location_id}"
metadata_dir="/data/leohsu/human_dataset/humans/metadata/init_frames"
room_mask_path="/home/agenuinedream/repo/yolov11/data/room_masks/loc_${location_id}_colored_mask.png"
output_dir="./results/tracking/yolov11x_ours_tuned/dino_vits8/loc_${location_id}"
# output_dir="./results/tracking/yolov8m_coco_ours_tuned/dino_vits8_iou_085_gen_feature_bank"
mkdir -p ${output_dir}

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path ${video_dir}/clip_01.mp4 \
    --output_path ./${output_dir}/loc_${location_id}_01.mp4 \
    --room_mask_path ${room_mask_path} &
    # --metadata_path ${metadata_dir}/loc${location_id}-frame0001.data &

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path ${video_dir}/clip_02.mp4 \
    --output_path ./${output_dir}/loc_${location_id}_02.mp4 \
    --room_mask_path ${room_mask_path} &
    # --metadata_path ${metadata_dir}/loc${location_id}-frame0901.data &

CUDA_VISIBLE_DEVICES=1 python leo_run_track.py \
    --video_path ${video_dir}/clip_03.mp4 \
    --output_path ./${output_dir}/loc_${location_id}_03.mp4 \
    --room_mask_path ${room_mask_path} &
    # --metadata_path ${metadata_dir}/loc${location_id}-frame1801.data &

CUDA_VISIBLE_DEVICES=1 python leo_run_track.py \
    --video_path ${video_dir}/clip_04.mp4 \
    --output_path ./${output_dir}/loc_${location_id}_04.mp4 \
    --room_mask_path ${room_mask_path} &
    # --metadata_path ${metadata_dir}/loc${location_id}-frame2701.data &

CUDA_VISIBLE_DEVICES=2 python leo_run_track.py \
    --video_path ${video_dir}/clip_05.mp4 \
    --output_path ./${output_dir}/loc_${location_id}_05.mp4 \
    --room_mask_path ${room_mask_path} &
    # --metadata_path ${metadata_dir}/loc${location_id}-frame3601.data &

CUDA_VISIBLE_DEVICES=2 python leo_run_track.py \
    --video_path ${video_dir}/clip_06.mp4 \
    --output_path ./${output_dir}/loc_${location_id}_06.mp4 \
    --room_mask_path ${room_mask_path} &
    # --metadata_path ${metadata_dir}/loc${location_id}-frame4501.data &

wait

python leo_utils/postprocess/combine.py \
    --root_dir ${output_dir} \
    --location_id ${location_id}

python leo_utils/postprocess/transform.py \
    --root_dir ${output_dir} \
    --location_id ${location_id}

python eval/eval.py \
    --pred_dir ${output_dir} \
    --loc_num ${location_id}

# CUDA_VISIBLE_DEVICES=1 python leo_run_track.py     --video_path /data/leohsu/human_dataset/humans/loc_02/clip_04.mp4     --output_path ./results/tracking/yolov11x_ours_tuned/loc_02_0.mp4     --metadata_path /data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame2701.data