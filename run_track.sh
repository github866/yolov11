FEATURE_SIMILARITY_THRESHOLD=0.0
IOU_THRESHOLD=0.0

CUDA_VISIBLE_DEVICES=0 python leo_run_track.py \
    --video_path /data/leohsu/human_dataset/humans/clips_02/clip_06.mp4 \
    --output_path ./results/iou_${IOU_THRESHOLD}_feature_${FEATURE_SIMILARITY_THRESHOLD}_clips_02_06.mp4 \
    --metadata_path /data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame4501.data \
    --iou_threshold $IOU_THRESHOLD \
    --feature_similarity_threshold $FEATURE_SIMILARITY_THRESHOLD \
    --ref_feature_path ./preprocess/subject_features.pt