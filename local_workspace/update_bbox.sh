#!/bin/bash

# Configuration
clips=("clip1" "clip2" "clip3" "clip4" "clip5" "clip6")
iterations=(15 15 15 15 15 15)
data_dir="loc01_data"

# Loop through all clips
for i in "${!clips[@]}"; do
    clip_name="${clips[$i]}"
    iteration="${iterations[$i]}"
    
    echo "Processing ${clip_name} with iteration ${iteration}..."
    
    python3 side_process/add_missing.py \
        --name "${clip_name}" \
        --missing_json_file "${data_dir}/cropped_images/${clip_name}_missing.json" \
        --iteration "${iteration}" \
        --yolo_path "${data_dir}/origin/"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed ${clip_name}"
    else
        echo "✗ Failed to process ${clip_name}"
        exit 1
    fi
done

echo "All clips processed successfully!"