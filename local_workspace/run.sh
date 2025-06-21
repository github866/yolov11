#!/bin/bash

#SETUP VARIABLES
name="clip1"
iteration=15
data_type="origin"

# CONSTANTS
loc_dir="loc03_data"
origin_json_dir="${loc_dir}/${data_type}/"
missing_json_dir="${loc_dir}/cropped_images/"
missing_json_file="${missing_json_dir}${name}_missing.json"

echo "updating output bounding box json file"

python3 side_process/add_missing.py --name ${name} \
--missing_json_file ${missing_json_file} \
--iteration ${iteration} \
--yolo_path ${origin_json_dir}


# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --part1     Run only part 1 (cropper.py)"
    echo "  --part2     Run only part 2 (remove_bbox.py)"
    echo "  --both      Run both parts in parallel (default)"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --part1          # Run only part 1"
    echo "  $0 --part2          # Run only part 2"
    echo "  $0 --both           # Run both parts in parallel"
    echo "  $0                  # Run both parts in parallel (default)"
}

# Default values
RUN_PART1=false
RUN_PART2=false
RUN_BOTH=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --part1)
            RUN_PART1=true
            shift
            ;;
        --part2)
            RUN_PART2=true
            shift
            ;;
        --both)
            RUN_BOTH=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# If no specific part is selected, default to both
if [[ "$RUN_PART1" == false && "$RUN_PART2" == false && "$RUN_BOTH" == false ]]; then
    RUN_BOTH=true
fi

# get new images
### part 1
if [[ "$RUN_PART1" == true || "$RUN_BOTH" == true ]]; then
    if [[ "$RUN_BOTH" == true ]]; then
        echo "Starting part 1 in background..."
        python3 cropper.py \
            --image_dir ${name}/ \
            --json_name ${loc_dir}/cropped_images/${name}_missing.json \
            --filter_txt ${origin_json_dir}frame_list_${name}.txt \
            --filter_frame_flag True &
    else
        echo "Running part 1..."
        python3 cropper.py \
            --image_dir ${name}/ \
            --json_name ${loc_dir}/cropped_images/${name}_missing.json \
            --filter_txt ${origin_json_dir}frame_list_${name}.txt \
            --filter_frame_flag True
    fi
fi
### part 1 ends

### part 2
if [[ "$RUN_PART2" == true || "$RUN_BOTH" == true ]]; then
    if [[ "$RUN_BOTH" == true ]]; then
        echo "Starting part 2 in background..."
        python3 remove_bbox.py --name ${name} --json_path ${origin_json_dir} &
    else
        echo "Running part 2..."
        python3 remove_bbox.py --name ${name} --json_path ${origin_json_dir}
    fi
fi
### part 2 ends

# Wait for background processes if running both
if [[ "$RUN_BOTH" == true ]]; then
    echo "Waiting for both parts to complete..."
    wait
    echo "Both parts completed!"
fi

# python3 side_process/add_missing.py --name clip1 --missing_json_file ${loc_dir}/cropped_images/clip1_missing.json
# python3 side_process/add_missing.py --name clip2 --missing_json_file ${loc_dir}/cropped_images/clip2_missing.json
# python3 side_process/add_missing.py --name clip3 --missing_json_file ${loc_dir}/cropped_images/clip3_missing.json
# python3 side_process/add_missing.py --name clip4 --missing_json_file ${loc_dir}/cropped_images/clip4_missing.json
# python3 side_process/add_missing.py --name clip5 --missing_json_file ${loc_dir}/cropped_images/clip5_missing.json
# python3 side_process/add_missing.py --name clip6 --missing_json_file ${loc_dir}/cropped_images/clip6_missing.json

# python3 side_process/create_video.py