#!/bin/bash

name="clip6"
output_name="output_bounding_boxes_clip6"
json_name="yolo_results_train_json"
# python3 cropper.py --image_dir ${name}/ --json_name ${name}_missing.json --filter_txt ${json_name}/frame_list_${name}.txt --filter_frame_flag True

# python3 side_process/add_missing.py --name clip1 
# python3 side_process/add_missing.py --name clip2 
# python3 side_process/add_missing.py --name clip3 
# python3 side_process/add_missing.py --name clip4 
python3 side_process/add_missing.py --name clip5 
python3 side_process/add_missing.py --name clip6

# python3 side_process/create_video.py