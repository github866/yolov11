#!/bin/bash

name="clip6"
json_name="yolo_results_train_json"
python3 cropper.py --image_dir ${name}/ --json_name ${name}_missing.json --filter_txt ${json_name}/frame_list_${name}.txt --filter_frame_flag True