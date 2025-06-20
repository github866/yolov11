#!/bin/bash
# get new images
### part 1
name="clip1"
json_name="loc03_data/origin/"
python3 cropper.py \
    --image_dir ${name}/ \
    --json_name local_workspace/loc03_data/cropped_images/${name}_missing.json \
    --filter_txt ${json_name}/frame_list_${name}.txt \
    --filter_frame_flag True
### part 1 ends

### part 2
python3 remove_bbox.py --name clip1
### part 2 ends


# python3 side_process/add_missing.py --name clip1 
# python3 side_process/add_missing.py --name clip2 
# python3 side_process/add_missing.py --name clip3 
# python3 side_process/add_missing.py --name clip4 
# python3 side_process/add_missing.py --name clip5 
# python3 side_process/add_missing.py --name clip6

# python3 side_process/create_video.py