import cv2
import json
import os
import numpy as np

RESIZE = 2/3

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_crops(data, output_path):
    for frame in data:
        frame_number = frame['frame']
        frame_path = f'frames/frame_{frame_number:04d}.png'
        frame_data = frame['persons']
        for person in frame_data:
            subject_id = person['subject_id']
            subject_name = person['subject_name']
            coordinate = person['coordinate']
            x1, y1, x2, y2 = coordinate
            #crop images
        img = cv2.imread(frame_path)
        if img is not None:
            h, w = img.shape[:2]  # should be 1080p: h=1080, w=1920
            # Ensure coordinates are in correct order and within bounds
            x1, x2 = sorted([max(0, min(x1, w-1)), max(0, min(x2, w-1))])
            y1, y2 = sorted([max(0, min(y1, h-1)), max(0, min(y2, h-1))])
            print(x1, x2, y1, y2)
            if x2 > x1 and y2 > y1:
                crop_img = img[y1:y2, x1:x2]
                # create subject dir for each subject
                subject_dir = f'{output_path}/{subject_name}'
                if not os.path.exists(subject_dir):
                    os.makedirs(subject_dir)
                cv2.imwrite(f'{subject_dir}/crop_{frame_number}.png', crop_img)

def create_rectangle(data, output_rect_path, start_frame, end_frame):
    # here we are drawing bounding boxes around the crops in the original images
    if not os.path.exists(output_rect_path):
        os.makedirs(output_rect_path)
    # Build a mapping from frame number to frame data for quick lookup
    frame_map = {frame['frame']: frame for frame in data}
    for frame_number in range(start_frame, end_frame + 1):
        frame_path = f'frames/frame_{frame_number:04d}.png'
        output_path = f'{output_rect_path}/frame_{frame_number:04d}.png'
        if frame_number in frame_map:
            frame_data = frame_map[frame_number]['persons']
            img = cv2.imread(frame_path)
            if img is not None:
                for person in frame_data:
                    subject_id = person['subject_id']
                    subject_name = person['subject_name']
                    coordinate = person['coordinate']
                    x1, y1, x2, y2 = coordinate
                    #add labels
                    cv2.putText(img, subject_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite(output_path, img)
            else:
                # If the original frame image is missing, create a blank image
                blank_img = 255 * np.ones((1080, 1920, 3), dtype=np.uint8)
                cv2.imwrite(output_path, blank_img)
        else:
            # If the frame is not in the JSON, create a blank image
            blank_img = 255 * np.ones((1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(output_path, blank_img)

def print_subject_count(data, subject_count={}):
    # 
    for frame in data:
        frame_data = frame['persons']
        for person in frame_data:
            subject_id = person['subject_id']
            subject_name = person['subject_name']
            if subject_name not in subject_count:
                subject_count[subject_name] = 0
            subject_count[subject_name] += 1
    for subject_name, count in subject_count.items():
        print(f'{subject_name}: {count}')
        
def execution(start_frame, end_frame, output_rect_path):
    # Collect all image filenames in order
    image_files = [
        os.path.join(output_rect_path, f"frame_{i:04d}.png")
        for i in range(start_frame, end_frame + 1)
        if os.path.exists(os.path.join(output_rect_path, f"frame_{i:04d}.png"))
    ]
    if not image_files:
        print("No images found in the specified frame range.")
        return

    # Read the first image to get frame size
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print("First image could not be read.")
        return
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(
        f"output_video_{start_frame}_{end_frame}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,  # fps
        (width, height)
    )

    for filename in image_files:
        img = cv2.imread(filename)
        if img is not None:
            out.write(img)
        else:
            print(f"Warning: Could not read {filename}, skipping.")

    out.release()
    print(f"Video saved as output_video_{start_frame}_{end_frame}.mp4")

def main():
    json_path = 'cropped_images/crops.json'
    data = load_json(json_path)
    #print the number of each subject and their corresponding total number of cropsin the data
    print_subject_count(data)

    # output_path = 'output_images'
    output_path_rect = 'output_images_rect'
    start_frame = 5000
    end_frame = 5300
    # visualize_crops(data, output_path)
    create_rectangle(data, output_path_rect, start_frame, end_frame)
    
    #convert the dir back to video
    execution(start_frame, end_frame, output_path_rect)

if __name__ == '__main__':
    main()