import json
import cv2
import os
import argparse

def load_data(file_path: str) -> list[dict]:
    """Load JSON data from file"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Creating empty file.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        empty_data = []  # Empty list for missing data
        with open(file_path, 'w') as file:
            json.dump(empty_data, file)
        return empty_data

def draw_bounding_box(frame_list: list[int], yolo_data: dict, image_dir: str, output_dir: str):
    """Draw bounding boxes on images using yolo format data"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for frame_key, detections in yolo_data.items():
        # Extract frame number from key like "frame_0001.png"
        frame_num = int(frame_key.split('_')[1].split('.')[0])
        
        # Only process frames that are in frame_list
        if frame_num not in frame_list:
            continue
        
        # Load the image
        image_path = f"{image_dir}/frame_{frame_num:04d}.png"
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
            
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image: {image_path}")
            continue
        
        # Draw bounding boxes for each detection
        for detection in detections:
            coordinates = detection['coordinates']
            conf = detection['conf']
            cls = detection['cls']
            
            x1, y1, x2, y2 = map(int, coordinates)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add confidence text
            conf_text = f"{conf:.2f}"
            cv2.putText(img, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the image with bounding boxes
        output_path = f"{output_dir}/frame_{frame_num:04d}.png"
        cv2.imwrite(output_path, img)
    
    print(f"Bounding boxes drawn and saved to {output_dir}")

# adding crop image bounding box into yolo_results_json_path in the format in yolo_result
def add_crop_image_bounding_box(crop_data: list[dict], origin_data: dict) -> dict:
    result = origin_data.copy()  # Start with original data
    
    for frame in crop_data:
        frame_num = frame['frame']
        frame_key = f"frame_{frame_num:04d}.png"
        
        # Get existing detections for this frame (if any)
        existing_detections = result.get(frame_key, [])
        
        # Add new detections from crop data
        for person in frame['persons']:
            coordinate = person['coordinate']
            x1, y1, x2, y2 = coordinate
            
            # Convert to yolo format: [x1, y1, x2, y2]
            detection = {
                "coordinates": [x1, y1, x2, y2],
                "conf": 1.0,  # Set confidence to 1.0 for crop data
                "cls": 0,     # Class 0 for person
                "frame_name": frame_num
            }
            existing_detections.append(detection)
        
        result[frame_key] = existing_detections
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Add missing frames to origin data')
    parser.add_argument('--name', type=str, default='clip1', help='Directory containing images')
    args = parser.parse_args()
    name = args.name
    yolo_results_json_path = 'yolo_results_train_json'
    crop_data = load_data(f'cropped_images/{name}_missing.json')

    frame_list = []
    # if crop_data:
    #     for frame in crop_data:
    #         frame_list.append(frame['frame'])
    #     frame_list.sort()
    # else:
    for i in range(1,901,15):
        frame_list.append(i)
    with open(f'{yolo_results_json_path}/frame_list_{name}.txt', 'w') as f:
        for frame in frame_list:
            f.write(f"{frame}\n")
    

    origin_data = load_data(f'{yolo_results_json_path}/{name}_result.json')

    
    # Convert crop data to yolo format
    crop_yolo_format = add_crop_image_bounding_box(crop_data, origin_data)
    
    print(f"Converted {len(crop_yolo_format)} frames to yolo format")
    print(f"Saved to {yolo_results_json_path}/{name}_with_missing.json")

    # save crop_yolo_format to json file
    with open(f'{yolo_results_json_path}/{name}_with_missing.json', 'w') as f:
        json.dump(crop_yolo_format, f, indent=2)

    print("Saving completed")
    # for frame_key, detections in crop_yolo_format.items():
    #     for obj in detections:
    #         if obj.get('conf', 0) == 1.0:
    #             print(obj['coordinates'])
    frame_id = 1
    frame_key = f'frame_{frame_id:04d}.png'
    if frame_key in crop_yolo_format:
        print(len(crop_yolo_format[frame_key]))
    else:
        print(f"No data for {frame_key}")    
    image_dir = name
    output_dir = f"output_bounding_boxes_{name}"
    draw_bounding_box(frame_list, crop_yolo_format, image_dir, output_dir)

if __name__ == '__main__':
    main()

