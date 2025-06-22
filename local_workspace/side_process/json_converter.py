import json
import os

def convert_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # convert to format like local_workspace/loc03_data/origin/clip1_result.json
    converted_data = {}
    
    for frame_name, detections in data.items():
        # Convert frame name from "frame_00000.png" to "frame_0001.png" format
        # Extract the number and add 1 to match the target format
        frame_num = int(frame_name.split('_')[1].split('.')[0])
        new_frame_name = f"frame_{frame_num + 1:04d}.png"
        
        converted_detections = []
        for detection in detections:
            converted_detection = {
                "coordinates": detection["bbox"],  # bbox -> coordinates
                "conf": detection["score"],        # score -> conf
                "cls": 0,                         # Add class (assuming class 0)
                "frame_name": frame_num + 1       # Add frame number
            }
            converted_detections.append(converted_detection)
        
        converted_data[new_frame_name] = converted_detections

    return converted_data

def split_json_by_frames(data, frames_per_file=900):
    """
    Split the JSON data into multiple files, each containing specified number of frames.
    
    Args:
        data: Dictionary with frame names as keys and detections as values
        frames_per_file: Number of frames per output file (default: 900)
    
    Returns:
        List of dictionaries, each containing frames_per_file frames
    """
    # Sort frames by frame number to ensure proper ordering
    sorted_frames = sorted(data.items(), key=lambda x: int(x[0].split('_')[1].split('.')[0]))
    
    split_data = []
    current_batch = {}
    frame_count = 0
    
    for frame_name, detections in sorted_frames:
        # Reset frame numbering for each clip (1-900)
        new_frame_num = frame_count + 1
        new_frame_name = f"frame_{new_frame_num:04d}.png"
        
        # Update detections with new frame numbers
        updated_detections = []
        for detection in detections:
            updated_detection = detection.copy()
            updated_detection["frame_name"] = new_frame_num
            updated_detections.append(updated_detection)
        
        current_batch[new_frame_name] = updated_detections
        frame_count += 1
        
        if frame_count >= frames_per_file:
            split_data.append(current_batch)
            current_batch = {}
            frame_count = 0
    
    # Add remaining frames if any
    if current_batch:
        split_data.append(current_batch)
    
    return split_data

if __name__ == "__main__":
    data = "loc04_data"
    json_file = f"{data}/ultralytics/output_inference_yolo11x_loc04.json"
    output_dir = f"{data}/origin"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the data
    data = convert_json(json_file)
    print(f"Total frames in converted data: {len(data)}")
    
    # Split into 6 files with 900 frames each
    split_data = split_json_by_frames(data, frames_per_file=900)
    print(f"Split into {len(split_data)} files")
    
    # Save each split as a separate JSON file
    for i, batch_data in enumerate(split_data, 1):
        output_file = os.path.join(output_dir, f"clip{i}_result.json")
        with open(output_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        print(f"Saved clip{i}_result.json with {len(batch_data)} frames")
    
    print(f"\nAll files saved to: {output_dir}")
    
    # Print sample data from first file
    print("\nSample from clip1_result.json:")
    first_file = os.path.join(output_dir, "clip1_result.json")
    with open(first_file, 'r') as f:
        sample_data = json.load(f)
    
    for i, (frame_name, detections) in enumerate(sample_data.items()):
        if i < 2:  # Show first 2 frames
            print(f"\n{frame_name}: {len(detections)} detections")
            for j, detection in enumerate(detections[:2]):  # Show first 2 detections per frame
                print(f"  Detection {j+1}: {detection}")
        else:
            break
