import os
import yaml
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import argparse
from run_track import process_frames, DINOPersonTracker

def select_frames(total_frames, num_frames=20):
    """Select frames evenly spaced throughout the video"""
    step = total_frames // num_frames
    selected_frames = list(range(0, total_frames, step))[:num_frames]
    return selected_frames

def extract_ground_truth(excel_file, image_dir, output_yaml, confidence_threshold=0.25):
    """
    Extract ground truth data from tracking results
    
    Args:
        excel_file: Path to Excel file with tracking results
        image_dir: Directory containing frame images
        output_yaml: Path to output YAML file
        confidence_threshold: Minimum confidence score for YOLO detections (default 0.25, matching YOLOv11)
    """
    # Read tracking results
    df = pd.read_excel(excel_file)
    
    # Filter by confidence threshold (matching YOLOv11's default)
    df = df[df['confidence'] >= confidence_threshold]
    
    # Get unique frames
    unique_frames = df['frame_number'].unique()
    total_frames = len(unique_frames)
    
    # Select frames for ground truth
    selected_frames = select_frames(total_frames)
    selected_frame_numbers = [unique_frames[i] for i in selected_frames]
    
    # Initialize our DINO-enhanced person tracker
    person_tracker = DINOPersonTracker(max_frames_missing=90, location_threshold=250,
                                     feature_similarity_threshold=0.7, max_people=10)
    
    # Prepare ground truth data
    ground_truth = {
        "dataset": {
            "train": [],
            "val": [],
            "test": []
        },
        "nc": 1,  # number of classes (person)
        "names": ["person"],  # class names
        "total_frames": total_frames,
        "selected_frames": selected_frame_numbers,
        "confidence_threshold": confidence_threshold,
        "model": "yolov11n",  # specify the model used
        "max_people": 10  # maximum number of people to track
    }
    
    # Process each selected frame
    for frame_num in selected_frame_numbers:
        frame_data = df[df['frame_number'] == frame_num]
        
        # Get corresponding image
        image_path = os.path.join(image_dir, f"frame_{frame_num}.png")
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for frame {frame_num}")
            continue
            
        # Read image to get dimensions
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Prepare detections for the tracker
        frame_detections = []
        for _, row in frame_data.iterrows():
            frame_detections.append({
                'frame_number': frame_num,
                'person_id': row['person_id'],
                'x': row['x'],
                'y': row['y'],
                'x1': row['x1'],
                'y1': row['y1'],
                'x2': row['x2'],
                'y2': row['y2'],
                'confidence': row['confidence']
            })
        
        # Update tracker with current frame's detections
        updated_detections = person_tracker.update(int(frame_num), frame_detections, image)
        
        # Extract person data from updated detections
        persons = []
        for det in updated_detections:
            # Convert bounding box to YOLO format (normalized x_center, y_center, width, height)
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            x_center = (x1 + x2) / (2 * width)
            y_center = (y1 + y2) / (2 * height)
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            
            person = {
                "class": 0,  # person class
                "bbox": [x_center, y_center, box_width, box_height],
                "confidence": float(det['confidence']),
                "person_id": int(det['person_id'])  # Use the tracked ID
            }
            persons.append(person)
        
        # Add frame data to ground truth
        frame_info = {
            "frame_number": int(frame_num),
            "image_path": image_path,
            "image_dimensions": {
                "width": width,
                "height": height
            },
            "persons": persons
        }
        
        # Split into train/val/test (80/10/10)
        if len(ground_truth["dataset"]["train"]) < len(selected_frames) * 0.8:
            ground_truth["dataset"]["train"].append(frame_info)
        elif len(ground_truth["dataset"]["val"]) < len(selected_frames) * 0.1:
            ground_truth["dataset"]["val"].append(frame_info)
        else:
            ground_truth["dataset"]["test"].append(frame_info)
    
    # Save ground truth to YAML
    with open(output_yaml, 'w') as f:
        yaml.dump(ground_truth, f, default_flow_style=False, sort_keys=False)
    
    print(f"Ground truth data saved to {output_yaml}")
    print(f"Total frames processed: {total_frames}")
    print(f"Selected frames for ground truth: {len(selected_frames)}")
    print(f"Train/Val/Test split: {len(ground_truth['dataset']['train'])}/{len(ground_truth['dataset']['val'])}/{len(ground_truth['dataset']['test'])}")
    print(f"Total persons annotated: {sum(len(frame['persons']) for frame in ground_truth['dataset']['train'] + ground_truth['dataset']['val'] + ground_truth['dataset']['test'])}")
    print(f"Confidence threshold used: {confidence_threshold} (matching YOLOv11)")
    print(f"Maximum person ID used: {max(p['person_id'] for frame in ground_truth['dataset']['train'] + ground_truth['dataset']['val'] + ground_truth['dataset']['test'] for p in frame['persons'])}")

def main():
    parser = argparse.ArgumentParser(description='Generate ground truth data from tracking results')
    parser.add_argument('--input_folder', type=str, required=True,
                      help='Input folder containing frame images')
    parser.add_argument('--output_excel', type=str, required=True,
                      help='Output Excel file path for tracking results')
    parser.add_argument('--output_images', type=str, required=True,
                      help='Output folder path for visualization images')
    parser.add_argument('--json_file', type=str, required=True,
                      help='JSON file containing room definitions')
    parser.add_argument('--ground_truth_yaml', type=str, required=True,
                      help='Output YAML file path for ground truth data')
    parser.add_argument('--confidence_threshold', type=float, default=0.25,
                      help='Minimum confidence score for YOLO detections (default: 0.25, matching YOLOv11)')
    
    args = parser.parse_args()
    
    # First run the tracking process
    print("Running human tracking process...")
    process_frames(args.input_folder, args.output_excel, args.output_images, args.json_file)
    
    # Then extract ground truth
    print("\nGenerating ground truth data...")
    extract_ground_truth(args.output_excel, args.output_images, args.ground_truth_yaml, args.confidence_threshold)

if __name__ == "__main__":
    main() 