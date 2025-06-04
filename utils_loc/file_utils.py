import yaml
import glob
import os
import random
import cv2
from pathlib import Path
import json

def save_detections_to_yaml(detections, output_yaml, image_dir, confidence_threshold=0.6):
    """Save high confidence detections to YAML file for YOLO training.
    
    Args:
        detections (list): List of all detections
        output_yaml (str): Path to save YAML file
        image_dir (str): Path to directory containing images
        confidence_threshold (float): Confidence threshold for pseudo-labels (higher for training)
    """
    
    print(f"\nPreparing YAML data with confidence threshold {confidence_threshold}...")
    print(f"Total detections before filtering: {len(detections)}")
    
    # Debug: Print some sample detections
    print("\nSample detection data:")
    if detections:
        print(f"First detection: {detections[0]}")
    
    # Get a mapping of frame numbers to actual filenames
    frame_to_filename = {}
    for filepath in glob.glob(os.path.join(image_dir, "frame_*.png")):
        filename = os.path.basename(filepath)
        # Extract frame number from filename (e.g., "frame_05393_time_00_02_59.767.png")
        try:
            frame_num = filename.split('_')[1]  # Get the frame number part
            frame_to_filename[frame_num] = filename
        except IndexError:
            print(f"Warning: Unexpected filename format: {filename}")
            continue
    
    print(f"\nFound {len(frame_to_filename)} frame files in directory")
    if frame_to_filename:
        print("Sample frame mappings:")
        sample_items = list(frame_to_filename.items())[:5]
        for frame_num, filename in sample_items:
            print(f"Frame {frame_num} -> {filename}")
    
    # Filter very high confidence detections for pseudo-labeling
    high_confidence_detections = [
        det for det in detections 
        if det['confidence'] > confidence_threshold
    ]
    
    print(f"\nHigh confidence detections after filtering: {len(high_confidence_detections)}")
    if high_confidence_detections:
        print("Sample high confidence detection:")
        print(high_confidence_detections[0])
    
    # Get unique frame numbers with high confidence detections
    frame_numbers = sorted(list(set(det['frame_number'] for det in high_confidence_detections)))
    print(f"Number of frames with high confidence detections: {len(frame_numbers)}")
    
    # Prepare YAML structure
    yaml_data = {
        'path': str(Path(image_dir).resolve()),  # Absolute path to images
        'train': [],  # List of training images
        'val': [],    # List of validation images (can be populated later)
        'names': {0: 'person'},  # Class names
        'nc': 1,  # Number of classes
        'frames': {}  # Frame-specific data including detections and features
    }
    
    processed_frames = 0
    total_detections = 0
    
    # Process each frame with high confidence detections
    for frame_num in frame_numbers:
        frame_detections = [det for det in high_confidence_detections if det['frame_number'] == frame_num]
        
        # Get actual filename for this frame
        if frame_num not in frame_to_filename:
            print(f"Warning: No matching file found for frame {frame_num}")
            continue
            
        frame_path = frame_to_filename[frame_num]
        
        # Add frame path to training set if not already added
        if frame_path not in yaml_data['train']:
            yaml_data['train'].append(frame_path)
        
        # Get image dimensions
        img_path = os.path.join(image_dir, frame_path)
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image: {img_path}")
            continue
            
        height, width = img.shape[:2]
        
        # Prepare frame data
        frame_data = {
            'detections': []
        }
        
        frame_detection_count = 0
        
        # Add each detection
        for det in frame_detections:
            try:
                # Get coordinates (already in original scale)
                x1 = float(det['x1'])
                y1 = float(det['y1'])
                x2 = float(det['x2'])
                y2 = float(det['y2'])
                
                # Normalize coordinates
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                # Ensure coordinates are valid
                if not all(0 <= coord <= 1 for coord in [x_center, y_center, bbox_width, bbox_height]):
                    print(f"Warning: Invalid normalized coordinates for detection in frame {frame_num}")
                    continue
                
                # Get feature vector if available
                feature_vector = None
                if 'feature_vector' in det:
                    feature_vector = det['feature_vector'].tolist() if hasattr(det['feature_vector'], 'tolist') else det['feature_vector']
                
                detection_data = {
                    'class': 0,  # person class
                    'bbox': [x_center, y_center, bbox_width, bbox_height],
                    'confidence': float(det['confidence']),
                    'person_id': int(det['person_id']) if isinstance(det['person_id'], (int, float)) else det['person_id'],
                    'room_id': det['room_id']
                }
                
                if feature_vector is not None:
                    detection_data['feature_vector'] = feature_vector
                
                frame_data['detections'].append(detection_data)
                frame_detection_count += 1
                total_detections += 1
            except Exception as e:
                print(f"Warning: Error processing detection in frame {frame_num}: {str(e)}")
                continue
        
        # Add frame data if it has detections
        if frame_data['detections']:
            yaml_data['frames'][frame_num] = frame_data
            processed_frames += 1
            
            # Print progress every 100 frames
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames} frames, {total_detections} total detections")
    
    # Save to YAML
    try:
        print(f"\nAttempting to save YAML file to: {output_yaml}")
        print("\nYAML data summary before saving:")
        print(f"- Number of training images: {len(yaml_data['train'])}")
        print(f"- Number of frames with detections: {len(yaml_data['frames'])}")
        print(f"- Total detections: {total_detections}")
        print(f"- Sample frame paths: {yaml_data['train'][:5]}")
        if yaml_data['frames']:
            sample_frame = next(iter(yaml_data['frames']))
            print(f"- Sample frame data: Frame {sample_frame} has {len(yaml_data['frames'][sample_frame]['detections'])} detections")
        
        with open(output_yaml, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        # Verify the saved file
        if os.path.exists(output_yaml):
            file_size = os.path.getsize(output_yaml)
            print(f"\nYAML file saved successfully:")
            print(f"- Path: {output_yaml}")
            print(f"- File size: {file_size} bytes")
            
            # Read back and verify content
            with open(output_yaml, 'r') as f:
                verify_data = yaml.safe_load(f)
                print("\nVerification of saved data:")
                print(f"- Training images: {len(verify_data['train'])}")
                print(f"- Frames with detections: {len(verify_data['frames'])}")
                if verify_data['frames']:
                    total_saved_detections = sum(len(frame_data['detections']) for frame_data in verify_data['frames'].values())
                    print(f"- Total saved detections: {total_saved_detections}")
        else:
            print(f"Error: Failed to create YAML file at {output_yaml}")
    except Exception as e:
        print(f"Error saving YAML file: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())

def save_detections_to_json(detections, output_json, num_frames=20, confidence_threshold=0.6):
    """Save high confidence detections to JSON file for training.
    
    Args:
        detections (list): List of all detections
        output_json (str): Path to save JSON file
        num_frames (int): Number of frames to randomly select
        confidence_threshold (float): Confidence threshold for pseudo-labels (higher for training)
    """
    # Filter high confidence detections (using higher threshold for training)
    high_confidence_detections = [
        det for det in detections 
        if det['confidence'] > confidence_threshold
    ]
    
    # Get unique frame numbers
    frame_numbers = list(set(int(det['frame_number']) for det in high_confidence_detections))
    
    # Randomly select frames
    selected_frames = sorted(random.sample(frame_numbers, min(num_frames, len(frame_numbers))))
    
    # Filter detections for selected frames
    selected_detections = [
        det for det in high_confidence_detections 
        if int(det['frame_number']) in selected_frames
    ]
    
    # Format detections for training
    training_data = []
    for det in selected_detections:
        # Get feature vector if available
        feature_vector = None
        if 'feature_vector' in det:
            feature_vector = det['feature_vector'].tolist() if hasattr(det['feature_vector'], 'tolist') else det['feature_vector']
        
        # Handle both bbox formats
        if 'bbox' in det:
            bbox = det['bbox']
        else:
            bbox = [det['x1'], det['y1'], det['x2'], det['y2']]
        
        training_data.append({
            'frame_number': int(det['frame_number']),
            'person_id': det['person_id'],
            'room_id': det['room_id'],
            'confidence': float(det['confidence']),
            'bbox': {
                'x1': float(bbox[0]),
                'y1': float(bbox[1]),
                'x2': float(bbox[2]),
                'y2': float(bbox[3])
            },
            'center': {
                'x': float(det['x']),
                'y': float(det['y'])
            },
            'feature_vector': feature_vector
        })
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Saved {len(training_data)} high-confidence detections from {len(selected_frames)} randomly selected frames to {output_json}")