import os
import json
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import colorsys
import argparse
from tqdm import tqdm

from utils.file_utils import save_detections_to_json, save_detections_to_yaml  
from tracker.tracker import DINOPersonTracker
from tracker.feature_extractor import DINOFeatureExtractor 
from preprocess.label import point_in_quad, draw_rooms

def yolo_init(ckpt='./leo/ckpt/yolo11x.pt'):
    model = YOLO(ckpt)
    model.conf = 0.01  # Much lower confidence threshold for detections
    model.iou = 0.1    # Lower IoU threshold for NMS
    model.agnostic_nms = True  # Use class-agnostic NMS
    model.max_det = 12
    model.classes = [0]  # Only track person class

    # Configure tracking parameters for better tracking
    model.tracker = {
        'track_high_thresh': 0.01,  # Much lower threshold to catch more detections
        'track_low_thresh': 0.005,  # Lower threshold for second association
        'new_track_thresh': 0.01,   # Lower threshold for new tracks
        'track_buffer': 150,        # Increase buffer to handle occlusions
        'match_thresh': 0.2,        # Lower matching threshold
        'fuse_score': True,
        'min_box_area': 3,          # Lower minimum box area
        'max_age': 150,             # Increase maximum age of a track
        'min_hits': 1               # Lower minimum hits to confirm a track
    }

    return model
    


def process_frames(
    input_folder, 
    output_excel, 
    output_images, 
    json_file, 
    output_json=None, 
    output_yaml=None
):
    model = yolo_init()
    
    # Load room definitions
    with open(json_file, 'r') as f:
        room_data = json.load(f)
    json_scale = room_data['image_scale']
    
    # Create DataFrame to store results
    results_data = []
    # Dictionary to store track histories (for visualization)
    track_history = {}
    
    # Create output directory for visualizations
    output_dir = Path(output_images)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize our DINO-enhanced person tracker with more lenient settings
    tracker = DINOPersonTracker(
        max_frames_missing=150,           # Increase frames before considering missing
        location_threshold=400,           # Increase distance threshold
        feature_similarity_threshold=0.3, # Lower feature similarity threshold
        max_people=12                     # Increase maximum number of people to track
    )
    
    # Initialize DINO feature extractor for saving features
    dino_extractor = DINOFeatureExtractor()
    
    # Frame counters for tracking
    frame_count = 0
    
    # Process each frame in input directory
    frames_dir = Path(input_folder)
    if not frames_dir.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
        
    all_frames = sorted(frames_dir.glob('*.png'))
    total_num = len(all_frames)

    print(f"Processing {total_num} frames...")
    
    # Optional: ID color mapping to visualize consistently
    id_colors = {}
    
    # Store all detections for JSON/YAML output
    all_detections = []
    
    # Store frames for feature extraction
    frame_images = {}
    
    for frame_path in tqdm(all_frames):
        # Read image for visualization
        image = cv2.imread(str(frame_path))

        
        # Store frame for feature extraction
        frame_num = frame_path.stem.split('_')[1]
        frame_images[frame_num] = image.copy()
        time_str = '_'.join(frame_path.stem.split('_')[3:])
        
        
        # Update frame count
        frame_count += 1
        
        # Get original image dimensions
        height, width = image.shape[:2]
        
        # Calculate display scale (for visualization)
        max_dimension = 1920  # Maximum dimension for visualization
        display_scale = min(max_dimension / width, max_dimension / height)
        
        # Scale image for display if needed
        if display_scale < 1:
            new_width = int(width * display_scale)
            new_height = int(height * display_scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Draw room boundaries with proper scaling
        image = draw_rooms(image, room_data, display_scale, json_scale)
        
        # Run YOLO detection with tracking enabled and more lenient settings
        results = model.track(
            frame_path,
            classes=[0],  # Track only person class
            persist=True,
            conf=0.1,     # Lower confidence threshold
            iou=0.3,      # Lower IoU threshold
            show=False,
            verbose=False
        )
        
        # Detections for current frame (to pass to our tracker)
        frame_detections = []
        
        # Process each detection with lower confidence threshold
        for result in results:
            boxes = result.boxes
            
            # Check if we have tracking IDs available
            if boxes.id is None:
                print(f"Warning: No tracking IDs found for frame {frame_num}. Skipping...")
                continue

            track_ids = boxes.id.int().cpu().tolist()
            for i, box in enumerate(boxes):
                # Get bounding box coordinates and tracking ID
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                track_id = track_ids[i]
                
                # Skip only very low confidence detections
                if conf < 0.1:  # Lower confidence threshold
                    continue
                
                # Scale the coordinates for display
                if display_scale < 1:
                    x1, y1, x2, y2 = [int(x * display_scale) for x in [x1, y1, x2, y2]]
                
                # Calculate center point of detection
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Original coordinates (unscaled)
                original_center_x = int(center_x / display_scale)
                original_center_y = int(center_y / display_scale)
                
                # Original bounding box coordinates (unscaled)
                original_x1 = int(x1 / display_scale)
                original_y1 = int(y1 / display_scale)
                original_x2 = int(x2 / display_scale)
                original_y2 = int(y2 / display_scale)
                
                # Determine room
                room_id = None
                for quad in room_data['quads']:
                    if point_in_quad((original_center_x, original_center_y), quad['points'], json_scale):
                        # Extract room number from shape name (e.g., "Shape 9" -> 9)
                        shape_name = quad['name']
                        room_number = int(shape_name.split()[-1])
                        room_id = room_number
                        break
                
                # Add to frame detections
                frame_detections.append({
                    'frame_number': frame_num,
                    'timestamp': time_str,
                    'person_id': track_id,
                    'room_id': room_id,
                    'confidence': conf,
                    'x': original_center_x,
                    'y': original_center_y,
                    'display_x': center_x,
                    'display_y': center_y,
                    'x1': original_x1,
                    'y1': original_y1,
                    'x2': original_x2,
                    'y2': original_y2
                })
        
        # Update our custom tracker with the current frame's detections
        updated_detections = tracker.update(int(frame_num), frame_detections, frame_images[frame_num])
        
        # Final check for duplicate IDs in the same frame
        ids_seen_in_frame = set()
        unique_detections = []
        
        for det in updated_detections:
            if det['person_id'] not in ids_seen_in_frame:
                ids_seen_in_frame.add(det['person_id'])
                unique_detections.append(det)
        
        # Add updated detections to results and visualize
        for det in unique_detections:
            # Add to overall results
            results_data.append({
                'frame_number': det['frame_number'],
                'timestamp': det['timestamp'],
                'person_id': det['person_id'],
                'room_id': det['room_id'],
                'confidence': det['confidence'],
                'x': det['x'],
                'y': det['y']
            })
            
            # Update track visualization history
            track_id = det['person_id']
            if track_id not in track_history:
                track_history[track_id] = []
                
                # Assign a consistent color for this ID
                if track_id not in id_colors:
                    try:
                        if isinstance(track_id, str):
                            if track_id.isdigit():
                                id_num = int(track_id)
                            else:
                                id_num = hash(track_id)
                        else:
                            id_num = int(track_id)
                        
                        hue = (id_num * 0.618033988749895) % 1.0
                        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.95)]
                        id_colors[track_id] = (b, g, r)  # OpenCV uses BGR
                    except (ValueError, TypeError):
                        id_colors[track_id] = (0, 255, 255)  # Yellow as default
            
            # Add to track history
            track_history[track_id].append((det['display_x'], det['display_y']))
            
            # Limit history length
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)
            
            # Get color for this ID
            id_color = id_colors.get(track_id, (0, 255, 255))  # Default to yellow if not found
            
            # Draw bounding box, center point, and person ID
            cv2.rectangle(image, (det['x1'], det['y1']), (det['x2'], det['y2']), id_color, 2)
            cv2.circle(image, (det['display_x'], det['display_y']), 5, id_color, -1)
            
            # Add person ID to the top of the bounding box
            cv2.putText(image, f"ID: {track_id}", (det['x1'], det['y1'] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)
            
            # Draw tracking lines
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [points], False, id_color, 2)
            
            # Draw room label if known
            if det['room_id'] is not None:
                cv2.putText(image, f"Room {det['room_id']}", (det['display_x'], det['display_y'] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save visualization
        output_path = output_dir / f"frame_{frame_num}.png"
        cv2.imwrite(str(output_path), image)
        
        # After processing detections, add them to all_detections
        for det in unique_detections:
            all_detections.append(det)
    
    print("\nProcessing complete. Saving outputs...")
    print(f"Total detections collected: {len(all_detections)}")
    
    # Save high-confidence detections to JSON/YAML if specified
    if output_json or output_yaml:
        print("\nExtracting feature vectors for detections...")
        feature_extraction_count = 0
        # Get feature vectors for detections
        for det in all_detections:
            frame_num = det['frame_number']
            if frame_num in frame_images:
                # Extract crop from frame
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                frame = frame_images[frame_num]
                crop = frame[y1:y2, x1:x2]
                
                # Skip if crop is invalid
                if crop.size == 0:
                    print(f"Warning: Invalid crop for detection in frame {frame_num}")
                    continue
                
                # Extract feature vector using DINO
                feature_vector = dino_extractor.extract_features(crop)
                det['feature_vector'] = feature_vector
                feature_extraction_count += 1
                
                if feature_extraction_count % 100 == 0:
                    print(f"Extracted features for {feature_extraction_count} detections")
        
        print(f"Successfully extracted features for {feature_extraction_count} detections")
        
        if output_json:
            print(f"\nSaving JSON output to {output_json}...")
            save_detections_to_json(all_detections, output_json, num_frames=20)
        
        if output_yaml:
            print(f"\nAttempting to save YAML output to {output_yaml}...")
            print(f"Input folder for YAML: {input_folder}")
            try:
                save_detections_to_yaml(all_detections, output_yaml, str(Path(input_folder).resolve()))
                # Verify the file was created
                if os.path.exists(output_yaml):
                    print(f"YAML file successfully created at {output_yaml}")
                    # Print file size
                    file_size = os.path.getsize(output_yaml)
                    print(f"YAML file size: {file_size} bytes")
                else:
                    print(f"Error: YAML file was not created at {output_yaml}")
            except Exception as e:
                print(f"Error while saving YAML file: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results_data)
    
    if df.empty:
        print("Warning: No detections were made. No data to save.")
        return
        
    # Count unique IDs
    unique_ids = df['person_id'].nunique()
    print(f"\nTracking Statistics:")
    print(f"Total unique person IDs: {unique_ids}")
    print(f"Total original YOLO ID mappings: {len(tracker.person_features)}")
    print(f"Maximum assigned ID: {tracker.next_perm_id - 1}")
    
    # Room occupancy statistics
    if 'room_id' in df.columns:
        room_occupancy = df.groupby(['frame_number', 'room_id']).size().reset_index(name='count')
        max_occupancy = room_occupancy.groupby('room_id')['count'].max().reset_index()
        print("\nMaximum room occupancy:")
        for _, row in max_occupancy.iterrows():
            if pd.notna(row['room_id']):
                print(f"Room {row['room_id']}: {row['count']} people")
    
    df.to_excel(output_excel, index=False)
    print(f"\nResults saved to {output_excel}")
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video frames for human tracking')
    parser.add_argument('--input_folder', type=str, required=True,
                      help='Input folder containing frame images')
    parser.add_argument('--output_excel', type=str, required=True,
                      help='Output Excel file path for tracking results')
    parser.add_argument('--output_images', type=str, required=True,
                      help='Output folder path for visualization images')
    parser.add_argument('--json_file', type=str, required=True,
                      help='JSON file containing room definitions')
    parser.add_argument('--output_json', type=str,
                      help='Output JSON file path for high-confidence detections')
    parser.add_argument('--output_yaml', type=str,
                      help='Output YAML file path for YOLO training')
    
    args = parser.parse_args()
    
    process_frames(args.input_folder, args.output_excel, args.output_images, 
                  args.json_file, args.output_json, args.output_yaml)