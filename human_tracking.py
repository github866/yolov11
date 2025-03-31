import os
import json
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def point_in_quad(point, quad_points, json_scale):
    """Check if a point is inside a quadrilateral"""
    x, y = point
    # Convert quad points to original scale
    original_points = [[int(x / json_scale), int(y / json_scale)] for x, y in quad_points]
    points = np.array(original_points, dtype=np.int32)
    
    # Get the bounds of the quad
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    # Create a mask just large enough for the quad
    mask = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    
    # Check if point is inside
    if x < 0 or x >= max_x + 1 or y < 0 or y >= max_y + 1:
        return False
    return mask[y, x] == 255

def draw_rooms(image, room_data, display_scale, json_scale):
    """Draw room boundaries on the image"""
    for quad in room_data['quads']:
        # The points in room_data are already scaled by json_scale
        # We need to scale them up to match the image size
        scaled_points = [[int(x / json_scale * display_scale), int(y / json_scale * display_scale)] for x, y in quad['points']]
        points = np.array(scaled_points, dtype=np.int32)
        cv2.polylines(image, [points], True, (0, 255, 0), 2)
        # Add room ID
        center = np.mean(points, axis=0).astype(int)
        cv2.putText(image, f"Room {quad['id']}", (center[0], center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def process_frames():
    # Load YOLO model
    model = YOLO('yolo11n.pt')
    
    # Load room definitions
    with open('loc01.json', 'r') as f:
        room_data = json.load(f)
    
    # Get image scale from JSON
    json_scale = room_data['image_scale']
    
    # Create DataFrame to store results
    results_data = []
    
    # Create output directory for visualizations
    output_dir = Path('output_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Process each frame in Camera-Loc01 directory
    frames_dir = Path('Camera-Loc01')
    for frame_path in sorted(frames_dir.glob('*.png')):
        # Read image for visualization
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Failed to read image: {frame_path}")
            continue
            
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
        
        # Get frame number and time from filename
        frame_info = frame_path.stem.split('_')
        frame_num = frame_info[1]
        time_str = '_'.join(frame_info[3:])
        
        # Draw room boundaries with proper scaling
        image = draw_rooms(image, room_data, display_scale, json_scale)
        
        # Run YOLO detection
        results = model(frame_path, classes=[0])  # class 0 is person
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Scale the coordinates for display
                if display_scale < 1:
                    x1, y1, x2, y2 = [int(x * display_scale) for x in [x1, y1, x2, y2]]
                
                # Calculate center point of detection
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Draw bounding box and center point
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Check which room the person is in (using original coordinates)
                original_center_x = int(center_x / display_scale)
                original_center_y = int(center_y / display_scale)
                room_id = None
                for quad in room_data['quads']:
                    if point_in_quad((original_center_x, original_center_y), quad['points'], json_scale):
                        room_id = quad['id']
                        # Draw room label
                        cv2.putText(image, f"Room {room_id}", (center_x, center_y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        break
                
                # Add to results
                results_data.append({
                    'frame_number': frame_num,
                    'timestamp': time_str,
                    'person_id': len([r for r in results_data if r['frame_number'] == frame_num]),
                    'room_id': room_id,
                    'confidence': conf,
                    'x': original_center_x,
                    'y': original_center_y
                })
        
        # Save visualization
        output_path = output_dir / f"frame_{frame_num}.png"
        cv2.imwrite(str(output_path), image)
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results_data)
    df.to_excel('human_tracking_results.xlsx', index=False)
    print(f"Results saved to human_tracking_results.xlsx")
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    process_frames() 