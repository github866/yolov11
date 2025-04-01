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

class PersonTracker:
    """Custom person tracker with ID memory for reappearing people"""
    
    def __init__(self, max_frames_missing=30, location_threshold=100):
        # Active tracks: dict of track_id -> {last_seen_frame, last_position, active}
        self.tracks = {}
        # Inactive (potentially reusable) tracks
        self.inactive_tracks = {}
        # Current frame number
        self.current_frame = 0
        # Maximum number of frames a person can be missing before considering a new ID
        self.max_frames_missing = max_frames_missing
        # Distance threshold for considering a person the same at a similar location (in pixels)
        self.location_threshold = location_threshold
        # Next ID to assign (only used if YOLO tracker fails)
        self.next_temp_id = 0
        # Track ID mapping to maintain consistent IDs (original_id -> our_id)
        self.id_mapping = {}
        # Counter for new permanent IDs we assign
        self.next_perm_id = 1
    
    def update(self, frame_number, detections):
        """
        Update tracker with new detections
        
        Args:
            frame_number: Current frame number
            detections: List of dictionaries with 'track_id', 'x', 'y' (center coordinates)
        
        Returns:
            Updated list of detections with potentially modified track_ids
        """
        self.current_frame = frame_number
        updated_detections = []
        
        # Track IDs seen in this frame
        seen_ids = set()
        
        # First, try to match all current detections with inactive tracks based on location
        for det in detections:
            original_id = det['person_id']
            current_pos = (det['x'], det['y'])
            
            # Always try to match with inactive tracks first, regardless of ID type
            best_match_id = None
            best_match_dist = float('inf')
            
            for inactive_id, inactive_info in self.inactive_tracks.items():
                # Calculate distance between current detection and inactive track
                last_pos = inactive_info['last_position']
                
                dist = np.sqrt((last_pos[0] - current_pos[0])**2 + (last_pos[1] - current_pos[1])**2)
                
                # Check if this is a good match (close enough and better than previous matches)
                if dist < self.location_threshold and dist < best_match_dist:
                    frames_gone = self.current_frame - inactive_info['last_seen_frame']
                    if frames_gone <= self.max_frames_missing:
                        best_match_id = inactive_id
                        best_match_dist = dist
            
            # Check if we found a good spatial match with an inactive track
            if best_match_id is not None:
                # Reuse the inactive ID
                assigned_id = best_match_id
                # Update ID mapping
                if isinstance(original_id, (int, np.integer)):
                    self.id_mapping[original_id] = assigned_id
                
                # Reactivate track
                self.tracks[assigned_id] = {
                    'last_seen_frame': self.current_frame,
                    'last_position': current_pos,
                    'active': True,
                    'original_id': original_id
                }
                # Remove from inactive tracks
                del self.inactive_tracks[best_match_id]
            else:
                # No spatial match found, check if we've seen this ID before
                if isinstance(original_id, (int, np.integer)) and original_id in self.id_mapping:
                    # We've seen this ID before, use our consistent mapping
                    assigned_id = self.id_mapping[original_id]
                    
                    # Check if this ID is currently inactive
                    if assigned_id in self.inactive_tracks:
                        # Reactivate from inactive
                        self.tracks[assigned_id] = {
                            'last_seen_frame': self.current_frame,
                            'last_position': current_pos,
                            'active': True,
                            'original_id': original_id
                        }
                        del self.inactive_tracks[assigned_id]
                    else:
                        # Just update the existing track
                        self.tracks[assigned_id] = {
                            'last_seen_frame': self.current_frame,
                            'last_position': current_pos,
                            'active': True,
                            'original_id': original_id
                        }
                elif isinstance(original_id, str) and original_id.startswith('temp_'):
                    # This is a temporary ID - assign a new permanent ID
                    assigned_id = self.next_perm_id
                    self.next_perm_id += 1
                    
                    # Add new track
                    self.tracks[assigned_id] = {
                        'last_seen_frame': self.current_frame,
                        'last_position': current_pos,
                        'active': True,
                        'original_id': original_id
                    }
                else:
                    # This is a new YOLO ID we haven't seen before
                    # Create a new permanent ID for it
                    assigned_id = self.next_perm_id
                    self.next_perm_id += 1
                    
                    # Update mapping
                    if isinstance(original_id, (int, np.integer)):
                        self.id_mapping[original_id] = assigned_id
                    
                    # Add new track
                    self.tracks[assigned_id] = {
                        'last_seen_frame': self.current_frame,
                        'last_position': current_pos,
                        'active': True,
                        'original_id': original_id
                    }
            
            # Add the assigned ID to the detection
            det['person_id'] = assigned_id
            seen_ids.add(assigned_id)
            updated_detections.append(det)
        
        # Move tracks not seen in this frame to inactive
        for track_id in list(self.tracks.keys()):
            if track_id not in seen_ids:
                # Move to inactive
                self.inactive_tracks[track_id] = self.tracks[track_id]
                self.inactive_tracks[track_id]['active'] = False
                del self.tracks[track_id]
        
        # Clean up old inactive tracks (those missing for too long)
        for track_id in list(self.inactive_tracks.keys()):
            track_info = self.inactive_tracks[track_id]
            if self.current_frame - track_info['last_seen_frame'] > self.max_frames_missing:
                # Also clean up ID mapping if necessary
                for orig_id, mapped_id in list(self.id_mapping.items()):
                    if mapped_id == track_id:
                        del self.id_mapping[orig_id]
                del self.inactive_tracks[track_id]
        
        return updated_detections
    
    def get_next_id(self):
        """Get the next available ID"""
        # This is only used as a fallback
        return 10000 + self.next_temp_id

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
    
    # Dictionary to store track histories (for visualization)
    track_history = {}
    
    # Initialize our custom person tracker with more aggressive settings
    # Increased max_frames_missing to 45 frames (1.5 seconds at 30fps)
    # Increased location_threshold to 150 pixels for more flexible matching
    person_tracker = PersonTracker(max_frames_missing=45, location_threshold=150)
    
    # Frame counters for tracking
    frame_count = 0
    
    # Process each frame in Camera-Loc01 directory
    frames_dir = Path('Camera-Loc01')
    all_frames = sorted(frames_dir.glob('*.png'))
    total_frames = len(all_frames)
    
    print(f"Processing {total_frames} frames...")
    
    for frame_path in all_frames:
        # Read image for visualization
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Failed to read image: {frame_path}")
            continue
        
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
        
        # Get frame number and time from filename
        frame_info = frame_path.stem.split('_')
        frame_num = frame_info[1]
        time_str = '_'.join(frame_info[3:])
        
        # Draw room boundaries with proper scaling
        image = draw_rooms(image, room_data, display_scale, json_scale)
        
        # Run YOLO detection with tracking enabled
        results = model.track(frame_path, classes=[0], persist=True)  # Enable tracking for people (class 0)
        
        # Detections for current frame (to pass to our tracker)
        frame_detections = []
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            
            # Check if we have tracking IDs available
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates and tracking ID
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    track_id = track_ids[i]
                    
                    # Scale the coordinates for display
                    if display_scale < 1:
                        x1, y1, x2, y2 = [int(x * display_scale) for x in [x1, y1, x2, y2]]
                    
                    # Calculate center point of detection
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Original coordinates (unscaled)
                    original_center_x = int(center_x / display_scale)
                    original_center_y = int(center_y / display_scale)
                    
                    # Determine room
                    room_id = None
                    for quad in room_data['quads']:
                        if point_in_quad((original_center_x, original_center_y), quad['points'], json_scale):
                            room_id = quad['id']
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
                        'x1': int(x1), 
                        'y1': int(y1), 
                        'x2': int(x2), 
                        'y2': int(y2)
                    })
            else:
                # Fallback to regular detection if tracking fails
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Scale the coordinates for display
                    if display_scale < 1:
                        x1, y1, x2, y2 = [int(x * display_scale) for x in [x1, y1, x2, y2]]
                    
                    # Calculate center point of detection
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Original coordinates (unscaled)
                    original_center_x = int(center_x / display_scale)
                    original_center_y = int(center_y / display_scale)
                    
                    # Determine room
                    room_id = None
                    for quad in room_data['quads']:
                        if point_in_quad((original_center_x, original_center_y), quad['points'], json_scale):
                            room_id = quad['id']
                            break
                    
                    # Add to frame detections with a temporary ID
                    temp_id = f"temp_{i}"
                    frame_detections.append({
                        'frame_number': frame_num,
                        'timestamp': time_str,
                        'person_id': temp_id,
                        'room_id': room_id,
                        'confidence': conf,
                        'x': original_center_x,
                        'y': original_center_y,
                        'display_x': center_x,
                        'display_y': center_y,
                        'x1': int(x1), 
                        'y1': int(y1), 
                        'x2': int(x2), 
                        'y2': int(y2)
                    })
        
        # Update our custom tracker with the current frame's detections
        updated_detections = person_tracker.update(int(frame_num), frame_detections)
        
        # Add updated detections to results and visualize
        for det in updated_detections:
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
            
            # Add to track history
            track_history[track_id].append((det['display_x'], det['display_y']))
            
            # Limit history length
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)
            
            # Draw bounding box, center point, and person ID
            cv2.rectangle(image, (det['x1'], det['y1']), (det['x2'], det['y2']), (0, 0, 255), 2)
            cv2.circle(image, (det['display_x'], det['display_y']), 5, (0, 0, 255), -1)
            
            # Add person ID to the top of the bounding box
            cv2.putText(image, f"ID: {track_id}", (det['x1'], det['y1'] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw tracking lines
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [points], False, (0, 255, 255), 2)
            
            # Draw room label if known
            if det['room_id'] is not None:
                cv2.putText(image, f"Room {det['room_id']}", (det['display_x'], det['display_y'] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save visualization
        output_path = output_dir / f"frame_{frame_num}.png"
        cv2.imwrite(str(output_path), image)
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%). Active tracks: {len(person_tracker.tracks)}, Inactive: {len(person_tracker.inactive_tracks)}")
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results_data)
    
    # Count unique IDs
    unique_ids = df['person_id'].nunique()
    print(f"\nTracking Statistics:")
    print(f"Total unique person IDs: {unique_ids}")
    print(f"Total original YOLO ID mappings: {len(person_tracker.id_mapping)}")
    print(f"Maximum assigned ID: {person_tracker.next_perm_id - 1}")
    
    # Room occupancy statistics
    if 'room_id' in df.columns:
        room_occupancy = df.groupby(['frame_number', 'room_id']).size().reset_index(name='count')
        max_occupancy = room_occupancy.groupby('room_id')['count'].max().reset_index()
        print("\nMaximum room occupancy:")
        for _, row in max_occupancy.iterrows():
            if pd.notna(row['room_id']):
                print(f"Room {row['room_id']}: {row['count']} people")
    
    df.to_excel('human_tracking_results.xlsx', index=False)
    print(f"\nResults saved to human_tracking_results.xlsx")
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    process_frames()