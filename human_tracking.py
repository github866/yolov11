import os
import json
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import colorsys
import argparse
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
import timm

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
    for i, quad in enumerate(room_data['quads']):
        # The points in room_data are already scaled by json_scale
        # We need to scale them up to match the image size
        scaled_points = [[int(x / json_scale * display_scale), int(y / json_scale * display_scale)] for x, y in quad['points']]
        points = np.array(scaled_points, dtype=np.int32)
        cv2.polylines(image, [points], True, (0, 255, 0), 2)
        # Add room number (using shape number)
        center = np.mean(points, axis=0).astype(int)
        cv2.putText(image, f"Room {i+1}", (center[0], center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

class PersonTracker:
    """Custom person tracker with ID memory for reappearing people"""
    
    def __init__(self, max_frames_missing=30, location_threshold=100, max_people=10):
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
        # Counter for new permanent IDs we assign
        self.next_perm_id = 1
        # Maximum number of people to track
        self.max_people = max_people
        # Track last seen frame for each person
        self.last_seen_frame = {}
    
    def get_next_id(self):
        """Get the next available ID, ensuring we don't exceed max_people"""
        # First try to find an unused ID between 1 and max_people
        for id in range(1, self.max_people + 1):
            if id not in self.tracks and id not in self.inactive_tracks:
                return id
        
        # If all IDs are in use, find the oldest track to replace
        oldest_id = None
        oldest_frame = float('inf')
        
        # Check both active and inactive tracks
        for track_id in list(self.tracks.keys()) + list(self.inactive_tracks.keys()):
            last_frame = self.last_seen_frame.get(track_id, 0)
            if last_frame < oldest_frame:
                oldest_frame = last_frame
                oldest_id = track_id
        
        # If we found an old track to replace
        if oldest_id is not None:
            # Remove old track
            if oldest_id in self.tracks:
                del self.tracks[oldest_id]
            if oldest_id in self.inactive_tracks:
                del self.inactive_tracks[oldest_id]
            if oldest_id in self.last_seen_frame:
                del self.last_seen_frame[oldest_id]
            return oldest_id
        
        # If all else fails, return 1
        return 1

class DINOFeatureExtractor:
    def __init__(self, model_name='dino_vits8', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_features(self, image_crop):
        """Extract DINO features from an image crop"""
        # Convert crop to RGB if needed
        if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        img_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.model(img_tensor)
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy()[0]

class DINOPersonTracker(PersonTracker):
    def __init__(self, max_frames_missing=30, location_threshold=100, feature_similarity_threshold=0.7, max_people=10):
        super().__init__(max_frames_missing, location_threshold, max_people)
        
        # Initialize DINO feature extractor
        self.feature_extractor = DINOFeatureExtractor()
        
        # Store person features
        self.person_features = {}  # person_id -> feature vector
        
        # Feature similarity threshold
        self.feature_similarity_threshold = feature_similarity_threshold
    
    def compute_feature_similarity(self, feature1, feature2):
        """Compute cosine similarity between two feature vectors"""
        return np.dot(feature1, feature2)
    
    def update(self, frame_number, detections, frame=None):
        """
        Update tracker with new detections using both spatial and feature information
        
        Args:
            frame_number: Current frame number
            detections: List of dictionaries with detection info
            frame: The full frame image for feature extraction
        """
        if frame is None:
            return super().update(frame_number, detections)
        
        self.current_frame = frame_number
        updated_detections = []
        
        # Extract features for all detections in current frame
        current_features = {}
        for det in detections:
            if 'x1' in det and 'y1' in det and 'x2' in det and 'y2' in det:
                # Extract crop from frame
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                crop = frame[y1:y2, x1:x2]
                
                # Skip if crop is invalid
                if crop.size == 0:
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_features(crop)
                current_features[(det['x'], det['y'])] = features
        
        # Initialize new person IDs in frame 0
        if frame_number == 0:
            # Reset all tracking data to start fresh
            self.tracks = {}
            self.inactive_tracks = {}
            self.person_features = {}
            self.next_perm_id = 1
            self.last_seen_frame = {}
            
            # Assign new IDs to all detections in the first frame (up to max_people)
            for i, det in enumerate(detections):
                if i >= self.max_people:
                    break  # Strictly enforce max_people limit
                    
                current_pos = (det['x'], det['y'])
                if current_pos not in current_features:
                    continue
                    
                new_id = self.get_next_id()  # Use get_next_id to ensure proper ID assignment
                
                self.tracks[new_id] = {
                    'last_seen_frame': frame_number,
                    'last_position': current_pos,
                    'active': True,
                    'original_id': det['person_id']
                }
                
                # Store features
                self.person_features[new_id] = current_features[current_pos]
                self.last_seen_frame[new_id] = frame_number
                
                # Update detection with new ID
                det['person_id'] = new_id
                updated_detections.append(det)
                
            return updated_detections
        
        # For all other frames, try to match with existing tracks
        matched_positions = set()
        matched_ids = set()
        
        # Process all detections to find matches
        for det in detections:
            current_pos = (det['x'], det['y'])
            
            # Skip if we couldn't extract features for this detection
            if current_pos not in current_features:
                continue
            
            current_feature = current_features[current_pos]
            
            # Try to find match in active and inactive tracks
            best_match_id = None
            best_match_score = -1
            
            # Check active tracks first
            for track_id, track_info in self.tracks.items():
                # Skip if already matched
                if track_id in matched_ids:
                    continue
                    
                # Get feature similarity
                if track_id in self.person_features:
                    similarity = self.compute_feature_similarity(current_feature, self.person_features[track_id])
                    
                    if similarity > best_match_score and similarity > self.feature_similarity_threshold:
                        best_match_score = similarity
                        best_match_id = track_id
            
            # Then check inactive tracks
            for track_id, track_info in self.inactive_tracks.items():
                # Skip if already matched
                if track_id in matched_ids:
                    continue
                    
                if track_id in self.person_features:
                    similarity = self.compute_feature_similarity(current_feature, self.person_features[track_id])
                    
                    if similarity > best_match_score and similarity > self.feature_similarity_threshold:
                        best_match_score = similarity
                        best_match_id = track_id
            
            if best_match_id is not None:
                # We found a match - update the track
                if best_match_id in self.tracks:
                    # Update active track
                    self.tracks[best_match_id]['last_seen_frame'] = frame_number
                    self.tracks[best_match_id]['last_position'] = current_pos
                else:
                    # Reactivate inactive track
                    track_info = self.inactive_tracks[best_match_id]
                    self.tracks[best_match_id] = {
                        'last_seen_frame': frame_number,
                        'last_position': current_pos,
                        'active': True,
                        'original_id': det['person_id']
                    }
                    del self.inactive_tracks[best_match_id]
                
                # Always update the feature vector with the latest features
                self.person_features[best_match_id] = current_feature
                self.last_seen_frame[best_match_id] = frame_number
                
                # Update detection with matched ID
                det['person_id'] = best_match_id
                updated_detections.append(det)
                
                matched_positions.add(current_pos)
                matched_ids.add(best_match_id)
                continue
            
            # No match found - create new track if possible
            total_tracks = len(self.tracks) + len(self.inactive_tracks)
            
            if total_tracks < self.max_people:
                # We can create a new ID
                new_id = self.get_next_id()  # Use get_next_id to ensure proper ID assignment
                
                self.tracks[new_id] = {
                    'last_seen_frame': frame_number,
                    'last_position': current_pos,
                    'active': True,
                    'original_id': det['person_id']
                }
                
                # Store features
                self.person_features[new_id] = current_feature
                self.last_seen_frame[new_id] = frame_number
                
                # Update detection with new ID
                det['person_id'] = new_id
                updated_detections.append(det)
                
                matched_positions.add(current_pos)
                matched_ids.add(new_id)
            else:
                # No space for new IDs - find closest match to replace
                closest_id = None
                closest_similarity = -1
                
                for track_id in list(self.tracks.keys()) + list(self.inactive_tracks.keys()):
                    if track_id in matched_ids:
                        continue
                        
                    if track_id in self.person_features:
                        similarity = self.compute_feature_similarity(current_feature, self.person_features[track_id])
                        if similarity > closest_similarity:
                            closest_similarity = similarity
                            closest_id = track_id
                
                if closest_id is not None:
                    # Remove old track
                    if closest_id in self.tracks:
                        del self.tracks[closest_id]
                    if closest_id in self.inactive_tracks:
                        del self.inactive_tracks[closest_id]
                    
                    # Replace with new track using the same ID
                    self.tracks[closest_id] = {
                        'last_seen_frame': frame_number,
                        'last_position': current_pos,
                        'active': True,
                        'original_id': det['person_id']
                    }
                    
                    # Update features
                    self.person_features[closest_id] = current_feature
                    self.last_seen_frame[closest_id] = frame_number
                    
                    # Update detection
                    det['person_id'] = closest_id
                    updated_detections.append(det)
                    
                    matched_positions.add(current_pos)
                    matched_ids.add(closest_id)
        
        # Move unmatched tracks to inactive
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_ids:
                self.inactive_tracks[track_id] = self.tracks[track_id]
                self.inactive_tracks[track_id]['active'] = False
                del self.tracks[track_id]
        
        # Clean up old inactive tracks
        for track_id in list(self.inactive_tracks.keys()):
            track_info = self.inactive_tracks[track_id]
            if frame_number - track_info['last_seen_frame'] > self.max_frames_missing:
                del self.inactive_tracks[track_id]
        
        return updated_detections

def process_frames(input_folder, output_excel, output_images, json_file):
    # Load YOLO model
    model = YOLO('yolo11n.pt')
    
    # Load room definitions
    with open(json_file, 'r') as f:
        room_data = json.load(f)
    
    # Get image scale from JSON
    json_scale = room_data['image_scale']
    
    # Create DataFrame to store results
    results_data = []
    
    # Create output directory for visualizations
    output_dir = Path(output_images)
    output_dir.mkdir(exist_ok=True)
    
    # Dictionary to store track histories (for visualization)
    track_history = {}
    
    # Initialize our DINO-enhanced person tracker
    person_tracker = DINOPersonTracker(max_frames_missing=90, location_threshold=250,
                                     feature_similarity_threshold=0.7, max_people=10)
    
    # Frame counters for tracking
    frame_count = 0
    
    # Process each frame in input directory
    frames_dir = Path(input_folder)
    if not frames_dir.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
        
    all_frames = sorted(frames_dir.glob('*.png'))
    total_frames = len(all_frames)
    
    if total_frames == 0:
        print(f"Error: No PNG files found in '{input_folder}'")
        return
        
    print(f"Processing {total_frames} frames...")
    
    # Optional: ID color mapping to visualize consistently
    id_colors = {}
    
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
                    
                    # Skip low confidence detections to avoid ID switching
                    if conf < 0.25:  # Slightly lower confidence threshold
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
                    
                    # Skip low confidence detections
                    if conf < 0.25:  # Slightly lower confidence threshold
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
                    
                    # Determine room
                    room_id = None
                    for quad in room_data['quads']:
                        if point_in_quad((original_center_x, original_center_y), quad['points'], json_scale):
                            # Extract room number from shape name (e.g., "Shape 9" -> 9)
                            shape_name = quad['name']
                            room_number = int(shape_name.split()[-1])
                            room_id = room_number
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
        updated_detections = person_tracker.update(int(frame_num), frame_detections, image)
        
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
                    # Generate a unique color based on ID (ensures same ID always has same color)
                    hue = (track_id * 0.618033988749895) % 1.0
                    r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.95)]
                    id_colors[track_id] = (b, g, r)  # OpenCV uses BGR
            
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
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%). Active tracks: {len(person_tracker.tracks)}, Inactive: {len(person_tracker.inactive_tracks)}")
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results_data)
    
    if df.empty:
        print("Warning: No detections were made. No data to save.")
        return
        
    # Count unique IDs
    unique_ids = df['person_id'].nunique()
    print(f"\nTracking Statistics:")
    print(f"Total unique person IDs: {unique_ids}")
    print(f"Total original YOLO ID mappings: {len(person_tracker.person_features)}")
    print(f"Maximum assigned ID: {person_tracker.next_perm_id - 1}")
    
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
    
    args = parser.parse_args()
    
    process_frames(args.input_folder, args.output_excel, args.output_images, args.json_file)