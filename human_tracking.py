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
import random
import yaml
import glob

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
    
    def enforce_max_people(self):
        """Enforce max_people limit by removing tracks with highest IDs"""
        all_tracks = list(self.tracks.keys()) + list(self.inactive_tracks.keys())
        if not all_tracks:
            return
            
        # Sort tracks by ID
        all_tracks.sort(reverse=True)  # Highest IDs first
        
        # Remove tracks until we're under the limit
        while len(all_tracks) >= self.max_people:
            track_id = all_tracks.pop(0)  # Remove highest ID
            if track_id in self.tracks:
                del self.tracks[track_id]
            if track_id in self.inactive_tracks:
                del self.inactive_tracks[track_id]
            if track_id in self.last_seen_frame:
                del self.last_seen_frame[track_id]
            if hasattr(self, 'person_features') and track_id in self.person_features:
                del self.person_features[track_id]
    
    def get_next_id(self, current_feature=None):
        """Get the next available ID, ensuring we don't exceed max_people.
        If current_feature is provided, use it to find the most similar track to replace."""
        # First enforce max_people limit
        self.enforce_max_people()
        
        # Try to find an unused ID between 1 and max_people
        for id in range(1, self.max_people + 1):
            if id not in self.tracks and id not in self.inactive_tracks:
                return id
        
        # If all IDs are in use and we have a feature vector, find the most similar track
        if current_feature is not None:
            best_match_id = None
            best_match_score = -1
            
            # Check both active and inactive tracks
            for track_id in list(self.tracks.keys()) + list(self.inactive_tracks.keys()):
                if track_id > self.max_people:
                    continue
                
                if track_id in self.person_features:
                    similarity = self.compute_feature_similarity(current_feature, self.person_features[track_id])
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Remove the matched track and its data
                if best_match_id in self.tracks:
                    del self.tracks[best_match_id]
                if best_match_id in self.inactive_tracks:
                    del self.inactive_tracks[best_match_id]
                if best_match_id in self.last_seen_frame:
                    del self.last_seen_frame[best_match_id]
                if best_match_id in self.person_features:
                    del self.person_features[best_match_id]
                return best_match_id
        
        # If no feature vector or no match found, find the oldest track to replace
        oldest_id = None
        oldest_frame = float('inf')
        
        # Check both active and inactive tracks
        for track_id in list(self.tracks.keys()) + list(self.inactive_tracks.keys()):
            if track_id > self.max_people:
                # Remove any IDs that somehow exceeded max_people
                if track_id in self.tracks:
                    del self.tracks[track_id]
                if track_id in self.inactive_tracks:
                    del self.inactive_tracks[track_id]
                if track_id in self.last_seen_frame:
                    del self.last_seen_frame[track_id]
                if hasattr(self, 'person_features') and track_id in self.person_features:
                    del self.person_features[track_id]
                continue
                
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
            if hasattr(self, 'person_features') and oldest_id in self.person_features:
                del self.person_features[oldest_id]
            return oldest_id
        
        # If all else fails, return 1
        return 1
    
    def update(self, frame_number, detections, frame=None):
        """Update tracker with new detections"""
        self.current_frame = frame_number
        updated_detections = []
        
        # First enforce max_people limit
        self.enforce_max_people()
        
        # Process each detection
        for det in detections:
            current_pos = (det['x'], det['y'])
            
            # Try to find match in active tracks first
            best_match_id = None
            best_match_dist = float('inf')
            
            for track_id, track_info in self.tracks.items():
                if track_id > self.max_people:
                    continue  # Skip IDs that exceed max_people
                    
                last_pos = track_info['last_position']
                dist = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
                
                if dist < self.location_threshold and dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = track_id
            
            # If no match in active tracks, try inactive tracks
            if best_match_id is None:
                for track_id, track_info in self.inactive_tracks.items():
                    if track_id > self.max_people:
                        continue  # Skip IDs that exceed max_people
                        
                    last_pos = track_info['last_position']
                    dist = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
                    
                    if dist < self.location_threshold and dist < best_match_dist:
                        best_match_dist = dist
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                if best_match_id in self.tracks:
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
                
                self.last_seen_frame[best_match_id] = frame_number
                det['person_id'] = best_match_id
            else:
                # Create new track
                new_id = self.get_next_id()
                self.tracks[new_id] = {
                    'last_seen_frame': frame_number,
                    'last_position': current_pos,
                    'active': True,
                    'original_id': det['person_id']
                }
                self.last_seen_frame[new_id] = frame_number
                det['person_id'] = new_id
            
            updated_detections.append(det)
        
        # Move old tracks to inactive
        for track_id in list(self.tracks.keys()):
            if track_id > self.max_people:
                del self.tracks[track_id]  # Remove any IDs that exceed max_people
                continue
                
            track_info = self.tracks[track_id]
            if frame_number - track_info['last_seen_frame'] > 1:  # More than 1 frame missing
                self.inactive_tracks[track_id] = track_info
                self.inactive_tracks[track_id]['active'] = False
                del self.tracks[track_id]
        
        # Clean up old inactive tracks
        for track_id in list(self.inactive_tracks.keys()):
            if track_id > self.max_people:
                del self.inactive_tracks[track_id]  # Remove any IDs that exceed max_people
                continue
                
            track_info = self.inactive_tracks[track_id]
            if frame_number - track_info['last_seen_frame'] > self.max_frames_missing:
                del self.inactive_tracks[track_id]
                if track_id in self.last_seen_frame:
                    del self.last_seen_frame[track_id]
        
        # Final enforcement of max_people
        self.enforce_max_people()
        
        return updated_detections

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
        
        # Minimum similarity threshold to consider changing bounding box
        self.min_similarity_for_update = 0.85  # Higher threshold for updating bounding boxes
    
    def compute_feature_similarity(self, feature1, feature2):
        """Compute cosine similarity between two feature vectors"""
        return np.dot(feature1, feature2)
    
    def get_next_id(self, current_feature=None):
        """Get the next available ID, ensuring we don't exceed max_people.
        If current_feature is provided, use it to find the most similar track to replace."""
        # First try to find an unused ID between 1 and max_people
        for id in range(1, self.max_people + 1):
            if id not in self.tracks and id not in self.inactive_tracks:
                return id
        
        # If all IDs are in use and we have a feature vector, find the most similar track
        if current_feature is not None:
            best_match_id = None
            best_match_score = -1
            
            # Check both active and inactive tracks
            for track_id in list(self.tracks.keys()) + list(self.inactive_tracks.keys()):
                if track_id > self.max_people:
                    continue
                
                if track_id in self.person_features:
                    similarity = self.compute_feature_similarity(current_feature, self.person_features[track_id])
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Remove the matched track and its data
                if best_match_id in self.tracks:
                    del self.tracks[best_match_id]
                if best_match_id in self.inactive_tracks:
                    del self.inactive_tracks[best_match_id]
                if best_match_id in self.last_seen_frame:
                    del self.last_seen_frame[best_match_id]
                if best_match_id in self.person_features:
                    del self.person_features[best_match_id]
                return best_match_id
        
        # If no feature vector or no match found, find the oldest track to replace
        oldest_id = None
        oldest_frame = float('inf')
        
        # Check both active and inactive tracks
        for track_id in list(self.tracks.keys()) + list(self.inactive_tracks.keys()):
            if track_id > self.max_people:
                # Remove any IDs that somehow exceeded max_people
                if track_id in self.tracks:
                    del self.tracks[track_id]
                if track_id in self.inactive_tracks:
                    del self.inactive_tracks[track_id]
                if track_id in self.last_seen_frame:
                    del self.last_seen_frame[track_id]
                if hasattr(self, 'person_features') and track_id in self.person_features:
                    del self.person_features[track_id]
                continue
                
            last_frame = self.last_seen_frame.get(track_id, 0)
            if last_frame < oldest_frame:
                oldest_frame = last_frame
                oldest_id = track_id
        
        if oldest_id is not None:
            # Remove the oldest ID track and its data
            if oldest_id in self.tracks:
                del self.tracks[oldest_id]
            if oldest_id in self.inactive_tracks:
                del self.inactive_tracks[oldest_id]
            if oldest_id in self.last_seen_frame:
                del self.last_seen_frame[oldest_id]
            if oldest_id in self.person_features:
                del self.person_features[oldest_id]
            return oldest_id
        
        # If all else fails, return 1
        return 1
    
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
                    
                new_id = self.get_next_id(current_features[current_pos])
                
                self.tracks[new_id] = {
                    'last_seen_frame': frame_number,
                    'last_position': current_pos,
                    'active': True,
                    'original_id': det['person_id']
                }
                
                # Store features
                self.person_features[new_id] = current_features[current_pos]
                self.last_seen_frame[new_id] = frame_number
                
                # Update detection with new ID and feature vector
                det['person_id'] = new_id
                det['feature_vector'] = current_features[current_pos]
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
                # We found a match - update the track only if similarity is high enough
                if best_match_score >= self.min_similarity_for_update:
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
                    
                    # Update detection with matched ID and feature vector
                    det['person_id'] = best_match_id
                    det['feature_vector'] = current_feature
                    updated_detections.append(det)
                    
                    matched_positions.add(current_pos)
                    matched_ids.add(best_match_id)
                else:
                    # Similarity is too low - try to get a new ID
                    new_id = self.get_next_id(current_feature)
                    
                    self.tracks[new_id] = {
                        'last_seen_frame': frame_number,
                        'last_position': current_pos,
                        'active': True,
                        'original_id': det['person_id']
                    }
                    
                    # Store features
                    self.person_features[new_id] = current_feature
                    self.last_seen_frame[new_id] = frame_number
                    
                    # Update detection
                    det['person_id'] = new_id
                    det['feature_vector'] = current_feature
                    updated_detections.append(det)
                    
                    matched_positions.add(current_pos)
                    matched_ids.add(new_id)
            else:
                # No match found - try to get a new ID
                new_id = self.get_next_id(current_feature)
                
                self.tracks[new_id] = {
                    'last_seen_frame': frame_number,
                    'last_position': current_pos,
                    'active': True,
                    'original_id': det['person_id']
                }
                
                # Store features
                self.person_features[new_id] = current_feature
                self.last_seen_frame[new_id] = frame_number
                
                # Update detection
                det['person_id'] = new_id
                det['feature_vector'] = current_feature
                updated_detections.append(det)
                
                matched_positions.add(current_pos)
                matched_ids.add(new_id)
        
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
                if track_id in self.person_features:
                    del self.person_features[track_id]
        
        return updated_detections

def save_detections_to_yaml(detections, output_yaml, image_dir, confidence_threshold=0.6):
    """Save high confidence detections to YAML file for YOLO training.
    
    Args:
        detections (list): List of all detections
        output_yaml (str): Path to save YAML file
        image_dir (str): Path to directory containing images
        confidence_threshold (float): Confidence threshold for pseudo-labels (higher for training)
    """
    import yaml
    import glob
    
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

def process_frames(input_folder, output_excel, output_images, json_file, output_json=None, output_yaml=None):
    print("\nInitializing process_frames with parameters:")
    print(f"- input_folder: {input_folder}")
    print(f"- output_excel: {output_excel}")
    print(f"- output_images: {output_images}")
    print(f"- json_file: {json_file}")
    print(f"- output_json: {output_json}")
    print(f"- output_yaml: {output_yaml}")
    
    # Load YOLO model
    model = YOLO('yolo11n.pt')
    
    # Configure YOLO model for better detection
    model.conf = 0.01  # Much lower confidence threshold for detections
    model.iou = 0.1    # Lower IoU threshold for NMS
    model.agnostic_nms = True  # Use class-agnostic NMS
    model.max_det = 300  # Increase maximum number of detections
    
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
    
    # Initialize our DINO-enhanced person tracker with more lenient settings
    person_tracker = DINOPersonTracker(
        max_frames_missing=150,           # Increase frames before considering missing
        location_threshold=400,           # Increase distance threshold
        feature_similarity_threshold=0.3, # Lower feature similarity threshold
        max_people=20                     # Increase maximum number of people to track
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
    total_frames = len(all_frames)
    
    if total_frames == 0:
        print(f"Error: No PNG files found in '{input_folder}'")
        return
        
    print(f"Processing {total_frames} frames...")
    
    # Optional: ID color mapping to visualize consistently
    id_colors = {}
    
    # Store all detections for JSON/YAML output
    all_detections = []
    
    # Store frames for feature extraction
    frame_images = {}
    
    for frame_path in all_frames:
        # Read image for visualization
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Failed to read image: {frame_path}")
            continue
        
        # Store frame for feature extraction
        frame_num = frame_path.stem.split('_')[1]
        frame_images[frame_num] = image.copy()
        
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
            if boxes.id is not None:
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
        updated_detections = person_tracker.update(int(frame_num), frame_detections, frame_images[frame_num])
        
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
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%). Active tracks: {len(person_tracker.tracks)}, Inactive: {len(person_tracker.inactive_tracks)}")
        
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
    parser.add_argument('--output_json', type=str,
                      help='Output JSON file path for high-confidence detections')
    parser.add_argument('--output_yaml', type=str,
                      help='Output YAML file path for YOLO training')
    
    args = parser.parse_args()
    
    process_frames(args.input_folder, args.output_excel, args.output_images, 
                  args.json_file, args.output_json, args.output_yaml)