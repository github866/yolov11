import numpy as np

from feature_extractor import DINOFeatureExtractor

class PersonTracker:
    def __init__(self, max_frames_missing=30, distance_threshold=100, max_people=10):
        self.tracks = {} # Active tracks: dict of track_id -> {last_seen_frame, last_position, active}
        self.inactive_tracks = {} # Inactive (potentially reusable) tracks
        self.current_frame = 0
        self.max_frames_missing = max_frames_missing
        self.distance_threshold = distance_threshold 
        self.next_perm_id = 1 # Counter for new permanent IDs we assign
        self.max_people = max_people
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
                
                if dist < self.distance_threshold and dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = track_id
            
            # If no match in active tracks, try inactive tracks
            if best_match_id is None:
                for track_id, track_info in self.inactive_tracks.items():
                    if track_id > self.max_people:
                        continue  # Skip IDs that exceed max_people
                        
                    last_pos = track_info['last_position']
                    dist = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
                    
                    if dist < self.distance_threshold and dist < best_match_dist:
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

class DINOPersonTracker(PersonTracker):
    def __init__(
            self,
            max_frames_missing=30, 
            distance_threshold=100, 
            feature_similarity_threshold=0.7, 
            max_people=12
        ):
        super().__init__(max_frames_missing, distance_threshold, max_people)
        
        self.feature_extractor = DINOFeatureExtractor()
        self.person_features = {}  # person_id -> feature vector 
        self.feature_similarity_threshold = feature_similarity_threshold
        self.min_similarity_for_update = 0.85  # Minimum similarity threshold to consider changing bounding box
    
    def compute_feature_similarity(self, feature1, feature2):
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