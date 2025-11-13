#!/usr/bin/env python3
"""
Human Detection and Room Localization with Tracking

This script processes a video to:
1. Detect humans using YOLO
2. Track them across frames using a simple distance-based tracker
3. Determine which room each person is in using a room segmentation map
4. Output results with visualization and data export

Optimized for low time complexity with efficient room lookup and tracking.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict

# Import utility for room color mapping
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from utils_loc.utility import color_to_room


class RoomSegmentationMap:
    """Efficient room segmentation map handler for O(1) room lookup."""
    
    def __init__(self, mask_path, room_names_map=None):
        """
        Initialize room segmentation map.
        
        Args:
            mask_path: Path to colored room segmentation image
            room_names_map: Optional dict mapping RGB tuples to room names.
                          If None, uses default color_to_room function.
        """
        self.mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if self.mask is None:
            raise ValueError(f"Could not load room mask from {mask_path}")
        
        self.height, self.width = self.mask.shape[:2]
        self.room_names_map = room_names_map
        
        # Pre-compute room lookup for efficiency (optional optimization)
        # For now, we'll do direct pixel lookup which is already O(1)
        print(f"Loaded room segmentation map: {self.width}x{self.height}")
    
    def get_room_at_point(self, x, y):
        """
        Get room name at a given point (x, y).
        O(1) lookup - direct pixel access.
        
        Args:
            x: X coordinate (must be within mask bounds)
            y: Y coordinate (must be within mask bounds)
            
        Returns:
            Room name string or "Unknown Room" if outside bounds
        """
        # Clamp coordinates to valid range
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        
        # Get BGR color at this point (OpenCV uses BGR)
        bgr_color = self.mask[y, x]
        rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
        
        # Map color to room name
        if self.room_names_map:
            room_name = self.room_names_map.get(rgb_color, "Unknown Room")
        else:
            room_name = color_to_room(rgb_color)
        
        return room_name
    
    def get_room_for_bbox(self, x1, y1, x2, y2):
        """
        Get room name for a bounding box by checking center point.
        More robust: could check multiple points or majority vote.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            Room name string
        """
        # Use center point of bounding box (bottom-center for better ground contact)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2 + (y2 - y1) * 0.3)  # Slightly below center
        
        return self.get_room_at_point(center_x, center_y)


# Using YOLO's built-in tracker - no custom tracker needed!
# YOLO's tracker is efficient, well-tested, and handles all the complexity
class HumanRoomTracker:
    """Main class for human detection, tracking, and room localization."""
    
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.25, 
                 max_people=10, room_mask_path=None, room_names_map=None,
                ):
        """
        Initialize the tracker.
        
        Args:
            model_path: Path to YOLO model checkpoint
            conf_threshold: Confidence threshold for detections
            max_people: Maximum number of people to track
            room_mask_path: Path to room segmentation map
            room_names_map: Optional custom room name mapping
        """
        # Initialize YOLO model
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.conf = conf_threshold
        self.model.iou = 0.45
        self.model.classes = [0]  # Only person class
        self.model.max_det = max_people
        
        # Configure YOLO's built-in tracker (BYTETrack)
        # This is efficient and handles all tracking automatically
        self.model.tracker = {
            'track_high_thresh': 0.5,      # High confidence threshold for tracks
            'track_low_thresh': 0.1,        # Low confidence threshold for tracks
            'new_track_thresh': 0.6,        # Threshold for new tracks
            'track_buffer': 30,              # Frames to keep lost tracks
            'match_thresh': 0.8,            # Matching threshold
            'fuse_score': True,             # Fuse detection score with tracking
            'min_box_area': 10,             # Minimum box area
            'max_age': 30,                  # Maximum age of a track
            'min_hits': 3                   # Minimum hits to confirm a track
        }
                
        # Initialize room segmentation map
        if room_mask_path:
            self.room_map = RoomSegmentationMap(room_mask_path, room_names_map)
        else:
            self.room_map = None
        
        # Storage for results
        self.results_data = []
        self.track_history = defaultdict(list)
        
    def process_video(self, video_path=None, camera_index=None, output_video_path=None, 
                     output_json_path=None, output_csv_path=None,
                     show_progress=True):
        """
        Process video or camera stream for human detection, tracking, and room localization.
        
        Args:
            video_path: Path to input video (None if using camera)
            camera_index: Camera index (0 for default camera, None if using video)
            output_video_path: Optional path to save output video with visualizations
            output_json_path: Optional path to save results as JSON
            output_csv_path: Optional path to save results as CSV
            show_progress: Whether to show progress bar (only for video files)
        """
        # Open video or camera
        if camera_index is not None:
            print(f"Opening camera {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            is_camera = True
        elif video_path:
            print(f"Opening video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            is_camera = False
        else:
            raise ValueError("Either video_path or camera_index must be provided")
        
        if not cap.isOpened():
            if is_camera:
                raise ValueError(f"Could not open camera {camera_index}")
            else:
                raise ValueError(f"Could not open video: {video_path}")
        
        # Get video/camera properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 fps for camera
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if is_camera:
            # For camera, set a reasonable FPS if not detected
            if fps == 0:
                fps = 30
            total_frames = None  # Camera has no fixed frame count
            print(f"Camera properties: {width}x{height} @ {fps} fps (real-time)")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")
        
        # Setup video writer if needed
        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        id_colors = {}
        
        # Create iterator based on input type
        if is_camera:
            # For camera, use infinite loop with manual break
            iterator = iter(int, 1)  # Infinite iterator
            print("Press 'q' to quit, 's' to save and quit")
        else:
            # For video, use frame count
            iterator = tqdm(range(total_frames)) if show_progress else range(total_frames)
        
        try:
            for _ in iterator:
                ret, frame = cap.read()
                if not ret:
                    if is_camera:
                        print("Camera disconnected or error reading frame")
                    break
                
                # Run YOLO detection with built-in tracking
                # YOLO's tracker automatically assigns track IDs
                results = self.model.track(
                    frame,
                    classes=[0],  # Person class only
                    persist=True,  # Maintain tracks across frames
                    conf=self.model.conf,
                    iou=0.45,
                    verbose=False
                )
                
                # Process detections with track IDs from YOLO
                updated_detections = []
                
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    
                    # Check if tracking IDs are available
                    if boxes.id is not None:
                        track_ids = boxes.id.int().cpu().tolist()
                        
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf[0])
                            track_id = int(track_ids[i])
                            
                            # Calculate center
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Determine room
                            room_name = "Unknown Room"
                            if self.room_map:
                                room_name = self.room_map.get_room_for_bbox(x1, y1, x2, y2)
                            
                            # Create detection with track ID
                            det = {
                                'frame_number': frame_count,
                                'person_id': track_id,
                                'room_name': room_name,
                                'confidence': conf,
                                'x': center_x,
                                'y': center_y,
                                'x1': int(x1),
                                'y1': int(y1),
                                'x2': int(x2),
                                'y2': int(y2)
                            }
                            updated_detections.append(det)
                
                # Count people in current frame
                num_people = len(updated_detections)
                
                # Draw people count in top-left corner
                count_text = f"People: {num_people}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5  # Large, visible text
                thickness = 3
                color = (0, 255, 0)  # Green color
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    count_text, font, font_scale, thickness
                )
                
                # Draw semi-transparent background for better visibility
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (10, 10),
                    (20 + text_width, 20 + text_height + baseline),
                    (0, 0, 0),
                    -1
                )
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Draw text
                cv2.putText(
                    frame,
                    count_text,
                    (15, 15 + text_height),
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )
                
                # Store results and visualize
                for det in updated_detections:
                    # Store data
                    self.results_data.append({
                        'frame': det['frame_number'],
                        'person_id': det['person_id'],
                        'room': det['room_name'],
                        'confidence': det['confidence'],
                        'x': det['x'],
                        'y': det['y'],
                        'x1': det['x1'],
                        'y1': det['y1'],
                        'x2': det['x2'],
                        'y2': det['y2']
                    })
                    
                    # Update track history for visualization
                    track_id = det['person_id']
                    self.track_history[track_id].append((det['x'], det['y']))
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)
                    
                    # Get or assign color for this track
                    if track_id not in id_colors:
                        hue = (track_id * 0.618033988749895) % 1.0
                        import colorsys
                        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.95)]
                        id_colors[track_id] = (b, g, r)  # BGR for OpenCV
                    
                    color = id_colors[track_id]
                    
                    # Draw on frame (always draw, whether saving or displaying)
                    # Draw bounding box
                    cv2.rectangle(frame, (det['x1'], det['y1']), 
                                (det['x2'], det['y2']), color, 2)
                    
                    # Draw center point
                    cv2.circle(frame, (det['x'], det['y']), 5, color, -1)
                    
                    # Draw room name
                    cv2.putText(frame, det['room_name'], 
                              (det['x1'], det['y1'] - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw tracking trail
                    if len(self.track_history[track_id]) > 1:
                        points = np.array(self.track_history[track_id], 
                                        dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], False, color, 2)
                
                # For camera, display frame in real-time
                if is_camera:
                    # Display frame
                    cv2.imshow('Human Room Tracking', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('s'):
                        print("Saving results and quitting...")
                        break
                
                # Write frame to output video if specified
                if out is not None:
                    out.write(frame)
                
                frame_count += 1
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            if is_camera:
                cv2.destroyAllWindows()
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
            print(f"Output video saved to: {output_video_path}")
        
        # Save results
        if output_json_path:
            self._save_json(output_json_path)
        
        if output_csv_path:
            self._save_csv(output_csv_path)
        
        # Print statistics
        self._print_statistics()
    
    def _save_json(self, output_path):
        """Save results to JSON file."""
        # Group by person_id for easier analysis
        results_by_person = defaultdict(list)
        for result in self.results_data:
            results_by_person[result['person_id']].append(result)
        
        output_data = {
            'total_frames': len(set(r['frame'] for r in self.results_data)),
            'total_detections': len(self.results_data),
            'unique_people': len(results_by_person),
            'results_by_person': dict(results_by_person)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to JSON: {output_path}")
    
    def _save_csv(self, output_path):
        """Save results to CSV file."""
        df = pd.DataFrame(self.results_data)
        df.to_csv(output_path, index=False)
        print(f"Results saved to CSV: {output_path}")
    
    def _print_statistics(self):
        """Print tracking statistics."""
        if not self.results_data:
            print("No detections found.")
            return
        
        df = pd.DataFrame(self.results_data)
        
        print("\n" + "="*50)
        print("TRACKING STATISTICS")
        print("="*50)
        print(f"Total frames processed: {df['frame'].nunique()}")
        print(f"Total detections: {len(df)}")
        print(f"Unique people tracked: {df['person_id'].nunique()}")
        
        if 'room' in df.columns:
            print("\nRoom occupancy:")
            room_counts = df['room'].value_counts()
            for room, count in room_counts.items():
                print(f"  {room}: {count} detections")
        
        # print("\nPer-person statistics:")
        # person_stats = df.groupby('person_id').agg({
        #     'frame': ['min', 'max', 'count'],
        #     'room': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
        # })
        # print(person_stats)
        print("="*50)

def main():
    parser = argparse.ArgumentParser(
        description='Human detection, tracking, and room localization from video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input video file (use --camera for real-time camera)')
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera index for real-time input (e.g., 0 for default camera)')
    parser.add_argument('--room_mask', type=str, required=True,
                       help='Path to room segmentation map (colored image)')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Path to YOLO model checkpoint (default: yolo11n.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--max_people', type=int, default=10,
                       help='Maximum number of people to track (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video with visualizations')
    parser.add_argument('--json', type=str, default=None,
                       help='Path to save results as JSON')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to save results as CSV')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.video is None and args.camera is None:
        raise ValueError("Either --video or --camera must be provided")
    if args.video is not None and args.camera is not None:
        raise ValueError("Cannot use both --video and --camera. Choose one.")
    
    if args.video and not Path(args.video).exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")
    if not Path(args.room_mask).exists():
        raise FileNotFoundError(f"Room mask file not found: {args.room_mask}")
    
    # Initialize tracker
    tracker = HumanRoomTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        max_people=args.max_people,
        room_mask_path=args.room_mask,
    )
    
    # Process video or camera
    tracker.process_video(
        video_path=args.video,
        camera_index=args.camera,
        output_video_path=args.output,
        output_json_path=args.json,
        output_csv_path=args.csv
    )
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

