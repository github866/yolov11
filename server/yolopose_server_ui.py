#!/usr/bin/env python3
"""
Real-time YOLO-Pose Server with UI Control Support
Processes frames with YOLO-Pose and respects display options from client
(This version does NOT include a tracker)
"""

import os
import sys
import cv2
import socket
import struct
import pickle
import time
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict

# Import utility for room color mapping
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def color_to_room(bgr):
    """Map BGR color to room name (OpenCV uses BGR, not RGB)."""
    
    if bgr == (255, 0, 0):  # Blue
        return "Patient Area"
    elif bgr == (0, 255, 0):  # Green
        return 'Nursing Station (Open)'
    elif bgr == (0, 0, 255):  # Red
        return 'Nursing Station (Closed)'
    elif bgr == (255, 255, 0):  # Yellow
        return 'Console Room'
    else:
        return 'Unknown Room'

def room_name_to_color(room_name):
    """Map room name to BGR color."""
    if room_name == "Patient Area":
        return (255, 0, 0)
    elif room_name == "Nursing Station (Open)":
        return (0, 255, 0)
    elif room_name == "Nursing Station (Closed)":
        return (0, 0, 255)
    elif room_name == "Console Room":
        return (255, 255, 0)
    elif room_name == "Unknown Room":
        return (0, 0, 0)

def check_gpu_availability():
    """Check if GPU is available for PyTorch"""
    print("=" * 50)
    print("GPU Availability Check")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  WARNING: CUDA is not available! Running on CPU.")
    print("=" * 50)


class RoomSegmentationMap:
    """Efficient room segmentation map handler for O(1) room lookup."""
    
    def __init__(self, mask_image):
        """Initialize room segmentation map from image."""
        self.mask = mask_image
        print(f"Mask shape received: {mask_image.shape}")
        
        # CRITICAL FIX: Handle potential extra dimension
        assert len(self.mask.shape) == 3, "Mask must be 3-dimensional"
        
        self.height, self.width = self.mask.shape[:2]
        print(f"📋 Loaded room segmentation map: {self.width}x{self.height}")
        
        # Compute room centers for visualization
        self.room_centers = self._compute_room_centers()
        print(f"📋 Found {len(self.room_centers)} unique rooms")
    
    def get_room_at_point(self, x, y):
        """Get room name at a given point (x, y). O(1) lookup."""
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        
        # Get BGR color at this point (OpenCV uses BGR)
        bgr_color = self.mask[y, x]
        
        # CRITICAL FIX: Pass BGR tuple directly to color_to_room
        bgr_tuple = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
        
        # Map color to room name
        room_name = color_to_room(bgr_tuple)
        return room_name
    
    def get_room_for_bbox(self, x1, y1, x2, y2):
        """Get room name for a bounding box by checking foot position."""
        # CRITICAL FIX: Use bottom-center point (feet) instead of bbox center
        # This is more accurate for human localization since feet touch the ground
        center_x = int((x1 + x2) / 2)
        bottom_y = int(y2 - (y2 - y1) * 0.05)  # 
        
        return self.get_room_at_point(center_x, bottom_y)
    
    def _compute_room_centers(self):
        """Compute the center point of each room by finding the centroid."""
        room_centers = {}
        room_pixels = defaultdict(list)
        
        # Collect all pixels for each room (sample for efficiency)
        sample_step = 5  # Sample every 5 pixels for speed
        for y in range(0, self.height, sample_step):
            for x in range(0, self.width, sample_step):
                bgr_color = self.mask[y, x]
                bgr_tuple = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
                room_name = color_to_room(bgr_tuple)
                
                room_pixels[room_name].append((x, y))
        
        # Calculate centroid for each room
        for room_name, pixels in room_pixels.items():
            if len(pixels) > 0:
                center_x = int(np.mean([p[0] for p in pixels]))
                center_y = int(np.mean([p[1] for p in pixels]))
                room_centers[room_name] = (center_x, center_y)
        
        # Manually set a position for "Unknown Room" for consistent display
        if "Unknown Room" in room_pixels:
            room_centers["Unknown Room"] = (30, 30) # Top-left corner
        
        return room_centers
        
    def get_visualization_mask(self, room_counts):
        """
        Create a visualization mask with room counts displayed at the center.
        
        Args:
            room_counts: A dictionary mapping room names to person counts.
            
        Returns:
            A transparent overlay image with counts displayed.
        """
        # Create a transparent overlay
        overlay = self.mask.copy()

        for room_name, count in room_counts.items():
            if room_name in self.room_centers and count > 0:
                center_x, center_y = self.room_centers[room_name]
                
                # Create text to display
                text = str(count)
                
                # Choose font and scale
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 3
                
                # Get text size to position it correctly
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                
                # Position text at the center of the room
                text_x = center_x - text_width // 2
                text_y = center_y + text_height // 2
                
                # Draw white text with a black border for visibility
                cv2.putText(
                    overlay, text, (text_x, text_y), font, font_scale, 
                    (0, 0, 0), font_thickness + 2, cv2.LINE_AA
                )
                cv2.putText(
                    overlay, text, (text_x, text_y), font, font_scale, 
                    (255, 255, 255), font_thickness, cv2.LINE_AA
                )
        
        return overlay
        
class RealtimeYoloPoseServer:
    def __init__(
        self, 
        model_path, 
        server_port,
        device=None,
        cuda_device=0,
    ):
        """Initialize real-time YOLO-Pose server with UI control support."""
        self.model_path = model_path
        self.server_port = server_port
        self.device = device
        self.cuda_device = cuda_device
        
        # Network
        self.server_socket = None
        self.client_socket = None
        self._recv_buffer = b""
        
        # Processing
        self.frame_count = 0
        self.model = None
        self.device = device
        
        # Room segmentation
        self.room_map = None
        self.segmentation_map_received = False
        
        # Performance tracking
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.performance_metrics = {
            'frames': 0,
            'inference_ms_total': 0.0,
            'overlay_ms_totals': defaultdict(float)
        }
        
        print(f"📁 YOLO Model: {self.model_path}")
        print(f"🎯 Target: Real-time multi-person pose detection with room localization")
        print(f"🎨 UI Controls: Supports dynamic pose and bounding box toggling")
    
    def initialize_yolo(self):
        """Initialize YOLO wrapper with persistent model loading."""
        try:
            print("🧠 Initializing YOLO model (this will take a few seconds)...")
            check_gpu_availability()
            
            # Load YOLO model
            self.model = YOLO(self.model_path)
            
            # Set device - use CUDA if available, otherwise fall back to CPU
            if self.device is None:
                if torch.cuda.is_available():
                    self.device = f'cuda:{self.cuda_device}'
                else:
                    self.device = 'cpu'
            
            # Move model to specified device
            self.model.to(self.device)
            print(f"\n✓ Using device: {self.device}")
            
            # Run a dummy inference to warm up the model
            print("🔥 Warming up model...")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False, device=self.device)
            
            print("✅ YOLO model loaded successfully! Model will persist in memory.")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize YOLO: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_server(self):
        """Start server."""
        try:
            print(f"🚀 Starting server on port {self.server_port}...")
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.server_port))
            self.server_socket.listen(1)
            
            print(f"👂 Server listening...")
            self.client_socket, addr = self.server_socket.accept()
            print(f"✅ Connected to client at {addr}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def receive_data_packet(self):
        """Receive a data packet with frame and display options."""
        try:
            data = self._recv_buffer
            payload_size = struct.calcsize("!L")
            
            # Receive packet size
            while len(data) < payload_size:
                packet = self.client_socket.recv(4096)
                if not packet:
                    self._recv_buffer = data
                    return None
                data += packet
            
            # Extract packet size
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("!L", packed_msg_size)[0]
            
            # Receive packet data
            while len(data) < msg_size:
                packet = self.client_socket.recv(4096)
                if not packet:
                    self._recv_buffer = data
                    return None
                data += packet
            
            # Extract packet
            packet_data = data[:msg_size]
            self._recv_buffer = data[msg_size:]
            data_packet = pickle.loads(packet_data)
            
            return data_packet
            
        except Exception as e:
            print(f"❌ Error receiving data packet: {e}")
            self._recv_buffer = data if 'data' in locals() else self._recv_buffer
            return None
    
    def send_frame_with_metadata(self, frame, metadata):
        """Send a processed frame with metadata back to client."""
        try:
            # Create response packet with frame and metadata
            response_packet = {
                'frame': frame,
                'metadata': metadata
            }
            
            # Serialize packet
            data = pickle.dumps(response_packet)
            
            # Send packet size
            size = len(data)
            self.client_socket.sendall(struct.pack("!L", size))
            
            # Send packet data
            self.client_socket.sendall(data)
            return True
            
        except Exception as e:
            print(f"❌ Error sending frame: {e}")
            return False
    
    def draw_pose_keypoints_yolo(self, frame, keypoints_obj, render_threshold=0.5):
        """Manually draw YOLO (COCO) pose keypoints on frame."""
        if keypoints_obj is None:
            return frame
            
        # COCO keypoint pairs for skeleton
        skeleton_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        try:
            for person_kpts in keypoints_obj:
                # Get keypoint coordinates (x, y, confidence)
                kpts = person_kpts.xy[0].cpu().numpy()  # Shape: (17, 2)
                kpts_conf = person_kpts.conf[0].cpu().numpy()  # Shape: (17,)
                
                # Draw skeleton connections
                for pair in skeleton_pairs:
                    pt1_idx, pt2_idx = pair
                    if (pt1_idx < len(kpts_conf) and pt2_idx < len(kpts_conf) and
                        kpts_conf[pt1_idx] > render_threshold and kpts_conf[pt2_idx] > render_threshold):
                        
                        pt1 = tuple(map(int, kpts[pt1_idx]))
                        pt2 = tuple(map(int, kpts[pt2_idx]))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                
                # Draw keypoints
                for i, (kpt, conf) in enumerate(zip(kpts, kpts_conf)):
                    if conf > render_threshold:
                        x, y = map(int, kpt)
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                        
        except Exception as e:
            print(f"⚠️  Error drawing YOLO keypoints: {e}")
        
        return frame
    
    def process_frame_yolo(self, frame, frame_number, fps=15):
        """Process frame using persistent YOLO model with tracking and display options."""
        try:
            # Process frame with YOLO
            start_time = time.time()
            results = self.model(
                frame, 
                verbose=False, 
                device=self.device
            )
            process_time = time.time() - start_time
            
            # Start with original frame (we'll draw on it selectively)
            processed_frame = frame.copy()
            
            # Extract pose information
            current_bboxes = []
            bbox_rooms = []  # Store room name for each bbox
            poses_detected = 0
            overlay_timings = {}
            
            # YOLO results are a list (usually of 1 for a single image)
            if not results or len(results) == 0:
                print("⚠️  No results from YOLO model.")
                # Create empty metadata
                metadata = { 
                    'frame_number': frame_number, 
                    'process_time_ms': round(process_time * 1000, 1), 
                    'fps': 0,
                    'poses_detected': 0, 
                }
                return processed_frame, metadata

            result = results[0]  # Get the first result object
            boxes = result.boxes
            keypoints = result.keypoints

            # Get bounding boxes and determine rooms
            if boxes is not None and len(boxes) > 0:
                poses_detected = len(boxes)
                for idx, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    current_bboxes.append((x1, y1, x2, y2))
                    
                    # Determine room for this bounding box
                    room_name = "Unknown Room"
                    if self.room_map:
                        room_name = self.room_map.get_room_for_bbox(x1, y1, x2, y2)
                    bbox_rooms.append(room_name)
            
            # Draw pose keypoints if enabled
            if keypoints is not None:
                pose_start = time.time()
                processed_frame = self.draw_pose_keypoints_yolo(processed_frame, keypoints)
                overlay_timings['pose_overlay_ms'] = round((time.time() - pose_start) * 1000, 2)

            # Draw detection bounding boxes with room labels if enabled
            bbox_start = time.time()
            for idx, bbox in enumerate(current_bboxes):
                min_x, min_y, max_x, max_y = bbox
                # Draw bounding box                
                # Draw room name on bounding box
                if idx < len(bbox_rooms):
                    room_name = bbox_rooms[idx]
                    room_color = room_name_to_color(room_name)
                    
                    cv2.rectangle(processed_frame, (min_x, min_y), (max_x, max_y), room_color, 2)

                    # Draw background for room name
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(
                        room_name, font, font_scale, thickness
                    )
                    
                    # Draw semi-transparent background
                    overlay = processed_frame.copy()
                    if room_name != "Unknown Room":
                        cv2.rectangle(
                            overlay,
                            (min_x, min_y - text_height - 10),
                            (min_x + text_width + 5, min_y),
                            (0, 0, 0),
                            -1
                        )
                        cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
                        
                        # Draw room name text
                        text_x = min_x + 2
                        text_y = min_y - 5 if min_y - 5 - text_height > 0 else min_y + text_height + 10
                        cv2.putText(
                            processed_frame,
                            room_name,
                            (text_x, text_y),
                            font,
                            font_scale,
                            (0, 255, 0),  # Green text
                            thickness,
                            cv2.LINE_AA
                        )
            overlay_timings['bbox_overlay_ms'] = round((time.time() - bbox_start) * 1000, 2)
            
            # Count people per room
            room_counts = defaultdict(int)
            for room_name in bbox_rooms:
                room_counts[room_name] += 1
            
            # Create combined visualization with room mask below
            if self.room_map:
                # Get room mask visualization with counts
                mask_vis = self.room_map.get_visualization_mask(dict(room_counts))
                # Combine frame and mask horizontally
                mask_vis = cv2.resize(mask_vis, (processed_frame.shape[1]//4*3, processed_frame.shape[0]))
                top_panel = np.hstack([processed_frame, mask_vis])

                # Create info panel and stack it vertically
                info_panel_height = 40
                info_panel = self._create_info_panel(
                    top_panel.shape[1], 
                    info_panel_height, 
                    dict(room_counts), 
                    frame_number,
                    fps
                )
                combined_frame = np.vstack([top_panel, info_panel])
            else:
                combined_frame = processed_frame
            
            # Calculate FPS
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.fps_counter / (current_time - self.last_fps_time)
                self.last_fps_time = current_time
                self.fps_counter = 0
            else:
                fps = self.fps_counter / (current_time - self.last_fps_time) if current_time > self.last_fps_time else 0
            
            # Create metadata dict
            metadata = {
                'frame_number': frame_number,
                'process_time_ms': round(process_time * 1000, 1),
                'fps': round(fps, 1),
                'poses_detected': poses_detected,
                'room_counts': dict(room_counts)
            }

            avg_process_time_ms, avg_overlay_ms = self.update_performance_metrics(process_time, overlay_timings)
            metadata['avg_process_time_ms'] = round(avg_process_time_ms, 1)
            if overlay_timings:
                metadata['overlay_timings_ms'] = overlay_timings
            if avg_overlay_ms:
                metadata['avg_overlay_ms'] = {key: round(value, 2) for key, value in avg_overlay_ms.items()}
            
            # print(f"✅ Frame {frame_number} processed in {process_time*1000:.1f}ms (FPS: {fps:.1f}) - "
            #       f"Poses: {poses_detected} "
            #       f"Room counts: {room_counts}")
            if room_counts and frame_number % 5 == 0:
                room_text = ", ".join(f"{room}: {count}" for room, count in room_counts.items())
                print(f"   🏠 Room occupancy: {room_text} at second {int(frame_number/15)} ")
            # if overlay_timings:
            #     timings_text = ", ".join(f"{key}={value}ms" for key, value in overlay_timings.items())
            #     print(f"   ⏱️ Overlay timings: {timings_text}")
            # if avg_overlay_ms:
            #     avg_timings_text = ", ".join(f"{key}_avg={value:.2f}ms" for key, value in avg_overlay_ms.items())
            #     print(f"   📊 Average overlay timings: {avg_timings_text}")
            # print(f"   📈 Average inference time: {avg_process_time_ms:.1f}ms")
            
            return combined_frame, metadata
                
        except Exception as e:
            print(f"⚠️  Processing error: {e}")
            import traceback
            traceback.print_exc()
            error_frame = self.add_error_overlay(frame, frame_number, f"Error: {str(e)}")
            error_metadata = {
                'frame_number': frame_number,
                'process_time_ms': 0,
                'fps': 0,
                'poses_detected': 0,
                'error': str(e)
            }
            return error_frame, error_metadata
    
    def add_error_overlay(self, frame, frame_number, error_msg):
        """Add error message to frame."""
        overlay_frame = frame.copy()
        cv2.putText(overlay_frame, f"Frame: {frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(overlay_frame, error_msg, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return overlay_frame
    
    def _create_info_panel(self, width, height, room_counts, frame_number, fps):
        """Create a small panel to display room occupancy information."""
        panel = np.zeros((height, width, 3), dtype="uint8")
        
        if not room_counts:
            info_text = "No people detected."
        else:
            room_text = ", ".join(f"{room}: {count}" for room, count in room_counts.items())
            info_text = f"Room Occupancy: {room_text} (Frame: {frame_number})"

        # Set font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 1
        text_color = (255, 255, 255)  # White

        # Add text to the panel
        cv2.putText(panel, info_text, (10, int(height * 0.6)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        return panel

    def run(self):
        """Main server loop with persistent YOLO model and UI control support."""
        print("🚀 Starting Real-time YOLO-Pose Server with UI Control Support...")
        
        # Initialize OpenPose
        if not self.initialize_yolo():
            print("❌ Failed to initialize YOLO")
            return False
        
        # Start server
        if not self.start_server():
            return False
        
        print("🎥 Starting real-time YOLO-Pose processing...")
        print("🎨 Server will respect display options from client UI")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Receive data packet (frame + display options + optional segmentation map)
                data_packet = self.receive_data_packet()
                if data_packet is None:
                    print("❌ Failed to receive data packet, stopping...")
                    break
                
                # Check if this is the first packet with segmentation map
                if not self.segmentation_map_received:
                    segmentation_map = data_packet.get('segmentation_map')
                    if segmentation_map is not None:
                        try:
                            print("📋 Received room segmentation map, initializing...")
                            self.room_map = RoomSegmentationMap(segmentation_map)
                            self.segmentation_map_received = True
                            print("✅ Room segmentation map initialized successfully!")
                        except Exception as e:
                            print(f"⚠️  Failed to initialize room map: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Extract frame and display options
                frame = data_packet.get('frame')
                
                if frame is None:
                    print("⚠️  Received empty frame, skipping...")
                    continue
                
                self.frame_count += 1
                
                # Process frame with display options
                processed_frame, metadata = self.process_frame_yolo(
                    frame=frame,
                    frame_number=self.frame_count,
                )
                
                # Send processed frame with metadata back
                if not self.send_frame_with_metadata(processed_frame, metadata):
                    print("❌ Failed to send processed frame, stopping...")
                    break
                
                # print(f"📤 Sent processed frame {self.frame_count}")
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            # Cleanup
            print("🧹 Cleaning up...")
            if self.model:
                del self.model
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()
            self.print_performance_summary()
            
            print("✅ Real-time YOLO-Pose server stopped")
        
        return True

    def update_performance_metrics(self, process_time_s, overlay_timings_ms):
        """Update rolling performance metrics and return current averages."""
        process_time_ms = process_time_s * 1000.0
        self.performance_metrics['frames'] += 1
        self.performance_metrics['inference_ms_total'] += process_time_ms
        for key, value in overlay_timings_ms.items():
            self.performance_metrics['overlay_ms_totals'][key] += value

        frames = self.performance_metrics['frames']
        avg_process_time_ms = self.performance_metrics['inference_ms_total'] / frames if frames else 0.0
        avg_overlay_ms = {
            key: total / frames
            for key, total in self.performance_metrics['overlay_ms_totals'].items()
            if frames
        }
        return avg_process_time_ms, avg_overlay_ms

    def print_performance_summary(self):
        """Print a summary of collected performance metrics."""
        frames = self.performance_metrics['frames']
        if frames == 0:
            print("ℹ️ No frames processed, performance summary unavailable.")
            return

        avg_inference_ms = self.performance_metrics['inference_ms_total'] / frames
        print("📊 Performance Summary:")
        print(f"   • Frames processed: {frames}")
        print(f"   • Average inference time: {avg_inference_ms:.2f}ms")

        if self.performance_metrics['overlay_ms_totals']:
            print("   • Average overlay times:")
            for key, total in self.performance_metrics['overlay_ms_totals'].items():
                print(f"     - {key}: {total / frames:.2f}ms")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time YOLO-Pose Server with UI Control Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    parser.add_argument('--port', type=int, default=9988, 
                        help='Server port')
    parser.add_argument('--model', type=str, default='/home/agenuinedream/repo/yolov11/yolo11m-pose.pt',
                        help='Path to YOLO-Pose model (e.g., /home/agenuinedream/repo/yolov11/yolo11m-pose.pt)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda:0" or "cpu"). Default: auto-detect.')
    
    args = parser.parse_args()
    
    # Check if model file exists before initializing the server
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found at '{args.model}'")
        sys.exit(1)

    # Create and run server
    server = RealtimeYoloPoseServer(
        server_port=args.port,
        model_path=args.model,
        device=args.device
    )
    
    try:
        success = server.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
