#!/usr/bin/env python3
"""
Real-time OpenPose Client with UI Controls
Sends camera frames to server and receives processed frames with pose detection.
Includes Tkinter UI for controlling display options.
"""

import cv2
import socket
import struct
import pickle
import time
import threading
import queue
import argparse
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class RealtimeOpenPoseClientUI:
    def __init__(self, server_host, server_port=9999, camera_source='0'):
        """Initialize real-time OpenPose client with UI controls."""
        self.server_host = server_host
        self.server_port = server_port
        self.camera_source = camera_source
        
        # Network
        self.client_socket = None
        self.connected = False
        
        # Camera
        self.cap = None
        
        # Frame queues
        self.send_queue = queue.Queue(maxsize=5)
        self.receive_queue = queue.Queue(maxsize=5)
        
        # Threading
        self.running = False
        self.frame_count = 0
        self.received_count = 0
        
        # Display options (controlled by UI)
        self.show_pose = True
        self.show_bbox = True
        self.show_hand = False
        self.show_face = False
        
        # Metadata from server
        self.latest_metadata = {
            'frame_number': 0,
            'process_time_ms': 0.0,
            'fps': 0.0,
            'poses_detected': 0,
            'tracked_count': 0
        }
        
        # Current frame for saving
        self.current_display_frame = None
        
        # UI
        self.root = None
        self.ui_thread = None
        self.video_label = None
        self.photo = None  # Keep reference to prevent garbage collection
    
    def create_ui(self):
        """Create Tkinter UI with embedded video and controls side-by-side."""
        self.root = tk.Tk()
        self.root.title("OpenPose Real-time - Combined View")
        
        # Main container
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Video display
        video_frame = ttk.Frame(container, padding="10")
        video_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        
        # Video label with black background
        self.video_label = tk.Label(video_frame, bg='black', width=640, height=480)
        self.video_label.pack()
        
        # Video title
        video_title = ttk.Label(video_frame, text="📹 Live Video Feed", 
                               font=('Helvetica', 10, 'bold'))
        video_title.pack(pady=(5, 0))
        
        # Right side: Controls and statistics
        control_frame = ttk.Frame(container, padding="10")
        control_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E))
        
        # Title
        title_label = ttk.Label(control_frame, text="OpenPose Controls", 
                               font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Connection status
        self.status_label = ttk.Label(control_frame, text="Status: Connecting...", 
                                     font=('Helvetica', 10))
        self.status_label.pack(pady=(0, 10))
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=(0, 10))
        
        # Display options section
        options_frame = ttk.LabelFrame(control_frame, text="Display Options", padding="10")
        options_frame.pack(fill='x', pady=(0, 10))
        
        # Pose checkbox
        self.pose_var = tk.BooleanVar(value=True)
        pose_check = ttk.Checkbutton(options_frame, text="Show Pose Skeleton", 
                                    variable=self.pose_var,
                                    command=self.on_pose_toggle)
        pose_check.pack(anchor='w', pady=3)
        
        # Bounding box checkbox
        self.bbox_var = tk.BooleanVar(value=True)
        bbox_check = ttk.Checkbutton(options_frame, text="Show Tracking Bounding Boxes", 
                                    variable=self.bbox_var,
                                    command=self.on_bbox_toggle)
        bbox_check.pack(anchor='w', pady=3)
        
        # Hand detection checkbox
        self.hand_var = tk.BooleanVar(value=False)
        hand_check = ttk.Checkbutton(options_frame, text="Show Hand Keypoints", 
                                    variable=self.hand_var,
                                    command=self.on_hand_toggle)
        hand_check.pack(anchor='w', pady=3)
        
        # Face detection checkbox
        self.face_var = tk.BooleanVar(value=False)
        face_check = ttk.Checkbutton(options_frame, text="Show Face Keypoints", 
                                    variable=self.face_var,
                                    command=self.on_face_toggle)
        face_check.pack(anchor='w', pady=3)
        
        # Performance Statistics section
        perf_frame = ttk.LabelFrame(control_frame, text="Performance Statistics", padding="10")
        perf_frame.pack(fill='x', pady=(0, 10))
        
        # Frame info
        self.frame_label = ttk.Label(perf_frame, text="Frame: 0", 
                                     font=('Helvetica', 10))
        self.frame_label.pack(anchor='w', pady=2)
        
        # FPS
        self.fps_label = ttk.Label(perf_frame, text="FPS: 0.0", 
                                   font=('Helvetica', 10))
        self.fps_label.pack(anchor='w', pady=2)
        
        # Process time
        self.process_time_label = ttk.Label(perf_frame, text="Process Time: 0.0 ms", 
                                           font=('Helvetica', 10))
        self.process_time_label.pack(anchor='w', pady=2)
        
        # Detection Statistics section
        detect_frame = ttk.LabelFrame(control_frame, text="Detection Statistics", padding="10")
        detect_frame.pack(fill='x', pady=(0, 10))
        
        # Poses detected
        self.poses_label = ttk.Label(detect_frame, text="Poses Detected: 0", 
                                    font=('Helvetica', 10))
        self.poses_label.pack(anchor='w', pady=2)
        
        # Tracked people
        self.tracked_label = ttk.Label(detect_frame, text="People Tracked: 0", 
                                      font=('Helvetica', 10))
        self.tracked_label.pack(anchor='w', pady=2)
        
        # Network Statistics section
        network_frame = ttk.LabelFrame(control_frame, text="Network Statistics", padding="10")
        network_frame.pack(fill='x', pady=(0, 10))
        
        self.stats_label = ttk.Label(network_frame, text="Sent: 0 | Received: 0", 
                                    font=('Helvetica', 10))
        self.stats_label.pack(anchor='w', pady=2)
        
        # Instructions
        instructions = ttk.Label(control_frame, 
                                text="Press 'Q' to quit\nPress 'S' to save frame",
                                font=('Helvetica', 9), foreground='gray')
        instructions.pack(pady=(10, 0))
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_ui_close)
        
        # Bind keyboard shortcuts
        self.root.bind('<q>', lambda e: self.on_ui_close())
        self.root.bind('<Q>', lambda e: self.on_ui_close())
        self.root.bind('<s>', lambda e: self.save_current_frame())
        self.root.bind('<S>', lambda e: self.save_current_frame())
        
        # Configure grid weights
        container.columnconfigure(0, weight=2)  # Video gets more space
        container.columnconfigure(1, weight=1)  # Controls get less space
        container.rowconfigure(0, weight=1)
    
    def on_pose_toggle(self):
        """Handle pose checkbox toggle."""
        self.show_pose = self.pose_var.get()
        print(f"🎨 Pose display: {'ON' if self.show_pose else 'OFF'}")
    
    def on_bbox_toggle(self):
        """Handle bounding box checkbox toggle."""
        self.show_bbox = self.bbox_var.get()
        print(f"📦 Bounding box display: {'ON' if self.show_bbox else 'OFF'}")
    
    def on_hand_toggle(self):
        """Handle hand detection checkbox toggle."""
        self.show_hand = self.hand_var.get()
        print(f"✋ Hand detection: {'ON' if self.show_hand else 'OFF'}")
    
    def on_face_toggle(self):
        """Handle face detection checkbox toggle."""
        self.show_face = self.face_var.get()
        print(f"👤 Face detection: {'ON' if self.show_face else 'OFF'}")
    
    def save_current_frame(self):
        """Save the current frame to disk."""
        if self.current_display_frame is not None:
            frame_number = self.latest_metadata.get('frame_number', 0)
            filename = f"openpose_frame_{frame_number:06d}.png"
            cv2.imwrite(filename, self.current_display_frame)
            print(f"💾 Saved frame to {filename}")
        else:
            print("⚠️  No frame available to save")
    
    def on_ui_close(self):
        """Handle UI window close."""
        print("🛑 UI window closed, stopping client...")
        self.running = False
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def update_ui_status(self):
        """Update UI status labels with latest metadata."""
        if self.root and self.root.winfo_exists():
            try:
                # Update connection status
                if self.connected:
                    status_text = "Status: ✅ Connected"
                    self.status_label.config(foreground='green')
                else:
                    status_text = "Status: ❌ Disconnected"
                    self.status_label.config(foreground='red')
                self.status_label.config(text=status_text)
                
                # Update performance statistics
                self.frame_label.config(text=f"Frame: {self.latest_metadata['frame_number']}")
                self.fps_label.config(text=f"FPS: {self.latest_metadata['fps']}")
                self.process_time_label.config(text=f"Process Time: {self.latest_metadata['process_time_ms']} ms")
                
                # Update detection statistics
                self.poses_label.config(text=f"Poses Detected: {self.latest_metadata['poses_detected']}")
                self.tracked_label.config(text=f"People Tracked: {self.latest_metadata['tracked_count']}")
                
                # Update network statistics
                stats_text = f"Sent: {self.frame_count} | Received: {self.received_count}"
                self.stats_label.config(text=stats_text)
                
                # Schedule next update (more frequent for smoother updates)
                self.root.after(100, self.update_ui_status)
            except tk.TclError:
                pass  # Window was destroyed
    
    def run_ui(self):
        """Run Tkinter UI in main thread."""
        self.create_ui()
        self.update_ui_status()  # Start status updates
        self.update_video_display()  # Start video updates
        self.root.mainloop()
    
    def connect_to_server(self):
        """Connect to OpenPose server."""
        try:
            print(f"🔌 Connecting to {self.server_host}:{self.server_port}...")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.client_socket.settimeout(10)
            self.client_socket.connect((self.server_host, self.server_port))
            self.client_socket.settimeout(None)
            
            print("✅ Connected to OpenPose server!")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera."""
        try:
            # Determine if camera_source is an integer (camera ID) or string (RTSP URL)
            if self.camera_source.isdigit():
                camera_input = int(self.camera_source)
                print(f"📷 Initializing camera ID {camera_input}...")
            else:
                camera_input = self.camera_source
                print(f"📷 Initializing RTSP stream {camera_input}...")
            
            self.cap = cv2.VideoCapture(camera_input)
            
            if not self.cap.isOpened():
                print(f"❌ Could not open camera source {self.camera_source}")
                return False
            
            # Set camera properties for both local cameras and RTSP streams
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if self.camera_source.isdigit():
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                # For RTSP streams, set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("✅ Camera initialized!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            return False
    
    def capture_frames(self):
        """Capture frames from camera and add to send queue."""
        print("📹 Starting frame capture thread...")
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Failed to capture frame")
                    continue
                
                # Add to send queue with display options
                try:
                    self.send_queue.put_nowait((frame, self.frame_count, 
                                               self.show_pose, self.show_bbox,
                                               self.show_hand, self.show_face))
                    self.frame_count += 1
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error capturing frames: {e}")
                break
    
    def send_frames(self):
        """Send frames to server with display options."""
        print("📤 Starting frame sending thread...")
        
        while self.running and self.connected:
            try:
                # Get frame from send queue
                frame, frame_number, show_pose, show_bbox, show_hand, show_face = self.send_queue.get(timeout=1.0)
                
                # Create data packet with frame and display options
                data_packet = {
                    'frame': frame,
                    'show_pose': show_pose,
                    'show_bbox': show_bbox,
                    'show_hand': show_hand,
                    'show_face': show_face
                }
                
                # Serialize packet
                data = pickle.dumps(data_packet)
                
                # Send packet size
                size = len(data)
                self.client_socket.sendall(struct.pack("!L", size))
                
                # Send packet data
                self.client_socket.sendall(data)
                
                if frame_number % 30 == 0:
                    print(f"📤 Sent {frame_number} frames (pose={show_pose}, bbox={show_bbox}, hand={show_hand}, face={show_face})")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error sending frames: {e}")
                self.connected = False
                break
    
    def receive_frames(self):
        """Receive processed frames with metadata from server."""
        print("📥 Starting frame receiving thread...")
        data = b""
        payload_size = struct.calcsize("!L")
        
        while self.running and self.connected:
            try:
                # Receive packet size
                while len(data) < payload_size:
                    packet = self.client_socket.recv(4096)
                    if not packet:
                        print("❌ Connection lost while receiving")
                        self.connected = False
                        return
                    data += packet
                
                # Extract packet size
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("!L", packed_msg_size)[0]
                
                # Receive packet data
                while len(data) < msg_size:
                    data += self.client_socket.recv(4096)
                
                # Extract packet
                packet_data = data[:msg_size]
                data = data[msg_size:]
                
                # Deserialize response packet
                response_packet = pickle.loads(packet_data)
                processed_frame = response_packet.get('frame')
                metadata = response_packet.get('metadata', {})
                
                # Update latest metadata
                if metadata:
                    self.latest_metadata = metadata
                
                # Add to receive queue
                try:
                    self.receive_queue.put_nowait((processed_frame, self.received_count))
                    self.received_count += 1
                    if self.received_count % 30 == 0:
                        print(f"📥 Received {self.received_count} processed frames")
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.receive_queue.get_nowait()
                        self.receive_queue.put_nowait((processed_frame, self.received_count))
                        self.received_count += 1
                    except queue.Empty:
                        pass
                
            except Exception as e:
                print(f"❌ Error receiving frames: {e}")
                self.connected = False
                break
    
    def update_video_display(self):
        """Update video display in Tkinter."""
        try:
            # Try to get a frame from the queue (non-blocking)
            if not self.receive_queue.empty():
                processed_frame, frame_number = self.receive_queue.get_nowait()
                
                # Store current frame for saving
                self.current_display_frame = processed_frame.copy()
                
                # Convert BGR (OpenCV) to RGB (PIL/Tkinter)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and resize to match UI label size (640x480)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                
                # Convert to ImageTk
                self.photo = ImageTk.PhotoImage(image=img)
                
                # Update label
                if self.video_label:
                    self.video_label.config(image=self.photo)
                    
        except queue.Empty:
            pass
        except Exception as e:
            print(f"❌ Error updating video display: {e}")
        
        # Schedule next update (30 FPS = ~33ms)
        if self.running and self.root and self.root.winfo_exists():
            self.root.after(33, self.update_video_display)
    
    def run(self):
        """Main client loop with Tkinter in main thread."""
        print("🚀 Starting Real-time OpenPose Client with Combined UI...")
        
        # Initialize camera
        if not self.initialize_camera():
            return False
        
        # Connect to server
        if not self.connect_to_server():
            self.cleanup()
            return False
        
        # Start worker threads
        self.running = True
        
        # Frame capture thread
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Frame sending thread
        send_thread = threading.Thread(target=self.send_frames)
        send_thread.daemon = True
        send_thread.start()
        
        # Frame receiving thread
        receive_thread = threading.Thread(target=self.receive_frames)
        receive_thread.daemon = True
        receive_thread.start()
        
        print("✅ All threads started!")
        print("🎥 Real-time OpenPose client active...")
        print("📺 Video and controls combined in one window!")
        print("Press 'Q' to quit, 'S' to save frame")
        
        try:
            # Run UI in main thread (Tkinter requirement)
            self.run_ui()
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        if self.client_socket:
            self.client_socket.close()
        
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
        
        print("✅ Cleanup completed")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time OpenPose Client with UI Controls')
    parser.add_argument('--server_host', type=str, default="10.55.164.170",
                        help='OpenPose server hostname or IP address')
    parser.add_argument('--port', type=int, default=9999,
                        help='Server port (default: 9999)')
    parser.add_argument('--camera', type=str, default='0',
                        help='Camera source: integer for camera ID or RTSP URL (default: 0)')

    
    args = parser.parse_args()
    
    # Create and run client
    client = RealtimeOpenPoseClientUI(
        server_host=args.server_host,
        server_port=args.port,
        camera_source=args.camera
    )
    
    try:
        success = client.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

