import os
import cv2
import json
import gradio as gr
from pathlib import Path
import time
import tempfile
import shutil
import numpy as np
from PIL import Image

# Set Gradio cache directory to user's home directory
os.environ['GRADIO_TEMP_DIR'] = str(Path.home() / '.gradio_cache')
os.environ['GRADIO_CACHE_DIR'] = str(Path.home() / '.gradio_cache')

class VideoLabeler:
    def __init__(self, video_path, ids, frame_dir="frames", log_dir="logs"):
        self.video_path = video_path
        self.ids = ids
        self.frame_dir = Path(frame_dir)
        self.log_dir = Path(log_dir)
        self.frames = []
        self.frame_index = 0
        self.current_frame_image = None
        self.grid_size = 50  # Default grid size
        
        # Create temp directory in user's home directory
        self.temp_dir = Path.home() / ".video_labeler_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Gradio cache directory
        gradio_cache = Path.home() / '.gradio_cache'
        gradio_cache.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing frames
        self.load_existing_frames()

    def load_existing_frames(self):
        """Load existing frames from the frames directory"""
        if not self.frame_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frame_dir}")
            
        # Get all PNG files in the frames directory
        self.frames = sorted([str(f) for f in self.frame_dir.glob("frame_*.png")])
        
        if not self.frames:
            raise ValueError("No frames found in the frames directory")
            
        print(f"Loaded {len(self.frames)} frames from {self.frame_dir}")

    def load_first_frame(self):
        try:
            self.frame_index = 0
            self.current_frame_image = cv2.imread(self.frames[0])
            if self.current_frame_image is None:
                raise ValueError(f"Failed to load frame: {self.frames[0]}")
            return self.frames[0]
        except Exception as e:
            print(f"Error loading first frame: {str(e)}")
            return None

    def next_frame(self):
        if not self.frames:
            return None
        self.frame_index = min(self.frame_index + 1, len(self.frames) - 1)
        self.current_frame_image = cv2.imread(self.frames[self.frame_index])
        return self.frames[self.frame_index]

    def prev_frame(self):
        if not self.frames:
            return None
        self.frame_index = max(self.frame_index - 1, 0)
        self.current_frame_image = cv2.imread(self.frames[self.frame_index])
        return self.frames[self.frame_index]

    def draw_grid(self, image, grid_size):
        """Draw grid lines on the image"""
        height, width = image.shape[:2]
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(image, (x, 0), (x, height), (200, 200, 200), 1)
            
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(image, (0, y), (width, y), (200, 200, 200), 1)
            
        # Add coordinate labels
        for x in range(0, width, grid_size):
            cv2.putText(image, str(x), (x + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for y in range(0, height, grid_size):
            cv2.putText(image, str(y), (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def preview_box(self, x1, y1, x2, y2, grid_size):
        """Show a preview of the bounding box with grid"""
        if self.current_frame_image is None:
            return None
            
        try:
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            grid_size = int(grid_size)
            
            # Create a copy of the current frame
            preview = self.current_frame_image.copy()
            
            # Draw grid
            self.draw_grid(preview, grid_size)
            
            # Ensure coordinates are within image bounds
            height, width = preview.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            # Draw red rectangle with thicker line
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Add semi-transparent overlay
            overlay = preview.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, preview, 0.8, 0, preview)
            
            # Draw red rectangle again on top
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Add coordinate labels
            cv2.putText(preview, f"({x1},{y1})", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(preview, f"({x2},{y2})", (x2 - 100, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save preview to temp file
            preview_path = self.temp_dir / f"preview_{int(time.time())}.png"
            cv2.imwrite(str(preview_path), preview)
            
            return str(preview_path)
        except Exception as e:
            print(f"Error in preview_box: {str(e)}")
            return None

    def save_box(self, x1, y1, x2, y2, role):
        """Save bounding box coordinates to JSON file"""
        if self.current_frame_image is None:
            return "Frame not loaded"
            
        if not role:
            return "Please select a role first"
            
        try:
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Ensure coordinates are within image bounds
            height, width = self.current_frame_image.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1 or y2 <= y1:
                return "Invalid coordinates: x2 must be greater than x1 and y2 must be greater than y1"
            
            # Create entry for JSON
            entry = {
                "frame": self.frame_index,
                "frame_path": self.frames[self.frame_index],
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "timestamp": int(time.time())
            }
            
            # Save to JSON file
            log_file = self.log_dir / f"{role}.json"
            
            # Read existing data or create new list
            try:
                if log_file.exists() and log_file.stat().st_size > 0:
                    with open(log_file, "r") as f:
                        data = json.load(f)
                else:
                    data = []
            except json.JSONDecodeError:
                print(f"Invalid JSON in {log_file}, creating new file")
                data = []
                
            data.append(entry)
            with open(log_file, "w") as f:
                json.dump(data, f, indent=2)
            
            return f"Saved box coordinates for {role} at frame {self.frame_index}"
        except Exception as e:
            print(f"Error saving box: {str(e)}")
            return f"Error saving box: {str(e)}"

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            gradio_cache = Path.home() / '.gradio_cache'
            if gradio_cache.exists():
                shutil.rmtree(gradio_cache)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {str(e)}")

    def load_frame(self, frame_number):
        """Load a specific frame by number"""
        if not self.frames:
            return None
        try:
            frame_number = int(frame_number)
            if 0 <= frame_number < len(self.frames):
                self.frame_index = frame_number
                self.current_frame_image = cv2.imread(self.frames[self.frame_index])
                return self.frames[self.frame_index]
            else:
                return None
        except Exception as e:
            print(f"Error loading frame {frame_number}: {str(e)}")
            return None

    def parse_coordinates(self, coord_text):
        """Parse combined coordinates text into individual values, omitting decimals"""
        try:
            # Remove any whitespace, parentheses, and split by comma
            coord_text = coord_text.replace(" ", "").replace("(", "").replace(")", "")
            coords = coord_text.split(",")
            if len(coords) != 4:
                return None, None, None, None
            # Convert to float first, then int (omit decimal part)
            return [int(float(x)) for x in coords]
        except Exception as e:
            return None, None, None, None

    def launch(self):
        try:
            with gr.Blocks() as demo:
                with gr.Row():
                    image = gr.Image(label="Frame Viewer", type="filepath", interactive=True)
                    with gr.Column():
                        obj_id = gr.Dropdown(choices=self.ids, label="Select Object ID", value=self.ids[0])
                        with gr.Row():
                            frame_number = gr.Number(label="Frame Number", precision=0, value=0)
                            jump_btn = gr.Button("Jump to Frame")
                        with gr.Row():
                            prev_btn = gr.Button("⬅️ Prev")
                            next_btn = gr.Button("➡️ Next")
                        
                        # Add bounding box tools
                        gr.Markdown("### Bounding Box Tools")
                        gr.Markdown("Enter coordinates for the bounding box:")
                        combined_coords = gr.Textbox(label="Combined Coordinates (x1,y1,x2,y2)", placeholder="Enter coordinates as x1,y1,x2,y2")
                        with gr.Row():
                            x1 = gr.Number(label="X1", precision=0)
                            y1 = gr.Number(label="Y1", precision=0)
                        with gr.Row():
                            x2 = gr.Number(label="X2", precision=0)
                            y2 = gr.Number(label="Y2", precision=0)
                        grid_size = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Grid Size")
                        preview_btn = gr.Button("👁️ Preview Box")
                        save_btn = gr.Button("💾 Save Box")
                        status = gr.Textbox(label="Status", interactive=False)

                def update_coordinates(coord_text):
                    x1_val, y1_val, x2_val, y2_val = self.parse_coordinates(coord_text)
                    return x1_val, y1_val, x2_val, y2_val

                next_btn.click(self.next_frame, outputs=image)
                prev_btn.click(self.prev_frame, outputs=image)
                jump_btn.click(self.load_frame, inputs=[frame_number], outputs=image)
                preview_btn.click(self.preview_box, inputs=[x1, y1, x2, y2, grid_size], outputs=image)
                save_btn.click(self.save_box, inputs=[x1, y1, x2, y2, obj_id], outputs=status)
                combined_coords.change(update_coordinates, inputs=[combined_coords], outputs=[x1, y1, x2, y2])
                demo.load(self.load_first_frame, outputs=image)

            demo.launch(share=False)
        except Exception as e:
            print(f"Error launching Gradio interface: {str(e)}")
            self.cleanup()
            raise

if __name__ == "__main__":
    try:
        labeler = VideoLabeler(
            video_path="/data/zhaoheng_zhu/origin/Camera-Loc02.mp4",
            ids=["patient_1","patient_2","patient_3","nurse_1","nurse_2","psychiatrist","Psychologist","person_1", "person_2", "person_3", "person_4", "person_5"]
        )
        labeler.launch()
    except Exception as e:
        print(f"Error initializing VideoLabeler: {str(e)}")
    finally:
        if 'labeler' in locals():
            labeler.cleanup()