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
    def __init__(self, video_path, ids, frame_dir="frames", log_dir="logs", clip_dir="clips"):
        self.video_path = video_path
        self.ids = ids
        self.frame_dir = Path(frame_dir)
        self.log_dir = Path(log_dir)
        self.clip_dir = Path(clip_dir)
        self.frames = []
        self.frame_index = 0
        self.current_frame_image = None
        
        # Create temp directory in user's home directory
        self.temp_dir = Path.home() / ".video_labeler_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Gradio cache directory
        gradio_cache = Path.home() / '.gradio_cache'
        gradio_cache.mkdir(parents=True, exist_ok=True)
        
        # Create clip directory
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify video file exists
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

    def extract_frames(self):
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"Video properties:")
        print(f"- Total frames: {total_frames}")
        print(f"- FPS: {fps}")
        print(f"- Duration: {duration:.2f} seconds")
        print(f"Starting frame extraction...")
        
        start_time = time.time()
        count = 0
        self.frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_file = self.frame_dir / f"frame_{count:04d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            self.frames.append(str(frame_file))
            
            # Print progress every 100 frames
            if count % 100 == 0:
                elapsed_time = time.time() - start_time
                frames_per_second = count / elapsed_time if elapsed_time > 0 else 0
                remaining_frames = total_frames - count
                estimated_time = remaining_frames / frames_per_second if frames_per_second > 0 else 0
                
                print(f"Progress: {count}/{total_frames} frames ({count/total_frames*100:.1f}%)")
                print(f"Processing speed: {frames_per_second:.1f} frames/second")
                print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
            
            count += 1
            
        cap.release()
        
        if not self.frames:
            raise ValueError("No frames were extracted from the video")
            
        total_time = time.time() - start_time
        print(f"\nFrame extraction completed:")
        print(f"- Total frames extracted: {len(self.frames)}")
        print(f"- Total processing time: {total_time/60:.1f} minutes")
        print(f"- Average processing speed: {len(self.frames)/total_time:.1f} frames/second")

    def load_first_frame(self):
        try:
            self.extract_frames()
            self.frame_index = 0
            self.current_frame_image = cv2.imread(self.frames[0])
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

    def save_annotations(self, boxes, object_id):
        if not boxes:
            return "No boxes to save"
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"{object_id}.json"
        entry = {"frame": self.frame_index, "boxes": boxes}
        if log_file.exists():
            with open(log_file, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)
        return f"Saved {len(boxes)} boxes for {object_id} at frame {self.frame_index}"

    def clip_and_save(self, clip_box):
        if not clip_box or not self.current_frame_image is not None:
            return "No region selected or frame not loaded"
            
        try:
            # Get coordinates from the clip box
            x1, y1, x2, y2 = clip_box
            
            # Ensure coordinates are within image bounds
            height, width = self.current_frame_image.shape[:2]
            x1 = max(0, min(int(x1), width))
            y1 = max(0, min(int(y1), height))
            x2 = max(0, min(int(x2), width))
            y2 = max(0, min(int(y2), height))
            
            # Extract the region
            clipped = self.current_frame_image[y1:y2, x1:x2]
            
            # Save the clipped image
            clip_filename = f"clip_frame_{self.frame_index:04d}_{int(time.time())}.jpg"
            clip_path = self.clip_dir / clip_filename
            cv2.imwrite(str(clip_path), clipped)
            
            return f"Saved clip to {clip_path}"
        except Exception as e:
            return f"Error saving clip: {str(e)}"

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

    def launch(self):
        try:
            with gr.Blocks() as demo:
                with gr.Row():
                    image = gr.Image(label="Frame Viewer", type="filepath", interactive=True, tool="editor")
                    with gr.Column():
                        obj_id = gr.Dropdown(choices=self.ids, label="Select Object ID")
                        save_btn = gr.Button("💾 Save Box")
                        msg_box = gr.Textbox(label="Status", interactive=False)
                        prev_btn = gr.Button("⬅️ Prev")
                        next_btn = gr.Button("➡️ Next")
                        
                        # Add clipping tools
                        gr.Markdown("### Clipping Tools")
                        clip_btn = gr.Button("✂️ Clip Selected Region")
                        clip_status = gr.Textbox(label="Clip Status", interactive=False)

                save_btn.click(self.save_annotations, inputs=[image, obj_id], outputs=msg_box)
                next_btn.click(self.next_frame, outputs=image)
                prev_btn.click(self.prev_frame, outputs=image)
                clip_btn.click(self.clip_and_save, inputs=[image], outputs=clip_status)
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