import json
import os
import cv2
import argparse
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
        
class BBoxDisplay:
    def __init__(self, img_path, json_path, frame_id=1):
        self.json_path = json_path
        self.img_dir = os.path.dirname(img_path)
        self.frame_id = frame_id
        self.data = self.load_json()
        self.available_frames = self.get_available_frames_with_images()
        self.root = None
        self.tk_img = None
        self.label = None
        self.frame_label = None

    def load_json(self):
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def get_available_frames_with_images(self):
        # Only include frames that have both JSON and image file
        frames = []
        for key in self.data.keys():
            if key.startswith('frame_') and key.endswith('.png'):
                try:
                    num = int(key[6:10])
                    img_path = os.path.join(self.img_dir, f'frame_{num:04d}.png')
                    if os.path.exists(img_path):
                        frames.append(num)
                except ValueError:
                    continue
        frames.sort()
        return frames

    def load_bboxes(self, frame_id=None):
        if frame_id is None:
            frame_id = self.frame_id
        frame_key = f"frame_{frame_id:04d}.png"
        if frame_key in self.data:
            bboxes = [obj['coordinates'] for obj in self.data[frame_key]]
            if not bboxes:
                print(f"No bounding boxes found for {frame_key}.")
            return bboxes
        else:
            print(f"Frame key {frame_key} not found in JSON.")
        return []

    def display(self):
        self.root = tk.Tk()
        self.root.title('BBox Display')
        self.update_image()
        # Add navigation buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, pady=10)
        prev_btn = tk.Button(btn_frame, text="Previous", command=self.show_previous_frame)
        prev_btn.pack(side=tk.LEFT, padx=10)
        next_btn = tk.Button(btn_frame, text="Next", command=self.show_next_frame)
        next_btn.pack(side=tk.LEFT, padx=10)
        # Add frame number label
        self.frame_label = tk.Label(btn_frame, text=f"Frame: {self.frame_id}")
        self.frame_label.pack(side=tk.LEFT, padx=10)
        self.root.mainloop()

    def update_image(self):
        img_path = f'{self.img_dir}/frame_{self.frame_id:04d}.png'
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist.")
            return
        image = Image.open(img_path).convert('RGB')
        bboxes = self.load_bboxes(self.frame_id)
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        self.tk_img = ImageTk.PhotoImage(image)
        if self.label is None:
            self.label = tk.Label(self.root, image=self.tk_img)
            self.label.pack()
        else:
            self.label.configure(image=self.tk_img)
            self.label.image = self.tk_img
        self.root.title(f'BBox Display - Frame {self.frame_id}')
        if self.frame_label is not None:
            self.frame_label.config(text=f"Frame: {self.frame_id}")

    def find_next_frame(self):
        # Find the next available frame after current (with image)
        for f in self.available_frames:
            if f > self.frame_id:
                return f
        return self.frame_id  # If none, stay

    def find_previous_frame(self):
        # Find the previous available frame before current (with image)
        for f in reversed(self.available_frames):
            if f < self.frame_id:
                return f
        return self.frame_id  # If none, stay

    def show_next_frame(self):
        next_frame = self.find_next_frame()
        if next_frame != self.frame_id:
            self.frame_id = next_frame
            self.update_image()

    def show_previous_frame(self):
        prev_frame = self.find_previous_frame()
        if prev_frame != self.frame_id:
            self.frame_id = prev_frame
            self.update_image()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='clip1', required=True)
    parser.add_argument('--frame', type=int, default=1, required=False)
    args = parser.parse_args()
    json_path = f'yolo_results_train_json/{args.name}_with_missing.json'
    img_dir = f'output_bounding_boxes_{args.name}'
    img_path = f'{img_dir}/frame_{args.frame:04d}.png'
    bbox_display = BBoxDisplay(img_path, json_path, frame_id=args.frame)
    bbox_display.display()


