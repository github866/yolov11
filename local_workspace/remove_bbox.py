import json
import os
import cv2
import argparse
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from tkinter import messagebox
        
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
        self.listbox = None
        self.scrollbar = None
        self.bbox_indices = []  # To map listbox index to bbox index
        self.selected_bbox_index = None
        self.modified_data = json.loads(json.dumps(self.data))  # Deep copy for modifications
        self.new_json_path = self.json_path.replace('.json', '_removed.json')

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
        # Add image and bbox list
        img_frame = tk.Frame(self.root)
        img_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.update_image()
        # Listbox for bboxes
        list_frame = tk.Frame(self.root)
        list_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        tk.Label(list_frame, text="Bounding Boxes").pack()
        self.scrollbar = tk.Scrollbar(list_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(list_frame, height=20, width=30, yscrollcommand=self.scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        self.scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self.on_bbox_select)
        self.populate_bbox_list()
        # Delete button
        del_btn = tk.Button(list_frame, text="Delete Selected", command=self.delete_selected_bbox)
        del_btn.pack(pady=10)
        # Navigation buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, pady=10)
        prev_btn = tk.Button(btn_frame, text="Previous", command=self.show_previous_frame)
        prev_btn.pack(side=tk.LEFT, padx=10)
        next_btn = tk.Button(btn_frame, text="Next", command=self.show_next_frame)
        next_btn.pack(side=tk.LEFT, padx=10)
        self.frame_label = tk.Label(btn_frame, text=f"Frame: {self.frame_id}")
        self.frame_label.pack(side=tk.LEFT, padx=10)
        self.root.mainloop()

    def populate_bbox_list(self):
        self.listbox.delete(0, tk.END)
        self.bbox_indices = []
        frame_key = f"frame_{self.frame_id:04d}.png"
        bboxes = self.modified_data.get(frame_key, [])
        for idx, obj in enumerate(bboxes):
            coords = obj['coordinates']
            self.listbox.insert(tk.END, f"BBox {idx+1}: {coords}")
            self.bbox_indices.append(idx)
        self.selected_bbox_index = None

    def on_bbox_select(self, event):
        selection = self.listbox.curselection()
        if selection:
            self.selected_bbox_index = self.bbox_indices[selection[0]]
            self.update_image(highlight_bbox=self.selected_bbox_index)
        else:
            self.selected_bbox_index = None
            self.update_image()

    def delete_selected_bbox(self):
        if self.selected_bbox_index is None:
            messagebox.showwarning("No selection", "Please select a bounding box to delete.")
            return
        frame_key = f"frame_{self.frame_id:04d}.png"
        bboxes = self.modified_data.get(frame_key, [])
        if 0 <= self.selected_bbox_index < len(bboxes):
            del bboxes[self.selected_bbox_index]
            self.modified_data[frame_key] = bboxes
            self.save_modified_json()
            self.populate_bbox_list()
            self.update_image()
            self.selected_bbox_index = None
        else:
            messagebox.showerror("Error", "Invalid bounding box selection.")

    def save_modified_json(self):
        with open(self.new_json_path, 'w') as f:
            json.dump(self.modified_data, f, indent=2)

    def update_image(self, highlight_bbox=None):
        img_path = f'{self.img_dir}/frame_{self.frame_id:04d}.png'
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist.")
            return
        image = Image.open(img_path).convert('RGB')
        frame_key = f"frame_{self.frame_id:04d}.png"
        bboxes = [obj['coordinates'] for obj in self.modified_data.get(frame_key, [])]
        draw = ImageDraw.Draw(image)
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            color = 'red'
            width = 3
            if highlight_bbox is not None and idx == highlight_bbox:
                color = 'yellow'
                width = 5
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        self.tk_img = ImageTk.PhotoImage(image)
        if self.label is None:
            self.label = tk.Label(self.root, image=self.tk_img)
            self.label.pack(side=tk.LEFT)
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
            self.populate_bbox_list()
            self.update_image()

    def show_previous_frame(self):
        prev_frame = self.find_previous_frame()
        if prev_frame != self.frame_id:
            self.frame_id = prev_frame
            self.populate_bbox_list()
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


