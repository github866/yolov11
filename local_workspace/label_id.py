import json
import os
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from tkinter import ttk, messagebox

IDENTITY_SUBJECT = [
    'Unlabeled',
    'nurse_1','nurse_2',
    'patient_1','patient_2','patient_3',
    'psychiatrist','psychologist','researcher',
    'person_1','person_2','person_3','person_4'
]

class IDLabeler:
    def __init__(self, img_dir, json_path, output_json, prefill_json=None, frame_id=1):
        self.img_dir = img_dir
        self.json_path = json_path
        self.output_json = output_json
        self.prefill_json = prefill_json
        self.frame_id = frame_id
        self.data = self.load_json(self.json_path)
        self.available_frames = self.get_available_frames_with_images()
        self.prefill = self.load_prefill(self.prefill_json) if self.prefill_json else {}
        self.root = None
        self.tk_img = None
        self.label = None
        self.frame_label = None
        self.bbox_rows_frame = None
        self.bbox_row_widgets = []
        self.selected_bbox_index = None
        self.img_frame = None
        self.subject_vars = []
        self.modified = {}
        self.load_existing_labels()
        self.bbox_coords_cache = []  # For click selection

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def load_prefill(self, path):
        with open(path, 'r') as f:
            arr = json.load(f)
        prefill = {}
        for entry in arr:
            frame = entry['frame']
            prefill[frame] = entry['persons']
        return prefill

    def get_available_frames_with_images(self):
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

    def load_existing_labels(self):
        if os.path.exists(self.output_json):
            with open(self.output_json, 'r') as f:
                arr = json.load(f)
            for entry in arr:
                self.modified[entry['frame']] = entry['persons']
        elif self.prefill:
            for frame, persons in self.prefill.items():
                self.modified[frame] = persons

    def display(self):
        self.root = tk.Tk()
        self.root.title('ID Labeler')
        self.root.configure(bg='#f0f0f0')
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        img_container = tk.Frame(left_frame, bg='white', relief=tk.RAISED, bd=2)
        img_container.pack(fill=tk.BOTH, expand=True)
        img_frame = tk.Frame(img_container, bg='white')
        img_frame.pack(fill=tk.BOTH, expand=True)
        self.img_frame = img_frame
        self.update_image()
        nav_frame = tk.Frame(img_container, bg='#e8e8e8', relief=tk.SUNKEN, bd=1)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        prev_btn = tk.Button(nav_frame, text="← Previous", command=self.show_previous_frame,
                           width=12, height=2, font=('Arial', 10, 'bold'),
                           bg='#4CAF50', fg='white', relief=tk.RAISED,
                           activebackground='#45a049', activeforeground='white')
        prev_btn.pack(side=tk.LEFT, padx=(10, 5), pady=5)
        self.frame_label = tk.Label(nav_frame, text=f"Frame: {self.frame_id}",
                                  font=('Arial', 12, 'bold'), bg='#2196F3', fg='white',
                                  relief=tk.RAISED, padx=15, pady=5)
        self.frame_label.pack(side=tk.LEFT, padx=10, pady=5)
        next_btn = tk.Button(nav_frame, text="Next →", command=self.show_next_frame,
                           width=12, height=2, font=('Arial', 10, 'bold'),
                           bg='#4CAF50', fg='white', relief=tk.RAISED,
                           activebackground='#45a049', activeforeground='white')
        next_btn.pack(side=tk.LEFT, padx=(5, 10), pady=5)
        save_btn = tk.Button(nav_frame, text="💾 Save", command=self.save_modified_json,
                           width=12, height=2, font=('Arial', 10, 'bold'),
                           bg='#2196F3', fg='white', relief=tk.RAISED,
                           activebackground='#1976D2', activeforeground='white')
        save_btn.pack(side=tk.LEFT, padx=10, pady=5)
        list_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        list_frame.pack(side=tk.RIGHT, padx=(20, 0), fill=tk.Y)
        title_label = tk.Label(list_frame, text="Bounding Boxes",
                             font=('Arial', 12, 'bold'), bg='#2196F3', fg='white',
                             relief=tk.RAISED, padx=10, pady=5)
        title_label.pack(fill=tk.X, pady=(0, 10))
        if self.bbox_rows_frame:
            self.bbox_rows_frame.destroy()
        self.bbox_rows_frame = tk.Frame(list_frame, bg='white')
        self.bbox_rows_frame.pack(fill=tk.BOTH, expand=True)
        self.subject_vars = []
        self.bbox_row_widgets = []
        self.populate_bbox_list()
        self.root.bind('<Left>', lambda event: self.show_previous_frame())
        self.root.bind('<Right>', lambda event: self.show_next_frame())
        self.root.bind('<s>', lambda event: self.save_modified_json())
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.mainloop()

    def populate_bbox_list(self):
        for widgets in self.bbox_row_widgets:
            for w in widgets:
                w.destroy()
        self.bbox_row_widgets = []
        self.subject_vars = []
        frame_key = f"frame_{self.frame_id:04d}.png"
        bboxes = self.data.get(frame_key, [])
        frame_persons = self.modified.get(self.frame_id, [])
        self.bbox_coords_cache = [obj['coordinates'] for obj in bboxes]
        for idx, obj in enumerate(bboxes):
            coords = obj['coordinates']
            match = None
            for p in frame_persons:
                if p['coordinate'] == coords:
                    match = p
                    break
            if match:
                subject_name = match['subject_name']
            else:
                subject_name = IDENTITY_SUBJECT[0]  # 'Unlabeled'
            var = tk.StringVar()
            var.set(subject_name)
            self.subject_vars.append(var)
            label = tk.Label(self.bbox_rows_frame, text=f"BBox {idx+1}: {coords}", font=('Arial', 10), bg='white')
            cb = ttk.Combobox(self.bbox_rows_frame, textvariable=var, values=IDENTITY_SUBJECT, state="readonly", width=15)
            cb.grid(row=idx, column=1, padx=5, pady=2, sticky='w')
            label.grid(row=idx, column=0, padx=5, pady=2, sticky='w')
            cb.bind('<<ComboboxSelected>>', lambda e, i=idx: self.on_subject_change(i))
            # Make label clickable to select bbox
            label.bind('<Button-1>', lambda e, i=idx: self.on_bbox_select(i))
            # Highlight if selected
            if self.selected_bbox_index == idx:
                label.configure(bg='#ffe082')
                cb.configure(style='Selected.TCombobox')
            else:
                label.configure(bg='white')
                cb.configure(style='TCombobox')
            self.bbox_row_widgets.append((label, cb))
        self.selected_bbox_index = None
        # Bind click event to image for bbox selection
        if self.label is not None:
            self.label.unbind('<Button-1>')
        if self.label is not None:
            self.label.bind('<Button-1>', self.on_image_click)

    def on_subject_change(self, idx):
        frame_key = f"frame_{self.frame_id:04d}.png"
        bboxes = self.data.get(frame_key, [])
        if idx >= len(bboxes):
            return
        coords = bboxes[idx]['coordinates']
        subject_name = self.subject_vars[idx].get()
        subject_id = IDENTITY_SUBJECT.index(subject_name)
        persons = self.modified.get(self.frame_id, [])
        
        # Check for duplicate subjects
        duplicate_found = False
        for p in persons:
            if p['subject_name'] == subject_name and p['coordinate'] != coords:
                print(f"WARNING: Subject '{subject_name}' already has another bounding box at {p['coordinate']}")
                print(f"         Trying to assign to {coords} - this will create a duplicate!")
                duplicate_found = True
                break
        
        # Update or add the person
        found = False
        for p in persons:
            if p['coordinate'] == coords:
                p['subject_id'] = subject_id
                p['subject_name'] = subject_name
                found = True
                break
        if not found:
            persons.append({
                'subject_id': subject_id,
                'subject_name': subject_name,
                'coordinate': coords
            })
        
        self.modified[self.frame_id] = persons
        
        # Debug: Print current state
        print(f"\nFrame {self.frame_id} current assignments:")
        for p in persons:
            print(f"  {p['subject_name']}: {p['coordinate']}")
        
        self.selected_bbox_index = idx
        self.populate_bbox_list()
        self.update_image(highlight_bbox=idx)

    def on_bbox_select(self, idx):
        self.selected_bbox_index = idx
        self.populate_bbox_list()
        self.update_image(highlight_bbox=idx)

    def on_image_click(self, event):
        # Map click to bbox
        if not self.bbox_coords_cache:
            return
        img_path = os.path.join(self.img_dir, f'frame_{self.frame_id:04d}.png')
        image = Image.open(img_path).convert('RGB')
        max_width, max_height = 1280, 720
        img_w, img_h = image.size
        scale = min(max_width / img_w, max_height / img_h, 1.0)
        click_x = int(event.x / scale)
        click_y = int(event.y / scale)
        for idx, bbox in enumerate(self.bbox_coords_cache):
            x1, y1, x2, y2 = bbox
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                self.on_bbox_select(idx)
                # Focus the corresponding combobox
                if idx < len(self.bbox_row_widgets):
                    self.bbox_row_widgets[idx][1].focus_set()
                break

    def validate_no_duplicates(self):
        """Check for duplicate subjects within each frame"""
        print("\n=== VALIDATING FOR DUPLICATES ===")
        has_duplicates = False
        for frame_id in sorted(self.modified.keys()):
            persons = self.modified[frame_id]
            subject_counts = {}
            for person in persons:
                subject_name = person['subject_name']
                if subject_name != 'Unlabeled':
                    if subject_name in subject_counts:
                        subject_counts[subject_name] += 1
                    else:
                        subject_counts[subject_name] = 1
            
            # Check for duplicates
            for subject_name, count in subject_counts.items():
                if count > 1:
                    print(f"ERROR: Frame {frame_id} has {count} bounding boxes for '{subject_name}'")
                    has_duplicates = True
                    # Show the coordinates
                    for person in persons:
                        if person['subject_name'] == subject_name:
                            print(f"  - {person['coordinate']}")
        
        if not has_duplicates:
            print("✓ No duplicates found!")
        return not has_duplicates

    def save_modified_json(self, event=None):
        # Validate before saving
        if not self.validate_no_duplicates():
            response = messagebox.askyesno("Duplicates Found", 
                                         "Duplicate subjects found in some frames. Save anyway?")
            if not response:
                return
        
        out = []
        for frame in sorted(self.modified.keys()):
            persons = [p for p in self.modified[frame] if p['subject_name'] != 'Unlabeled']
            if persons:
                out.append({
                    'frame': frame,
                    'persons': persons
                })
        with open(self.output_json, 'w') as f:
            json.dump(out, f, indent=2)
        messagebox.showinfo("Saved", f"Labels saved to {self.output_json}.")

    def update_image(self, highlight_bbox=None):
        img_path = os.path.join(self.img_dir, f'frame_{self.frame_id:04d}.png')
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist.")
            return
        image = Image.open(img_path).convert('RGB')
        frame_key = f"frame_{self.frame_id:04d}.png"
        bboxes = self.data.get(frame_key, [])
        draw = ImageDraw.Draw(image)
        for idx, obj in enumerate(bboxes):
            bbox = obj['coordinates']
            color = 'red'
            width = 3
            if highlight_bbox is not None and idx == highlight_bbox:
                color = 'yellow'
                width = 5
            draw.rectangle(bbox, outline=color, width=width)
            frame_persons = self.modified.get(self.frame_id, [])
            label = None
            for p in frame_persons:
                if p['coordinate'] == bbox and p['subject_name'] != 'Unlabeled':
                    label = p['subject_name']
                    break
            if label:
                draw.text((bbox[0], bbox[1]-15), str(label), fill=color)
        max_width, max_height = 1280, 720
        img_w, img_h = image.size
        scale = min(max_width / img_w, max_height / img_h, 1.0)
        if scale < 1.0:
            new_size = (int(img_w * scale), int(img_h * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(image)
        if self.label is None:
            self.label = tk.Label(self.img_frame, image=self.tk_img)
            self.label.pack(expand=True, fill=tk.BOTH)
        else:
            self.label.configure(image=self.tk_img)
            self.label.image = self.tk_img
        self.label.bind('<Button-1>', self.on_image_click)
        self.root.title(f'ID Labeler - Frame {self.frame_id}')
        if self.frame_label is not None:
            self.frame_label.config(text=f"Frame: {self.frame_id}")

    def find_next_frame(self):
        for f in self.available_frames:
            if f > self.frame_id:
                return f
        return self.frame_id

    def find_previous_frame(self):
        for f in reversed(self.available_frames):
            if f < self.frame_id:
                return f
        return self.frame_id

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
    img_dir = "clip1"
    json_path = "loc01_data/origin/clip1_with_missing.json"
    output_json = "loc01_data/cropped_images/clip1_missing.json"
    prefill_json = "loc01_data/cropped_images/clip1_missing.json"
    frame_id = 1
    labeler = IDLabeler(img_dir, json_path, output_json, prefill_json=prefill_json, frame_id=frame_id)
    labeler.display()
