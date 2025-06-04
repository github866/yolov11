import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
import json
import argparse
from tkinter import ttk

class Cropper:
    def __init__(self, image_dir='frames/',json_name='crops.json'):
        """
        Cropper is a graphical tool for interactively cropping regions from a sequence of images.

        Features:
        - Displays images from a specified directory and allows navigation between frames.
        - Lets users select rectangular crop regions using the mouse.
        - Supports saving multiple crop regions per image, with coordinates stored in a single JSON file.
        - Includes a dropdown menu for annotating each crop with an identity label (e.g., nurse, patient, etc.).
        - User interface built with Tkinter, including navigation buttons, frame selection, and identity selection.

        Usage:
        - Run the tool, select frames, draw crop rectangles, choose an identity, and save crops.
        - All crop data is saved in a structured JSON file for easy downstream processing.

        This tool is useful for preparing datasets for computer vision tasks, such as object detection or tracking, where precise region annotation and identity labeling are required.
        """
        
        # load the image directory
        self.image_dir = image_dir
        # load the image paths
        self.image_paths = self.load_image(image_dir)
        # set the current frame to 1
        self.current_frame = 0
        self.end_frame = len(self.image_paths) - 1

        # crop the image parameters
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.crop_coords = None

        # loading cropped data
        self.cropped_dir = os.path.join(os.getcwd(), 'cropped_images')
        if not os.path.exists(self.cropped_dir):
            os.makedirs(self.cropped_dir)
        self.json_name = json_name
        self.json_path = os.path.join(self.cropped_dir, self.json_name)
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w') as f:
                json.dump([], f)

        # loading the provided identity subjects
        self.IDENTITY_SUBJECT = (['nurse_1','nurse_2',
            'patient_1','patient_2','patient_3',
            'psychiatrist','psychologist','researcher',
            'person_1','person_2','person_3','person_4'])
        self.current_subject_index = 0
        self.min_subject_index = 0
        self.max_subject_index = len(self.IDENTITY_SUBJECT) - 1
        self.identity_var = None
        self.identity_menu = None

        # Constant for image to fit in UI
        self.RESIZE = 2/3

    @staticmethod
    def load_image(image_dir):
        # print the image name in sequence
        image_paths = sorted(os.listdir(image_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
        # for image_path in image_paths:
        #     image = cv2.imread(os.path.join(image_dir, image_path))
        #     # print(image_dir + image_path)
        return image_paths


    def load_buttons(self):
        '''
        This function loads the buttons for the GUI interface.
        It includes navigation buttons (Previous and Next) to move through the images,
        an entry field and a button to directly select and display a specific frame,
        a label to display the current frame number,
        and a button to display the selected identity.
        '''
        # Add navigation buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, pady=10)
        prev_btn = tk.Button(btn_frame, text="Previous", command=self.previous_image)
        prev_btn.pack(side=tk.LEFT, padx=10)
        next_btn = tk.Button(btn_frame, text="Next", command=self.next_image)
        next_btn.pack(side=tk.LEFT, padx=10)

        # Add entry and button for selecting a frame
        self.frame_entry = tk.Entry(btn_frame, width=5)
        self.frame_entry.pack(side=tk.LEFT, padx=10)
        display_select_frame_btn = tk.Button(btn_frame, text="Display Select Frame", command=self.display_select_frame)
        display_select_frame_btn.pack(side=tk.LEFT, padx=10)
        
        # Display frame number
        self.frames_number_label = tk.Label(btn_frame, text=f"Frame: {self.current_frame}")
        self.frames_number_label.pack(side=tk.LEFT, padx=10)

        # display_identity_btn = tk.Button(btn_frame, text="Display Identity", command=self.get_selected_identity_index)
        # display_identity_btn.pack(side=tk.LEFT, padx=10)

    def identity_selection(self):
        """
        This function initializes the identity selection menu for the GUI interface.
        It sets the default identity to the first subject in the list and creates a combobox
        with all the subjects as options. The combobox is then packed to the right side of the window.
        """
        self.identity_var = tk.StringVar()
        self.identity_var.set(self.IDENTITY_SUBJECT[0])
        self.identity_menu = ttk.Combobox(self.root, textvariable=self.identity_var, values=self.IDENTITY_SUBJECT, state="readonly")
        self.identity_menu.pack(side=tk.RIGHT, pady=10)

    def get_selected_identity_index(self):
        # print(self.IDENTITY_SUBJECT.index(self.identity_var.get()))
        return self.IDENTITY_SUBJECT.index(self.identity_var.get())

    def get_selected_identity_name(self):
        # print(self.identity_var.get())
        return self.identity_var.get()
        

    def run(self):
        """
        UIs are here
        """
        self.root = tk.Tk()
        self.root.title(self.image_dir)
        self.root.geometry("1280x920")

        self.canvas = tk.Canvas(self.root, width=1920*self.RESIZE, height=1080*self.RESIZE)
        self.canvas.pack()

        # load all necessary buttons
        self.load_buttons()

        # Adding identity selection on the side
        self.identity_selection()

        # Load and display the first image as default
        self.image = Image.open(os.path.join(self.image_dir, self.image_paths[self.current_frame]))
        # Resize the image for display
        display_width = int(self.image.width * self.RESIZE)
        display_height = int(self.image.height * self.RESIZE)
        self.display_image = self.image.resize((display_width, display_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.crop_image()

        self.root.mainloop()
    def crop_image(self):
        self.rect = None
        self.handles = []
        self.active_handle = None
        self.start_x = self.start_y = None
        self.crop_coords = None
        HANDLE_SIZE = 8

        def draw_handles(x1, y1, x2, y2):
            # Remove old handles
            for h in self.handles:
                self.canvas.delete(h)
            self.handles = []
            # Draw new handles at corners
            for (cx, cy) in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                handle = self.canvas.create_rectangle(
                    cx-HANDLE_SIZE, cy-HANDLE_SIZE, cx+HANDLE_SIZE, cy+HANDLE_SIZE,
                    fill='blue', outline='white'
                )
                self.handles.append(handle)

        def get_handle_at(x, y):
            for idx, handle in enumerate(self.handles):
                hx1, hy1, hx2, hy2 = self.canvas.coords(handle)
                if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                    return idx
            return None

        def on_button_press(event):
            handle_idx = get_handle_at(event.x, event.y)
            if handle_idx is not None:
                self.active_handle = handle_idx
            else:
                self.start_x, self.start_y = event.x, event.y
                if self.rect:
                    self.canvas.delete(self.rect)
                for h in self.handles:
                    self.canvas.delete(h)
                self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')
                self.handles = []
                self.active_handle = None

        def on_move_press(event):
            # Clamp event.x and event.y to canvas size
            max_x = int(self.image.width * self.RESIZE)
            max_y = int(self.image.height * self.RESIZE)
            clamped_x = max(0, min(event.x, max_x - 1))
            clamped_y = max(0, min(event.y, max_y - 1))
            scale = 1.0 / self.RESIZE
            if self.active_handle is not None and self.rect:
                x1, y1, x2, y2 = self.canvas.coords(self.rect)
                coords = [x1, y1, x2, y2]
                if self.active_handle == 0:  # top-left
                    coords[0], coords[1] = clamped_x, clamped_y
                elif self.active_handle == 1:  # top-right
                    coords[2], coords[1] = clamped_x, clamped_y
                elif self.active_handle == 2:  # bottom-right
                    coords[2], coords[3] = clamped_x, clamped_y
                elif self.active_handle == 3:  # bottom-left
                    coords[0], coords[3] = clamped_x, clamped_y
                self.canvas.coords(self.rect, *coords)
                draw_handles(*coords)
                self.crop_coords = (int(min(coords[0], coords[2])), int(min(coords[1], coords[3])),
                                    int(max(coords[0], coords[2])), int(max(coords[1], coords[3])))
                # Calculate original coordinates
                rx1, ry1, rx2, ry2 = self.crop_coords
                ox1 = int(round(rx1 * scale))
                oy1 = int(round(ry1 * scale))
                ox2 = int(round(rx2 * scale))
                oy2 = int(round(ry2 * scale))
                if hasattr(self, 'coords_label'):
                    self.coords_label.config(text=f"Crop: <{rx1},{ry1},{rx2},{ry2}> | Original: <{ox1},{oy1},{ox2},{oy2}>")
            elif self.rect:
                self.canvas.coords(self.rect, self.start_x, self.start_y, clamped_x, clamped_y)

        def on_button_release(event):
            max_x = int(self.image.width * self.RESIZE)
            max_y = int(self.image.height * self.RESIZE)
            scale = 1.0 / self.RESIZE
            if self.rect:
                x1, y1, x2, y2 = self.canvas.coords(self.rect)
                # Clamp all coordinates
                x1 = max(0, min(x1, max_x - 1))
                y1 = max(0, min(y1, max_y - 1))
                x2 = max(0, min(x2, max_x - 1))
                y2 = max(0, min(y2, max_y - 1))
                self.crop_coords = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
                draw_handles(*self.crop_coords)
                rx1, ry1, rx2, ry2 = self.crop_coords
                ox1 = int(round(rx1 * scale))
                oy1 = int(round(ry1 * scale))
                ox2 = int(round(rx2 * scale))
                oy2 = int(round(ry2 * scale))
                if hasattr(self, 'coords_label'):
                    self.coords_label.config(text=f"Crop: <{rx1},{ry1},{rx2},{ry2}> | Original: <{ox1},{oy1},{ox2},{oy2}>")
                else:
                    self.coords_label = tk.Label(self.root, text=f"Crop: <{rx1},{ry1},{rx2},{ry2}> | Original: <{ox1},{oy1},{ox2},{oy2}>")
                    self.coords_label.pack(side=tk.BOTTOM, pady=5)
            self.active_handle = None

        self.canvas.bind('<ButtonPress-1>', on_button_press)
        self.canvas.bind('<B1-Motion>', on_move_press)
        self.canvas.bind('<ButtonRelease-1>', on_button_release)

        # Add a button to save the crop coordinates
        def save_crop():
            '''
            This function is used to save the crop coordinates to the JSON file.
            It also checks if the person already exists in the JSON file and updates the coordinate if it does.
            If the person does not exist, it adds a new person to the JSON file.
            '''
            if self.crop_coords:
                x1, y1, x2, y2 = self.crop_coords
                # Scale coordinates back to original image size
                scale = 1.0 / self.RESIZE
                orig_x1 = int(round(x1 * scale))
                orig_y1 = int(round(y1 * scale))
                orig_x2 = int(round(x2 * scale))
                orig_y2 = int(round(y2 * scale))
                # Clamp to original image size
                orig_x1 = max(0, min(orig_x1, self.image.width - 1))
                orig_y1 = max(0, min(orig_y1, self.image.height - 1))
                orig_x2 = max(0, min(orig_x2, self.image.width - 1))
                orig_y2 = max(0, min(orig_y2, self.image.height - 1))
                crop = [orig_x1, orig_y1, orig_x2, orig_y2]
                frame_number = self.current_frame + 1  # 1-based
                subject_id = self.get_selected_identity_index()
                subject_name = self.get_selected_identity_name()
                data = []
                if os.path.exists(self.json_path):
                    with open(self.json_path, 'r') as f:
                        try:
                            data = json.load(f)
                        except Exception:
                            data = []
                found = False
                for entry in data:
                    if entry.get('frame') == frame_number:
                        person_found = False
                        for person in entry.setdefault('persons', []):
                            if person['subject_id'] == subject_id:
                                person['coordinate'] = crop
                                person_found = True
                                break
                        if not person_found:
                            entry['persons'].append({
                                "subject_id": subject_id,
                                "subject_name": subject_name,
                                "coordinate": crop
                            })
                        found = True
                        break
                if not found:
                    data.append({
                        "frame": frame_number,
                        "persons": [{
                            "subject_id": subject_id,
                            "subject_name": subject_name,
                            "coordinate": crop
                        }]
                    })
                with open(self.json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                if hasattr(self, 'coords_label'):
                    self.coords_label.config(text=f"Saved: <{orig_x1},{orig_y1},{orig_x2},{orig_y2}> | Resized: <{x1},{y1},{x2},{y2}>")

        if not hasattr(self, 'save_btn'):
            self.save_btn = tk.Button(self.root, text="Save Crop", command=save_crop)
            self.save_btn.pack(side=tk.BOTTOM, pady=5)

        self.canvas.bind('<ButtonPress-1>', on_button_press)
        self.canvas.bind('<B1-Motion>', on_move_press)
        self.canvas.bind('<ButtonRelease-1>', on_button_release)

    def next_image(self):
        self.current_frame += 1
        if self.current_frame > self.end_frame:
            self.current_frame = 0
        self.update_image()

    def previous_image(self):
        self.current_frame -= 1
        if self.current_frame < 0:
            self.current_frame = 0
        self.update_image()

    def display_select_frame(self):
        try:
            frame_number = int(self.frame_entry.get())
            # User enters 1 for frame_0001.png, so subtract 1 for zero-based index
            if 1 <= frame_number <= self.end_frame + 1:
                self.current_frame = frame_number - 1
            else:
                self.current_frame = 0
        except ValueError:
            self.current_frame = 0
        self.update_image()

    def update_image(self):
        self.image = Image.open(os.path.join(self.image_dir, self.image_paths[self.current_frame]))
        # Resize the image for display
        display_width = int(self.image.width * self.RESIZE)
        display_height = int(self.image.height * self.RESIZE)
        self.display_image = self.image.resize((display_width, display_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo)
        # Display frame number as 1-based for user clarity
        self.frames_number_label.config(text=f"Frame: {self.current_frame + 1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="frames/")
    parser.add_argument("--output_dir", type=str, default="cropped_images")
    parser.add_argument("--json_name", type=str, default="crops.json")
    args = parser.parse_args()
    
    image_dir = args.image_dir

    cropper = Cropper(args.image_dir,args.json_name)
    cropper.run()

if __name__ == "__main__":
    main()