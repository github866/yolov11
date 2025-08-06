import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
import argparse

class ImageViewer:
    def __init__(self, image_dir='Loc01/'):
        """
        Simple image viewer for displaying images from a directory with second-based navigation.
        
        Features:
        - Displays images from a specified directory
        - Navigate between images using Previous/Next buttons or arrow keys
        - Shows current second number
        - Supports seconds_XXXX.png naming convention
        
        Usage:
        - Run the tool to view images in sequence
        - Use Previous/Next buttons or left/right arrow keys to navigate
        """
        
        # Load the image directory
        self.image_dir = image_dir
        # Load the image paths
        self.image_paths = self.load_images(image_dir)
        # Set the current second to 0
        self.current_second = 0
        self.end_second = len(self.image_paths) - 1

        # Constant for image to fit in UI
        self.RESIZE = 2/3
        
        print(f"Loaded {len(self.image_paths)} images from {image_dir}")
    
    @staticmethod
    def load_images(image_dir):
        # Get all files and filter for seconds_XXXX pattern
        all_files = os.listdir(image_dir)
        image_paths = [f for f in all_files if f.startswith('seconds_') and (f.endswith('.png') or f.endswith('.jpg'))]
        # Sort by second number
        image_paths = sorted(image_paths, key=lambda x: int(x.split('_')[1].split('.')[0]))
        return image_paths

    def load_buttons(self):
        '''
        This function loads the buttons for the GUI interface.
        It includes navigation buttons (Previous and Next) to move through the images,
        an entry field and a button to directly select and display a specific second,
        and a label to display the current second number.
        '''
        # Add navigation buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, pady=10)
        prev_btn = tk.Button(btn_frame, text="Previous", command=self.previous_image)
        prev_btn.pack(side=tk.LEFT, padx=10)
        next_btn = tk.Button(btn_frame, text="Next", command=self.next_image)
        next_btn.pack(side=tk.LEFT, padx=10)

        # Add entry and button for selecting a second
        self.second_entry = tk.Entry(btn_frame, width=5)
        self.second_entry.pack(side=tk.LEFT, padx=10)
        display_select_second_btn = tk.Button(btn_frame, text="Display Select Second", command=self.display_select_second)
        display_select_second_btn.pack(side=tk.LEFT, padx=10)
        
        # Display second number
        self.second_number_label = tk.Label(btn_frame, text=f"Second: {self.current_second}")
        self.second_number_label.pack(side=tk.LEFT, padx=10)

    def run(self):
        """
        Main UI setup and execution
        """
        self.root = tk.Tk()
        self.root.title(f"Image Viewer - {self.image_dir}")
        self.root.geometry("1280x920")

        # Bind left and right arrow keys to navigation functions
        self.root.bind('<Left>', lambda event: self.previous_image())
        self.root.bind('<Right>', lambda event: self.next_image())

        self.canvas = tk.Canvas(self.root, width=1920*self.RESIZE, height=1080*self.RESIZE)
        self.canvas.pack()

        # Load all necessary buttons
        self.load_buttons()

        # Load and display the first image as default
        if self.image_paths:
            self.image = Image.open(os.path.join(self.image_dir, self.image_paths[self.current_second]))
            # Resize the image for display
            display_width = int(self.image.width * self.RESIZE)
            display_height = int(self.image.height * self.RESIZE)
            self.display_image = self.image.resize((display_width, display_height), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.display_image)
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.root.mainloop()

    def next_image(self):
        self.current_second += 1
        if self.current_second > self.end_second:
            self.current_second = 0
        self.update_image()

    def previous_image(self):
        self.current_second -= 1
        if self.current_second < 0:
            self.current_second = self.end_second
        self.update_image()

    def display_select_second(self):
        try:
            second_number = int(self.second_entry.get())
            if 0 <= second_number <= self.end_second:
                self.current_second = second_number
            else:
                self.current_second = 0
        except ValueError:
            self.current_second = 0
        self.update_image()

    def update_image(self):
        if self.image_paths:
            self.image = Image.open(os.path.join(self.image_dir, self.image_paths[self.current_second]))
            # Resize the image for display
            display_width = int(self.image.width * self.RESIZE)
            display_height = int(self.image.height * self.RESIZE)
            self.display_image = self.image.resize((display_width, display_height), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.display_image)
            self.canvas.itemconfig(self.image_on_canvas, image=self.photo)
            # Display second number
            self.second_number_label.config(text=f"Second: {self.current_second}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="Loc01/")
    args = parser.parse_args()

    image_dir = args.image_dir
    viewer = ImageViewer(image_dir)
    viewer.run()

if __name__ == "__main__":
    main()