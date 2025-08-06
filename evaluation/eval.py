import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
import argparse
from tkinter import ttk

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


class Evaluator(ImageViewer):
    def __init__(self, image_dir='Loc01/', current_subject_index=0):
        super().__init__(image_dir)
        # Identity subjects list
        self.IDENTITY_SUBJECT = (['nurse_1','nurse_2',
            'patient_1','patient_2','patient_3',
            'psychiatrist','psychologist','researcher',
            'person_1','person_2','person_3','person_4'])
        
        # Identity selection parameters
        self.current_subject_index = current_subject_index
        self.min_subject_index = 0
        self.max_subject_index = len(self.IDENTITY_SUBJECT) - 1
        self.identity_var = None
        self.identity_menu = None

    def evaluate(self):
        """
        Evaluation functionality that uses the selected identity.
        """
        selected_identity = self.get_selected_identity_name()
        selected_identity_index = self.get_selected_identity_index()
        print(f"Evaluating image at second {self.current_second} for identity: {selected_identity} (index: {selected_identity_index})")
        # Add your evaluation logic here
        # You can now use selected_identity and selected_identity_index in your evaluation

    def load_evaluation_buttons(self):
        """
        Load buttons for evaluation metrics tracking.
        """
        # Create a frame for evaluation buttons
        eval_frame = tk.Frame(self.root)
        eval_frame.pack(side=tk.LEFT, pady=10, padx=10)
        
        # Add labels for the evaluation section
        eval_label = tk.Label(eval_frame, text="Evaluation Metrics", font=("Arial", 12, "bold"))
        eval_label.pack(pady=5)
        
        # Create buttons for each metric
        self.correct_btn = tk.Button(eval_frame, text="Correct", command=self.mark_correct, bg="green", fg="white")
        self.correct_btn.pack(pady=2)
        
        self.incorrect_btn = tk.Button(eval_frame, text="Incorrect", command=self.mark_incorrect, bg="red", fg="white")
        self.incorrect_btn.pack(pady=2)
        
        self.missing_btn = tk.Button(eval_frame, text="Missing", command=self.mark_missing, bg="orange", fg="white")
        self.missing_btn.pack(pady=2)
        
        # Add a button to show results
        self.show_results_btn = tk.Button(eval_frame, text="Show Results", command=self.show_results, bg="blue", fg="white")
        self.show_results_btn.pack(pady=5)
        
        # Initialize evaluation data
        self.evaluation_data = {}
        for identity in self.IDENTITY_SUBJECT:
            self.evaluation_data[identity] = {
                'correct': 0,
                'incorrect': 0,
                'missing': 0,
                'total_seconds': 180  # Default total seconds
            }

    def mark_correct(self):
        """
        Mark the current evaluation as correct for the selected identity.
        """
        identity = self.get_selected_identity_name()
        self.evaluation_data[identity]['correct'] += 1
        print(f"Marked as CORRECT for {identity} at second {self.current_second}")

    def mark_incorrect(self):
        """
        Mark the current evaluation as incorrect for the selected identity.
        """
        identity = self.get_selected_identity_name()
        self.evaluation_data[identity]['incorrect'] += 1
        print(f"Marked as INCORRECT for {identity} at second {self.current_second}")

    def mark_missing(self):
        """
        Mark the current evaluation as missing for the selected identity.
        """
        identity = self.get_selected_identity_name()
        self.evaluation_data[identity]['missing'] += 1
        print(f"Marked as MISSING for {identity} at second {self.current_second}")

    def calculate_accuracy(self, correct, incorrect, missing):
        """
        Calculate accuracy based on correct, incorrect, and missing values.
        """
        total = correct + incorrect + missing
        if total == 0:
            return 0.0
        return round(correct / total, 3)

    def show_results(self):
        """
        Display evaluation results in a table format.
        """
        # Create a new window for results
        results_window = tk.Toplevel(self.root)
        results_window.title("Evaluation Results")
        results_window.geometry("800x600")
        
        # Create a frame for the table
        table_frame = tk.Frame(results_window)
        table_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        # Create headers
        headers = ["Name", "Accuracy", "Correct", "Incorrect", "Missing", "Total Seconds"]
        for i, header in enumerate(headers):
            label = tk.Label(table_frame, text=header, font=("Arial", 10, "bold"), 
                           relief=tk.RAISED, borderwidth=2, width=12)
            label.grid(row=0, column=i, sticky="ew", padx=1, pady=1)
        
        # Populate data rows
        row = 1
        for identity in self.IDENTITY_SUBJECT:
            data = self.evaluation_data[identity]
            correct = data['correct']
            incorrect = data['incorrect']
            missing = data['missing']
            total_seconds = data['total_seconds']
            accuracy = self.calculate_accuracy(correct, incorrect, missing)
            
            # Determine if accuracy should be bold (high performance)
            accuracy_text = f"{accuracy:.3f}"
            if accuracy >= 0.8:  # Bold high accuracy values
                accuracy_text = f"**{accuracy:.3f}**"
            
            # Create row data
            row_data = [identity, accuracy_text, str(correct), str(incorrect), str(missing), str(total_seconds)]
            
            for i, value in enumerate(row_data):
                # Use bold font for high accuracy values
                font_weight = "bold" if i == 1 and accuracy >= 0.8 else "normal"
                label = tk.Label(table_frame, text=value, font=("Arial", 9, font_weight),
                               relief=tk.SUNKEN, borderwidth=1, width=12)
                label.grid(row=row, column=i, sticky="ew", padx=1, pady=1)
            
            row += 1
        
        # Configure grid weights
        for i in range(len(headers)):
            table_frame.columnconfigure(i, weight=1)

    def load_buttons(self):
        super().load_buttons()
        # Add evaluate button
        self.evaluate_btn = tk.Button(self.root, text="Evaluate", command=self.evaluate)
        self.evaluate_btn.pack(side=tk.LEFT, padx=10)
        
        # Load evaluation buttons
        self.load_evaluation_buttons()

    def identity_selection(self):
        """
        This function initializes the identity selection menu for the GUI interface.
        It sets the default identity to the first subject in the list and creates a combobox
        with all the subjects as options. The combobox is then packed to the right side of the window.
        """
        self.identity_var = tk.StringVar()
        self.identity_var.set(self.IDENTITY_SUBJECT[self.current_subject_index])
        self.identity_menu = ttk.Combobox(self.root, textvariable=self.identity_var, values=self.IDENTITY_SUBJECT, state="readonly")
        self.identity_menu.pack(side=tk.RIGHT, pady=10)

    def get_selected_identity_index(self):
        """
        Returns the index of the currently selected identity in the IDENTITY_SUBJECT list.
        """
        return self.IDENTITY_SUBJECT.index(self.identity_var.get())

    def get_selected_identity_name(self):
        """
        Returns the name of the currently selected identity.
        """
        return self.identity_var.get()

    def run(self):
        """
        Main UI setup and execution with identity selection
        """
        self.root = tk.Tk()
        self.root.title(f"Evaluator - {self.image_dir}")
        self.root.geometry("1280x920")

        # Bind left and right arrow keys to navigation functions
        self.root.bind('<Left>', lambda event: self.previous_image())
        self.root.bind('<Right>', lambda event: self.next_image())

        self.canvas = tk.Canvas(self.root, width=1920*self.RESIZE, height=1080*self.RESIZE)
        self.canvas.pack()

        # Load all necessary buttons
        self.load_buttons()

        # Adding identity selection on the side
        self.identity_selection()

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="Loc01/")
    parser.add_argument("--current_subject_index", type=int, default=0)
    args = parser.parse_args()

    image_dir = args.image_dir
    current_subject_index = args.current_subject_index
    viewer = Evaluator(image_dir, current_subject_index)
    viewer.run()

if __name__ == "__main__":
    main()