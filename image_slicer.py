import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk

class ImageSlicer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Slicer Tool")
        
        # Configure window size
        self.root.geometry("1200x800")
        
        # Variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.image_scale = 1.0
        self.regions = []  # List of region polygons
        self.current_region_points = []  # Points of the region being drawn
        self.current_region_id = 0
        
        # Mouse coordinates
        self.prev_x = 0
        self.prev_y = 0
        
        # Drawing mode
        self.drawing_mode = "rectangle"  # "rectangle" or "select"
        
        # Create GUI components
        self.create_menu()
        self.create_toolbar()
        self.create_canvas()
        self.create_statusbar()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Export Slices", command=self.export_slices)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_exit)
        
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_toolbar(self):
        toolbar_frame = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Region control buttons
        self.btn_new_region = tk.Button(toolbar_frame, text="New Rectangle", command=self.start_new_region)
        self.btn_new_region.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.btn_delete_region = tk.Button(toolbar_frame, text="Delete Rectangle", command=self.delete_selected_region)
        self.btn_delete_region.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Export button
        self.btn_export = tk.Button(toolbar_frame, text="Export Slices", command=self.export_slices)
        self.btn_export.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Mode selection
        self.draw_mode_var = tk.StringVar(value="rectangle")
        self.rb_draw = tk.Radiobutton(toolbar_frame, text="Draw Mode", variable=self.draw_mode_var, 
                                     value="rectangle", command=self.set_draw_mode)
        self.rb_draw.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.rb_select = tk.Radiobutton(toolbar_frame, text="Select Mode", variable=self.draw_mode_var, 
                                       value="select", command=self.set_select_mode)
        self.rb_select.pack(side=tk.LEFT, padx=5, pady=2)
    
    def create_canvas(self):
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray", cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas scrollbars
        v_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = tk.Scrollbar(self.root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Region info panel
        self.region_info_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        self.region_info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        tk.Label(self.region_info_frame, text="Regions:").pack(anchor='w')
        
        self.region_listbox = tk.Listbox(self.region_info_frame, width=30, height=20)
        self.region_listbox.pack(fill=tk.BOTH, expand=True)
        self.region_listbox.bind("<<ListboxSelect>>", self.on_region_select)
    
    def create_statusbar(self):
        self.statusbar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        
        if file_path:
            self.image_path = file_path
            self.load_image()
    
    def load_image(self):
        if self.image_path and os.path.exists(self.image_path):
            # Clear previous data
            self.regions = []
            self.current_region_points = []
            self.canvas.delete("all")
            self.region_listbox.delete(0, tk.END)
            
            # Load image using OpenCV
            self.original_image = cv2.imread(self.image_path)
            self.display_image = self.original_image.copy()
            
            # Convert BGR to RGB
            self.display_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            
            # Resize image if too large
            h, w = self.display_image.shape[:2]
            max_size = 800
            
            if h > max_size or w > max_size:
                if h > w:
                    self.image_scale = max_size / h
                    new_h = max_size
                    new_w = int(w * self.image_scale)
                else:
                    self.image_scale = max_size / w
                    new_w = max_size
                    new_h = int(h * self.image_scale)
                
                self.display_image = cv2.resize(self.display_image, (new_w, new_h))
                self.statusbar.config(text=f"Loaded image: {self.image_path} (scaled to {new_w}x{new_h})")
            else:
                self.image_scale = 1.0
                self.statusbar.config(text=f"Loaded image: {self.image_path} ({w}x{h})")
            
            # Convert to PhotoImage for canvas
            self.photo_image = ImageTk.PhotoImage(image=Image.fromarray(self.display_image))
            
            # Reset canvas and display image
            self.canvas.config(scrollregion=(0, 0, self.photo_image.width(), self.photo_image.height()))
            self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW, tags="image")
    
    def start_new_region(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        # Set draw mode
        self.drawing_mode = "rectangle"
        self.draw_mode_var.set("rectangle")
        self.canvas.config(cursor="crosshair")
        
        self.statusbar.config(text="Click and drag to draw a rectangle")
    
    def delete_selected_region(self):
        selection = self.region_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No region selected")
            return
        
        index = selection[0]
        region = self.regions[index]
        
        # Delete from canvas
        self.canvas.delete(f"region_{region['id']}")
        
        # Remove from data structures
        self.regions.pop(index)
        
        # Update listbox
        self.region_listbox.delete(index)
        
        self.statusbar.config(text=f"Region '{region['name']}' deleted")
    
    def set_draw_mode(self):
        self.drawing_mode = "rectangle"
        self.canvas.config(cursor="crosshair")
        self.statusbar.config(text="Draw Mode: Click and drag to draw a rectangle")
    
    def set_select_mode(self):
        self.drawing_mode = "select"
        self.canvas.config(cursor="arrow")
        self.statusbar.config(text="Select Mode: Click to select regions")
    
    def on_mouse_down(self, event):
        if not self.image_path:
            return
            
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        if self.drawing_mode == "rectangle":
            # Start a new rectangle
            self.start_x, self.start_y = x, y
            self.current_rect = self.canvas.create_rectangle(
                x, y, x, y, outline="red", tags="temp_rect"
            )
        elif self.drawing_mode == "select":
            # Check if clicked inside any region polygon
            for i, region in enumerate(self.regions):
                points = region["points"]
                if self.point_in_rect(x, y, points):
                    self.region_listbox.selection_clear(0, tk.END)
                    self.region_listbox.selection_set(i)
                    self.region_listbox.see(i)
                    self.statusbar.config(text=f"Selected {region['name']}")
                    break
    
    def on_mouse_drag(self, event):
        if not self.image_path or self.drawing_mode != "rectangle" or not hasattr(self, 'current_rect'):
            return
        
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Update rectangle as user drags
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, x, y)
    
    def on_mouse_up(self, event):
        if not self.image_path or self.drawing_mode != "rectangle" or not hasattr(self, 'current_rect'):
            return
        
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Get rectangle coords
        x1, y1, x2, y2 = self.start_x, self.start_y, x, y
        
        # Make sure x1,y1 is the top-left and x2,y2 is the bottom-right
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # If rectangle is too small, ignore it
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.canvas.delete("temp_rect")
            return
        
        # Create points for the rectangle
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        # Ask for region name
        region_name = simpledialog.askstring("Region Name", "Enter name for this region:", 
                                          initialvalue=f"Region {self.current_region_id}")
        
        if not region_name:
            region_name = f"Region {self.current_region_id}"
        
        # Delete temporary rectangle
        self.canvas.delete("temp_rect")
        
        # Create final rectangle
        polygon_id = self.canvas.create_polygon(
            points, outline="red", fill="", tags=f"region_{self.current_region_id}"
        )
        
        # Add the region to the list
        region = {
            "id": self.current_region_id,
            "polygon_id": polygon_id,
            "points": points,
            "name": region_name
        }
        
        self.regions.append(region)
        
        # Add to listbox
        self.region_listbox.insert(tk.END, region_name)
        
        # Increment region ID
        self.current_region_id += 1
        
        self.statusbar.config(text=f"Created region '{region_name}'")
    
    def on_mouse_move(self, event):
        if not self.image_path:
            return
        
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.statusbar.config(text=f"Cursor: ({int(x)}, {int(y)})")
    
    def on_region_select(self, event):
        selection = self.region_listbox.curselection()
        if selection:
            index = selection[0]
            region = self.regions[index]
            
            # Highlight the selected region
            for r in self.regions:
                self.canvas.itemconfig(f"region_{r['id']}", width=1)
            
            self.canvas.itemconfig(f"region_{region['id']}", width=3)
            
            self.statusbar.config(text=f"Selected {region['name']}")
    
    def point_in_rect(self, x, y, rect_points):
        """Check if a point is inside a rectangle defined by its 4 corner points"""
        x1, y1 = rect_points[0]
        x2, y2 = rect_points[2]
        
        return (x1 <= x <= x2) and (y1 <= y <= y2)
    
    def export_slices(self):
        """Export each defined region as a separate PNG image"""
        if not self.image_path or not self.regions:
            messagebox.showwarning("Warning", "No image loaded or no regions defined")
            return
        
        # Ask for export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        # Make sure original image is loaded
        if self.original_image is None:
            self.original_image = cv2.imread(self.image_path)
        
        # Process each region
        for region in self.regions:
            region_id = region["id"]
            region_name = region["name"]
            points = region["points"]
            
            # Create safe filename from region name
            safe_name = "".join(c if c.isalnum() else "_" for c in region_name)
            filename = f"{safe_name}.png"
            output_path = os.path.join(export_dir, filename)
            
            # Scale points back to original image size if needed
            scaled_points = []
            for x, y in points:
                orig_x = int(x / self.image_scale)
                orig_y = int(y / self.image_scale)
                scaled_points.append((orig_x, orig_y))
            
            # Get rectangle coordinates
            x1, y1 = scaled_points[0]
            x2, y2 = scaled_points[2]
            
            # Crop to the rectangle
            cropped = self.original_image[y1:y2, x1:x2]
            
            # Save as PNG
            cv2.imwrite(output_path, cropped)
            
        messagebox.showinfo("Export Complete", f"Exported {len(self.regions)} slices to {export_dir}")
        self.statusbar.config(text=f"Exported slices to {export_dir}")
    
    def show_instructions(self):
        instructions = """
        Image Slicer Tool Instructions:
        
        1. Open an image using File > Open Image
        2. Click 'New Rectangle' to start drawing a rectangle region
        3. Click and drag on the image to create a rectangular selection
        4. Enter a name for the region when prompted
        5. Use 'Select Mode' to select regions for editing or deletion
        6. Export all regions as PNG files using 'Export Slices'
        
        Note: Each slice will be saved as a separate PNG file named after the region.
        """
        messagebox.showinfo("Instructions", instructions)
    
    def on_exit(self):
        if messagebox.askokcancel("Exit", "Do you want to exit?"):
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSlicer(root)
    root.mainloop() 