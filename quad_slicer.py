import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
import os
import json
from PIL import Image, ImageTk

class QuadSlicer:
    def __init__(self, root):
        self.root = root
        self.root.title("Quad Slicer Tool")
        
        # Configure window size
        self.root.geometry("1200x800")
        
        # Variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.quads = []  # List of quadrilateral shapes
        self.current_quad_points = []  # Points of the quad being drawn (max 4)
        self.current_quad_id = 0
        self.temp_line_ids = []  # Store IDs of temporary lines
        self.export_sizes = [(800, 600), (1024, 768), (1280, 720), (1920, 1080)]  # Default export sizes
        
        # Drawing mode
        self.drawing_mode = "draw"  # "draw" or "select"
        
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
        file_menu.add_separator()
        file_menu.add_command(label="Save Shapes to JSON", command=self.save_to_json)
        file_menu.add_command(label="Load Shapes from JSON", command=self.load_from_json)
        file_menu.add_separator()
        file_menu.add_command(label="Export Slices as PNG", command=self.export_slices)
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
        
        # Quad control buttons
        self.btn_new_quad = tk.Button(toolbar_frame, text="New Shape", command=self.start_new_quad)
        self.btn_new_quad.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.btn_delete_quad = tk.Button(toolbar_frame, text="Delete Shape", command=self.delete_selected_quad)
        self.btn_delete_quad.pack(side=tk.LEFT, padx=2, pady=2)
        
        # JSON buttons
        self.btn_save_json = tk.Button(toolbar_frame, text="Save to JSON", command=self.save_to_json)
        self.btn_save_json.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.btn_load_json = tk.Button(toolbar_frame, text="Load from JSON", command=self.load_from_json)
        self.btn_load_json.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Export button
        self.btn_export = tk.Button(toolbar_frame, text="Export as PNG", command=self.export_slices)
        self.btn_export.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Size selection
        tk.Label(toolbar_frame, text="Export Size:").pack(side=tk.LEFT, padx=5, pady=2)
        self.size_var = tk.StringVar(value="800x600")
        self.size_menu = tk.OptionMenu(toolbar_frame, self.size_var, 
                                     *[f"{w}x{h}" for w, h in self.export_sizes],
                                     command=self.on_size_change)
        self.size_menu.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Mode selection
        self.draw_mode_var = tk.StringVar(value="draw")
        self.rb_draw = tk.Radiobutton(toolbar_frame, text="Draw Mode", variable=self.draw_mode_var, 
                                     value="draw", command=self.set_draw_mode)
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
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_move)
        
        # Quad info panel
        self.quad_info_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        self.quad_info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        tk.Label(self.quad_info_frame, text="Shapes:").pack(anchor='w')
        
        self.quad_listbox = tk.Listbox(self.quad_info_frame, width=30, height=20)
        self.quad_listbox.pack(fill=tk.BOTH, expand=True)
        self.quad_listbox.bind("<<ListboxSelect>>", self.on_quad_select)
    
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
            self.quads = []
            self.current_quad_points = []
            self.canvas.delete("all")
            self.quad_listbox.delete(0, tk.END)
            
            # Load image using OpenCV
            self.original_image = cv2.imread(self.image_path)
            self.display_image = self.original_image.copy()
            
            # Convert BGR to RGB
            self.display_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage for canvas
            self.photo_image = ImageTk.PhotoImage(image=Image.fromarray(self.display_image))
            
            # Reset canvas and display image
            self.canvas.config(scrollregion=(0, 0, self.photo_image.width(), self.photo_image.height()))
            self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW, tags="image")
            
            # Update status
            h, w = self.display_image.shape[:2]
            self.statusbar.config(text=f"Loaded image: {self.image_path} ({w}x{h})")
    
    def start_new_quad(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        # Clean up any existing temporary points
        self.canvas.delete("temp_point")
        self.canvas.delete("temp_line")
        
        # Reset points and set draw mode
        self.current_quad_points = []
        self.temp_line_ids = []
        self.drawing_mode = "draw"
        self.draw_mode_var.set("draw")
        self.canvas.config(cursor="crosshair")
        
        self.statusbar.config(text="Click to place up to 4 points for your shape")
    
    def delete_selected_quad(self):
        selection = self.quad_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No shape selected")
            return
        
        index = selection[0]
        quad = self.quads[index]
        
        # Delete from canvas
        self.canvas.delete(f"quad_{quad['id']}")
        
        # Remove from data structures
        self.quads.pop(index)
        
        # Update listbox
        self.quad_listbox.delete(index)
        
        self.statusbar.config(text=f"Shape '{quad['name']}' deleted")
    
    def set_draw_mode(self):
        self.drawing_mode = "draw"
        self.canvas.config(cursor="crosshair")
        self.statusbar.config(text="Draw Mode: Click to place points (max 4)")
    
    def set_select_mode(self):
        self.drawing_mode = "select"
        self.canvas.config(cursor="arrow")
        self.statusbar.config(text="Select Mode: Click to select shapes")
    
    def on_canvas_click(self, event):
        if not self.image_path:
            return
        
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        if self.drawing_mode == "draw":
            # Check if we already have 4 points
            if len(self.current_quad_points) >= 4:
                messagebox.showinfo("Shape Complete", "Already have 4 points. Complete or discard this shape.")
                return
            
            # Add the point
            self.current_quad_points.append((x, y))
            
            # Draw a point at the click location
            point_id = self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", tags="temp_point")
            
            # If this is not the first point, draw a line from the previous point
            if len(self.current_quad_points) > 1:
                prev_x, prev_y = self.current_quad_points[-2]
                line_id = self.canvas.create_line(prev_x, prev_y, x, y, fill="red", tags="temp_line")
                self.temp_line_ids.append(line_id)
            
            # If this is the 4th point, draw a line back to the first point and complete the shape
            if len(self.current_quad_points) == 4:
                first_x, first_y = self.current_quad_points[0]
                last_line_id = self.canvas.create_line(x, y, first_x, first_y, fill="red", tags="temp_line")
                self.temp_line_ids.append(last_line_id)
                
                # Ask for quad name
                quad_name = simpledialog.askstring("Shape Name", "Enter name for this shape:", 
                                                initialvalue=f"Shape {self.current_quad_id}")
                
                if not quad_name:
                    quad_name = f"Shape {self.current_quad_id}"
                
                # Create final polygon
                polygon_id = self.canvas.create_polygon(
                    self.current_quad_points, outline="red", fill="", width=2, tags=f"quad_{self.current_quad_id}"
                )
                
                # Add the quad to the list
                quad = {
                    "id": self.current_quad_id,
                    "polygon_id": polygon_id,
                    "points": self.current_quad_points.copy(),
                    "name": quad_name
                }
                
                self.quads.append(quad)
                
                # Add to listbox
                self.quad_listbox.insert(tk.END, quad_name)
                
                # Clean up temporary objects
                self.canvas.delete("temp_point")
                self.canvas.delete("temp_line")
                
                # Increment quad ID and reset points
                self.current_quad_id += 1
                self.current_quad_points = []
                self.temp_line_ids = []
                
                self.statusbar.config(text=f"Created shape '{quad_name}'")
            else:
                self.statusbar.config(text=f"Added point {len(self.current_quad_points)}/4 at ({int(x)}, {int(y)})")
                
        elif self.drawing_mode == "select":
            # Check if clicked inside any quad polygon
            for i, quad in enumerate(self.quads):
                if self.point_in_quad(x, y, quad["points"]):
                    self.quad_listbox.selection_clear(0, tk.END)
                    self.quad_listbox.selection_set(i)
                    self.quad_listbox.see(i)
                    self.statusbar.config(text=f"Selected {quad['name']}")
                    break
    
    def on_canvas_move(self, event):
        if not self.image_path:
            return
        
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.statusbar.config(text=f"Cursor: ({int(x)}, {int(y)})")
        
        # If we have at least one point, draw a temporary guiding line
        if self.drawing_mode == "draw" and self.current_quad_points:
            self.canvas.delete("temp_rubber_line")
            last_x, last_y = self.current_quad_points[-1]
            self.canvas.create_line(last_x, last_y, x, y, dash=(4, 4), fill="blue", tags="temp_rubber_line")
    
    def on_quad_select(self, event):
        selection = self.quad_listbox.curselection()
        if selection:
            index = selection[0]
            quad = self.quads[index]
            
            # Highlight the selected quad
            for q in self.quads:
                self.canvas.itemconfig(f"quad_{q['id']}", width=1)
            
            self.canvas.itemconfig(f"quad_{quad['id']}", width=3)
            
            self.statusbar.config(text=f"Selected {quad['name']}")
    
    def point_in_quad(self, x, y, quad_points):
        """Check if a point is inside a polygon using the ray casting algorithm"""
        n = len(quad_points)
        inside = False
        
        p1x, p1y = quad_points[0]
        for i in range(1, n + 1):
            p2x, p2y = quad_points[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def save_to_json(self):
        """Save shapes to a JSON file"""
        if not self.image_path or not self.quads:
            messagebox.showwarning("Warning", "No image loaded or no shapes defined")
            return
        
        # Ask for save file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Create data structure
        data = {
            "image_path": self.image_path,
            "quads": []
        }
        
        # Add shapes
        for quad in self.quads:
            quad_data = {
                "id": quad["id"],
                "name": quad["name"],
                "points": quad["points"]
            }
            data["quads"].append(quad_data)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.statusbar.config(text=f"Saved shapes to {file_path}")
    
    def load_from_json(self):
        """Load shapes from a JSON file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Load data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get image path
            image_path = data.get("image_path", "")
            
            # If image doesn't exist, ask for a new one
            if not os.path.exists(image_path):
                messagebox.showinfo("Image Not Found", "The original image was not found. Please select the image file.")
                image_path = filedialog.askopenfilename(
                    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
                )
                if not image_path:
                    return
            
            # Set image path and load it
            self.image_path = image_path
            self.load_image()
            
            # Clear existing shapes
            self.quads = []
            self.current_quad_id = 0
            self.canvas.delete("quad_*")
            self.quad_listbox.delete(0, tk.END)
            
            # Load shapes
            for quad_data in data.get("quads", []):
                quad_id = quad_data.get("id", self.current_quad_id)
                quad_name = quad_data.get("name", f"Shape {quad_id}")
                points = quad_data.get("points", [])
                
                # Create polygon on canvas
                polygon_id = self.canvas.create_polygon(
                    points, outline="red", fill="", width=2, tags=f"quad_{quad_id}"
                )
                
                # Add to quads list
                quad = {
                    "id": quad_id,
                    "polygon_id": polygon_id,
                    "points": points,
                    "name": quad_name
                }
                
                self.quads.append(quad)
                
                # Add to listbox
                self.quad_listbox.insert(tk.END, quad_name)
                
                # Update current_quad_id
                self.current_quad_id = max(self.current_quad_id, quad_id + 1)
            
            self.statusbar.config(text=f"Loaded {len(self.quads)} shapes from {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load from JSON: {str(e)}")
    
    def on_size_change(self, *args):
        """Handle size selection change"""
        size_str = self.size_var.get()
        width, height = map(int, size_str.split('x'))
        self.statusbar.config(text=f"Selected export size: {width}x{height}")
    
    def create_perspective_transform(self, src_points, target_size):
        """Create a perspective transform from the quadrilateral to a rectangle"""
        # Convert to numpy arrays
        src = np.array(src_points, dtype=np.float32)
        
        # Define destination points (rectangle)
        width, height = target_size
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        return M, width, height
    
    def export_slices(self):
        """Export each defined quadrilateral as a separate PNG image"""
        if not self.image_path or not self.quads:
            messagebox.showwarning("Warning", "No image loaded or no shapes defined")
            return
        
        # Ask for export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        # Get selected size
        size_str = self.size_var.get()
        target_width, target_height = map(int, size_str.split('x'))
        
        # Make sure original image is loaded
        if self.original_image is None:
            self.original_image = cv2.imread(self.image_path)
        
        # Process each quad
        for quad in self.quads:
            quad_id = quad["id"]
            quad_name = quad["name"]
            points = quad["points"]
            
            # Create safe filename from quad name
            safe_name = "".join(c if c.isalnum() else "_" for c in quad_name)
            filename = f"{safe_name}_{target_width}x{target_height}.png"
            output_path = os.path.join(export_dir, filename)
            
            # Get perspective transform
            M, width, height = self.create_perspective_transform(points, (target_width, target_height))
            
            # Apply perspective transform
            warped = cv2.warpPerspective(self.original_image, M, (target_width, target_height))
            
            # Save as PNG
            cv2.imwrite(output_path, warped)
            
        messagebox.showinfo("Export Complete", f"Exported {len(self.quads)} slices to {export_dir}")
        self.statusbar.config(text=f"Exported slices to {export_dir}")
    
    def show_instructions(self):
        instructions = """
        Quad Slicer Tool Instructions:
        
        1. Open an image using File > Open Image
        2. Click 'New Shape' to start defining a shape
        3. Click on the image to place exactly 4 points that define your shape
        4. Enter a name for the shape when prompted
        5. Use 'Select Mode' to select shapes for deletion
        6. Save your shapes to a JSON file using 'Save to JSON'
        7. Load previously saved shapes using 'Load from JSON'
        8. Export shapes as PNG files using 'Export as PNG' (optional)
        
        Note: JSON files store all shape information and can be loaded later.
        """
        messagebox.showinfo("Instructions", instructions)
    
    def on_exit(self):
        if messagebox.askokcancel("Exit", "Do you want to exit?"):
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuadSlicer(root)
    root.mainloop() 