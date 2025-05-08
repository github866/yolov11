import os
import cv2
import json
import gradio as gr
from pathlib import Path
import numpy as np
from PIL import Image
import time

# Set Gradio cache directory to user's home directory
os.environ['GRADIO_TEMP_DIR'] = str(Path.home() / '.gradio_cache')
os.environ['GRADIO_CACHE_DIR'] = str(Path.home() / '.gradio_cache')

class RoomSegmenter:
    def __init__(self, image_path, output_dir="room_segments"):
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.points = []  # List of all available points
        self.polygons = []  # List of dictionaries containing polygon points and names
        self.current_image = None
        self.temp_dir = Path.home() / ".room_segmenter_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Gradio cache directory
        gradio_cache = Path.home() / '.gradio_cache'
        gradio_cache.mkdir(parents=True, exist_ok=True)
        
        # Load the image
        self.load_image()

    def load_image(self):
        """Load the input image"""
        self.current_image = cv2.imread(self.image_path)
        if self.current_image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        return self.image_path

    def add_points(self, text):
        """Add multiple points from text input (format: name1:x1,y1;name2:x2,y2;name3:x3,y3)"""
        if not text:
            return None, "Please enter points in format: name1:x1,y1;name2:x2,y2;name3:x3,y3"
        
        try:
            points = text.split(';')
            for point in points:
                if point.strip():
                    if ':' in point:
                        name, coords = point.strip().split(':')
                        x, y = map(int, coords.split(','))
                    else:
                        x, y = map(int, point.strip().split(','))
                        name = f"P{len(self.points)+1}"
                    
                    # Add point to available points
                    self.points.append({
                        "x": x,
                        "y": y,
                        "name": name
                    })
            
            return self.update_preview(), f"Added {len(points)} points"
        except Exception as e:
            return None, f"Error adding points: {str(e)}"

    def create_polygon(self, polygon_text):
        """Create a polygon from text input (format: polygon_name:point1,point2,point3)"""
        if not polygon_text:
            return None, "Please enter polygon in format: polygon_name:point1,point2,point3"
        
        try:
            # Split polygon name and points
            if ':' not in polygon_text:
                return None, "Invalid format. Use: polygon_name:point1,point2,point3"
            
            name, points_text = polygon_text.split(':')
            point_names = points_text.split(',')
            
            if len(point_names) < 3:
                return None, "Need at least 3 points to create a polygon"
            
            # Find points by name
            polygon_points = []
            for point_name in point_names:
                point = next((p for p in self.points if p["name"] == point_name.strip()), None)
                if not point:
                    return None, f"Point {point_name} not found"
                polygon_points.append(point)
            
            # Add the polygon
            self.polygons.append({
                "name": name.strip(),
                "points": polygon_points
            })
            
            # Save to JSON and get save message
            save_message = self.save_polygons()
            
            return self.update_preview(), f"Created room: {name}\n{save_message}"
        except Exception as e:
            return None, f"Error creating polygon: {str(e)}"

    def update_preview(self):
        """Update the preview image with points and polygons"""
        if self.current_image is None:
            return None
            
        try:
            preview = self.current_image.copy()
            
            # Draw all points
            for point in self.points:
                x, y = point["x"], point["y"]
                # Draw larger point
                cv2.circle(preview, (x, y), 8, (0, 0, 255), -1)  # Red for points
                # Draw point name and coordinates with background
                text = f"{point['name']}:({x},{y})"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Draw background rectangle
                cv2.rectangle(preview, (x+5, y-text_height-5), (x+5+text_width, y+5), (255, 255, 255), -1)
                # Draw text
                cv2.putText(preview, text, (x+5, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw polygons
            for polygon in self.polygons:
                points = np.array([(p["x"], p["y"]) for p in polygon["points"]], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(preview, [points], True, (0, 255, 0), 2)
                # Add polygon name with background
                if points.size > 0:
                    x, y = points[0][0]
                    text = polygon["name"]
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # Draw background rectangle
                    cv2.rectangle(preview, (x, y-text_height-10), (x+text_width, y), (255, 255, 255), -1)
                    # Draw text
                    cv2.putText(preview, text, (x, y-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save preview to temp file
            preview_path = self.temp_dir / f"preview_{int(time.time())}.png"
            cv2.imwrite(str(preview_path), preview)
            
            return str(preview_path)
        except Exception as e:
            print(f"Error updating preview: {str(e)}")
            return None

    def preview_points(self, text):
        """Preview points before adding them"""
        if not text:
            return None, "Please enter points to preview"
        
        try:
            preview = self.current_image.copy()
            points = text.split(';')
            preview_points = []
            
            for point in points:
                if point.strip():
                    if ':' in point:
                        name, coords = point.strip().split(':')
                        x, y = map(int, coords.split(','))
                    else:
                        x, y = map(int, point.strip().split(','))
                        name = f"P{len(preview_points)+1}"
                    
                    preview_points.append({
                        "x": x,
                        "y": y,
                        "name": name
                    })
            
            # Draw preview points
            for point in preview_points:
                x, y = point["x"], point["y"]
                # Draw larger point
                cv2.circle(preview, (x, y), 8, (255, 0, 0), -1)  # Blue for preview
                # Draw point name and coordinates with background
                text = f"{point['name']}:({x},{y})"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Draw background rectangle
                cv2.rectangle(preview, (x+5, y-text_height-5), (x+5+text_width, y+5), (255, 255, 255), -1)
                # Draw text
                cv2.putText(preview, text, (x+5, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Save preview to temp file
            preview_path = self.temp_dir / f"preview_{int(time.time())}.png"
            cv2.imwrite(str(preview_path), preview)
            
            return str(preview_path), f"Previewing {len(preview_points)} points"
        except Exception as e:
            return None, f"Error previewing points: {str(e)}"

    def save_polygons(self):
        """Save all polygons to a JSON file with room information"""
        output_file = self.output_dir / "polygons.json"
        
        # Create a more detailed output structure
        output_data = {
            "image_path": str(self.image_path),
            "points": self.points,
            "rooms": []
        }
        
        # Add each polygon as a room
        for polygon in self.polygons:
            room_data = {
                "room_name": polygon["name"],
                "points": polygon["points"],
                "coordinates": [(p["x"], p["y"]) for p in polygon["points"]]
            }
            output_data["rooms"].append(room_data)
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        return f"Saved to {output_file}"

    def clear_all(self):
        """Clear all points and polygons"""
        self.points = []
        self.polygons = []
        return self.update_preview(), "Cleared all points and polygons"

    def get_points_list(self):
        """Get a formatted string of all points"""
        if not self.points:
            return "No points added yet"
        
        points_list = "Available Points:\n"
        for point in self.points:
            points_list += f"{point['name']}: ({point['x']}, {point['y']})\n"
        return points_list

    def launch(self):
        try:
            with gr.Blocks() as demo:
                with gr.Row():
                    image = gr.Image(label="Image Viewer", type="filepath", interactive=True)
                    with gr.Column():
                        gr.Markdown("### Point Creation")
                        gr.Markdown("Enter points in format: name1:x1,y1;name2:x2,y2;name3:x3,y3")
                        gr.Markdown("Example: corner1:100,100;corner2:200,100;corner3:200,200")
                        points_text = gr.Textbox(label="Points", lines=3)
                        preview_points_btn = gr.Button("👁️ Preview Points")
                        add_points_btn = gr.Button("Add Points")
                        
                        gr.Markdown("### Available Points")
                        points_list = gr.Textbox(label="Points List", lines=10, interactive=False)
                        
                        gr.Markdown("### Room Creation")
                        gr.Markdown("Enter room in format: room_name:point1,point2,point3")
                        gr.Markdown("Example: living_room:corner1,corner2,corner3,corner4")
                        room_text = gr.Textbox(label="Room", lines=2)
                        create_room_btn = gr.Button("Create Room")
                        
                        gr.Markdown("### Save Location")
                        gr.Markdown(f"Data is automatically saved to: `{self.output_dir}/polygons.json`")
                        
                        clear_btn = gr.Button("Clear All")
                        status = gr.Textbox(label="Status", interactive=False)

                def update_points_list():
                    return self.get_points_list()

                preview_points_btn.click(
                    self.preview_points,
                    inputs=[points_text],
                    outputs=[image, status]
                )
                add_points_btn.click(
                    self.add_points,
                    inputs=[points_text],
                    outputs=[image, status]
                ).then(
                    update_points_list,
                    outputs=[points_list]
                )
                create_room_btn.click(
                    self.create_polygon,
                    inputs=[room_text],
                    outputs=[image, status]
                )
                clear_btn.click(
                    self.clear_all,
                    outputs=[image, status]
                ).then(
                    update_points_list,
                    outputs=[points_list]
                )
                demo.load(self.load_image, outputs=image).then(
                    update_points_list,
                    outputs=[points_list]
                )

            demo.launch(share=False)
        except Exception as e:
            print(f"Error launching Gradio interface: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        segmenter = RoomSegmenter(
            image_path="frames/frame_0000.png"  # Replace with your image path
        )
        segmenter.launch()
    except Exception as e:
        print(f"Error initializing RoomSegmenter: {str(e)}") 