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
    def __init__(self, image_path, output_dir="room_segments",output_name="polygons.json"):
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.points = []  # List of all available points
        self.polygons = []  # List of dictionaries containing polygon points and names
        self.current_image = None
        self.temp_dir = Path(".room_segmenter_temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Gradio cache directory
        gradio_cache = Path.home() / '.gradio_cache'
        gradio_cache.mkdir(parents=True, exist_ok=True)
        
        # Load the image
        self.load_image()
        self.output_name = output_name

    def load_image(self):
        """Load the input image"""
        self.current_image = cv2.imread(self.image_path)
        if self.current_image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        # Return the image as a numpy array (BGR to RGB for Gradio)
        return cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

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
        """Create multiple polygons from text input (format: room1:point1,point2,point3;room2:point1,point2,point3)"""
        if not polygon_text:
            return None, "Please enter rooms in format: room1:point1,point2,point3;room2:point1,point2,point3"
        
        try:
            # Split multiple room definitions
            room_definitions = polygon_text.split(';')
            success_messages = []
            
            for room_def in room_definitions:
                if not room_def.strip():
                    continue
                    
                # Split room name and points
                if ':' not in room_def:
                    return None, f"Invalid format in: {room_def}. Use: room_name:point1,point2,point3"
                
                name, points_text = room_def.split(':')
                point_names = points_text.split(',')
                
                if len(point_names) < 3:
                    return None, f"Need at least 3 points to create room: {name}"
                
                # Find points by name
                polygon_points = []
                for point_name in point_names:
                    point = next((p for p in self.points if p["name"] == point_name.strip()), None)
                    if not point:
                        return None, f"Point {point_name} not found for room {name}"
                    polygon_points.append(point)
                
                # Add the polygon
                self.polygons.append({
                    "name": name.strip(),
                    "points": polygon_points
                })
                success_messages.append(f"Created room: {name}")
            
            # Save to JSON and get save message
            save_message = self.save_polygons()
            
            return self.update_preview(), "\n".join(success_messages) + f"\n{save_message}"
        except Exception as e:
            return None, f"Error creating polygons: {str(e)}"

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
                # Draw background rectangle with more padding and a border
                pad_x, pad_y = 8, 8
                rect_start = (x+5, y-text_height-5-pad_y)
                rect_end = (x+5+text_width+pad_x, y+5+pad_y)
                cv2.rectangle(preview, rect_start, rect_end, (255, 255, 255), -1)  # White background
                cv2.rectangle(preview, rect_start, rect_end, (0, 0, 0), 2)  # Black border
                # Draw text
                cv2.putText(preview, text, (x+5+pad_x//2, y),
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
                    cv2.rectangle(preview, (x, y-text_height-10), (x+text_width, y), (0, 0, 0), 2)
                    # Draw text
                    cv2.putText(preview, text, (x, y-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Return the preview as a numpy array (BGR to RGB for Gradio)
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            return preview_rgb
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
                pad_x, pad_y = 8, 8
                rect_start = (x+5, y-text_height-5-pad_y)
                rect_end = (x+5+text_width+pad_x, y+5+pad_y)
                cv2.rectangle(preview, rect_start, rect_end, (255, 255, 255), -1)
                cv2.rectangle(preview, rect_start, rect_end, (0, 0, 0), 2)
                # Draw text
                cv2.putText(preview, text, (x+5+pad_x//2, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # Return the preview as a numpy array (BGR to RGB for Gradio)
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            return preview_rgb, f"Previewing {len(preview_points)} points"
        except Exception as e:
            return None, f"Error previewing points: {str(e)}"

    def save_polygons(self):
        """Save all polygons to a JSON file with simplified room information"""
        output_file = self.output_dir / self.output_name
        
        # Create a simplified output structure
        output_data = {}
        
        # Add each polygon as a room with just coordinates
        for polygon in self.polygons:
            room_name = polygon["name"]
            coordinates = [(p["x"], p["y"]) for p in polygon["points"]]
            output_data[room_name] = coordinates
        
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
            points_list += f"{point['name']}: {point['x']}, {point['y']};\n"
        return points_list

    def add_point_by_click(self, evt: gr.SelectData):
        """Add a point by clicking on the image."""
        x, y = int(evt.index[0]), int(evt.index[1])
        name = f"P{len(self.points)+1}"
        self.points.append({
            "x": x,
            "y": y,
            "name": name
        })
        return self.update_preview(), f"Added point {name}: ({x}, {y})"

    def delete_point(self, point_name):
        """Delete a point by name, update preview and points list."""
        before = len(self.points)
        self.points = [p for p in self.points if p["name"] != point_name]
        # Also remove the point from any polygons
        for polygon in self.polygons:
            polygon["points"] = [p for p in polygon["points"] if p["name"] != point_name]
        after = len(self.points)
        msg = f"Deleted point {point_name}" if before != after else f"Point {point_name} not found"
        return self.update_preview(), msg, self.get_points_list(), self.get_point_names()

    def get_point_names(self):
        """Return a list of all point names for the dropdown."""
        return [p["name"] for p in self.points]

    def launch(self):
        try:
            with gr.Blocks() as demo:
                with gr.Row():
                    image = gr.Image(label="Image Viewer", interactive=True)
                    with gr.Column():
                        gr.Markdown("### Point Creation")
                        gr.Markdown("Enter points in format: name1:x1,y1;name2:x2,y2;name3:x3,y3")
                        gr.Markdown("Example: corner1:100,100;corner2:200,100;corner3:200,200")
                        points_text = gr.Textbox(label="Points", lines=3)
                        preview_points_btn = gr.Button("👁️ Preview Points")
                        add_points_btn = gr.Button("Add Points")
                        
                        gr.Markdown("### Available Points")
                        points_list = gr.Textbox(label="Points List", lines=10, interactive=False)
                        
                        gr.Markdown("### Delete Point")
                        point_dropdown = gr.Dropdown(label="Select Point to Delete", choices=self.get_point_names(), interactive=True, allow_custom_value=True)
                        delete_point_btn = gr.Button("Delete Point")

                        gr.Markdown("### Room Creation")
                        gr.Markdown("Enter room in format: room1:point1,point2,point3;room2:point1,point2,point3")
                        gr.Markdown("Example: living_room:corner1,corner2,corner3,corner4")
                        room_text = gr.Textbox(label="Room", lines=2)
                        create_room_btn = gr.Button("Create Room")
                        
                        gr.Markdown("### Save Location")
                        gr.Markdown(f"Data is automatically saved to: `{self.output_dir}/polygons.json`")
                        
                        clear_btn = gr.Button("Clear All")
                        status = gr.Textbox(label="Status", interactive=False)

                def update_points_list():
                    return self.get_points_list()
                def update_point_dropdown():
                    return self.get_point_names()

                # Add click event handler for image
                image.select(
                    self.add_point_by_click,
                    outputs=[image, status]
                ).then(
                    update_points_list,
                    outputs=[points_list]
                ).then(
                    update_point_dropdown,
                    outputs=[point_dropdown]
                )

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
                ).then(
                    update_point_dropdown,
                    outputs=[point_dropdown]
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
                ).then(
                    update_point_dropdown,
                    outputs=[point_dropdown]
                )
                delete_point_btn.click(
                    self.delete_point,
                    inputs=[point_dropdown],
                    outputs=[image, status, points_list, point_dropdown]
                )
                demo.load(self.load_image, outputs=image).then(
                    update_points_list,
                    outputs=[points_list]
                ).then(
                    update_point_dropdown,
                    outputs=[point_dropdown]
                )

            demo.launch(share=False)
        except Exception as e:
            print(f"Error launching Gradio interface: {str(e)}")
            raise

if __name__ == "__main__":
    name = "loc02"
    segmenter = RoomSegmenter(
        image_path=f"first_frame_img/{name}.png",output_name=f"{name}.json"
    )   
    segmenter.launch()
