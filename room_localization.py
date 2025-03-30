import json
import cv2
import numpy as np
from shapely.geometry import Point, Polygon

class RoomLocalizer:
    def __init__(self, boundaries_file):
        # Load room boundaries from JSON file
        with open(boundaries_file, 'r') as f:
            data = json.load(f)
        
        self.rooms = []
        self.room_polygons = []
        
        # Create polygon for each room
        for room in data['rooms']:
            name = room['name']
            boundary = room['boundary']
            self.rooms.append(name)
            self.room_polygons.append(Polygon(boundary))
    
    def locate_person(self, person_bbox):
        """
        Determine which room a person is in based on their bounding box
        
        Args:
            person_bbox: [x1, y1, x2, y2] format bounding box
            
        Returns:
            room_name: Name of the room the person is in, or "Unknown" if not in any room
        """
        # Get bottom center point (feet) of the person
        x1, y1, x2, y2 = person_bbox
        feet_point = Point((x1 + x2) / 2, y2)
        
        # Check which room contains this point
        for i, polygon in enumerate(self.room_polygons):
            if polygon.contains(feet_point):
                return self.rooms[i]
        
        # If not in any room, find the closest room
        min_distance = float('inf')
        closest_room = "Unknown"
        
        for i, polygon in enumerate(self.room_polygons):
            distance = polygon.exterior.distance(feet_point)
            if distance < min_distance:
                min_distance = distance
                closest_room = self.rooms[i]
        
        # If the person is very close to a room (within 20 pixels), consider them in that room
        if min_distance < 20:
            return f"{closest_room} (near boundary)"
        
        return "Unknown"
    
    def visualize_localization(self, image, detections):
        """
        Visualize people and their room locations on an image
        
        Args:
            image: OpenCV image
            detections: List of [x1, y1, x2, y2, conf, class_id] format detections
            
        Returns:
            Annotated image
        """
        img = image.copy()
        
        # Draw room boundaries
        for i, polygon in enumerate(self.room_polygons):
            coords = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.polylines(img, [coords], True, (0, 255, 0), 2)
            
            # Draw room name
            centroid = np.mean(coords, axis=0).astype(int)
            cv2.putText(img, self.rooms[i], tuple(centroid), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw people and their locations
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            
            # Only process person class (typically class 0 in COCO)
            if int(class_id) != 0:  # Assuming 0 is the person class
                continue
                
            # Determine room location
            room = self.locate_person([x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # Draw room information
            text = f"Person: {room}"
            cv2.putText(img, text, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img

def process_yolo_results(yolo_results, room_boundaries_file, image_path, output_path=None):
    """
    Process YOLOv11 results with room boundaries
    
    Args:
        yolo_results: Detection results from YOLOv11
        room_boundaries_file: JSON file with room boundaries
        image_path: Path to the original image
        output_path: Path to save the visualization (optional)
    
    Returns:
        List of dictionaries with person detections and their room locations
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Initialize room localizer
    localizer = RoomLocalizer(room_boundaries_file)
    
    # Get all person detections from YOLO results
    person_detections = []
    for detection in yolo_results:
        # Check if the detection format matches what's expected
        if len(detection) >= 6:  # Should have at least x1,y1,x2,y2,conf,class_id
            x1, y1, x2, y2, conf, class_id = detection[:6]
            if int(class_id) == 0:  # Assuming 0 is the person class
                person_detections.append(detection)
    
    # Determine room for each person
    results = []
    for det in person_detections:
        x1, y1, x2, y2, conf, _ = det[:6]
        room = localizer.locate_person([x1, y1, x2, y2])
        results.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "room": room
        })
    
    # Visualize results if requested
    if output_path:
        vis_img = localizer.visualize_localization(image, person_detections)
        cv2.imwrite(output_path, vis_img)
    
    return results

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process YOLOv11 results with room boundaries")
    parser.add_argument("--boundaries", required=True, help="Path to room boundaries JSON file")
    parser.add_argument("--image", required=True, help="Path to the original image")
    parser.add_argument("--yolo_results", required=True, help="Path to YOLOv11 results JSON file")
    parser.add_argument("--output", help="Path to save the visualization")
    
    args = parser.parse_args()
    
    # Load YOLOv11 results
    with open(args.yolo_results, 'r') as f:
        yolo_results = json.load(f)
    
    results = process_yolo_results(yolo_results, args.boundaries, args.image, args.output)
    
    # Print results
    for i, res in enumerate(results):
        print(f"Person {i+1}: Room = {res['room']}, Confidence = {res['confidence']:.2f}") 