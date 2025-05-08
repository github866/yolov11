import cv2
import numpy as np
import json
import os

class PolygonAnnotator:
    def __init__(self):
        self.points = []
        self.polygons = []
        self.current_image = None
        self.window_name = "Polygon Annotator"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.draw_current_state()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) >= 3:
                self.polygons.append(self.points.copy())
                self.points = []
                self.draw_current_state()

    def draw_current_state(self):
        if self.current_image is None:
            return

        display_image = self.current_image.copy()
        
        # Draw all completed polygons
        for polygon in self.polygons:
            points = np.array(polygon, np.int32)
            cv2.polylines(display_image, [points], True, (0, 255, 0), 2)
            for point in polygon:
                cv2.circle(display_image, point, 3, (0, 0, 255), -1)

        # Draw current polygon
        if len(self.points) > 0:
            points = np.array(self.points, np.int32)
            cv2.polylines(display_image, [points], False, (255, 0, 0), 2)
            for point in self.points:
                cv2.circle(display_image, point, 3, (0, 0, 255), -1)

        cv2.imshow(self.window_name, display_image)

    def load_image(self, image_path):
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.draw_current_state()

    def save_annotations(self, output_file):
        annotations = {
            "polygons": [{"points": points} for points in self.polygons]
        }
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)

    def run(self):
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('c'):  # Clear current polygon
                self.points = []
                self.draw_current_state()
            elif key == ord('s'):  # Save annotations
                self.save_annotations('annotations.json')
                print("Annotations saved to annotations.json")

        cv2.destroyAllWindows()

def main():
    annotator = PolygonAnnotator()
    
    # Get image path from user
    image_path = input("Enter the path to your image: ")
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return

    try:
        annotator.load_image(image_path)
        print("\nInstructions:")
        print("- Left click to add points")
        print("- Right click to complete a polygon")
        print("- Press 'c' to clear current polygon")
        print("- Press 's' to save annotations")
        print("- Press ESC to quit")
        
        annotator.run()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
