import cv2
import numpy as np
import os
import json

class RoomSegmenter:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.original = self.image.copy()
        self.room_boundaries = []
        self.current_boundary = []
        self.room_names = []
        self.current_room_name = ""
        self.drawing = False
        
        # Window setup
        cv2.namedWindow('Room Segmentation')
        cv2.setMouseCallback('Room Segmentation', self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_boundary.append((x, y))
            cv2.circle(self.image, (x, y), 3, (0, 255, 0), -1)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.line(self.image, self.current_boundary[-1], (x, y), (0, 255, 0), 2)
            self.current_boundary.append((x, y))
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Complete the boundary if it has at least 3 points
            if len(self.current_boundary) >= 3:
                cv2.line(self.image, self.current_boundary[-1], self.current_boundary[0], (0, 255, 0), 2)
    
    def run(self):
        print("Instructions:")
        print("- Left click and drag to draw room boundaries")
        print("- Press 'n' to name the current room")
        print("- Press 'c' to clear current boundary")
        print("- Press 'r' to reset all boundaries")
        print("- Press 's' to save the boundaries")
        print("- Press 'q' to quit")
        
        while True:
            cv2.imshow('Room Segmentation', self.image)
            key = cv2.waitKey(1) & 0xFF
            
            # Name the current room
            if key == ord('n'):
                self.current_room_name = input("Enter room name: ")
                if self.current_boundary and len(self.current_boundary) >= 3:
                    self.room_boundaries.append(self.current_boundary)
                    self.room_names.append(self.current_room_name)
                    # Draw the room name in the image
                    centroid = np.mean(np.array(self.current_boundary), axis=0).astype(int)
                    cv2.putText(self.image, self.current_room_name, tuple(centroid), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    self.current_boundary = []
                    print(f"Room '{self.current_room_name}' added.")
                else:
                    print("Draw a valid boundary first (at least 3 points)")
            
            # Clear current boundary
            elif key == ord('c'):
                self.image = self.original.copy()
                # Redraw existing boundaries
                for i, boundary in enumerate(self.room_boundaries):
                    for j in range(len(boundary)-1):
                        cv2.line(self.image, boundary[j], boundary[j+1], (0, 255, 0), 2)
                    cv2.line(self.image, boundary[-1], boundary[0], (0, 255, 0), 2)
                    centroid = np.mean(np.array(boundary), axis=0).astype(int)
                    cv2.putText(self.image, self.room_names[i], tuple(centroid), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                self.current_boundary = []
                print("Current boundary cleared.")
            
            # Reset all
            elif key == ord('r'):
                self.image = self.original.copy()
                self.current_boundary = []
                self.room_boundaries = []
                self.room_names = []
                print("All boundaries reset.")
            
            # Save boundaries
            elif key == ord('s'):
                self.save_boundaries()
            
            # Quit
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def save_boundaries(self):
        if not self.room_boundaries:
            print("No boundaries to save.")
            return
        
        data = {
            "rooms": []
        }
        
        for i, boundary in enumerate(self.room_boundaries):
            room_data = {
                "name": self.room_names[i],
                "boundary": boundary
            }
            data["rooms"].append(room_data)
        
        filename = input("Enter filename to save (without extension): ")
        with open(f"{filename}.json", "w") as f:
            json.dump(data, f)
        
        # Also save a visualization
        visual = self.original.copy()
        for i, boundary in enumerate(self.room_boundaries):
            points = np.array(boundary, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(visual, [points], True, (0, 255, 0), 2)
            centroid = np.mean(np.array(boundary), axis=0).astype(int)
            cv2.putText(visual, self.room_names[i], tuple(centroid), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imwrite(f"{filename}_visual.png", visual)
        print(f"Boundaries saved to {filename}.json and visualization to {filename}_visual.png")

if __name__ == "__main__":
    image_path = input("Enter path to the image: ")
    if os.path.exists(image_path):
        segmenter = RoomSegmenter(image_path)
        segmenter.run()
    else:
        print("Image file not found.") 