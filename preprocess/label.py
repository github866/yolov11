import cv2
import numpy as np
import json
import os

def point_in_quad(point, quad_points, json_scale):
    """Check if a point is inside a quadrilateral"""
    x, y = point
    # Convert quad points to original scale
    original_points = [[int(x / json_scale), int(y / json_scale)] for x, y in quad_points]
    points = np.array(original_points, dtype=np.int32)
    
    # Get the bounds of the quad
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    # Create a mask just large enough for the quad
    mask = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    
    # Check if point is inside
    if x < 0 or x >= max_x + 1 or y < 0 or y >= max_y + 1:
        return False
    return mask[y, x] == 255

def draw_rooms(image, room_data, display_scale, json_scale):
    """Draw room boundaries on the image"""
    for i, quad in enumerate(room_data['quads']):
        # The points in room_data are already scaled by json_scale
        # We need to scale them up to match the image size
        scaled_points = [[int(x / json_scale * display_scale), int(y / json_scale * display_scale)] for x, y in quad['points']]
        points = np.array(scaled_points, dtype=np.int32)
        cv2.polylines(image, [points], True, (0, 255, 0), 2)
        # Add room number (using shape number)
        center = np.mean(points, axis=0).astype(int)
        cv2.putText(image, f"Room {i+1}", (center[0], center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def create_complete_room_data(location_name):
    """
    Create complete room data by analyzing points and defining missing rooms
    Based on the reference mask, we need to define:
    - CSU Miliue (Blue)
    - Nurse Station (Green) 
    - Quiet Room (Red)
    - Sally Port/Entrance (Light Blue)
    """
    # Load existing room data
    room_data_path = f'room_segments/{location_name}.json'
    if os.path.exists(room_data_path):
        with open(room_data_path, 'r') as f:
            room_data = json.load(f)
    else:
        room_data = {}
    

    
    return room_data

def create_room_mask(image_size, room_data, json_scale=1.0):
    """
    Create a mask where each room is filled with its specific color
    Colors: CSU Miliue (Blue), Nurse Station (Green), Quiet Room (Red), Sally Port/Entrance (Light Blue)
    """
    # Define colors in BGR format (OpenCV uses BGR)
    colors = {
        'CSU Miliue': (255, 0, 0),      # Blue
        'CSU Miliue2': (255, 0, 0),      # Blue
        'CSU Miliue3': (255, 0, 0),      # Blue
        'Nursing Station': (0, 255, 0),   # Green
        'Quiet Room': (0, 0, 255),      # Red
        'Sally Port / Entrance': (255, 255, 0)  # Light Blue (Cyan)
    }
    
    # Create blank mask
    mask = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Process each room
    for room_name, points in room_data.items():
        if room_name in colors:
            # Convert points to numpy array and scale if needed
            if json_scale != 1.0:
                scaled_points = [[int(x / json_scale), int(y / json_scale)] for x, y in points]
            else:
                scaled_points = [[int(x), int(y)] for x, y in points]
            
            points_array = np.array(scaled_points, dtype=np.int32)
            
            # Fill the room area with the specified color
            cv2.fillPoly(mask, [points_array], colors[room_name])
    
    return mask

def main():
    name = 'loc04'
    image_path = f'first_frame_img/{name}.png'
    
    # Create complete room data
    room_data = create_complete_room_data(name)
    
    # Load the original image to get dimensions
    if os.path.exists(image_path):
        original_image = cv2.imread(image_path)
        image_size = (original_image.shape[1], original_image.shape[0])
    else:
        # If image doesn't exist, estimate size from room data
        all_points = []
        for points in room_data.values():
            all_points.extend(points)
        all_points = np.array(all_points)
        max_x = np.max(all_points[:, 0])
        max_y = np.max(all_points[:, 1])
        image_size = (max_x + 50, max_y + 50)  # Add some padding
    
    # Create room mask
    room_mask = create_room_mask(image_size, room_data)
    
    # Save the mask
    output_path = f'room_mask/{name}_colored_mask.png'
    os.makedirs('room_mask', exist_ok=True)
    cv2.imwrite(output_path, room_mask)
    
    print(f"Room mask saved to {output_path}")

if __name__ == "__main__":
    main()
