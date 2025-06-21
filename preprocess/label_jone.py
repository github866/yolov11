import cv2
import numpy as np

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