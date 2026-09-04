import cv2
import numpy as np
from pathlib import Path

# --- Global Variables ---
# A list to store completed polygons. Each polygon is a list of points.
polygons = []
# A list of points for the polygon currently being drawn.
current_points = []
# A predefined list of colors (in BGR format) to cycle through for each new polygon.
colors = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (255, 255, 255) # White
]

# The original image, loaded once.
original_image = None
# A transparent overlay where we will draw the final colored, filled polygons.
final_overlay = None

def redraw_canvas():
    """Redraws the main display image by blending the original image with the final overlay."""
    # Start with a fresh copy of the original image
    display_image = original_image.copy()

    # Blend the overlay containing all completed polygons with the original image
    # The alpha (transparency) can be adjusted here (e.g., 0.4 for overlay, 0.6 for original)
    cv2.addWeighted(final_overlay, 0.4, display_image, 0.6, 0, display_image)
    
    # Draw the lines of the polygon currently being created
    if len(current_points) > 1:
        for i in range(len(current_points) - 1):
            cv2.line(display_image, current_points[i], current_points[i+1], (0, 255, 255), 2)
    
    # Draw vertices for the current polygon
    for point in current_points:
        cv2.circle(display_image, point, 4, (0, 0, 255), -1)

    cv2.imshow("Segmentation Tool", display_image)

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to handle drawing polygons."""
    global current_points, polygons, final_overlay

    # Left-click to add a new vertex to the current polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Vertex added at ({x}, {y}). Current vertices: {len(current_points)}")
        redraw_canvas()

    # Right-click to finish the current polygon
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_points) >= 3:
            # Add the completed polygon to our main list
            polygons.append(np.array(current_points, dtype=np.int32))
            
            # Select the next color from our palette
            color_index = (len(polygons) - 1) % len(colors)
            color = colors[color_index]
            
            # Draw the filled polygon onto our persistent overlay
            cv2.fillPoly(final_overlay, [polygons[-1]], color)
            
            print(f"--- Polygon #{len(polygons)} finished with {len(current_points)} vertices. ---")
            
            # Clear the current points to start a new polygon
            current_points = []
            redraw_canvas()
        else:
            print("Cannot finish polygon: at least 3 vertices are required.")


def manual_multi_segmentation(image_path):
    """Initializes and runs the multi-region manual segmentation interface."""
    global original_image, final_overlay, polygons, current_points

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Initialize a transparent overlay with the same dimensions as the original image
    final_overlay = np.zeros(original_image.shape, dtype=np.uint8)
    
    window_name = "Segmentation Tool"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- Instructions ---")
    print("1. Left-Click (LMB) to add vertices for a polygon.")
    print("2. Right-Click (RMB) to complete the current polygon and start a new one.")
    print("3. Press 'r' to reset and clear all segmented regions.")
    print("4. Press 's' to generate and save the final segmented image.")
    print("5. Press 'q' to quit.")
    print("-" * 20)
    
    redraw_canvas()

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset everything
            print("Resetting all segmented regions.")
            polygons = []
            current_points = []
            final_overlay = np.zeros(original_image.shape, dtype=np.uint8)
            redraw_canvas()
        elif key == ord('s'):
            # Generate and save the final image
            if not polygons:
                print("Cannot save: No regions have been segmented yet.")
                continue

            # Create the final blended image for saving
            final_result = cv2.addWeighted(original_image, 0.6, final_overlay, 0.4, 0)
            
            base_name = Path(image_path).stem
            
            # Define file paths for both outputs
            save_path_blended = f"{base_name}_multi_segmented.png"
            save_path_mask = f"{base_name}_mask_only.png"
            
            # Save both the blended image and the mask-only image
            cv2.imwrite(save_path_blended, final_result)
            cv2.imwrite(save_path_mask, final_overlay)
            print(f"Final segmented image saved to: {save_path_blended}")
            print(f"Mask-only image saved to: {save_path_mask}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # NOTE: You MUST replace this path with the actual location of your image.
    IMAGE_FILE = "/Users/leohsuinthehouse/random_code/new__first_frame_789.png"
    
    image_path = Path(IMAGE_FILE)
    if not image_path.exists():
        print(f"!!! WARNING: Image '{IMAGE_FILE}' not found.")
        print("Please ensure the image is in the correct directory or update the IMAGE_FILE variable.")
        # Create a blank image to demonstrate the tool if the file is missing
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), dummy_img)
        print("A blank dummy image has been created to test the interface.")

    manual_multi_segmentation(IMAGE_FILE)