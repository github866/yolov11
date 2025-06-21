import json
import cv2
import os

def extract_cropped_imgs(img_dir, json_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open (json_path, 'r') as f:
        data = json.load(f)

    for item in data:
        frame_name = item['frame_path'].split('/')[-1]
        frame_path = f"{img_dir}/{frame_name}"
        coord = item['coordinates']

        cropped_img = cv2.imread(frame_path)[coord['y1']:coord['y2'], coord['x1']:coord['x2']]

        if cropped_img is not None:
            output_path = f"{output_dir}/{frame_name}"
            cv2.imwrite(output_path, cropped_img)
            print(f"Saved cropped image to {output_path}")
        else:
            print(f"Failed to crop image from {frame_path} with coordinates {coord}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract cropped images from frames based on JSON coordinates.")
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing the original images.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file with coordinates.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the cropped images.')

    args = parser.parse_args()

    extract_cropped_imgs(args.img_dir, args.json_path, args.output_dir)