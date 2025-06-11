import cv2
import os
import json
from ultralytics import YOLO
import re

model = YOLO("yolo11n.pt")

image_dir = "frames"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

def detect_human(image_dir):
    all_results = {}
    for image_file in sorted(os.listdir(image_dir)):
        if not (image_file.lower().endswith('.jpg') or image_file.lower().endswith('.png')):
            continue
        image_path = os.path.join(image_dir, image_file)
        # Only detect human (class 0)
        results = model.predict(image_path, save=False, conf=0.2, iou=0.7, classes=[0])
        # Save results as JSON
        result_data = []
        for r in results:
            for box in r.boxes:
                match = re.search(r'frame_(\d+)', image_file)
                frame_number = int(match.group(1)) if match else -1
                box_data = {
                    "coordinates": box.xyxy[0].tolist(),
                    "conf": float(box.conf[0]),
                    "cls": int(box.cls[0]),
                    # make frame number an int
                    "frame_name": frame_number
                }
                result_data.append(box_data)
        all_results[image_file] = result_data
        # Draw boxes and save image
        img = cv2.imread(image_path)
        for box in result_data:
            x1, y1, x2, y2 = map(int, box["coordinates"])
            conf = box["conf"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"Human {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        out_img_path = os.path.join(output_dir, image_file)
        cv2.imwrite(out_img_path, img)
    # Write all results to a single JSON file
    json_path = os.path.join(output_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

def convert_to_mp4():
    # Convert processed images back to mp4
    # Assumes output images are named in order (e.g., frame_0001.png, frame_0002.png, ...)
    os.system(f"ffmpeg -framerate 30 -i results/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' results/output.mp4")

if __name__ == "__main__":
    detect_human(image_dir)
    convert_to_mp4()
    
