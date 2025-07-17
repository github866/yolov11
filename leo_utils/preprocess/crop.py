import json
import cv2
import os
import sys 
sys.path.append('/home/agenuinedream/repo/yolov11/src')
from detector.yolo import Detector

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

def extract_by_detection(ckpt, json_path, video_path, output_dir, base_num=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    detector = Detector(ckpt)
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(data.keys())

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_cnt = 0
    while True:
        frame_cnt += 1
        # print(f"Processing frame {frame_cnt}")
        ret, frame = cap.read()
        if not ret:
            break

        if str(frame_cnt) not in data.keys():
            continue
        print(f"Processing frame {frame_cnt} with data: {data[str(frame_cnt)]}")

        results = detector.detect(frame)
        bboxs = results[0].boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = map(int, bboxs[data[str(frame_cnt)]])
        cropped_img = frame[y1:y2, x1:x2]
        cv2.imwrite(f"{output_dir}/frame_{frame_cnt:04d}.png", cropped_img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract cropped images from frames based on JSON coordinates.")
    parser.add_argument('--img_dir', type=str, help='Directory containing the original images.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file with coordinates.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the cropped images.')
    parser.add_argument('--ckpt', type=str, help='Path to the YOLO model checkpoint.')
    parser.add_argument('--video_path', type=str, help='Path to the video file for detection.')

    args = parser.parse_args()

    # extract_cropped_imgs(args.img_dir, args.json_path, args.output_dir)
    extract_by_detection(
        json_path=args.json_path, # /home/agenuinedream/repo/yolov11/data/feature_crop_by_detect/patient_3.json
        video_path=args.video_path, # /home/agenuinedream/repo/yolov11/data/feature_crop_by_detect/patient_3
        ckpt=args.ckpt,
        output_dir=args.output_dir, # /home/agenuinedream/repo/yolov11/data/feature_bank/patient_3
    )

'''
python /home/agenuinedream/repo/yolov11/preprocess/crop.py --json_path /home/agenuinedream/repo/yolov11/data/feature_crop_by_detect/yolov8n_coco_ours_tuned/person_1.json --video_path /data/leohsu/human_dataset/humans/Camera-Loc02.mp4 --ckpt /home/agenuinedream/repo/yolov11/leo/ckpt/yolov8n_coco_ours_tuned.pt --output_dir /home/agenuinedream/repo/yolov11/data/feature_bank/person_1
python /home/agenuinedream/repo/yolov11/preprocess/crop.py --json_path /home/agenuinedream/repo/yolov11/data/feature_crop_by_detect/yolov8n_coco_ours_tuned/person_2.json --video_path /data/leohsu/human_dataset/humans/Camera-Loc02.mp4 --ckpt /home/agenuinedream/repo/yolov11/leo/ckpt/yolov8n_coco_ours_tuned.pt --output_dir /home/agenuinedream/repo/yolov11/data/feature_bank/person_2
python /home/agenuinedream/repo/yolov11/preprocess/crop.py --json_path /home/agenuinedream/repo/yolov11/data/feature_crop_by_detect/yolov8n_coco_ours_tuned/person_3.json --video_path /data/leohsu/human_dataset/humans/Camera-Loc02.mp4 --ckpt /home/agenuinedream/repo/yolov11/leo/ckpt/yolov8n_coco_ours_tuned.pt --output_dir /home/agenuinedream/repo/yolov11/data/feature_bank/person_3
python /home/agenuinedream/repo/yolov11/preprocess/crop.py --json_path /home/agenuinedream/repo/yolov11/data/feature_crop_by_detect/yolov8n_coco_ours_tuned/person_4.json --video_path /data/leohsu/human_dataset/humans/Camera-Loc02.mp4 --ckpt /home/agenuinedream/repo/yolov11/leo/ckpt/yolov8n_coco_ours_tuned.pt --output_dir /home/agenuinedream/repo/yolov11/data/feature_bank/person_4


'''