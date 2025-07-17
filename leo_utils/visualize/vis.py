import cv2
import numpy as np
import os
import sys
sys.path.append("/home/agenuinedream/repo/yolov11/src/")
from detector.yolo import Detector

def combine_two_videos_horizontally(video1_path, video2_path, output_path, tag1='', tag2=''):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error opening video files.")
        return

    fps = min(cap1.get(cv2.CAP_PROP_FPS), cap2.get(cv2.CAP_PROP_FPS))

    # Get frame size for both videos
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use the same height for both videos, pick the minimum height to avoid cropping
    out_height = min(h1, h2)
    out_width = w1 + w2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    color = (0, 255, 255)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Resize both frames to same height if needed
        if h1 != out_height:
            frame1 = cv2.resize(frame1, (w1, out_height))
        if h2 != out_height:
            frame2 = cv2.resize(frame2, (w2, out_height))

        # Add tags
        if tag1:
            cv2.putText(frame1, tag1, (15, 40), font, font_scale, color, thickness, cv2.LINE_AA)
        if tag2:
            cv2.putText(frame2, tag2, (15, 40), font, font_scale, color, thickness, cv2.LINE_AA)

        # Concatenate frames horizontally
        combined_frame = np.hstack((frame1, frame2))

        out.write(combined_frame)

    cap1.release()
    cap2.release()
    out.release()
    print(f"Combined video saved to {output_path}")

def visualize_detection_result(video_path, ckpt, output_dir):
    detector = Detector(ckpt=ckpt)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = video_path.split('/')[-1].replace('.mp4', '_detection.mp4')
    output_path = os.path.join(output_dir, video_name)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1
        results = detector.detect(frame)
        
        for i, (box, conf) in enumerate(zip (results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy())):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {i} Conf: {conf:.2f}', (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.imwrite(f'{output_dir}/frames/frame_{frame_cnt:04d}.png', frame)  # Save each frame as an image
        out.write(frame)

    cap.release()
    out.release()
    print(f"Detection video saved to {video_path.replace('.mp4', '_detection.mp4')}")

# Example usage:
# combine_two_videos_horizontally("video1.mp4", "video2.mp4", "output.mp4")

def main():
    loc_num = '02'
    clip_num = '06'
    video1_path = f'/home/agenuinedream/repo/yolov11/results/tracking/yolov8n_coco_ours_tuned/feature_bank_7/loc_{loc_num}_{clip_num}.mp4'
    video2_path = f'/home/agenuinedream/repo/yolov11/results/tracking/yolov11x_ours_tuned/loc_{loc_num}_{clip_num}.mp4'
    output_path = f'/home/agenuinedream/repo/yolov11/results/tracking/comparison/yolov8n_yolov11x_loc_{loc_num}_{clip_num}.mp4'
    combine_two_videos_horizontally(video1_path, video2_path, output_path, tag1='YOLOv8n', tag2='YOLOv11x')

    # video_path = '/data/leohsu/human_dataset/humans/Camera-Loc02.mp4'
    # ckpt = '/home/agenuinedream/repo/yolov11/leo/ckpt/yolov11x_ours_best.pt'
    # output_dir = '/home/agenuinedream/repo/yolov11/results/detection/yolov11x_ours_best'
    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
    # visualize_detection_result(video_path, ckpt, output_dir)

if __name__ == "__main__":
    main()
    # combine_two_videos_horizontally(video1_path, video2_path, 'output_combined.mp4')