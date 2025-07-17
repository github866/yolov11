import cv2
import json
import os
from argparse import ArgumentParser

from leo_utils.preprocess.init_bbox_jone import read_init_frames
from utils_loc.utility import xywh_to_xyxy, id_to_name, color_to_room   
from src.detector.yolo import Detector
from src.tracker.leo_tracker import PersonTracker

class MOTWorker:
    def __init__(self, video_path: str, output_path: str, ckpt='./leo/ckpt/yolov11x_ours_best.pt'): # .mp4
        self.video_path = video_path
        self.output_path = output_path
        self.detection_output_path = output_path.replace('.mp4', '_detection.mp4')
        self.detector = Detector(ckpt=ckpt) 
        # './leo/ckpt/yolov11x_ours_best.pt'
        # './leo/ckpt/yolov8m_coco_ours_tuned.pt'
        self.tracker = PersonTracker()
        self.missing_log = {}
        self.similarity_log = {}
        self.write_video = True
    
    def process_video(self, first_frame_bbxs, room_mask_path):
        room_mask = cv2.imread(room_mask_path, cv2.IMREAD_COLOR)
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.write_video:
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        # You can use the same VideoWriter instance (out) for writing frames to a single output video file.
        # If you want to write to a different output video (e.g., detection_output_path), you need to create a separate VideoWriter:

        # You can use the same VideoWriter instance (out) for writing frames to a single output video file.
        # If you want to write to a different output video (e.g., detection_output_path), you need to create a separate VideoWriter:
        detection_out = cv2.VideoWriter(self.detection_output_path, fourcc, fps, (w, h))

        data_log = {}
        for i in range(1, 13):
            data_log[i] = {
                'frame_id': [],
                'bboxs': [],
                'room': []
            }
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count == 0:
                # Initialize tracker with the first frame bounding boxes
                if self.tracker.use_feature_bank:
                    self.tracker.init_feature_bank_by_average('/home/agenuinedream/repo/yolov11/data/feature_bank')
                else:
                    self.tracker.initialize_first_frame(first_frame_bbxs, frame)
                print(f"Initialized tracker with {len(first_frame_bbxs)} bounding boxes.")
            
            # Perform detection
            results = self.detector.detect(frame)
            bboxs = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes in xyxy format

            # Update tracker
            if self.tracker.use_feature_bank:
                self.tracker.associate(frame, bboxs, frame_count, self.missing_log, self.similarity_log)
            else:
                self.tracker.update(frame, bboxs)
            
            # Draw bounding boxes and IDs on the frame
            if self.write_video:
                for person_id, bbox in self.tracker.all_people_bboxs.items():
                    if bbox is None or person_id > 7:
                        print(f"Skipping person_id {person_id} with bbox {bbox}")
                        continue
                    if person_id in self.missing_log.keys() and frame_count in self.missing_log[person_id]:
                        print(f"Skipping person_id {person_id} at frame {frame_count} due to missing log")
                        continue
                    x1, y1, x2, y2 = bbox
                    ref_x = (x1 + x2) / 2
                    ref_y = ((y1+y2)/2 + y2) / 2

                    data_log[person_id]['bboxs'].append(list(map(float, [x1, y1, x2, y2])))
                    data_log[person_id]['frame_id'].append(frame_count)

                    # Check if the center of the bounding box is inside the room mask
                    mask_value = room_mask[int(ref_y), int(ref_x)]
                    mask_value = tuple(map(int, mask_value))  # Convert to tuple of integers
                    room_name= color_to_room(mask_value)
                    data_log[person_id]['room'].append(room_name)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f'{id_to_name(person_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                    cv2.putText(frame, room_name, (int(x1), int(y1)-35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mask_value, 2)

                out.write(frame)
            frame_count += 1
            print(f'Processing frame {frame_count}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')
        
        # Save the data log
        output_json_path = self.output_path.replace('.mp4', '.json')

        with open(output_json_path, 'w') as f:
            json.dump(data_log, f, indent=4)
        print(f"Data log saved to {output_json_path}")

        missing_log_path = self.output_path.replace('.mp4', '_missing_log.json')
        with open(missing_log_path, 'w') as f:
            json.dump(self.missing_log, f, indent=4)

        similarity_log_path = self.output_path.replace('.mp4', '_similarity_log.json')
        with open(similarity_log_path, 'w') as f:
            json.dump(self.similarity_log, f, indent=4)

        cap.release()
        if self.write_video:
            out.release()
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str, default='/data/leohsu/human_dataset/humans/loc_02/clip_01.mp4')
    parser.add_argument('--output_path', type=str, default='./results/loc_02_01.mp4')
    parser.add_argument('--metadata_path', type=str, default='/data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame0001.data')
    parser.add_argument('--room_mask_path', type=str, default='/home/agenuinedream/repo/yolov11/leo_utils/preprocess/mask_visualization.png')
    args = parser.parse_args()

    video_path = args.video_path
    output_path = args.output_path
    metadata_path = args.metadata_path
    room_mask_path = args.room_mask_path
    
    # Example bounding boxes for the first frame (x1, y1, x2, y2)
    # initial_bounding_boxes = read_init_frames(metadata_path)
    initial_bounding_boxes = read_init_frames(metadata_path)
    
    first_frame_bbxs = []
    for bbox in initial_bounding_boxes:
        xyxy = xywh_to_xyxy(bbox)
        if xyxy is not None:
            first_frame_bbxs.append(xyxy)
        else:
            first_frame_bbxs.append([0, 0, 0, 0])  # Placeholder for invalid bbox
    
    # ckpt = './leo/ckpt/yolov11x_ours_best.pt'
    ckpt = './leo/ckpt/yolov8m_coco_ours_tuned.pt'
    mot_worker = MOTWorker(video_path, output_path, ckpt)
    mot_worker.process_video(first_frame_bbxs, room_mask_path)
    