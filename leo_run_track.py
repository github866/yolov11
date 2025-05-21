import cv2
import json
import os
from argparse import ArgumentParser

from preprocess.init_bbox import read_init_frames
from utils_loc.utility import xywh_to_xyxy, id_to_name, color_to_room   
from src.detector.yolo import Detector
from src.tracker.leo_tracker import PersonTracker

class MOTWorker:
    def __init__(self, video_path: str, output_path: str, ref_feature_path=None):
        self.video_path = video_path
        self.output_path = output_path
        self.detector = Detector(ckpt='./leo/ckpt/yolo11x.pt')
        
        # Use the reference feature file if provided
        if ref_feature_path and os.path.exists(ref_feature_path):
            print(f"Using reference feature file: {ref_feature_path}")
            self.tracker = PersonTracker(reference_feature_path=ref_feature_path)
        else:
            # Default to the subject_features.pt in the tracker directory
            default_ref_path = os.path.join(os.path.dirname(__file__), 'src/tracker/subject_features.pt')
            print(f"Using default reference feature file: {default_ref_path}")
            self.tracker = PersonTracker(reference_feature_path=default_ref_path)
    
    def process_video(self, first_frame_bbxs, room_mask_path):
        room_mask = cv2.imread(room_mask_path, cv2.IMREAD_COLOR)
        self.tracker.initialize_first_frame(first_frame_bbxs)
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        data_log = {}
        for i in range(1, 8):
            data_log[i] = {
                'frame_id': [],
                'bboxs': [],
                'room': [],
                'identity': []
            }
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = self.detector.track(frame)
            bboxs = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes in xyxy format
            
            # Update tracker
            self.tracker.update(frame, bboxs)
            
            # Draw bounding boxes and IDs on the frame
            for person_id, bbox in self.tracker.all_people_bboxs.items():
                if bbox is None:
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
                
                # Get the reference identity if available
                ref_identity = self.tracker.get_identity(person_id)
                if ref_identity:
                    person_name = ref_identity
                    data_log[person_id]['identity'].append(ref_identity)
                else:
                    person_name = id_to_name(person_id)
                    data_log[person_id]['identity'].append(person_name)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, person_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.putText(frame, room_name, (int(x1), int(y1)-35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mask_value, 2)

            out.write(frame)
            frame_count += 1
            print(f'Processing frame {frame_count}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')
        
        # Save the data log
        output_json_path = self.output_path.replace('.mp4', '.json')
        with open(output_json_path, 'w') as f:
            json.dump(data_log, f, indent=4)
        print(f"Data log saved to {output_json_path}")

        cap.release()
        out.release()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str, default='/data/leohsu/human_dataset/humans/clips_02/clip_01.mp4')
    parser.add_argument('--output_path', type=str, default='./results/clips_02_01.mp4')
    parser.add_argument('--metadata_path', type=str, default='/data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame0001.data')
    parser.add_argument('--ref_feature_path', type=str, default='src/tracker/subject_features.pt', 
                        help='Path to the reference feature file')
    args = parser.parse_args()

    video_path = args.video_path
    output_path = args.output_path
    metadata_path = args.metadata_path
    ref_feature_path = args.ref_feature_path
    
    # Example bounding boxes for the first frame (x1, y1, x2, y2)
    initial_bounding_boxes = read_init_frames(metadata_path)
    
    first_frame_bbxs = []
    for bbox in initial_bounding_boxes:
        xyxy = xywh_to_xyxy(bbox)
        if xyxy is not None:
            first_frame_bbxs.append(xyxy)
        else:
            first_frame_bbxs.append([0, 0, 0, 0])  # Placeholder for invalid bbox
    
    mot_worker = MOTWorker(video_path, output_path, ref_feature_path)
    room_mask_path = './mask_visualization.png'
    mot_worker.process_video(first_frame_bbxs, room_mask_path)


    # 01
    # initial_bounding_boxes = [
        # [595.36, 417.17, 134.28, 320.03], 
        # [135.94, 347.41, 110.53, 197.61], 
        # [875.75, 419.34, 238.53, 169.06], 
        # [230.74, 414.42, 182.90, 213.61], 
        # [1044.48, 243.93, 149.47, 119.39],
        # [713.41, 78.57, 72.63, 185.01],
        # [769.82, 95.63, 71.31, 181.95], 
        # [489.89, 133.93, 91.32, 235.12],
        # [614.39, 86.20, 78.36, 197.15],
        # [862.51, 81.09, 88.11, 216.49], 
        # [758.02, 418.76, 200.42, 350.79], 
        # [220, 200, 60, 100]
    # ]
    # 901
    # 1801
    # 2701
    # 3601
    # 4501
    