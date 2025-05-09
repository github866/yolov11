from ultralytics import YOLO
import json
import cv2

class CustomTracker:
    def __init__(self, model, classes=[0], conf=0.1, iou=0.95):
        self.model = model
        self.classes = classes
        self.conf = conf
        self.iou = iou

    def track(self, video_path, save=True, project='runs/track', name='camera_loc'):
        results = self.model.track(
            source=video_path,
            conf=self.conf,
            iou=self.iou,
            save=save,
            project=project,
            name=name,
            classes=self.classes,
            exist_ok=True
        )
        return results
    
    def track_by_detection(self, video_path, target_id=['']):
        cv2.VideoCapture(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # detect the video frame
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = self.model(frame, conf=self.conf)
            detections = results[0].boxes.xyxy.cpu().numpy()

            # check if the target_id is in the detections
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                if int(cls) in target_id:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {int(cls)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    model = YOLO('ckpt/yolo11x.pt')  # Load a pretrained YOLOv11 model
    src_list = [
        # '/data/leohsu/human_dataset/humans/test/Camera-Loc01_38to48.mp4',
        # '/data/leohsu/human_dataset/humans/test/Camera-Loc02_38to48.mp4', 
        # '/data/leohsu/human_dataset/humans/test/Camera-Loc03_38to48.mp4',
        # '/data/leohsu/human_dataset/humans/test/Camera-Loc04_38to48.mp4',
        # '/data/leohsu/human_dataset/humans/Camera-Loc01.mp4',
        '/data/leohsu/human_dataset/humans/Camera-Loc02.mp4',
        '/data/leohsu/human_dataset/humans/Camera-Loc03.mp4',
        # '/data/leohsu/human_dataset/humans/Camera-Loc04.mp4',
    ]
    
    for src in src_list:
        results = model.track(
            source=src,
            conf=0.1, 
            iou=0.95, 
            save=True,
            project='runs/track',
            name='camera_loc',
            classes=[0], 
            exist_ok=True,
        )

        # print(f"Results: {results}")

        output = []

        for frame_id, frame in enumerate(results):
            frame_data = []
            for det in frame.boxes:
                item = {
                    "frame_id": frame_id,
                    "track_id": int(det.id.item()) if det.id is not None else -1,
                    "class_id": int(det.cls.item()),
                    "conf": float(det.conf.item()),
                    "bbox": [float(x) for x in det.xyxy[0].tolist()]  # [x1, y1, x2, y2]
                }
                frame_data.append(item)
            output.extend(frame_data)

            # Save to JSON
            with open(src.split('/')[-1]+'.json', 'w') as f:
                json.dump(output, f, indent=2)

if __name__=="__main__":
    main()
