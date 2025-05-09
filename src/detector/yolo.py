from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, ckpt: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(ckpt).to(self.device)
        self.model.conf = 0.01  # Much lower confidence threshold for detections
        self.model.iou = 0.1    # Lower IoU threshold for NMS
        self.model.agnostic_nms = True  # Use class-agnostic NMS
        self.model.max_det = 12
        self.model.classes = [0]  # Only track person class

    def detect(self, img_path: str):
        results = self.model(
            img_path, 
            conf=0.01, 
            iou=0.1, 
            agnostic_nms=True,
            max_det=12, 
            classes=[0], 
            device=self.device
        )
        return results
    
    def track(self, img_path: str):
        results = self.model.track(
            img_path, 
            conf=0.01, 
            iou=0.1, 
            agnostic_nms=True,
            max_det=12, 
            classes=[0], 
            device=self.device
        )
        return results

