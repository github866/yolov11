from ultralytics import YOLO
import torch
import numpy as np

class Detector:
    def __init__(self, ckpt: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(ckpt).to(self.device)
        self.model.conf = 0.01  # Much lower confidence threshold for detections
        self.model.iou = 0.1    # Lower IoU threshold for NMS
        self.model.agnostic_nms = True  # Use class-agnostic NMS
        self.model.max_det = 30
        self.model.classes = [0]  # Only track person class

    def detect(self, img: np.ndarray):
        results = self.model(
            img, 
            conf=0.6, 
            iou=0.6, # if iou > threshold, the detection will be discarded by nms
            agnostic_nms=False, #  Use class-specific NMS
            max_det=30, 
            classes=[0], 
            device=self.device
        )
        return results
    
    def track(self, img: np.ndarray):
        results = self.model.track(
            img, 
            conf=0.2, 
            iou=0.9, 
            agnostic_nms=False,  # Use class-specific NMS
            max_det=30, 
            classes=[0], 
            device=self.device, 
            cfg='botsort.yaml',  # Use the custom tracker configuration
        )
        return results

