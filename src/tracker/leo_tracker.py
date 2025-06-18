import numpy as np
from typing import Literal
from .feature_extractor import DINOFeatureExtractor

class PersonTracker:
    def __init__(
            self,
            distance_threshold=100, 
            feature_similarity_threshold=0.8, 
            iou_threshold=0.7,
        ):
        
        self.all_people_bboxs = {} # person_id -> [x1, y1, x2, y2]
        self.all_people_feature = {} # person_id -> feature vector 

        # thresholds for different metrics
        self.feature_similarity_threshold = feature_similarity_threshold
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        # Initialize the feature extractor
        self.feature_extractor = DINOFeatureExtractor()
        # feature bank
        self.reference_feature = None  # Reference feature for similarity comparison
    
    def compute_feature_similarity(self, feature1, feature2):
        return np.dot(feature1, feature2)
    
    def compute_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def compute_iou_ratio(self, box1, box2):
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

        # Calculate intersection
        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)
        intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

        # Calculate union
        x1_j = min(x1_a, x1_b)
        y1_j = min(y1_a, y1_b)
        x2_j = max(x2_a, x2_b)
        y2_j = max(y2_a, y2_b)
        union_area = (x2_j - x1_j) * (y2_j - y1_j)
        return intersection_area / union_area if union_area > 0 else 0
    
    def initialize_first_frame(self, GT_bboxs, frame): # x1, y1, x2, y2
        for i, bbox in enumerate(GT_bboxs):
            if bbox == [0, 0, 0, 0]:
                self.all_people_bboxs[i+1] = None
                continue
            x1, y1, x2, y2 = bbox
            self.all_people_bboxs[i+1] = [x1, y1, x2, y2]
            self.all_people_feature[i+1] = self.feature_extractor.extract_features(
                frame[y1:y2, x1:x2]
            )

    def update(self, frame, bboxs, metrics: Literal["feature", "distance", "iou"] = "feature"):
        """
        Update tracker with new detections using both spatial and feature information
        
        Args:
            frame_number: Current frame number
            detections: List of dictionaries with detection info
            frame: The full frame image for feature extraction
        """
        if frame is None:
            return None
        
        if self.all_people_bboxs is None:
            print("No initial bounding boxes provided. Cannot update tracker.")
            return 
        
        candidate_bboxs = {}
        for key in self.all_people_bboxs.keys():
            candidate_bboxs[key] = None

        if metrics == "feature":
            for k, v in self.all_people_feature.items():
                similarity = 0
                for i, bbox in enumerate(bboxs):
                    x1_new, y1_new, x2_new, y2_new = map(int, bbox)
                    feature_vector_new = self.feature_extractor.extract_features(frame[y1_new:y2_new, x1_new:x2_new])
                        
                    # Compute similarity
                    new_similarity = self.compute_feature_similarity(v, feature_vector_new)
                        
                    if new_similarity > similarity and new_similarity > self.feature_similarity_threshold:
                        candidate_bboxs[k] = [i, similarity]
                        similarity = new_similarity

        elif metrics == "iou":
            for k, bbox in self.all_people_bboxs.items():
                iou_ratio = 0
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                for i, new_bbox in enumerate(bboxs):
                    x1_new, y1_new, x2_new, y2_new = new_bbox
                    new_iou_ratio = self.compute_iou_ratio((x1, y1, x2, y2), (x1_new, y1_new, x2_new, y2_new))
                    if new_iou_ratio > iou_ratio and new_iou_ratio > self.iou_threshold:
                        candidate_bboxs[k] = [i, new_iou_ratio]
                        iou_ratio = new_iou_ratio

        elif metrics == "distance":
            for k, bbox in self.all_people_bboxs.items():
                distance = 1e10
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                for i, new_bbox in enumerate(bboxs):
                    x1_new, y1_new, x2_new, y2_new = new_bbox
                    x_c = int((x1 + x2) / 2)
                    y_c = int((y1 + y2) / 2)
                    x_c_new = int((x1_new + x2_new) / 2)
                    y_c_new = int((y1_new + y2_new) / 2)
                    new_distance = self.compute_distance((x_c, y_c), (x_c_new, y_c_new))
                    if new_distance < distance and new_distance < self.distance_threshold:
                        candidate_bboxs[k] = [i, new_distance]
                        distance = new_distance

        # Check if the same bbx will be assigned to multiple keys
        bbx_to_keys = {}
        for k in self.all_people_bboxs.keys():
            if k in candidate_bboxs.keys() and candidate_bboxs[k] is not None:
                bbx_i, score = candidate_bboxs[k]
                if bbx_i not in bbx_to_keys:
                    bbx_to_keys[bbx_i] = [[k, score]]
                else:
                    bbx_to_keys[bbx_i].append([k, score])
        
        # Decide which key to assign the bbx to
        for bbx_i, keys in bbx_to_keys.items():
            # If multiple keys are assigned to the same bbx, choose the one with the highest score
            best_key = max(keys, key=lambda x: x[1])[0]
            self.all_people_bboxs[best_key] = bboxs[bbx_i]
            self.all_people_feature[best_key] = self.feature_extractor.extract_features(
                frame[int(bboxs[bbx_i][1]):int(bboxs[bbx_i][3]), int(bboxs[bbx_i][0]):int(bboxs[bbx_i][2])]
            )
    
