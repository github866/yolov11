import numpy as np
import torch
from typing import Literal
from .feature_extractor import DINOFeatureExtractor

class PersonTracker:
    def __init__(
            self,
            distance_threshold=100, 
            feature_similarity_threshold=0.8, 
            iou_threshold=0.7,
            reference_feature_path="subject_features.pt"
        ):
        
        self.all_people_bboxs = {}  # person_id -> feature vector 
        self.feature_similarity_threshold = feature_similarity_threshold
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.feature_extractor = DINOFeatureExtractor()
        self.reference_feature_path = reference_feature_path
        
        # Load the reference features
        self.reference_features = self.load_reference_features()
        
        # Map of person_id to reference feature name
        self.id_to_ref_name = {}
    
    def load_reference_features(self):
        # Load the reference feature from the .pt file
        print(f"Loading reference features from {self.reference_feature_path}")
        try:
            reference_features = torch.load(self.reference_feature_path)
            print(f"Successfully loaded features for: {list(reference_features.keys())}")
            return reference_features
        except Exception as e:
            print(f"Error loading reference features: {e}")
            return {}
    
    def compute_feature_similarity(self, feature1, feature2):
        # if they came from a “.pt” checkpoint you’ll typically have torch.Tensors:
        if isinstance(feature1, torch.Tensor):
            feature1 = feature1.cpu().numpy()
        if isinstance(feature2, torch.Tensor):
            feature2 = feature2.cpu().numpy()
            
        # guard against zero‐vectors
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # normalize and dot–product = cosine similarity
        feature1_norm = feature1 / norm1
        feature2_norm = feature2 / norm2
        similarity = np.dot(feature1_norm, feature2_norm)
        
        return float(similarity)
    
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
    
    def initialize_first_frame(self, GT_bboxs): # x1, y1, x2, y2
        for i, bbox in enumerate(GT_bboxs):
            if bbox == [0, 0, 0, 0]:
                self.all_people_bboxs[i+1] = None
                continue
            x1, y1, x2, y2 = bbox
            self.all_people_bboxs[i+1] = [x1, y1, x2, y2]

    def identify_with_reference(self, feature_vector):
        """
        Identify a person by comparing their feature vector with reference features
        
        Args:
            feature_vector: Feature vector extracted from the current frame
            
        Returns:
            best_name: The name of the best matching reference person
            best_similarity: The similarity score
        """
        best_name = None
        best_similarity = 0
        
        for ref_name, ref_feature in self.reference_features.items():
            # Convert PyTorch tensor to numpy if needed
            if isinstance(ref_feature, torch.Tensor):
                ref_feature = ref_feature.cpu().numpy()
                
            similarity = self.compute_feature_similarity(feature_vector, ref_feature)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = ref_name
                
        return best_name, best_similarity

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
            all_people_bboxs_feature = {}
            for k, v in self.all_people_bboxs.items():
                if self.all_people_bboxs[k] is None:
                    continue
                x1, y1, x2, y2 = map(int, v)
                # Extract features from the current frame
                feature_vector = self.feature_extractor.extract_features(frame[y1:y2, x1:x2])
                all_people_bboxs_feature[k] = feature_vector
                
                # Compare with reference features if available
                if self.reference_features and k not in self.id_to_ref_name:
                    ref_name, similarity = self.identify_with_reference(feature_vector)
                    if similarity > self.feature_similarity_threshold:
                        self.id_to_ref_name[k] = (ref_name, similarity)
                        print(f"Person {k} identified as {ref_name} with similarity {similarity:.4f}")

            for k, v in all_people_bboxs_feature.items():
                similarity, iou_ratio, distance = 0, 0, 1e10
                for i, bbox in enumerate(bboxs):
                    x1_new, y1_new, x2_new, y2_new = map(int, bbox)
                    feature_vector_new = self.feature_extractor.extract_features(frame[y1_new:y2_new, x1_new:x2_new])
                        
                    # Compute similarity
                    new_similarity = self.compute_feature_similarity(v, feature_vector_new)
                        
                    if new_similarity > similarity and new_similarity > self.feature_similarity_threshold:
                        candidate_bboxs[k] = [i, new_similarity]

        elif metrics == "iou":
                    new_iou_ratio = self.compute_iou_ratio((x1, y1, x2, y2), (x1_new, y1_new, x2_new, y2_new))
                    if new_iou_ratio > iou_ratio and new_iou_ratio > self.iou_threshold:
                        candidate_bboxs[k] = [i, new_iou_ratio]

        elif metrics == "distance":
            new_distance = self.compute_distance((x1, y1), (x1_new, y1_new))
            if new_distance < distance and self.distance_threshold:
                candidate_bboxs[k] = [i, new_distance*-1]

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
            if len(keys) == 1:
                k, score = keys[0]
                self.all_people_bboxs[k] = bboxs[bbx_i]
            elif len(keys) > 1:
                # If multiple keys are assigned to the same bbx, choose the one with the highest score
                best_key = max(keys, key=lambda x: x[1])[0]
                self.all_people_bboxs[best_key] = bboxs[bbx_i]
    
    def get_identity(self, person_id):
        """Get the identified reference name for a person_id if available"""
        if person_id in self.id_to_ref_name:
            return self.id_to_ref_name[person_id][0]
        return None
    
