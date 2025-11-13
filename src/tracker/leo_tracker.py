import numpy as np
import os
import cv2
import json
from typing import Literal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .feature_extractor import DINOFeatureExtractor, CLIPFeatureExtractor
from scipy.optimize import linear_sum_assignment

class PersonTracker:
    def __init__(
            self,
            distance_threshold=100, 
            feature_similarity_threshold=0.5, 
            iou_threshold=0.7,
            feature_extractor=DINOFeatureExtractor(model_name='dino_vits8')
            # feature_extractor=CLIPFeatureExtractor(weights_path="/home/agenuinedream/repo/HSfM_RELEASE/checkpoints/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth")
        ):
        
        self.all_people_bboxs = {} # person_id -> [x1, y1, x2, y2]
        self.all_people_feature = {} # person_id -> feature vector
        self.feature_bank = {} # person_id -> feature vector 
        self.feature_bank_list = {}

        # thresholds for different metrics
        self.feature_similarity_threshold = feature_similarity_threshold
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        # Initialize the feature extractor
        self.feature_extractor = feature_extractor

        # feature bank
        self.use_feature_bank = True 
    
    def compute_feature_similarity(self, feature1, feature2):
        return np.dot(feature1, feature2)

    def compute_feature_similarity_clustering(self, feature_bank_features, query_feature):
        """
        Compute similarity between query feature and clustered feature bank.
        
        Args:
            feature_bank_features: Cluster centers for an identity (array of feature vectors)
            query_feature: Feature vector to compare against
            
        Returns:
            Maximum similarity score across all clusters
        """
        assert isinstance(feature_bank_features, np.ndarray) and feature_bank_features.ndim == 2
        # Multiple cluster centers
        similarities = [self.compute_feature_similarity(center, query_feature) 
                        for center in feature_bank_features]
        return max(similarities)

    def compute_feature_similarity_by_average(self, feature_bank_features:list, query_feature):
        """
        Compute similarity between query feature and average feature vector of a person.
        """
        assert isinstance(feature_bank_features, list)
        similarity = 0
        for feature in feature_bank_features:
            similarity += self.compute_feature_similarity(feature, query_feature)
        average_similarity = similarity / len(feature_bank_features)

        return average_similarity
    
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
            if not self.use_feature_bank:
                self.all_people_feature[i+1] = self.feature_extractor.extract_features(
                    frame[y1:y2, x1:x2]
                )
            else:
                if str(i+1) not in self.feature_bank:
                    self.feature_bank[str(i+1)] = self.feature_extractor.extract_features(
                        frame[y1:y2, x1:x2]
                    )

    def init_feature_bank(self, feature_bank_dir): 
        # get the directory of the feature bank
        for dir in os.listdir(feature_bank_dir):
            img_dir = os.path.join(feature_bank_dir, dir)
            if not os.path.isdir(img_dir):
                continue
            key = get_key_from_dir(dir)

            feature_list = []
            for img in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img)
                if not img.endswith('.jpg') and not img.endswith('.png'):
                    continue
                image_crop = cv2.imread(img_path)
                feature_vector = self.feature_extractor.extract_features(image_crop)
                feature_list.append(feature_vector)
            
            if len(feature_list) > 0:
                # Average the features to create a representative feature vector for the person
                self.feature_bank[key] = feature_list

    def init_feature_bank_by_average(self, feature_bank_dir_list):
        # get the directory of the feature bank
        for feature_bank_dir in feature_bank_dir_list:
            for dir in os.listdir(feature_bank_dir):
                img_dir = os.path.join(feature_bank_dir, dir)
                if not os.path.isdir(img_dir):
                    continue
                key = get_key_from_dir(dir)

                feature_list = []
                # recursively get all images in the directory
                for dir_path, dir_names, img_names in os.walk(img_dir):
                    for img in img_names:
                        img_path = os.path.join(dir_path, img)

                        if not img.endswith('.jpg') and not img.endswith('.png'):
                            continue
                        image_crop = cv2.imread(img_path)
                        feature_vector = self.feature_extractor.extract_features(image_crop)
                        feature_list.append(feature_vector)
                
                if len(feature_list) > 0:
                    # Average the features to create a representative feature vector for the person
                    if key not in self.feature_bank_list:
                        self.feature_bank_list[key] = []
                    self.feature_bank_list[key].append(np.mean(feature_list, axis=0))

        for key, feature_list in self.feature_bank_list.items():
            self.feature_bank[key] = np.mean(feature_list, axis=0)
            print(f"Initialized feature bank for {key} with {len(feature_list)} features.")

    def init_feature_bank_clustering(self, feature_bank_dir, max_clusters=5, min_clusters=1):
        """
        Initialize feature bank using K-means clustering to create multiple representative features per identity.
        
        Args:
            feature_bank_dir: Directory containing identity folders with images
            max_clusters: Maximum number of clusters per identity
            min_clusters: Minimum number of clusters per identity
        """
        for dir in os.listdir(feature_bank_dir):
            img_dir = os.path.join(feature_bank_dir, dir)
            if not os.path.isdir(img_dir):
                continue
                
            # Map directory names to keys
            key = get_key_from_dir(dir)

            feature_list = []
            for img in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img)
                if not img.endswith('.jpg') and not img.endswith('.png'):
                    continue
                image_crop = cv2.imread(img_path)
                feature_vector = self.feature_extractor.extract_features(image_crop)
                feature_list.append(feature_vector)
            
            if len(feature_list) > 0:
                features_array = np.array(feature_list)
                
                # Determine optimal number of clusters
                optimal_k = self._find_optimal_clusters(features_array, min_clusters, max_clusters)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(features_array)
                cluster_centers = kmeans.cluster_centers_
                
                # Store cluster centers as representative features
                self.feature_bank[key] = cluster_centers                

    def _find_optimal_clusters(self, features, min_k, max_k):
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            features: Array of feature vectors
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters
            
        Returns:
            Optimal number of clusters
        """
        if len(features) < 3:
            return 1
            
        max_k = min(max_k, len(features) - 1)
        if max_k < min_k:
            return min_k
            
        best_score = -1
        optimal_k = min_k
        
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(features)
                
                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(features, cluster_labels)
                    if score > best_score:
                        best_score = score
                        optimal_k = k
            except:
                continue
                
        return optimal_k

        
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
                    print(f"Comparing feature {k} with bbox {i}: similarity = {new_similarity:.4f}")
                    if new_similarity > similarity and new_similarity > self.feature_similarity_threshold:
                        similarity = new_similarity
                        candidate_bboxs[k] = [i, similarity]

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
            print(f"Assigning bbox {bbx_i} to key {best_key} with score {keys[0][1]}")
            self.all_people_bboxs[best_key] = bboxs[bbx_i]
            self.all_people_feature[best_key] = self.feature_extractor.extract_features(
                frame[int(bboxs[bbx_i][1]):int(bboxs[bbx_i][3]), int(bboxs[bbx_i][0]):int(bboxs[bbx_i][2])]
            )
    
    def associate(self, frame, bboxs, frame_cnt, missing_log, similarity_log):
        similarity_log[frame_cnt] = {}
        bbox_features = [
            self.feature_extractor.extract_features(frame[y1:y2, x1:x2])
            for (x1, y1, x2, y2) in map(lambda b: map(int, b), bboxs)
        ]

        person_prefs = {}  # key: person, value: sorted [(bbox_id, similarity)]
        assigned_bboxes = set()
        assignments = {}
        pointer = {}

        # Precompute all similarity lists
        for k, v in self.feature_bank.items():
            if v is None:
                print(f"Feature for person {k} is None, skipping association.")
                continue

            similarity_list = []
            for j, feature_vector_new in enumerate(bbox_features):
                sim = float(self.compute_feature_similarity(v, feature_vector_new))
                similarity_list.append((j, sim))
            similarity_list.sort(key=lambda x: x[1], reverse=True)
            person_prefs[k] = similarity_list
            pointer[k] = 0

        # Greedy matching with fallback
        unassigned = set(person_prefs.keys())
        while unassigned:
            proposals = {}  # bbox_id -> list of (person, similarity)
            to_remove = set()
            for person in unassigned:
                # Move pointer to next unassigned bbox
                while pointer[person] < len(person_prefs[person]) and \
                    person_prefs[person][pointer[person]][0] in assigned_bboxes:
                    pointer[person] += 1
                if pointer[person] >= len(person_prefs[person]):
                    # No bboxes left to assign
                    to_remove.add(person)
                    continue
                bbox_id, sim = person_prefs[person][pointer[person]]
                if sim < self.feature_similarity_threshold:
                    to_remove.add(person)
                    continue
                proposals.setdefault(bbox_id, []).append((person, sim))
            
            if not proposals:
                break

            # Resolve conflicts
            for bbox_id, candidates in proposals.items():
                # Assign bbox to person with highest similarity
                best_person, best_sim = max(candidates, key=lambda x: x[1])
                assignments[best_person] = bbox_id
                assigned_bboxes.add(bbox_id)
                to_remove.add(best_person)

            # Remove assigned people
            unassigned -= to_remove

        # Update self.all_people_bboxs, similarity_log, and missing_log as needed
        key_updated = set()
        for person, bbox_id in assignments.items():
            new_x1, new_y1, new_x2, new_y2 = map(int, bboxs[bbox_id])
            self.all_people_bboxs[person] = [new_x1, new_y1, new_x2, new_y2]
            key_updated.add(person)
            # Optionally update similarity_log here for tracking assignments

        # Update missing logs for people who did not get assigned
        for key in self.all_people_bboxs.keys():
            if key not in key_updated:
                if key not in missing_log:
                    missing_log[key] = []
                missing_log[key].append(frame_cnt)
        
    def hungarian_associate(self, frame, bboxs, frame_cnt, missing_log, similarity_log):
        # Get all person IDs and all bbox IDs
        person_keys = list(self.feature_bank.keys())
        n_persons = len(person_keys)
        n_bboxes = len(bboxs)
        
        # Precompute feature vectors for all detections
        bbox_features = []
        for bbox in bboxs:
            x1, y1, x2, y2 = map(int, bbox)
            feat = self.feature_extractor.extract_features(frame[y1:y2, x1:x2])
            bbox_features.append(feat)
        
        # Build similarity matrix [n_persons x n_bboxes]
        similarity_matrix = np.zeros((n_persons, n_bboxes))
        for i, person_id in enumerate(person_keys):
            person_feat = self.feature_bank[person_id]
            if person_feat is None:
                similarity_matrix[i, :] = -np.inf  # Very bad match
                continue
            for j, det_feat in enumerate(bbox_features):
                similarity = self.compute_feature_similarity(person_feat, det_feat)
                similarity_matrix[i, j] = similarity

        # If you want to threshold low similarities, set them to a large negative number (optional)
        similarity_matrix[similarity_matrix < self.feature_similarity_threshold] = -np.inf

        # Convert to cost matrix for Hungarian (maximize similarity -> minimize negative similarity)
        cost_matrix = -similarity_matrix

        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Log similarities for top 5 bboxes per person (for compatibility)
        similarity_log[frame_cnt] = {}
        for i, person_id in enumerate(person_keys):
            person_sim_list = [(j, float(np.round(similarity_matrix[i, j], 4))) for j in range(n_bboxes)]
            person_sim_list.sort(key=lambda x: x[1], reverse=True)
            similarity_log[frame_cnt][person_id] = {
                'bbox_id': [pair[0] for pair in person_sim_list[:5]],
                'similarity_list': [pair[1] for pair in person_sim_list[:5]]
            }

        # Update assignments
        key_updated = set()
        print("min similarity", min(similarity_matrix.flatten()))
        for i, j in zip(row_ind, col_ind):
            # Only assign if similarity is above threshold
            if similarity_matrix[i, j] > -np.inf:
                person_id = person_keys[i]
                key_updated.add(person_id)
                x1, y1, x2, y2 = map(int, bboxs[j])
                self.all_people_bboxs[person_id] = [x1, y1, x2, y2]
        
        # Update missing logs
        for key in self.all_people_bboxs.keys():
            if key not in key_updated:
                if key not in missing_log:
                    missing_log[key] = []
                missing_log[key].append(frame_cnt)

def get_key_from_dir(dir):
    if dir == "nurse_1":
        key = 1
    elif dir == "nurse_2":
        key = 2
    elif dir == "patient_1":
        key = 3
    elif dir == "patient_2":
        key = 4
    elif dir == "patient_3":
        key = 5
    elif dir == "psychiatrist":
        key = 6
    elif dir == "psychologist":
        key = 7
    elif dir == "person_1":
        key = 8
    elif dir == "person_2":
        key = 9
    elif dir == "person_3":
        key = 10
    elif dir == "researcher":
        key = 11
    else: # "person_4"
        key = 12

    return key

