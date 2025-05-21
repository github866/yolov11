import torch
import torch.nn as nn
import json
import os
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

class DINOFeatureExtractor(nn.Module):
    def __init__(self, model_name="dino_vits16"):
        super(DINOFeatureExtractor, self).__init__()
        self.model = torch.hub.load("facebookresearch/dino:main", model_name)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        # Apply transforms and add batch dimension
        tensor = self.transform(image)
        return tensor.unsqueeze(0)

    def forward(self, x):
        with torch.no_grad():
            # Get the features from the last layer
            features = self.model(x)
            # If features is a tuple, take the first element (usually the main output)
            if isinstance(features, tuple):
                features = features[0]
            return features

def save_cropped_image(image, coords, output_path):
    """Save the cropped image for verification"""
    cropped = image.crop((coords['x1'], coords['y1'], coords['x2'], coords['y2']))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cropped.save(output_path)

def process_subject(json_path, extractor, save_crops=False):
    # Read JSON file
    with open(json_path, 'r') as f:
        frames = json.load(f)
    
    features_list = []
    subject_name = Path(json_path).stem
    
    # Create directory for cropped images if needed
    if save_crops:
        crop_dir = os.path.join('cropped_images', subject_name)
        os.makedirs(crop_dir, exist_ok=True)
    
    # Process each frame
    for frame_info in tqdm(frames, desc=f"Processing {subject_name}"):
        frame_path = frame_info['frame_path']
        
        # # Extract frame number from the frame path
        # frame_name = Path(frame_path).stem
        # try:
        #     frame_number = int(frame_name.split('_')[-1])
        #     # Skip frames before 4501
        #     if frame_number < 4501:
        #         continue
        # except (ValueError, IndexError):
        #     # If frame number can't be extracted, process the frame anyway
        #     pass
            
        coords = frame_info['coordinates']
        
        # Load and crop image
        try:
            
            image = Image.open(frame_path).convert('RGB')
            
            # Save cropped image if requested
            if save_crops:
                frame_name = Path(frame_path).stem
                crop_path = os.path.join(crop_dir, f"{frame_name}_cropped.jpg")
                save_cropped_image(image, coords, crop_path)
            
            # Crop the image
            cropped_image = image.crop((coords['x1'], coords['y1'], coords['x2'], coords['y2']))
            
            # Preprocess and extract features
            tensor = extractor.preprocess(cropped_image)
            features = extractor(tensor)
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            continue
    
    if not features_list:
        return None
    
    # Stack and average features
    features_tensor = torch.cat(features_list, dim=0)
    # avg_features = torch.mean(features_tensor, dim=0)
    return avg_features

def main():
    parser = argparse.ArgumentParser(description="Extract and average DINO features for subjects")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory containing JSON files")
    parser.add_argument("--output_path", type=str, default="subject_features.pt", help="Path to save the features")
    parser.add_argument("--save_crops", action="store_true", default=False, help="Save cropped images for verification")
    args = parser.parse_args()

    # Initialize feature extractor
    extractor = DINOFeatureExtractor()
    
    # Process all JSON files
    subject_features = {}
    for json_file in os.listdir(args.logs_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(args.logs_dir, json_file)
            subject_name = Path(json_file).stem
            
            print(f"\nProcessing subject: {subject_name}")
            features = process_subject(json_path, extractor, args.save_crops)
            
            if features is not None:
                subject_features[subject_name] = features
    
    # Reorder subject features according to the specified sequence
    ordered_subject_features = {}
    desired_order = ["nurse_1", "nurse_2", "patient_1","patient_2","patient_3", "psychiatrist", "psychologist"]
    
    for key in desired_order:
        if key in subject_features:
            ordered_subject_features[key] = subject_features[key]
            print(f"Added {key} to ordered features")
        else:
            print(f"Warning: {key} not found in processed features")
    
    # Include any additional keys not in the desired order at the end
    for key in subject_features:
        if key not in ordered_subject_features:
            ordered_subject_features[key] = subject_features[key]
            print(f"Added additional subject: {key}")
    
    # Save all features
    torch.save(ordered_subject_features, args.output_path)
    print(f"\nFeatures saved to {args.output_path}")
    if args.save_crops:
        print(f"Cropped images saved in 'cropped_images' directory")

if __name__ == "__main__":
    main()