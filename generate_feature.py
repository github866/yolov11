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
        return self.transform(image).unsqueeze(0)

    def forward(self, x):
        with torch.no_grad():
            return self.model.forward_features(x)

def process_subject(json_path, extractor):
    # Read JSON file
    with open(json_path, 'r') as f:
        frames = json.load(f)
    
    features_list = []
    
    # Process each frame
    for frame_info in tqdm(frames, desc=f"Processing {Path(json_path).stem}"):
        frame_path = frame_info['frame_path']
        coords = frame_info['coordinates']
        
        # Load and crop image
        try:
            image = Image.open(frame_path).convert('RGB')
            cropped_image = image.crop((coords['x1'], coords['y1'], coords['x2'], coords['y2']))

            
            # Extract features
            features = extractor(cropped_image)
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            continue
    
    if not features_list:
        return None
    
    # Stack and average features
    features_tensor = torch.cat(features_list, dim=0)
    avg_features = torch.mean(features_tensor, dim=0)
    return avg_features

def main():
    parser = argparse.ArgumentParser(description="Extract and average DINO features for subjects")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory containing JSON files")
    parser.add_argument("--output_path", type=str, default="subject_features.pt", help="Path to save the features")
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
            features = process_subject(json_path, extractor)
            
            if features is not None:
                subject_features[subject_name] = features
    
    # Save all features
    torch.save(subject_features, args.output_path)
    print(f"\nFeatures saved to {args.output_path}")

if __name__ == "__main__":
    main()