import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import clip

class DINOFeatureExtractor:
    def __init__(self, model_name='dino_vits8', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], 
            #     std=[0.229, 0.224, 0.225]
            # )
        ])
    
    @torch.no_grad()
    def extract_features(self, image_crop):
        """Extract DINO features from an image crop"""
        # Convert crop to RGB if needed
        if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        img_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.model(img_tensor)
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy()[0]
    
class CLIPFeatureExtractor:
    def __init__(self, model_name='ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_features(self, image_crop):
        """Extract CLIP features from an image crop"""
        # Convert crop to PIL Image
        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        image_pil = transforms.ToPILImage()(image_crop)
        
        # Preprocess the image
        img_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.model.encode_image(img_tensor)
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy()[0]