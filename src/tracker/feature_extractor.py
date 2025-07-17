import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import clip

class DINOFeatureExtractor:
    # 'dino_vits8': (8, 16) https://huggingface.co/facebook/dino-vitb8
    # 'dinov2_vitb14': (14) https://huggingface.co/collections/facebook/dinov2-6526c98554b3d2576e071ce3
    # 'facebookresearch/dino:main'
    def __init__(self, model_name='dino_vits8', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        if 'dinov2' in model_name:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
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
    def __init__(
        self, 
        model_name='ViT-B/16', 
        device='cuda' if torch.cuda.is_available() else 'cpu', 
        weights_path="None"
    ):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
    
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