"""
CLIP Image Encoder
Implements image encoding using OpenAI's CLIP model
"""

import torch
import torch.nn as nn
from typing import Union, List, Any, Dict
from PIL import Image
import numpy as np

from ..base import ImageEncoder

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class CLIPImageEncoder(ImageEncoder):
    """CLIP-based image encoder"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        
        self.model_name = config.get('model_name', 'ViT-B/32')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = None
        self.preprocess = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the CLIP model"""
        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            print(f"Loaded CLIP model {self.model_name} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model {self.model_name}: {e}")
    
    def encode(self, images: Union[torch.Tensor, List[str], List[Image.Image]]) -> torch.Tensor:
        """
        Encode images to embeddings
        
        Args:
            images: Can be:
                - torch.Tensor: Preprocessed image tensor (B, C, H, W)
                - List[str]: List of image file paths
                - List[Image.Image]: List of PIL images
                
        Returns:
            torch.Tensor: Normalized image embeddings (B, D)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Handle different input types
        if isinstance(images, torch.Tensor):
            # Assume already preprocessed
            image_tensor = images.to(self.device)
        elif isinstance(images, list):
            # Process list of paths or PIL images
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    # Load from path
                    pil_img = Image.open(img).convert('RGB')
                elif isinstance(img, Image.Image):
                    pil_img = img.convert('RGB')
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                
                processed_images.append(self.preprocess(pil_img))
            
            image_tensor = torch.stack(processed_images).to(self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(images)}")
        
        # Encode images
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            # Normalize features (CLIP does this internally but being explicit)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings"""
        if 'ViT-B/32' in self.model_name or 'RN50' in self.model_name:
            return 512
        elif 'ViT-L/14' in self.model_name:
            return 768
        else:
            # Default fallback - could also query the model directly
            return 512
    
    def preprocess_single_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Preprocess a single image for encoding"""
        if isinstance(image, str):
            pil_img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_img = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return self.preprocess(pil_img)
    
    def encode_single(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Convenience method to encode a single image"""
        if isinstance(image, torch.Tensor):
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            return self.encode(image)
        else:
            return self.encode([image])
    
    def __repr__(self) -> str:
        return f"CLIPImageEncoder(model_name='{self.model_name}', device='{self.device}')"