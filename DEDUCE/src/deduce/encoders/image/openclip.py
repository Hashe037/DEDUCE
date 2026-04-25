"""
Fixed OpenCLIP Image Encoder
Properly handles pretrained parameter and model name formatting
"""

import torch
import torch.nn.functional as F
from typing import Union, List, Any, Dict
from PIL import Image

from ..base import ImageEncoder

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


class OpenCLIPImageEncoder(ImageEncoder):
    """
    OpenCLIP image encoder supporting all available models
    
    Examples:
        # Standard models
        config = {'model_name': 'ViT-B-32', 'pretrained': 'laion2b_s34b_b79k'}
        config = {'model_name': 'ViT-L-14', 'pretrained': 'laion2b_s32b_b82k'}
        
        # Perception models
        config = {'model_name': 'PE-Core-L-14-336', 'pretrained': 'metaclip_5_4b_b32k'}
        
        # SigLIP models
        config = {'model_name': 'ViT-SO400M-14-SigLIP-384', 'pretrained': 'webli'}
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "open_clip not available. Install with: pip install open-clip-torch"
            )
        
        self.model_name = config.get('model_name', 'ViT-B-32')
        self.pretrained = config.get('pretrained', 'laion2b_s34b_b79k')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Normalize model name format (replace slashes with dashes)
        self.model_name = self.model_name.replace('/', '-')
        
        # Model components
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the specified OpenCLIP model"""
        try:
            print(f"Loading OpenCLIP model: {self.model_name} with pretrained: {self.pretrained}")
            
            # Handle 'auto' pretrained - find best match for this model
            if self.pretrained == 'auto':
                available_pretrained = self._get_available_pretrained_for_model(self.model_name)
                if available_pretrained:
                    self.pretrained = available_pretrained[0]
                    print(f"Auto-selected pretrained: {self.pretrained}")
                else:
                    self.pretrained = None
                    print("No pretrained weights found, using random initialization")
            
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()
            
            print(f"✅ Loaded OpenCLIP image model: {self.model_name}")
            if self.pretrained:
                print(f"   Pretrained weights: {self.pretrained}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            # Provide helpful error message with available options
            print(f"❌ Failed to load model {self.model_name} with pretrained {self.pretrained}")
            print(f"Error: {e}")
            
            # Show available pretrained options for this specific model
            available_for_model = self._get_available_pretrained_for_model(self.model_name)
            if available_for_model:
                print(f"Available pretrained options for {self.model_name}: {available_for_model}")
            else:
                available_models = open_clip.list_models()
                print(f"Available models: {available_models[:10]}... (showing first 10)")
            
            raise RuntimeError(
                f"Failed to load OpenCLIP model '{self.model_name}' with pretrained '{self.pretrained}': {e}\n"
                f"Available pretrained for {self.model_name}: {available_for_model}"
            )
    
    def _get_available_pretrained_for_model(self, model_name: str) -> List[str]:
        """Get pretrained options specifically for this model"""
        try:
            # Get all pretrained options from OpenCLIP
            all_pretrained = open_clip.list_pretrained(model_name)
            return all_pretrained
        except Exception as e:
            print(f"Error getting pretrained options for {model_name}: {e}")
            return []
    
    def encode(self, images: Union[torch.Tensor, List[str], List[Image.Image]]) -> torch.Tensor:
        """
        Encode images to embeddings
        
        Args:
            images: Images in various formats
                
        Returns:
            torch.Tensor: Normalized image embeddings (B, D)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Handle different input types
        if isinstance(images, torch.Tensor):
            # Handle tensor input - may need preprocessing
            if images.dim() == 4:  # (B, C, H, W)
                # Check if image needs resizing to model's expected input size
                expected_size = getattr(self.model.visual, 'image_size', 224)
                if isinstance(expected_size, (list, tuple)):
                    expected_size = expected_size[0]  # Assume square
                
                if images.shape[-2:] != (expected_size, expected_size):
                    images = F.interpolate(
                        images, size=(expected_size, expected_size), 
                        mode='bilinear', align_corners=False
                    )
            image_tensor = images.to(self.device)
            
        elif isinstance(images, list):
            # Process list of paths or PIL images using model's preprocessing
            processed_images = []
            for img in images:
                if isinstance(img, str):
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
            # Normalize features (standard for similarity computation)
            image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings"""
        # Try various ways to get embedding dimension
        if hasattr(self.model.visual, 'output_dim'):
            return self.model.visual.output_dim
        elif hasattr(self.model, 'embed_dim'):
            return self.model.embed_dim
        elif hasattr(self.model.visual, 'embed_dim'):
            return self.model.visual.embed_dim
        else:
            # Fallback: test with dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                dummy_output = self.model.encode_image(dummy_input)
                return dummy_output.shape[-1]
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all available model architectures"""
        if not OPEN_CLIP_AVAILABLE:
            return []
        return open_clip.list_models()
    
    @classmethod 
    def get_available_pretrained(cls, model_name: str) -> List[str]:
        """Get available pretrained weights for a specific model"""
        if not OPEN_CLIP_AVAILABLE:
            return []
        
        # Normalize model name
        model_name = model_name.replace('/', '-')
        return open_clip.list_pretrained(model_name)
    
    @classmethod
    def get_model_info(cls, model_name: str = None) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not OPEN_CLIP_AVAILABLE:
            return {'error': 'OpenCLIP not available'}
        
        if model_name:
            # Normalize model name
            model_name = model_name.replace('/', '-')
            return {
                'model_name': model_name,
                'pretrained_options': cls.get_available_pretrained(model_name),
                'backend': 'open_clip'
            }
        else:
            return {
                'available_models': cls.get_available_models(),
                'backend': 'open_clip',
                'total_models': len(cls.get_available_models())
            }
    
    def __repr__(self) -> str:
        return f"OpenCLIPImageEncoder(model='{self.model_name}', pretrained='{self.pretrained}', device='{self.device}')"