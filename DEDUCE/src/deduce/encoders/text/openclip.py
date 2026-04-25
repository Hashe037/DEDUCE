"""
Unified OpenCLIP Text Encoder
Single encoder supporting all OpenCLIP models (CLIP, ViT, ConvNeXT, EVA, etc.)
Model selection via model_name parameter
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from ..base import TextEncoder

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


class OpenCLIPTextEncoder(TextEncoder):
    """
    Unified OpenCLIP text encoder supporting all available models
    
    Examples:
        # CLIP models
        config = {'model_name': 'ViT-B-32', 'pretrained': 'laion2b_s34b_b79k'}
        config = {'model_name': 'ViT-L-14', 'pretrained': 'laion2b_s32b_b82k'}
        
        # EVA models
        config = {'model_name': 'EVA02-L-14', 'pretrained': 'merged2b_s4b_b131k'}
        
        # Original CLIP (if available in open_clip)
        config = {'model_name': 'RN50', 'pretrained': 'openai'}
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
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.context_length = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the specified OpenCLIP model"""
        try:
            # Handle 'auto' pretrained - extract valid option for this model
            if self.pretrained == 'auto':
                all_options = open_clip.list_pretrained(self.model_name)
                valid_options = []
                for option in all_options:
                    if ':' in option:
                        model_part, dataset_part = option.split(':', 1)
                        if model_part == self.model_name:
                            valid_options.append(dataset_part)
                    else:
                        valid_options.append(option)
                
                self.pretrained = valid_options[0] if valid_options else None
            
            self.model, _, _ = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.context_length = self.model.context_length
            self.model.eval()
            
            print(f"Loaded OpenCLIP text model: {self.model_name}")
            if self.pretrained:
                print(f"  Pretrained weights: {self.pretrained}")
            print(f"  Context length: {self.context_length}")
            print(f"  Device: {self.device}")
            
        except Exception as e:
            available_models = open_clip.list_models()
            raise RuntimeError(
                f"Failed to load OpenCLIP model '{self.model_name}' with pretrained '{self.pretrained}': {e}\n"
                f"Available models: {available_models[:10]}... (showing first 10)"
            )
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            torch.Tensor: Normalized text embeddings (B, D)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings")
        
        # Tokenize texts using model-specific tokenizer
        text_tokens = self.tokenizer(texts).to(self.device)
        
        # Encode texts
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize features (standard for similarity computation)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features
    
    def encode_single(self, text: str) -> torch.Tensor:
        """Convenience method to encode a single text"""
        return self.encode([text])
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings"""
        # Try various ways to get embedding dimension
        if hasattr(self.model.text, 'output_dim'):
            return self.model.text.output_dim
        elif hasattr(self.model, 'embed_dim'):
            return self.model.embed_dim
        elif hasattr(self.model.text, 'embed_dim'):
            return self.model.text.embed_dim
        else:
            # Fallback: test with dummy input
            dummy_text = ["test"]
            with torch.no_grad():
                dummy_output = self.encode(dummy_text)
                return dummy_output.shape[-1]
    
    def get_max_length(self) -> int:
        """Get maximum sequence length"""
        return self.context_length or 77  # Default CLIP context length
    
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
        return open_clip.list_pretrained(model_name)
    
    @classmethod
    def get_model_info(cls, model_name: str = None) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not OPEN_CLIP_AVAILABLE:
            return {'error': 'OpenCLIP not available'}
        
        if model_name:
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
        return f"OpenCLIPTextEncoder(model='{self.model_name}', pretrained='{self.pretrained}', device='{self.device}')"