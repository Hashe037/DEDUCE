"""
CLIP Text Encoder
Implements text encoding using OpenAI's CLIP model
"""

import torch
from typing import List, Dict, Any, Union

from ..base import TextEncoder

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class CLIPTextEncoder(TextEncoder):
    """CLIP-based text encoder"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        
        self.model_name = config.get('model_name', 'ViT-B/32')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = config.get('max_length', 77)  # CLIP's context length
        
        # Load model
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the CLIP model"""
        try:
            self.model, _ = clip.load(self.model_name, device=self.device)
            self.model.eval()
            print(f"Loaded CLIP text model {self.model_name} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model {self.model_name}: {e}")
    
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
        
        # Tokenize texts
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        
        # Encode texts
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize features (CLIP does this internally but being explicit)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def encode_single(self, text: str) -> torch.Tensor:
        """Convenience method to encode a single text"""
        return self.encode([text])
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings"""
        if 'ViT-B/32' in self.model_name or 'RN50' in self.model_name:
            return 512
        elif 'ViT-L/14' in self.model_name:
            return 768
        else:
            # Default fallback
            return 512
    
    def tokenize(self, texts: List[str]) -> torch.Tensor:
        """Tokenize texts without encoding"""
        return clip.tokenize(texts, truncate=True)
    
    def get_max_length(self) -> int:
        """Get maximum sequence length"""
        return self.max_length
    
    def __repr__(self) -> str:
        return f"CLIPTextEncoder(model_name='{self.model_name}', device='{self.device}')"