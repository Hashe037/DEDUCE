"""
Base encoder classes
"""

from abc import ABC, abstractmethod
import torch
from typing import Union, List, Any, Dict


class BaseEncoder(ABC):
    """Abstract base class for all encoders"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def encode(self, input_data: Any) -> torch.Tensor:
        """Encode input data to embeddings"""
        pass
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the underlying model"""
        pass
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings"""
        # Default implementation - should be overridden
        return 512
    
    def to(self, device: str):
        """Move encoder to device"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.to(device)
        return self
    
    def eval(self):
        """Set encoder to evaluation mode"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set encoder to training mode"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.train(mode)
        return self


#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------


class ImageEncoder(BaseEncoder):
    """Base class for image encoders"""
    
    @abstractmethod
    def encode(self, images: Union[torch.Tensor, List[str]]) -> torch.Tensor:
        """Encode images to embeddings"""
        pass
    
    def encode_batch(self, images: Union[torch.Tensor, List[str]], batch_size: int = 32) -> torch.Tensor:
        """Encode images in batches for memory efficiency"""
        if isinstance(images, torch.Tensor):
            # Process tensor in batches
            all_embeddings = []
            for i in range(0, images.size(0), batch_size):
                batch = images[i:i+batch_size]
                embeddings = self.encode(batch)
                all_embeddings.append(embeddings)
            return torch.cat(all_embeddings, dim=0)
        elif isinstance(images, list):
            # Process list in batches
            all_embeddings = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                embeddings = self.encode(batch)
                all_embeddings.append(embeddings)
            return torch.cat(all_embeddings, dim=0)
        else:
            raise ValueError(f"Unsupported input type: {type(images)}")



class TextEncoder(BaseEncoder):
    """Base class for text encoders"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings"""
        pass
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts in batches for memory efficiency"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)
    
    def get_max_length(self) -> int:
        """Get maximum sequence length supported by the encoder"""
        # Default implementation - should be overridden
        return 512