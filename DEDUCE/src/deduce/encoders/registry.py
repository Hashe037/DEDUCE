"""
Fixed Encoder Registry System
Fixes the pretrained value override issue and model name format issue
"""

from typing import Dict, Type, Any, Optional, List
from .base import ImageEncoder, TextEncoder, BaseEncoder


class EncoderRegistry:
    """Registry for OpenCLIP encoders with unified model access"""
    
    def __init__(self):
        self._image_encoders: Dict[str, Type[ImageEncoder]] = {}
        self._text_encoders: Dict[str, Type[TextEncoder]] = {}
        self._register_builtin_encoders()
    
    def _register_builtin_encoders(self):
        """Register unified OpenCLIP encoder"""
        
        # Register OpenCLIP as the primary (and likely only) encoder
        try:
            from .image.openclip import OpenCLIPImageEncoder
            from .text.openclip import OpenCLIPTextEncoder
            
            # Register as primary encoder
            self.register_image_encoder("openclip", OpenCLIPImageEncoder)
            self.register_text_encoder("openclip", OpenCLIPTextEncoder)
            
        except ImportError:
            print("⚠️ OpenCLIP not available - no encoders registered")
        
    
    def register_image_encoder(self, name: str, encoder_class: Type[ImageEncoder]):
        """Register an image encoder class"""
        if not issubclass(encoder_class, ImageEncoder):
            raise ValueError(f"Encoder class must inherit from ImageEncoder")
        self._image_encoders[name] = encoder_class
    
    def register_text_encoder(self, name: str, encoder_class: Type[TextEncoder]):
        """Register a text encoder class"""
        if not issubclass(encoder_class, TextEncoder):
            raise ValueError(f"Encoder class must inherit from TextEncoder")
        self._text_encoders[name] = encoder_class
    
    def create_image_encoder(self, name: str, config: Dict[str, Any]) -> ImageEncoder:
        """Create an image encoder instance"""
        if name not in self._image_encoders:
            raise ValueError(f"Unknown image encoder: {name}. "
                           f"Available: {list(self._image_encoders.keys())}")
        
        encoder_class = self._image_encoders[name]
        
        # Auto-enhance config for OpenCLIP (backward compatibility)
        if encoder_class.__name__ == 'OpenCLIPImageEncoder':
            config = self._enhance_openclip_config(config)
        
        return encoder_class(config)
    
    def create_text_encoder(self, name: str, config: Dict[str, Any]) -> TextEncoder:
        """Create a text encoder instance"""
        if name not in self._text_encoders:
            raise ValueError(f"Unknown text encoder: {name}. "
                           f"Available: {list(self._text_encoders.keys())}")
        
        encoder_class = self._text_encoders[name]
        
        # Auto-enhance config for OpenCLIP (backward compatibility)
        if encoder_class.__name__ == 'OpenCLIPTextEncoder':
            config = self._enhance_openclip_config(config)
        
        return encoder_class(config)
    
    def _enhance_openclip_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-enhance config for OpenCLIP with sensible defaults and backward compatibility"""
        enhanced_config = config.copy()
        
        model_name = enhanced_config.get('model_name', 'ViT-B-32')
        
        # Fix model name format: Convert slashes to dashes for OpenCLIP
        # OpenCLIP expects 'ViT-B-32' not 'ViT-B/32'
        model_name = model_name.replace('/', '-')
        enhanced_config['model_name'] = model_name

        # IMPORTANT: Only set pretrained default if it's not already specified
        # This was the main bug - we were always overriding the user's pretrained choice
        if 'pretrained' not in enhanced_config or enhanced_config['pretrained'] is None:
            pretrained_defaults = {
                'ViT-B-32': 'laion2b_s34b_b79k',
                'ViT-B-16': 'laion2b_s34b_b88k', 
                'ViT-L-14': 'laion2b_s32b_b82k',
                'ViT-L-14-336': 'openai',
                'ViT-H-14': 'laion2b_s32b_b79k',
                'ViT-bigG-14': 'laion2b_s34b_b88k',
                'PE-Core-L-14-336': 'metaclip_5_4b_b32k',
                'PE-Core-bigG-14-448': 'metaclip_5_4b_b16k',
                'ViT-SO400M-14-SigLIP': 'webli',
                'ViT-SO400M-14-SigLIP-384': 'webli',
                'EVA02-L-14': 'merged2b_s4b_b131k',
                'EVA02-E-14': 'laion2b_s4b_b115k',
                'RN50': 'openai',
                'RN101': 'openai'
            }
            
            enhanced_config['pretrained'] = pretrained_defaults.get(model_name, 'auto')
        
        # Auto-detect image size if not specified
        if 'image_size' not in enhanced_config:
            # Extract size from model name if present
            for size_str in ['448', '384', '378', '336', '320', '256']:
                if size_str in model_name:
                    enhanced_config['image_size'] = int(size_str)
                    break
            else:
                # Default to 224px for standard models
                enhanced_config['image_size'] = 224
        
        # Set normalize default (false for OpenCLIP since it handles normalization internally)
        if 'normalize' not in enhanced_config:
            enhanced_config['normalize'] = False
        
        return enhanced_config
    
    def list_image_encoders(self) -> List[str]:
        """List available image encoder names"""
        return list(self._image_encoders.keys())
    
    def list_text_encoders(self) -> List[str]:
        """List available text encoder names"""
        return list(self._text_encoders.keys())
    
    def get_available_models(self) -> List[str]:
        """Get all available OpenCLIP models"""
        try:
            from .image.openclip import OpenCLIPImageEncoder
            return OpenCLIPImageEncoder.get_available_models()
        except ImportError:
            return []
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get comprehensive model information"""
        try:
            from .image.openclip import OpenCLIPImageEncoder
            return OpenCLIPImageEncoder.get_model_info(model_name)
        except ImportError:
            return {'error': 'OpenCLIP not available'}