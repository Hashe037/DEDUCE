"""
Encoders module
Provides image and text encoders for the semantic embeddings framework
"""

from .base import BaseEncoder, ImageEncoder, TextEncoder
from .registry import EncoderRegistry

# Import openclip encoders
try:
    from .image.openclip import OpenCLIPImageEncoder
except ImportError:
    OpenCLIPImageEncoder = None

try:
    from .text.openclip import OpenCLIPTextEncoder
except ImportError:
    OpenCLIPTextEncoder = None


__all__ = [
    'BaseEncoder',
    'ImageEncoder', 
    'TextEncoder',
    'EncoderRegistry',
    'OpenCLIPImageEncoder',
    'OpenCLIPTextEncoder',
]