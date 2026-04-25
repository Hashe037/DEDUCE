"""
Image encoders module
"""

try:
    from .openclip import OpenCLIPImageEncoder
    __all__ = ['OpenCLIPImageEncoder']
except ImportError:
    __all__ = []