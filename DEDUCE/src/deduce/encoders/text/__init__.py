"""
Text encoders module
"""

try:
    from .openclip import OpenCLIPTextEncoder
    __all__ = ['OpenCLIPTextEncoder']
except ImportError:
    __all__ = []