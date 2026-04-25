"""
Lighting Semantic Descriptor
Describes different lighting conditions and qualities
"""

from typing import Dict
from ..base import SemanticDescriptor


class LightingDescriptor(SemanticDescriptor):
    """Semantic descriptor for lighting condition categories"""
    
    @property
    def name(self) -> str:
        return "lighting"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define lighting categories and their semantic descriptions"""
        return {
            'bright': 'bright well-lit illumination',
            'dim': 'dim low-light conditions',
            'dramatic': 'dramatic high-contrast lighting',
            'soft': 'soft diffused lighting',
            'harsh': 'harsh direct lighting',
            'backlit': 'backlit silhouette lighting',
            'natural': 'natural daylight illumination',
            'artificial': 'artificial indoor lighting'
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown weather categories"""
        return f"{category} lighting"
        
