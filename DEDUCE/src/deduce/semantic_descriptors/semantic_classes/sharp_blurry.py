from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class SharpBlurryDescriptor(SemanticDescriptor):
    """Semantic descriptor for no damage vs damage"""
    
    @property
    def name(self) -> str:
        return "sharpy_blurry"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'sharp': 'sharp focused crisp detailed clear',
            'blurry': 'blurry unfocused soft motion blur hazy',
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} image conditions"