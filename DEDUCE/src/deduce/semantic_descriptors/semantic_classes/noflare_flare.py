from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class FlareDescriptor(SemanticDescriptor):
    """Semantic descriptor for clear vs rain weather"""
    
    @property
    def name(self) -> str:
        return "noflare_flare"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'no flare': 'clear image',
            'flare': 'lens flare oversaturation', 
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} flare conditions"