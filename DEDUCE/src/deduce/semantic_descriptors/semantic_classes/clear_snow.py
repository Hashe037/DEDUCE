from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class ClearSnowDescriptor(SemanticDescriptor):
    """Semantic descriptor for clear vs snow weather"""
    
    @property
    def name(self) -> str:
        return "clear_snow"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'clear': 'clear weather high visibility',
            'snow': 'snow cold ice', 
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} weather conditions"