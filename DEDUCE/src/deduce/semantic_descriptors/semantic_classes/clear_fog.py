from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class ClearFogDescriptor(SemanticDescriptor):
    """Semantic descriptor for clear vs fog weather"""
    
    @property
    def name(self) -> str:
        return "clear_fog"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'clear': 'clear weather high visibility',
            'fog': 'fog hazy low visibility', 
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} weather conditions"