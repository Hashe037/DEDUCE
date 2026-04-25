from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class DensityDescriptor(SemanticDescriptor):
    """Semantic descriptor for no damage vs damage"""
    
    @property
    def name(self) -> str:
        return "lowdense_highdense"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'low density': 'sparse scattered isolated few buildings',
            'high density': 'dense crowded packed urban cityscape',
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} density conditions"