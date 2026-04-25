from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class VegetationDescriptor(SemanticDescriptor):
    """Semantic descriptor for no damage vs damage"""
    
    @property
    def name(self) -> str:
        return " "
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'no vegetation': 'barren concrete paved urban treeless',
            'vegetation': 'trees greenery foliage plants nature vegetation',
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} vegetation conditions"