from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class SummerWinterDescriptor(SemanticDescriptor):
    """Semantic descriptor for clear vs snow weather"""
    
    @property
    def name(self) -> str:
        return "summer_winter"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'summer': 'clear summer',
            'winter': 'snow winter', 
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} weather conditions"