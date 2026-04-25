from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class DayNightDescriptor(SemanticDescriptor):
    """Semantic descriptor for day vs night"""
    
    @property
    def name(self) -> str:
        return "day_night"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'day': 'daytime',
            'night': 'nighttime', 
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} lighting conditions"