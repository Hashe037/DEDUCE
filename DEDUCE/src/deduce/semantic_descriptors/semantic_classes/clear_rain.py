from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class ClearRainDescriptor(SemanticDescriptor):
    """Semantic descriptor for clear vs rain weather"""
    
    @property
    def name(self) -> str:
        return "clear_rain"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'clear': 'clear weather',
            'rain': 'rain drizzle puddles', 
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} weather conditions"