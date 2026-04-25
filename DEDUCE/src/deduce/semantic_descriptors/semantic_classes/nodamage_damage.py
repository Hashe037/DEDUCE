from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class DamageDescriptor(SemanticDescriptor):
    """Semantic descriptor for no damage vs damage"""
    
    @property
    def name(self) -> str:
        return "nodamage_damage"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'no damage': 'intact standing whole',
            'damage': 'damaged rubble broken', 
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} damage conditions"