from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class PeopleAmountDescriptor(SemanticDescriptor):
    """Semantic descriptor for no damage vs damage"""
    
    @property
    def name(self) -> str:
        return "nopeople_people"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'no people': 'empty deserted unpopulated quiet',
            'people': 'crowd pedestrians populated busy people walking',
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} people conditions"