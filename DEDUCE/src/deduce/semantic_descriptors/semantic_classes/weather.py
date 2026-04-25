"""
Weather Semantic Descriptor
Describes different weather conditions and atmospheric states
"""

from typing import Dict
from ..base import SemanticDescriptor


class WeatherDescriptor(SemanticDescriptor):
    """Semantic descriptor for weather condition categories"""
    
    @property
    def name(self) -> str:
        return "weather"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default weather categories and their semantic descriptions"""
        return {
            'sunny': 'clear sunny weather',
            'cloudy': 'overcast cloudy sky',
            'rainy': 'rainy wet conditions',
            'foggy': 'foggy misty atmosphere',
            'stormy': 'stormy dramatic weather',
            'snowy': 'snowy winter conditions'
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown weather categories"""
        return f"{category} weather conditions"