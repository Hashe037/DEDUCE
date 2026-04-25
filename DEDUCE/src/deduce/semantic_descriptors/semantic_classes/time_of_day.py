from typing import List, Dict, Any, Optional
from ..base import SemanticDescriptor

class TimeOfDayDescriptor(SemanticDescriptor):
    """Semantic descriptor for time of day categories"""
    
    @property
    def name(self) -> str:
        return "time_of_day"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default time of day categories and their semantic descriptions"""
        return {
            'dawn': 'low sun angle dawn light',
            'morning': 'bright morning sunlight', 
            'midday': 'harsh midday overhead sun',
            'afternoon': 'warm afternoon light',
            'evening': 'golden evening light',
            'night': 'dark nighttime scene'
        }

        # More descriptive case
        # return {
        #     'dawn': 'soft low-angle dawn light with cool bluish tones and long shadows',
        #     'morning': 'bright clear morning sunlight with crisp contrast and slightly warm tones', 
        #     'midday': 'harsh overhead midday sun with strong highlights and short shadows',
        #     'afternoon': 'warm afternoon light with golden tones and elongated shadows',
        #     'evening': 'rich golden-hour evening light with soft contrast and orange hues',
        #     'night': 'dark nighttime scene with deep shadows, artificial lights, and low visibility'
        # }

        # Less descriptive case
        # return {
        #     "dawn": "cool soft light, long shadows",
        #     "morning": "bright clear light, crisp contrast",
        #     "midday": "harsh overhead light, short shadows",
        #     "afternoon": "warm light, long angled shadows",
        #     "evening": "golden soft light, orange glow",
        #     "night": "dark scene, artificial lights"
        # }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown time of day categories"""
        return f"{category} lighting conditions"