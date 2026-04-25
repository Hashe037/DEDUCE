"""
Object Pose Semantic Descriptor
Describes different viewing angles and poses of objects
"""

from typing import Dict
from ..base import SemanticDescriptor


class ObjectPoseDescriptor(SemanticDescriptor):
    """Semantic descriptor for object pose/viewpoint categories"""
    
    @property
    def name(self) -> str:
        return "object_pose"
    
    def _get_default_categories(self) -> Dict[str, str]:
        """Define default object pose categories and their semantic descriptions"""
        return {
            'front_view': 'front-facing viewpoint',
            'side_view': 'side profile view',
            'back_view': 'rear view from behind',
            'three_quarter': 'three-quarter angle view',
            'top_down': 'overhead top-down view',
            'bottom_up': 'upward bottom-up perspective'
        }
    
    def _get_fallback_description(self, category: str) -> str:
        """Generate fallback description for unknown pose categories"""
        return f"{category} viewing angle"