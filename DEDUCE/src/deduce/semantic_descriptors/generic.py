from typing import Dict
from .base import SemanticDescriptor


class GenericDescriptor(SemanticDescriptor):
    """Generic semantic descriptor that uses config-only definitions"""
    
    def __init__(self, config: Dict[str, any], global_semantics: Dict[str, any] = None):
        # Store the name from config before calling parent init
        self._descriptor_name = config.get('name', 'generic')
        super().__init__(config, global_semantics)
    
    @property
    def name(self) -> str:
        return self._descriptor_name
    
    def _get_default_categories(self) -> Dict[str, str]:
        # Return empty dict - everything comes from config
        return {}
    
    def _get_fallback_description(self, category: str) -> str:
        return f"{category} semantic feature"