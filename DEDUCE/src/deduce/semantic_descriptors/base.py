from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class SemanticTemplate:
    """Template structure for semantic descriptions"""
    part_a_object: Optional[str] = None          # User input object
    part_b_scene: Optional[str] = None           # User input scene descriptor  
    part_c_semantic: str = ""                    # Required semantic descriptor
    part_d_additional: Optional[str] = None      # Optional additional descriptor

    # Minimalistic
    def format_description(self) -> str:
        """Format the complete semantic description following the template:
        'Image [of a part_a] [in the setting of part_b] with [part_c] [and part_d]'
        """
        description = "" #"Image"
        
        # Part A: Object (optional)
        if self.part_a_object:
            description += f"{self.part_a_object}"
        
        # Part B: Scene setting (optional)  
        if self.part_b_scene:
            description += f"{self.part_b_scene} "
        elif self.part_a_object:
            # If we have object but no scene, adjust grammar
            description += ""
            
        # Part C: Main semantic descriptor (required)
        description += f"{self.part_c_semantic}"
        
        # Part D: Additional descriptor (optional)
        if self.part_d_additional:
            description += f"{self.part_d_additional}"
            
        return description


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------



class SemanticDescriptor(ABC):
    """Abstract base class for semantic descriptors"""
    
    def __init__(self, config: Dict[str, Any], global_semantics: Dict[str, Any] = None):
        self.config = config
        self.global_semantics = global_semantics or {}

        # Priority: descriptor-specific > global > None
        self.part_a_object = (
            config.get('part_a_object') or 
            self.global_semantics.get('part_a_object')
        )
        self.part_b_scene = (
            config.get('part_b_scene') or 
            self.global_semantics.get('part_b_scene')
        )
        self.part_d_additional = (
            config.get('part_d_additional') or 
            self.global_semantics.get('part_d_additional')
        )
        
        # Build categories using the helper method
        self.categories = self._build_categories_from_config()
        
    @abstractmethod
    def _get_default_categories(self) -> Dict[str, str]:
        """Define the default semantic categories and their descriptions"""
        pass
    
    @abstractmethod  
    def _get_fallback_description(self, category: str) -> str:
        """Generate a fallback description for unknown categories"""
        pass
    
    def _build_categories_from_config(self) -> Dict[str, str]:
        """Build categories dictionary from config, with fallbacks to defaults"""
        try:
            # Get default categories from subclass
            default_categories = self._get_default_categories()
        except NotImplementedError:
            # Handle case where abstract method isn't implemented (e.g., during registry info gathering)
            return {}
        
        # Parse config descriptions
        config_descriptions = self.config.get('descriptions', '').split(',')
        config_descriptions = [desc.strip() for desc in config_descriptions if desc.strip()]
        
        # Parse config categories
        config_categories = self.config.get('categories', '').split(',')
        config_categories = [cat.strip() for cat in config_categories if cat.strip()]
        
        if config_categories and len(config_categories) == len(config_descriptions):
            # Use both configured categories and descriptions
            return dict(zip(config_categories, config_descriptions))
        elif config_categories:
            # Use configured categories with default descriptions (mapped)
            result = {}
            for cat in config_categories:
                if cat in default_categories:
                    result[cat] = default_categories[cat]
                else:
                    # Use fallback for unknown categories
                    try:
                        result[cat] = self._get_fallback_description(cat)
                    except NotImplementedError:
                        result[cat] = f"{cat} semantic feature"
            return result
        else:
            # Use all defaults
            return default_categories
    
    # Keep the old method for backward compatibility, but mark as deprecated
    def _define_categories(self) -> Dict[str, str]:
        """
        DEPRECATED: Use _get_default_categories() instead.
        This method is kept for backward compatibility.
        """
        return self._get_default_categories()
        
    def get_descriptions(self) -> List[str]:
        """Generate complete formatted descriptions for each category"""
        descriptions = []
        for category_key, part_c_desc in self.categories.items():
            template = SemanticTemplate(
                part_a_object=self.part_a_object,
                part_b_scene=self.part_b_scene, 
                part_c_semantic=part_c_desc,
                part_d_additional=self.part_d_additional
            )
            descriptions.append(template.format_description())
        return descriptions
    
    def get_aggregated_descriptions(self) -> Dict[str, List[str]]:
        """Generate multiple descriptions per category for prompt aggregation
        
        Returns:
            Dict mapping category keys to lists of description variants
        """
        # Parse comma-separated values for aggregation
        part_a_variants = self._parse_variants(self.part_a_object)
        part_b_variants = self._parse_variants(self.part_b_scene)
        part_d_variants = self._parse_variants(self.part_d_additional)
        
        aggregated = {}
        for category_key, part_c_desc in self.categories.items():
            variants = []
            
            # Generate all combinations if multiple variants exist
            for a in part_a_variants:
                for b in part_b_variants:
                    for d in part_d_variants:
                        template = SemanticTemplate(
                            part_a_object=a,
                            part_b_scene=b,
                            part_c_semantic=part_c_desc,
                            part_d_additional=d
                        )
                        variants.append(template.format_description())
            
            aggregated[category_key] = variants
        
        return aggregated

    def _parse_variants(self, value: Optional[str]) -> List[Optional[str]]:
        """Parse pipe-separated variants or return single value"""
        if not value:
            return [None]
        if '|' in value:
            return [v.strip() for v in value.split('|')]
        return [value]
            
    def get_category_keys(self) -> List[str]:
        """Get the category keys (used for prediction mapping)"""
        return list(self.categories.keys())
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Descriptor name"""
        pass