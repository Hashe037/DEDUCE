"""
Semantic Descriptor Registry System
Provides registration and factory functionality for semantic descriptors
"""

from typing import Dict, Type, Any, List
from .base import SemanticDescriptor


class SemanticRegistry:
    """Registry for managing semantic descriptor classes and creation"""
    
    def __init__(self):
        self._descriptors: Dict[str, Type[SemanticDescriptor]] = {}
        self._register_builtin_descriptors()
    
    def _register_builtin_descriptors(self):
        """Register built-in semantic descriptors"""
        # Import and register built-in descriptors
        try:
            from .semantic_classes.time_of_day import TimeOfDayDescriptor
            self.register_descriptor("time_of_day", TimeOfDayDescriptor)
        except ImportError:
            pass
            
        try:
            from .semantic_classes.weather import WeatherDescriptor
            self.register_descriptor("weather", WeatherDescriptor)
        except ImportError:
            pass
            
        try:
            from .semantic_classes.object_pose import ObjectPoseDescriptor
            self.register_descriptor("object_pose", ObjectPoseDescriptor)
        except ImportError:
            pass
            
        try:
            from .semantic_classes.lighting import LightingDescriptor
            self.register_descriptor("lighting", LightingDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.day_night import DayNightDescriptor
            self.register_descriptor("day_night", DayNightDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.clear_rain import ClearRainDescriptor
            self.register_descriptor("clear_rain", ClearRainDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.noflare_flare import FlareDescriptor
            self.register_descriptor("noflare_flare", FlareDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.clear_snow import ClearSnowDescriptor
            self.register_descriptor("clear_snow", ClearSnowDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.clear_fog import ClearFogDescriptor
            self.register_descriptor("clear_fog", ClearFogDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.nodamage_damage import DamageDescriptor
            self.register_descriptor("nodamage_damage", DamageDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.sharp_blurry import SharpBlurryDescriptor
            self.register_descriptor("sharp_blurry", SharpBlurryDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.noveg_veg import VegetationDescriptor
            self.register_descriptor("noveg_veg", VegetationDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.density import DensityDescriptor
            self.register_descriptor("lowdense_highdense", DensityDescriptor)
        except ImportError:
            pass

        try:
            from .semantic_classes.nopeople_people import PeopleAmountDescriptor
            self.register_descriptor("nopeople_people", PeopleAmountDescriptor)

        except ImportError:
            pass
        
        try:
            from .semantic_classes.summer_winter import SummerWinterDescriptor
            self.register_descriptor("summer_winter", SummerWinterDescriptor)

        except ImportError:
            pass
    
    def register_descriptor(self, name: str, descriptor_class: Type[SemanticDescriptor]):
        """Register a semantic descriptor class"""
        if not issubclass(descriptor_class, SemanticDescriptor):
            raise ValueError(f"Descriptor class must inherit from SemanticDescriptor")
        
        self._descriptors[name] = descriptor_class
    
    def create_descriptor(self, name: str, config: Dict[str, Any], global_semantics: Dict[str, Any] = None) -> SemanticDescriptor:
        """Create a semantic descriptor instance"""
        # Check if it's a registered descriptor
        if name in self._descriptors:
            descriptor_class = self._descriptors[name]
            return descriptor_class(config, global_semantics)
        
        # If not registered, try to create a generic descriptor from config
        # This allows pure config-based descriptors
        try:
            from .generic import GenericDescriptor
            
            # Add the name to the config
            config_with_name = config.copy()
            config_with_name['name'] = name
            
            return GenericDescriptor(config_with_name, global_semantics)
        except Exception as e:
            raise ValueError(
                f"Unknown semantic descriptor: {name}. "
                f"Available: {list(self._descriptors.keys())}. "
                f"Could not create generic descriptor: {e}"
            )
    
    def list_descriptors(self) -> List[str]:
        """List available semantic descriptor names"""
        return list(self._descriptors.keys())
    
    def get_descriptor_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific descriptor"""
        if name not in self._descriptors:
            raise ValueError(f"Unknown semantic descriptor: {name}")
        
        descriptor_class = self._descriptors[name]
        
        # Create a temporary instance with empty config to get basic info
        try:
            temp_instance = descriptor_class({})
            categories = temp_instance.get_category_keys()
            return {
                'name': temp_instance.name,
                'class_name': descriptor_class.__name__,
                'categories': categories,
                'num_categories': len(categories),
                'docstring': descriptor_class.__doc__
            }
        except Exception as e:
            # Fallback: try to get basic info without full instantiation
            try:
                # Try to get default categories directly from class
                temp_instance = descriptor_class({})
                default_categories = temp_instance._get_default_categories()
                return {
                    'name': name,
                    'class_name': descriptor_class.__name__,
                    'categories': list(default_categories.keys()),
                    'num_categories': len(default_categories),
                    'error': f"Partial info only: {e}",
                    'docstring': descriptor_class.__doc__
                }
            except Exception as e2:
                return {
                    'name': name,
                    'class_name': descriptor_class.__name__,
                    'categories': [],
                    'num_categories': 0,
                    'error': f"Could not instantiate: {e2}",
                    'docstring': descriptor_class.__doc__
                }
    
    def validate_descriptor_config(self, name: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for a descriptor"""
        try:
            descriptor = self.create_descriptor(name, config)
            # Try to generate descriptions to validate
            descriptions = descriptor.get_descriptions()
            return len(descriptions) > 0
        except Exception:
            return False


# Decorator for registering semantic descriptors
def register_semantic_descriptor(name: str):
    """Decorator to register semantic descriptors"""
    def decorator(descriptor_class: Type[SemanticDescriptor]):
        # This would work with a global registry instance
        # For now, descriptors need to be manually registered in _register_builtin_descriptors
        descriptor_class._registry_name = name
        return descriptor_class
    return decorator


# Global registry instance for decorator usage
_global_registry = None

def get_global_registry() -> SemanticRegistry:
    """Get or create the global semantic registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = SemanticRegistry()
    return _global_registry