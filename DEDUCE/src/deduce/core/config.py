from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional
import configparser
import json
import pdb

"""
Minimalistic Configuration Manager
"""

import os

class ConfigManager:
    """Simple configuration manager for INI files"""
    
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
    def get(self, section: str, fallback: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get section as dictionary"""
        if not self.config.has_section(section):
            return fallback or {}
        return dict(self.config.items(section))
    
    def get_list(self, section: str, key: str, fallback: List[str] = None) -> List[str]:
        """Get comma-separated value as list"""
        value = self.config.get(section, key, fallback='')
        if not value:
            return fallback or []
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def get_int(self, section: str, key: str, fallback: int = None) -> int:
        """Get integer value from config"""
        value = self.config.get(section, key, fallback=str(fallback) if fallback is not None else '')
        if not value:
            return fallback
        try:
            return int(value)
        except ValueError:
            return fallback

    def get_float(self, section: str, key: str, fallback: float = None) -> float:
        """Get float value from config"""
        value = self.config.get(section, key, fallback=str(fallback) if fallback is not None else '')
        if not value:
            return fallback
        try:
            return float(value)
        except ValueError:
            return fallback

    def get_bool(self, section: str, key: str, fallback: bool = None) -> bool:
        """Get boolean value from config"""
        value = self.config.get(section, key, fallback=str(fallback) if fallback is not None else '')
        if not value:
            return fallback
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_dataset_paths(self) -> Union[str, List[str]]:
        """Get dataset path(s) - returns single string or list for multiple paths"""
        dataset_config = self.get('DATASET', {})
        
        # Try 'paths' first (for multiple), then 'path' (for single)
        if 'paths' in dataset_config:
            paths = self.get_list('DATASET', 'paths')
            return paths if len(paths) > 1 else paths[0] if paths else None
        elif 'path' in dataset_config:
            return dataset_config['path']
        else:
            return None
        
    def get_labeled_folders(self) -> Optional[Dict[str, str]]:
        """Parse labeled_folders config option"""
        
        # Get the string value directly from the section
        labeled_str = self.config['DATASET'].get('labeled_folders', None)
        
        # Ensure it's a string and not empty
        # if labeled_str is None:
        #     return None
        if not isinstance(labeled_str, str) or not labeled_str.strip():
            return None
        
        # Parse into folders dict
        folders = {}
        for item in labeled_str.split(','):
            item = item.strip()
            if '=' in item:
                path, label = item.split('=', 1)
                folders[path.strip()] = label.strip()
        
        return folders if folders else None

    def is_labeled_dataset(self) -> bool:
        """Check if config specifies labeled dataset"""
        return self.get_labeled_folders() is not None

    def get_synthetic_data_config(self) -> Optional[Dict[str, Any]]:
        """Get synthetic data config, returns None if section is missing or path is empty"""
        if not self.config.has_section('SYNTHETIC_DATA'):
            return None
        path = self.config.get('SYNTHETIC_DATA', 'synthetic_data_path', fallback='').strip()
        if not path:
            return None
        return {
            'synthetic_data_path': path,
            'semantic_descriptor': self.config.get('SYNTHETIC_DATA', 'semantic_descriptor', fallback=None) or None,
            'original_label': self.config.get('SYNTHETIC_DATA', 'original_label', fallback=None) or None,
            'synthetic_label': self.config.get('SYNTHETIC_DATA', 'synthetic_label', fallback=None) or None,
        }