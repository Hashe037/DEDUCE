"""
Simple logging utilities for the semantic evaluation framework
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(level: str = 'INFO', 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup basic logging for the framework
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        
    Returns:
        Configured root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Basic format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(numeric_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def setup_logging_from_config(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging from configuration dictionary
    
    Args:
        config: Configuration dictionary with optional LOGGING section
        
    Returns:
        Configured logger
    """
    log_config = config.get('LOGGING', {})
    
    level = log_config.get('level', 'INFO')
    log_file = log_config.get('file', None)
    
    # Handle log directory
    if log_file and 'directory' in log_config:
        log_file = Path(log_config['directory']) / log_file
    
    return setup_logging(level=level, log_file=log_file)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module"""
    return logging.getLogger(name)