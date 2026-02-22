"""
Logging utilities for REXI.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rexi.config.settings import get_settings

def setup_logging(
    name: str = "rexi",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration."""
    
    settings = get_settings()
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or settings.log_level.upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file or settings.log_file:
        log_path = Path(log_file or settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

# Configure root logger
root_logger = setup_logging()
