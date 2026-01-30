"""Logging configuration for the project."""

import os
import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_dir: str = "logs",
    log_file: str = "lulc_detection.log",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> logger:
    """
    Setup and configure logger.
    
    Args:
        log_dir: Directory to save log files
        log_file: Name of the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rotation: When to rotate log file
        retention: How long to keep old logs
        
    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
    )
    
    logger.add(
        log_path / log_file,
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=level,
    )
    
    logger.info(f"Logger initialized. Logs saved to: {log_path / log_file}")
    
    return logger


default_logger = setup_logger()
