"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_config(config_name: str = "config.yaml") -> OmegaConf:
    """Load configuration using OmegaConf."""
    config_path = Path("config") / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def merge_configs(*configs: OmegaConf) -> OmegaConf:
    """Merge multiple configuration objects."""
    return OmegaConf.merge(*configs)


def get_full_config() -> OmegaConf:
    """Load and merge all configuration files."""
    main_config = load_config("config.yaml")
    model_config = load_config("model_config.yaml")
    paths_config = load_config("paths.yaml")
    
    full_config = merge_configs(main_config, model_config, paths_config)
    return full_config


def save_config(config: OmegaConf, save_path: str) -> None:
    """Save configuration to file."""
    OmegaConf.save(config, save_path)
