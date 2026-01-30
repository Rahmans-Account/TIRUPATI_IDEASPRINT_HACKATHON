"""Input/Output utility functions."""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object as pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """Save dictionary as JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_numpy(array: np.ndarray, filepath: str) -> None:
    """Save numpy array."""
    np.save(filepath, array)


def load_numpy(filepath: str) -> np.ndarray:
    """Load numpy array."""
    return np.load(filepath)


def ensure_dir(directory: str) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
