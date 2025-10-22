"""
Helper utilities for the optimization framework
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np


def compute_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """
    Compute hash of a file
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
    
    Returns:
        str: Hexadecimal hash string
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 2):
    """Save dictionary to JSON file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_model_size_mb(file_path: Path) -> float:
    """Get model file size in MB"""
    return file_path.stat().st_size / (1024 * 1024)


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def count_parameters(params_dict: Dict[str, np.ndarray]) -> int:
    """Count total parameters from parameter dictionary"""
    return sum(np.prod(p.shape) for p in params_dict.values())
