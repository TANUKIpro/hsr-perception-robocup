"""
Model path resolution utilities.

Provides functions for resolving model names to full paths,
prioritizing local cached models over remote downloads.
"""

from pathlib import Path
from typing import Optional


def resolve_model_path(model_name: str, verbose: bool = False) -> str:
    """
    Resolve a model name to a full path, prioritizing local cache.

    This function helps avoid unnecessary model downloads by checking
    local directories first. If the model is found locally, its full
    path is returned. Otherwise, the original model name is returned,
    allowing Ultralytics/SAM2 to handle the download.

    Search order:
    1. If model_name is an absolute path that exists -> return as-is
    2. /workspace/models/pretrained/{model_name}
    3. /workspace/models/finetuned/{model_name}
    4. Current working directory/{model_name}
    5. Not found -> return original model_name (triggers auto-download)

    Args:
        model_name: Model filename (e.g., "yolov8m.pt") or full path
        verbose: If True, print resolution information

    Returns:
        Full path to the model if found locally, otherwise the original model_name

    Examples:
        >>> resolve_model_path("yolov8m.pt")
        '/workspace/models/pretrained/yolov8m.pt'

        >>> resolve_model_path("/custom/path/model.pt")
        '/custom/path/model.pt'

        >>> resolve_model_path("yolov8x.pt")  # Not cached
        'yolov8x.pt'
    """
    # If it's an absolute path that exists, use it directly
    model_path = Path(model_name)
    if model_path.is_absolute() and model_path.exists():
        if verbose:
            print(f"Using absolute path: {model_name}")
        return model_name

    # Search paths in priority order
    search_paths = [
        Path("/workspace/models/pretrained") / model_name,
        Path("/workspace/models/finetuned") / model_name,
        Path.cwd() / model_name,
    ]

    for path in search_paths:
        if path.exists():
            if verbose:
                print(f"Found cached model: {path}")
            return str(path)

    # Model not found locally - return original name for auto-download
    if verbose:
        print(f"Model not in cache, will download: {model_name}")
    return model_name


def get_pretrained_model_path(model_name: str) -> Path:
    """
    Get the path for a pretrained model in the standard location.

    Args:
        model_name: Model filename (e.g., "yolov8m.pt")

    Returns:
        Path to /workspace/models/pretrained/{model_name}
    """
    return Path("/workspace/models/pretrained") / model_name


def get_finetuned_model_path(model_name: str) -> Path:
    """
    Get the path for a finetuned model in the standard location.

    Args:
        model_name: Model filename (e.g., "best.pt")

    Returns:
        Path to /workspace/models/finetuned/{model_name}
    """
    return Path("/workspace/models/finetuned") / model_name


def is_model_cached(model_name: str) -> bool:
    """
    Check if a model is available in the local cache.

    Args:
        model_name: Model filename to check

    Returns:
        True if model exists locally, False otherwise
    """
    resolved = resolve_model_path(model_name)
    return resolved != model_name and Path(resolved).exists()
